#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/program/retrieval/adapter.h"
#include "core/device.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>

namespace ninfer::models::qwen3_5::detail {

void ProgramImpl::record_kvmem_placement(SequenceState& sequence, KvmemPlacementPhase phase,
                                         bool backend,
                                         const KVAddressSpaceStore::KVPlacementCounts& counts) {
    auto& diagnostics   = requests[sequence.lane].timings.kvmem;
    diagnostics.enabled = true;
    add_kvmem_placement(diagnostics.placement[static_cast<std::size_t>(phase)][backend ? 1 : 0],
                        counts.telemetry);
}

std::vector<std::uint32_t>
ProgramImpl::kvmem_restore_pages(const KVAddressSpaceStore& addresses, KVAddressSpaceHandle address,
                                 std::uint32_t frontier, const PreparedPromptData& prompt,
                                 std::span<const std::uint32_t> canonical_selection) const {
    const auto count = (frontier + kPagedKVPageSize - 1U) / kPagedKVPageSize;
    std::vector<std::uint32_t> selected;
    if (!addresses.active(address) && !canonical_selection.empty()) {
        // Canonical features retain scored history, not the complete executing
        // window. Restore and admit its recency/sink share as well: otherwise a
        // binding fits the smaller scored set and its first replay discovers
        // missing destinations after another lane has filled the shared pool.
        const auto budget =
            kvmem_window_pages +
            (&addresses == backend_kv_addresses.get() && speculative_backend == SpeculativeBackend::Mtp
                 ? (draft_window + kPagedKVPageSize - 1U) / kPagedKVPageSize
                 : 0U);
        selected = context_window_page_set(count, budget, canonical_selection,
                                           media_page_groups(prompt.vision_items),
                                           prompt.instruction_spans);
    } else if (addresses.active(address) || frontier == prompt.token_ids.size()) {
        for (std::uint32_t p = 0; p < count; ++p) {
            if (addresses.page_in_device_working_set(address, p)) { selected.push_back(p); }
        }
        if (!prompt.instruction_spans.empty()) {
            const auto budget = kvmem_window_pages +
                (&addresses == backend_kv_addresses.get() && speculative_backend == SpeculativeBackend::Mtp
                     ? (draft_window + kPagedKVPageSize - 1U) / kPagedKVPageSize : 0U);
            selected = context_window_page_set(count, budget, selected,
                                               media_page_groups(prompt.vision_items),
                                               prompt.instruction_spans);
        }
    } else {
        const auto groups = media_page_groups(prompt.vision_items);
        selected = groups.empty() && prompt.instruction_spans.empty()
                       ? prefill_window_page_set(count, 2U, kvmem_window_pages - 2U)
                       : context_window_page_set(count, kvmem_window_pages, {}, groups,
                                                 prompt.instruction_spans);
    }
    if (count && !std::binary_search(selected.begin(), selected.end(), count - 1U)) {
        selected.push_back(count - 1U);
    }
    return selected;
}

bool ProgramImpl::kvmem_same_prompt_source(const CheckpointState& source,
                                           const PreparedPromptData& prompt) const {
    // Native may resume an identical vision input at an image/rewrite boundary rather than
    // the complete prompt. Recomputing its shorter suffix must not change a completed query's
    // selected history. Use Native's complete typed identity (including media), not a session
    // key or just the reused token count; a changed/extended input needs its own retrieval.
    return source.kvmem_features && !source.kvmem_features->retrieved_pages.empty() &&
           source.identity && source.identity->ledger.size() >= prompt.token_ids.size() &&
           prefix_matches(prompt, source.identity->ledger, source.identity->prefix_identity,
                          static_cast<std::uint32_t>(prompt.token_ids.size()));
}

void ProgramImpl::initialize_kvmem(SequenceState& sequence, RequestControl::Prefill& staged,
                                   const KvmemPrefixFeatures* features, bool canonical) {
    if (!kvmem_window_pages) { return; }
    auto& sparse = kvmem_lanes_.at(sequence.lane);
    sparse.index.truncate_to(0);
    if (features) {
        sparse.index = features->index;
        sparse.index.truncate_to(staged.base);
    }
    if (sparse.index.total_tokens() < staged.base) {
        (void)sparse.index.append(staged.base - sparse.index.total_tokens());
    }
    sparse.capture_begin = features && features->key_frontier == staged.base
                               ? staged.base - staged.base % execution::kKvmemCaptureBlockTokens
                               : (staged.base + execution::kKvmemCaptureBlockTokens - 1U) /
                                     execution::kKvmemCaptureBlockTokens *
                                     execution::kKvmemCaptureBlockTokens;
    sparse.query.clear();
    sparse.query_count.clear();
    staged.canonical_retrieval    = canonical;
    sparse.retrieved_pages        = features && (canonical || staged.base == staged.prompt_tokens)
                                        ? features->retrieved_pages
                                        : std::vector<std::uint32_t>{};
    sparse.media_groups           = media_page_groups(staged.prompt.vision_items);
    sparse.instruction_spans      = staged.prompt.instruction_spans;
    sparse.query_checkpoint_valid = false;
    const auto query              = kvmem_query_span(staged.prompt_tokens, staged.base,
                                                     staged.prompt.retrieval_query, staged.prompt.vision_items);
    sparse.query_begin            = canonical ? staged.base : query.begin;
    sparse.query_end              = canonical ? staged.base : query.end;
    CUDA_CHECK(cudaMemsetAsync(sparse.query_sum.data, 0, sparse.query_sum.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(sparse.key_sums.data, 0, sparse.key_sums.bytes(), device.stream));
    if (features && features->key_frontier == staged.base) {
        if (features->key_sums.size() * sizeof(float) != sparse.key_sums.bytes()) {
            throw std::logic_error("cached KVMem capture geometry changed");
        }
        CUDA_CHECK(cudaMemcpyAsync(sparse.key_sums.data, features->key_sums.data(),
                                   sparse.key_sums.bytes(), cudaMemcpyHostToDevice, device.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    requests[sequence.lane].timings.kvmem.enabled = true;
    if (sparse.query_begin == staged.base && sparse.query_begin != 0 &&
        sparse.query_begin < sparse.query_end &&
        staged.prompt_tokens > kvmem_window_pages * kPagedKVPageSize) {
        copy_kvmem_query_state(sequence, false);
        sparse.query_checkpoint_valid = true;
    }
}

std::shared_ptr<const KvmemPrefixFeatures>
ProgramImpl::capture_kvmem_features(const SequenceState& sequence,
                                    std::shared_ptr<HostResidentCharge> charge) {
    if (!kvmem_window_pages) { return {}; }
    const auto& sparse = kvmem_lanes_.at(sequence.lane);
    auto features      = std::make_shared<KvmemPrefixFeatures>(
        KvmemPrefixFeatures{.key_frontier = sparse.index.total_tokens(), .index = sparse.index});
    features->index.truncate_to(sequence.text_kv_valid);
    if (features->index.total_tokens() < sequence.text_kv_valid) {
        (void)features->index.append(sequence.text_kv_valid - features->index.total_tokens());
    }
    features->retrieved_pages = sparse.retrieved_pages;
    features->metadata_charge =
        charge ? std::move(charge)
               : host_context_arena->charge_metadata(
                     features->index.descriptor_bytes() + sparse.key_sums.bytes() +
                     features->retrieved_pages.capacity() * sizeof(std::uint32_t) +
                     sizeof(KvmemPrefixFeatures));
    if (!features->metadata_charge) { throw std::bad_alloc(); }
    features->key_sums.resize(sparse.key_sums.bytes() / sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(features->key_sums.data(), sparse.key_sums.data,
                               sparse.key_sums.bytes(), cudaMemcpyDeviceToHost, device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    return features;
}

void ProgramImpl::roll_sparse_prefill_window(SequenceState& sequence, std::uint32_t,
                                             std::uint32_t cursor, std::uint32_t backend_valid,
                                             bool retrieved_history) {
    if (!kvmem_window_pages) { return; }
    const auto& sparse  = kvmem_lanes_.at(sequence.lane);
    const auto selected = retrieved_history ? std::span<const std::uint32_t>(sparse.retrieved_pages)
                                            : std::span<const std::uint32_t>{};
    const auto phase =
        retrieved_history ? KvmemPlacementPhase::Replay : KvmemPlacementPhase::Prefill;
    const auto place = [&](KVAddressSpaceStore& addresses, KVAddressSpaceHandle address,
                           std::uint32_t valid, std::uint32_t budget, bool backend) {
        const auto mapped    = addresses.mapped_pages(address);
        const auto committed = std::min(mapped, (valid + kPagedKVPageSize - 1U) / kPagedKVPageSize);
        if (committed <= budget && mapped <= budget) { return; }
        auto window = !sparse.media_groups.empty() || !sparse.instruction_spans.empty()
                          ? context_window_page_set(committed, budget, selected, sparse.media_groups,
                                                    sparse.instruction_spans)
                      : retrieved_history ? decode_window_page_set(committed, budget, selected)
                                          : prefill_window_page_set(committed, 2U, budget - 2U);
        append_prefill_growth_pages(window, committed, mapped);
        const auto counts = addresses.apply_device_placement(
            address, *host_kv_extents, window, device.transfer_stream,
            retrieved_history ? "replay" : "prefill");
        record_kvmem_placement(sequence, phase, backend, counts);
    };
    place(*text_kv_addresses, sequence.kv->text, cursor, kvmem_window_pages, false);
    if (sequence.kv->backend && speculative_backend == SpeculativeBackend::Mtp) {
        place(*backend_kv_addresses, *sequence.kv->backend, backend_valid,
              kvmem_window_pages + (draft_window + kPagedKVPageSize - 1U) / kPagedKVPageSize, true);
    }
}

void ProgramImpl::roll_sparse_decode_window(SequenceState& sequence) {
    if (!kvmem_window_pages) { return; }
    const auto& sparse = kvmem_lanes_.at(sequence.lane);
    const auto place   = [&](KVAddressSpaceStore& addresses, KVAddressSpaceHandle address,
                           std::uint32_t budget, bool backend) {
        const auto mapped = addresses.mapped_pages(address);
        if (mapped <= budget) { return; }
        auto selected =
            context_window_page_set(mapped, budget, sparse.retrieved_pages, sparse.media_groups,
                                    sparse.instruction_spans);
        const auto counts = addresses.apply_device_placement(address, *host_kv_extents, selected,
                                                               device.transfer_stream, "decode");
        record_kvmem_placement(sequence, KvmemPlacementPhase::Decode, backend, counts);
    };
    place(*text_kv_addresses, sequence.kv->text, kvmem_window_pages, false);
    if (sequence.kv->backend && speculative_backend == SpeculativeBackend::Mtp) {
        place(*backend_kv_addresses, *sequence.kv->backend,
              kvmem_window_pages + (draft_window + kPagedKVPageSize - 1U) / kPagedKVPageSize, true);
    }
}

void ProgramImpl::copy_kvmem_query_host(const SequenceState& sequence, HostContextAllocation& host,
                                        bool restore) {
    const auto& layout = state_images->host_layout();
    if (host.bytes() < layout.image_bytes || !host.data()) {
        throw std::logic_error("query checkpoint Host geometry changed");
    }
    const auto copy = [&](const Tensor& tensor, std::size_t offset) {
        if (offset > host.bytes() || tensor.bytes() > host.bytes() - offset) {
            throw std::logic_error("query checkpoint exceeds Host image");
        }
        CUDA_CHECK(cudaMemcpyAsync(restore ? tensor.data : host.data() + offset,
                                   restore ? host.data() + offset : tensor.data, tensor.bytes(),
                                   restore ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
                                   device.stream));
    };
    const auto& linear = *kvmem_query_checkpoint_;
    for (std::uint32_t layer = 0; layer < linear.layer_count(); ++layer) {
        copy(linear.conv_slot(layer, sequence.lane),
             layout.linear_conv.offset + layer * layout.linear_conv_layer_bytes);
        copy(linear.recurrent_slot(layer, sequence.lane),
             layout.linear_recurrent.offset + layer * layout.linear_recurrent_layer_bytes);
    }
    if (kvmem_draft_checkpoint_) {
        for (std::uint32_t layer = 0; layer < kvmem_draft_checkpoint_->layer_count(); ++layer) {
            const auto view = kvmem_draft_checkpoint_->layer_view(layer);
            copy(view.k.slice(3, sequence.lane, 1),
                 layout.dflash_local_k->offset + layer * layout.dflash_local_layer_bytes);
            copy(view.v.slice(3, sequence.lane, 1),
                 layout.dflash_local_v->offset + layer * layout.dflash_local_layer_bytes);
        }
    }
    copy(kvmem_lanes_.at(sequence.lane).query_tail_hidden, layout.continuation_hidden.offset);
    copy(kvmem_lanes_.at(sequence.lane).query_key_checkpoint, layout.image_bytes);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
}

void ProgramImpl::restore_kvmem_resume(SequenceState& sequence, ResumeStateImpl& saved) {
    if (!kvmem_window_pages || !saved.kvmem) { return; }
    auto& target                    = kvmem_lanes_.at(sequence.lane);
    const auto query_sum            = target.query_sum;
    const auto key_sums             = target.key_sums;
    const auto query_key_checkpoint = target.query_key_checkpoint;
    const auto query_tail_hidden    = target.query_tail_hidden;
    target                          = std::move(*saved.kvmem);
    target.query_sum                = query_sum;
    target.key_sums                 = key_sums;
    target.query_key_checkpoint     = query_key_checkpoint;
    target.query_tail_hidden        = query_tail_hidden;
    if (!saved.kvmem_features ||
        saved.kvmem_features->key_sums.size() * sizeof(float) != key_sums.bytes()) {
        throw std::logic_error("paused KVMem lost its partial capture");
    }
    CUDA_CHECK(cudaMemcpyAsync(key_sums.data, saved.kvmem_features->key_sums.data(),
                               key_sums.bytes(), cudaMemcpyHostToDevice, device.stream));
    // Query sums are required when a pause precedes completion of the unpublished probe.
    // The CPU query vector alone only describes a completed probe.
    if (saved.kvmem_query_sums.size() * sizeof(float) != query_sum.bytes()) {
        throw std::logic_error("paused KVMem query geometry changed");
    }
    CUDA_CHECK(cudaMemcpyAsync(query_sum.data, saved.kvmem_query_sums.data(), query_sum.bytes(),
                               cudaMemcpyHostToDevice, device.stream));
    if (saved.kvmem_query_host) { copy_kvmem_query_host(sequence, *saved.kvmem_query_host, true); }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
}

void ProgramImpl::copy_kvmem_query_state(SequenceState& sequence, bool restore) {
    auto& live           = state_images->linear();
    auto& snapshot       = *kvmem_query_checkpoint_;
    const auto selectors = state_selectors(sequence);
    const auto slot      = restore ? selectors.destination : selectors.source;
    auto& sparse         = kvmem_lanes_.at(sequence.lane);
    CUDA_CHECK(cudaMemcpyAsync(restore ? sequence.tail_hidden.data : sparse.query_tail_hidden.data,
                               restore ? sparse.query_tail_hidden.data : sequence.tail_hidden.data,
                               sparse.query_tail_hidden.bytes(), cudaMemcpyDeviceToDevice,
                               device.stream));
    if (restore) { sequence.tail_hidden_valid = true; }
    CUDA_CHECK(cudaMemcpyAsync(restore ? sparse.key_sums.data : sparse.query_key_checkpoint.data,
                               restore ? sparse.query_key_checkpoint.data : sparse.key_sums.data,
                               sparse.key_sums.bytes(), cudaMemcpyDeviceToDevice, device.stream));
    if (restore) { sparse.index.truncate_to(sparse.query_begin); }
    if (is_masked_draft_backend(speculative_backend)) {
        const auto query_begin = kvmem_lanes_.at(sequence.lane).query_begin;
        if (!dflash || !kvmem_draft_checkpoint_ || dflash->full ||
            (!restore && sequence.dflash_context_frontier != query_begin)) {
            throw std::logic_error("sparse draft checkpoint does not match its query frontier");
        }
        if (restore) {
            dflash->local.copy_slot_from(*kvmem_draft_checkpoint_, sequence.lane, slot,
                                         device.stream);
            sequence.dflash_context_frontier = query_begin;
            // No pending decode feature is valid at a prefill checkpoint. Replay
            // rebuilds local KV from target features; the probe publishes no draft.
            const auto pending = dflash->pending_features.slice(2, sequence.lane, 1);
            CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), device.stream));
        } else {
            kvmem_draft_checkpoint_->copy_slot_from(dflash->local, slot, sequence.lane,
                                                    device.stream);
        }
    }
    for (std::uint32_t layer = 0; layer < live.layer_count(); ++layer) {
        for (const bool recurrent : {false, true}) {
            const Tensor active =
                recurrent ? live.recurrent_slot(layer, slot) : live.conv_slot(layer, slot);
            const Tensor saved = recurrent ? snapshot.recurrent_slot(layer, sequence.lane)
                                           : snapshot.conv_slot(layer, sequence.lane);
            CUDA_CHECK(cudaMemcpyAsync(restore ? active.data : saved.data,
                                       restore ? saved.data : active.data, active.bytes(),
                                       cudaMemcpyDeviceToDevice, device.stream));
        }
    }
}

void ProgramImpl::rewind_kvmem_query_probe(SequenceState& sequence,
                                           RequestControl::Prefill& staged) {
    auto& sparse      = kvmem_lanes_.at(sequence.lane);
    const auto cursor = *staged.query_replay_cursor;
    copy_kvmem_query_state(sequence, true);
    text_kv_addresses->truncate_for_replay(sequence.kv->text, cursor, *host_kv_extents);
    if (sequence.kv->backend) {
        backend_kv_addresses->truncate_for_replay(*sequence.kv->backend, cursor, *host_kv_extents);
    }
    const auto historical_pages = (cursor + kPagedKVPageSize - 1U) / kPagedKVPageSize;
    sparse.retrieved_pages.erase(std::lower_bound(sparse.retrieved_pages.begin(),
                                                  sparse.retrieved_pages.end(), historical_pages),
                                 sparse.retrieved_pages.end());
    sequence.text_kv_valid = cursor;
    if (staged.prepare_mtp) { sequence.mtp_kv_valid = cursor; }
}

std::uint32_t ProgramImpl::advance_kvmem_query_replay(SequenceState& sequence,
                                                      RequestControl::Prefill& staged,
                                                      runtime::ExecutionTimingRecorder& timing,
                                                      std::uint32_t token_budget) {
    auto& sparse = kvmem_lanes_.at(sequence.lane);
    if (!sparse.query_checkpoint_valid || !staged.query_replay_cursor) {
        throw std::logic_error("sparse query replay has no private checkpoint");
    }
    auto& cursor = *staged.query_replay_cursor;
    if (cursor < sparse.query_begin || cursor >= staged.prompt_tokens) {
        throw std::logic_error("sparse query replay cursor is outside the prompt");
    }
    if (cursor == sparse.query_begin) {
        const auto* counts = requests[sequence.lane].sampling_host.token_counts;
        if (counts != nullptr && std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
            // Admission resets generated-token counts. The unpublished probe must not
            // change penalty history before replay samples the actual Begin token.
            TokenId probe_token      = 0;
            std::int32_t probe_count = 0;
            CUDA_CHECK(cudaMemcpy(&probe_token, io.token.data, sizeof(probe_token),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&probe_count, counts + probe_token, sizeof(probe_count),
                                  cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "KVMEM probe sample_history=%d\n", probe_count);
        }
        // Rewind the unpublished probe's complete causal state exactly once.
        rewind_kvmem_query_probe(sequence, staged);
        if (staged.vision) {
            staged.vision->retire_handoff();
            if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
                std::fprintf(stderr, "KVMEM vision replay begin=%u end=%u lane=%u\n", cursor,
                             staged.prompt_tokens, sequence.lane);
            }
        }
    }

    // One bounded execution unit. Placement republishes the truncated execution row
    // before this chunk; the caller synchronizes before returning control to Engine.
    {
        roll_sparse_prefill_window(sequence, staged.prompt_tokens, cursor,
                                   backend_kv_valid(sequence), true);
        const auto target = std::min(staged.prompt_tokens, cursor + prefill_chunk);
        text_kv_addresses->reserve_growth(
            sequence.kv->text,
            text_kv_addresses->growth_pages_for_tokens(sequence.kv->text, target));
        text_kv_addresses->ensure_mapped_to_tokens(sequence.kv->text, target, device.stream);
        if (sequence.kv->backend) {
            const auto backend_target = requests[sequence.lane].permit->backend_frontier;
            backend_kv_addresses->reserve_growth(*sequence.kv->backend,
                                                 backend_kv_addresses->growth_pages_for_tokens(
                                                     *sequence.kv->backend, backend_target));
            backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, backend_target,
                                                          device.stream);
        }
        const auto slots = state_selectors(sequence);
        execution::PrefillContext replay{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            nullptr,
            slots.source,
            slots.destination,
            staged.initial_mtp_extent,
            sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0,
            dflash_prefill_host_ingress,
            sequence.rope_delta};
        replay.prefill_gpu_timer = &prefill_gpu_timer_;
        // The probe determines selection, but the stored keys and checkpoints belong to
        // canonical replay. Rebuild K means from the saved partial block at query_begin;
        // leave the completed query sum unchanged.
        replay.execution.kvmem_q_sum         = static_cast<float*>(sparse.query_sum.data);
        replay.execution.kvmem_k_sum         = static_cast<float*>(sparse.key_sums.data);
        replay.execution.kvmem_capture_slots = kvmem_capture_slots_;
        if (dflash) { mark_workspace_usage(workspace_plan.dflash_context); }
        const auto count =
            std::min({prefill_chunk, token_budget == 0 ? prefill_chunk : token_budget,
                      staged.prompt_tokens - cursor});
        const auto split =
            std::upper_bound(staged.prompt.identity.rewrite_execution_frontiers.begin(),
                             staged.prompt.identity.rewrite_execution_frontiers.end(), cursor);
        auto frontier       = split == staged.prompt.identity.rewrite_execution_frontiers.end()
                                  ? std::optional<std::uint32_t>{}
                                  : std::optional<std::uint32_t>{*split};
        const auto& control = requests[sequence.lane];
        const auto capture  = control.next_capture < control.capture_groups.size()
                                  ? std::optional<std::uint32_t>(
                                       control.capture_groups[control.next_capture].frontier)
                                  : std::nullopt;
        if (capture && *capture > cursor && (!frontier || *capture < *frontier)) {
            frontier = capture;
        }
        timing.pause();
        const auto result =
            staged.vision
                ? execution::prefill_multimodal_chunk(replay, staged.prompt, *staged.vision, count,
                                                      frontier,
                                                      cursor + count == staged.prompt_tokens)
                : execution::prefill_text_chunk(replay, staged.prompt.token_ids, count, frontier,
                                                cursor + count == staged.prompt_tokens);
        timing.include(result.timing);
        timing.resume_post();
        auto& diagnostics = requests[sequence.lane].timings.kvmem;
        diagnostics.replay_tokens += result.processed_tokens;
        diagnostics.replay_execution_host_ns += result.timing.host_ns();
        diagnostics.replay_execution_device_wait_ns += result.timing.device_wait_ns;

        if (result.processed_tokens == 0 || result.processed_tokens > count) {
            throw std::logic_error("sparse query replay made invalid progress");
        }
        cursor += result.processed_tokens;
        sequence.text_kv_valid = cursor;
        if (staged.prepare_mtp) sequence.mtp_kv_valid = cursor;
        if (is_masked_draft_backend(speculative_backend)) {
            sequence.dflash_context_frontier = cursor;
        }
        commit_sequence_kv(sequence, cursor, backend_kv_valid(sequence));
        consume_kvmem_chunk_capture(sequence, cursor - result.processed_tokens, cursor);
        settle_state_fork(sequence);
        copy_tail(sequence, prefill_hidden.slice(
                                1, static_cast<std::int32_t>(result.processed_tokens) - 1, 1));
        sequence.tail_hidden_valid = true;
        if (capture && cursor == *capture && !result.finalized) {
            requests[sequence.lane].capture_pending = true;
        }
        if (cursor == staged.prompt_tokens && !result.finalized) {
            throw std::logic_error("sparse query replay did not replace the probe sample");
        }
        if (cursor == staged.prompt_tokens && std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
            std::fprintf(stderr, "KVMEM replay begin=%u end=%u tokens=%u\n", sparse.query_begin,
                         cursor, cursor - sparse.query_begin);
        }
        return result.processed_tokens;
    }
}

void ProgramImpl::consume_kvmem_chunk_capture(SequenceState& sequence, std::uint32_t chunk_begin,
                                              std::uint32_t chunk_end) {
    auto& diagnostics = requests[sequence.lane].timings.kvmem;
    ++diagnostics.key_capture_calls;
    KvmemHostWallTimer capture_timer(diagnostics.key_capture_host_wall_ns);
    auto& sparse                    = kvmem_lanes_.at(sequence.lane);
    const std::uint32_t kLayers     = sparse.index.layers();
    const std::uint32_t kKvWidth    = sparse.index.kv_heads() * sparse.index.head_dim();
    const std::uint32_t first_block = chunk_begin / execution::kKvmemCaptureBlockTokens;
    const std::uint32_t last_block  = chunk_end / execution::kKvmemCaptureBlockTokens;
    if (last_block <= first_block) {
        // No block completed; still advance the index so block numbering tracks tokens.
        while (sparse.index.total_tokens() < chunk_end) {
            const std::uint32_t step = std::min(execution::kKvmemCaptureBlockTokens,
                                                chunk_end - sparse.index.total_tokens());
            (void)sparse.index.append(step);
        }
        return;
    }
    while (sparse.index.total_tokens() < chunk_end) {
        const std::uint32_t step =
            std::min(execution::kKvmemCaptureBlockTokens, chunk_end - sparse.index.total_tokens());
        (void)sparse.index.append(step);
    }
    const std::uint32_t block_count = last_block - first_block;
    std::vector<float> sums(static_cast<std::size_t>(kLayers) * kvmem_capture_slots_ * kKvWidth);
    const auto copy_begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(sums.data(), sparse.key_sums.data, sums.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    diagnostics.key_capture_d2h_bytes += sums.size() * sizeof(float);
    diagnostics.key_capture_submit_wait_ns += kvmem_elapsed_ns(copy_begin);
    std::vector<float> mean(kKvWidth);
    for (std::uint32_t offset = 0; offset < block_count; ++offset) {
        const auto block = first_block + offset;
        const auto slot  = block % kvmem_capture_slots_;
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            const float* source =
                sums.data() +
                (static_cast<std::size_t>(layer) * kvmem_capture_slots_ + slot) * kKvWidth;
            for (std::uint32_t element = 0; element < kKvWidth; ++element) {
                mean[element] =
                    source[element] / static_cast<float>(execution::kKvmemCaptureBlockTokens);
            }
            if (block * execution::kKvmemCaptureBlockTokens >= sparse.capture_begin) {
                sparse.index.write_block_mean(block, layer, mean);
            }
            auto* device_sum =
                static_cast<float*>(sparse.key_sums.data) +
                (static_cast<std::size_t>(layer) * kvmem_capture_slots_ + slot) * kKvWidth;
            CUDA_CHECK(cudaMemsetAsync(device_sum, 0, kKvWidth * sizeof(float), device.stream));
        }
    }
}

void ProgramImpl::finalize_kvmem_query(SequenceState& sequence, std::uint32_t prompt_tokens) {
    auto& sparse                 = kvmem_lanes_.at(sequence.lane);
    const auto& attention        = *parameters.model.config().text.attention;
    const std::uint32_t kLayers  = sparse.index.layers();
    const std::uint32_t kQHeads  = attention.num_attention_heads;
    const std::uint32_t kKvHeads = attention.num_key_value_heads;
    const std::uint32_t kHeadDim = attention.head_dim;
    const auto query_tokens      = std::min(prompt_tokens, sparse.query_end) - sparse.query_begin;
    if (query_tokens == 0) { return; }
    auto& diagnostics = requests[sequence.lane].timings.kvmem;
    ++diagnostics.query_capture_calls;
    KvmemHostWallTimer capture_timer(diagnostics.query_capture_host_wall_ns);
    std::vector<float> sums(static_cast<std::size_t>(kLayers) * kQHeads * kHeadDim);
    const auto copy_begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(sums.data(), sparse.query_sum.data, sums.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    diagnostics.query_capture_d2h_bytes += sums.size() * sizeof(float);
    diagnostics.query_capture_submit_wait_ns += kvmem_elapsed_ns(copy_begin);
    sparse.query.assign(static_cast<std::size_t>(kLayers) * kKvHeads * kHeadDim, 0.0F);
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t head = 0; head < kQHeads; ++head) {
            const std::size_t source =
                (static_cast<std::size_t>(layer) * kQHeads + head) * kHeadDim;
            const std::size_t target =
                (static_cast<std::size_t>(layer) * kKvHeads + head / (kQHeads / kKvHeads)) *
                kHeadDim;
            for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
                sparse.query[target + dim] += sums[source + dim];
            }
        }
    }
    const float scale = 1.0F / static_cast<float>(query_tokens);
    for (float& value : sparse.query) { value *= scale; }
    sparse.query_count.assign(kLayers, query_tokens);
    double norm = 0;
    for (const auto value : sparse.query) { norm += static_cast<double>(value) * value; }
    if (!std::isfinite(norm) || norm == 0) {
        throw std::runtime_error("KVMem query capture is empty or nonfinite");
    }
    if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
        std::fprintf(stderr, "KVMEM capture query_tokens=%u blocks=%u query_norm=%.6g lane=%u\n",
                     query_tokens, sparse.index.block_count(), std::sqrt(norm), sequence.lane);
    }
    CUDA_CHECK(
        cudaMemsetAsync(sparse.query_sum.data, 0, sums.size() * sizeof(float), device.stream));
}

void ProgramImpl::apply_kvmem_retrieval_placement(SequenceState& sequence) {
    auto& sparse                         = kvmem_lanes_.at(sequence.lane);
    constexpr std::uint32_t kBlockTokens = 128U;
    const std::uint32_t mapped           = text_kv_addresses->mapped_pages(sequence.kv->text);
    if (mapped <= kvmem_window_pages) { return; }
    if (sparse.query_count.empty() || sparse.query_count[0] == 0) { return; }
    const std::uint32_t blocks = sparse.index.block_count();
    if (blocks == 0) { return; }
    const auto selection_begin = std::chrono::steady_clock::now();
    std::vector<float> scores(blocks, std::numeric_limits<float>::quiet_NaN());
    for (std::uint32_t block = 0; block < blocks; ++block) {
        if (!sparse.index.block(block).full) { continue; }
        (void)sparse.index.score(block, sparse.query, sparse.query_count, scores[block]);
    }
    detail::BlockSelectionConfig config;
    config.block_tokens = kBlockTokens;
    // The device budget is one window shared by sink, recency, and retrieval; the
    // selection must leave the recency and sink share inside the window or the
    // placement's promote side overflows the pool.
    const std::uint32_t recent_pages       = kvmem_window_pages / 4U;
    config.budget_blocks                   = (kvmem_window_pages - recent_pages - 2U) / 2U;
    config.sink_blocks                     = 1U;
    config.recent_blocks                   = 2U;
    const detail::BlockSelection selection = detail::select_blocks(sparse.index, scores, config);
    std::vector<std::uint32_t> pages       = detail::block_pages(selection.selected, kBlockTokens);
    pages.erase(std::remove_if(pages.begin(), pages.end(),
                               [mapped](std::uint32_t page) { return page >= mapped; }),
                pages.end());
    if (!sparse.media_groups.empty()) {
        // Normalize scored candidates into atomic media groups. Large latest media
        // may consume the usual recent share, but two boundary pages stay reserved.
        std::uint32_t media_budget = config.budget_blocks * 2U;
        for (const auto& group : sparse.media_groups) {
            if (group.begin < 2) { media_budget = std::max(media_budget, group.end); }
        }
        const auto& latest        = sparse.media_groups.back();
        std::uint32_t sink_extent = 2;
        for (const auto& group : sparse.media_groups) {
            if (group.begin < 2) { sink_extent = std::max(sink_extent, group.end); }
        }
        media_budget = std::max(media_budget,
                                sink_extent + (latest.begin < 2 ? 0 : latest.end - latest.begin));
        pages = context_window_page_set(mapped, media_budget, pages, sparse.media_groups, {}, false);
    }
    if (!sparse.instruction_spans.empty()) {
        // Keep scored features within their usual share unless mandatory context
        // is larger. Both the executing window and checkpoint restore apply this
        // same instruction policy; recency is filled only in the executing set.
        const auto required = context_window_page_set(mapped, kvmem_window_pages, {},
                                                       sparse.media_groups, sparse.instruction_spans,
                                                       false).size();
        const auto budget = std::max(config.budget_blocks * 2U,
                                     static_cast<std::uint32_t>(required));
        pages = context_window_page_set(mapped, budget, pages, sparse.media_groups,
                                         sparse.instruction_spans, false);
    }
    sparse.retrieved_pages           = pages;
    // The probe is unpublished. Place the complete first canonical replay window,
    // rather than adding an overlapping recent range at the probe's end. That
    // smaller set left holes which the next unit expanded after another binding
    // consumed the pool margin. The active row protects this complete set through
    // rewind; query-suffix replicas may be backed and retired, since replay rewrites
    // them. Scored features remain separate from the executing sink/recent window.
    const auto query_pages = (sparse.query_begin + kPagedKVPageSize - 1U) / kPagedKVPageSize;
    const auto replay_pages = sparse.query_checkpoint_valid ? query_pages : mapped;
    pages = context_window_page_set(replay_pages, kvmem_window_pages,
                                   sparse.retrieved_pages, sparse.media_groups,
                                   sparse.instruction_spans);
    auto& diagnostics = requests[sequence.lane].timings.kvmem;
    ++diagnostics.selection_calls;
    diagnostics.scored_blocks += selection.scored_blocks;
    diagnostics.selection_host_wall_ns += kvmem_elapsed_ns(selection_begin);
    const auto placement = text_kv_addresses->apply_device_placement(
        sequence.kv->text, *host_kv_extents, pages, device.transfer_stream, "retrieval");
    record_kvmem_placement(sequence, KvmemPlacementPhase::Retrieval, false, placement);
    if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
        std::fprintf(stderr,
                     "KVMEM retrieval scored=%u selected=%zu promoted=%u demoted=%u lane=%u\n",
                     selection.scored_blocks, pages.size(), placement.promoted, placement.demoted,
                     sequence.lane);
        std::uint32_t visible = 0, retained = 0;
        for (const auto& span : sparse.instruction_spans) {
            for (auto p = span.begin / kPagedKVPageSize;
                 p < std::min(replay_pages, static_cast<std::uint32_t>((span.begin + span.count + 63U) / 64U)); ++p) {
                ++visible;
                retained += std::binary_search(pages.begin(), pages.end(), p);
            }
        }
        std::fprintf(stderr, "KVMEM instructions lane=%u spans=%zu visible=%u retained=%u\n",
                     sequence.lane, sparse.instruction_spans.size(), visible, retained);
    }
    if (sequence.kv->backend && speculative_backend == SpeculativeBackend::Mtp) {
        const auto backend_mapped = backend_kv_addresses->mapped_pages(*sequence.kv->backend);
        const auto backend_replay_pages =
            sparse.query_checkpoint_valid ? query_pages : backend_mapped;
        const auto backend_budget =
            kvmem_window_pages + (draft_window + kPagedKVPageSize - 1U) / kPagedKVPageSize;
        pages = context_window_page_set(backend_replay_pages, backend_budget,
                                       sparse.retrieved_pages, sparse.media_groups,
                                       sparse.instruction_spans);
        const auto backend_placement = backend_kv_addresses->apply_device_placement(
            *sequence.kv->backend, *host_kv_extents, pages, device.transfer_stream, "retrieval");
        record_kvmem_placement(sequence, KvmemPlacementPhase::Retrieval, true, backend_placement);
    }
}

} // namespace ninfer::models::qwen3_5::detail
