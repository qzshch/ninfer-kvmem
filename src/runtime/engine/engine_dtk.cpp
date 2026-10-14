// Finite resident GLM/DTK implementation of the public Engine route.
#include "ninfer/engine.h"
#include "models/glm5_next/model.h"
#include "models/glm5_next/program.h"
#include "models/glm5_next/frontend.h"
#include "runtime/contract/sampling.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <algorithm>

namespace ninfer {
namespace {
using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point from) {
    return std::chrono::duration<double>(Clock::now() - from).count();
}

struct Job {
    std::vector<TokenId> prompt;
    PromptSummary summary;
    PromptPreparationStats preparation;
    RequestOptions options;
    ResolvedSamplingParameters sampling;
    std::promise<GenerationResult> result;
    std::atomic<bool> cancel{false};
    std::uint64_t id = 0;
    Clock::time_point submitted, deadline;
};
} // namespace

class PreparedPrompt::Impl {
public:
    std::vector<TokenId> tokens;
    PromptSummary summary;
    PromptPreparationStats prepare;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> p) noexcept : impl_(std::move(p)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return bool(impl_); }

class GenerationHandle::Impl {
public:
    std::shared_ptr<void> owner;
    std::shared_ptr<Job> job;
    std::future<GenerationResult> future;

    ~Impl() {
        if (job) job->cancel = true;
    }
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> p) noexcept : impl_(std::move(p)) {}

GenerationHandle::operator bool() const noexcept { return bool(impl_); }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ ? impl_->job->sampling : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (!impl_) throw std::logic_error("empty GenerationHandle");
    if (sink)
        throw std::invalid_argument("DTK resident GLM currently supports Aggregate consumers");
    auto p = std::move(impl_);
    while (p->future.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready)
        if (cancellation.requested()) p->job->cancel = true;
    return p->future.get();
}

class Engine::Impl {
public:
    EngineOptions options;
    std::array<std::unique_ptr<DeviceContext>, 3> devices;
    std::unique_ptr<models::glm5_next::Model> model;
    std::unique_ptr<models::glm5_next::Frontend> frontend;
    std::unique_ptr<models::glm5_next::Program> program;
    LoadSummary load;
    MemorySummary memory;
    RuntimeStats runtime;
    std::atomic<bool> available{true};
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::shared_ptr<Job>> pending;
    std::shared_ptr<Job> active;
    bool stopping = false;
    std::thread worker;
    std::uint64_t next_id = 1;

    explicit Impl(EngineOptions input) : options(std::move(input)) {
        if (options.device != 0 || options.max_concurrency != 1 || options.max_context < 1 ||
            options.max_context > 32768 || options.max_pending_requests < 1 ||
            options.use_cuda_graph || options.enable_vision ||
            options.purpose != EnginePurpose::Generation ||
            options.kv_cache != KvCacheStorage::BFloat16 ||
            options.kv_capacity.mode != KvCapacityMode::Explicit ||
            options.kv_capacity.explicit_tokens != options.max_context)
            throw std::invalid_argument(
                "DTK resident GLM requires devices 0,1,2; one lane; explicit BF16 "
                "KV=context<=32768; generation; graphs/vision disabled");
        if (options.speculative.backend != SpeculativeBackend::None &&
            (options.speculative.backend != SpeculativeBackend::DFlash2 ||
             options.speculative.draft_tokens != 7))
            throw std::invalid_argument("DTK GLM speculation requires its DFlash2 K7 companion");
        const auto start = Clock::now();
        int count        = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count != 3)
            throw std::invalid_argument("DTK resident GLM requires exactly three visible devices");
        std::array<DeviceContext*, 3> pointers{};
        for (int d = 0; d < 3; ++d) {
            devices[d]  = std::make_unique<DeviceContext>(d);
            pointers[d] = devices[d].get();
        }
        artifact::Reader reader(options.artifact_path);
        model =
            std::make_unique<models::glm5_next::Model>(reader, pointers, &options.startup_observer);
        frontend = std::make_unique<models::glm5_next::Frontend>(model->tokenizer());
        program  = std::make_unique<models::glm5_next::Program>(
            *model, pointers,
            models::glm5_next::ProgramOptions{static_cast<int>(options.max_context),
                                              options.kvmem_window_pages});
        load.architecture       = "Glm5NextForConditionalGeneration";
        load.model_name         = "GLM-5.3-Flash UD-IQ4_XS with DFlash2";
        load.cuda_sync_mode     = devices[0]->sync_mode();
        load.prefill_signature  = "DTK; three resident layer stages; one lane; greedy";
        load.load_seconds       = seconds(start);
        memory.device           = 0;
        memory.max_context      = options.max_context;
        memory.kv_capacity      = options.max_context;
        memory.kv_capacity_mode = KvCapacityMode::Explicit;
        memory.kv_cache         = KvCacheStorage::BFloat16;
        for (int d = 0; d < 3; ++d) {
            const auto& s = model->stats(d);
            load.artifact_bytes_read += s.read_bytes;
            load.host_to_device_bytes += s.h2d_bytes;
            load.upload_seconds += s.upload_seconds;
            load.peak_staging_bytes =
                std::max<std::uint64_t>(load.peak_staging_bytes, s.peak_staging_bytes);
            load.device_object_count += s.device_object_count;
            load.host_object_count += s.host_object_count;
            memory.weights.capacity_bytes += s.device_capacity_bytes;
            memory.sequence.capacity_bytes += program->stats().persistent_bytes[d];
        }
        memory.weights.used_bytes = memory.weights.peak_used_bytes = memory.weights.capacity_bytes;
        memory.sequence.used_bytes = memory.sequence.peak_used_bytes =
            memory.sequence.capacity_bytes;
        memory.workspace.capacity_bytes = 6ULL * 1024 * 1024 * 1024;
        runtime.lane_count              = 1;
        worker                          = std::thread([this] { run(); });
    }

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
            for (auto& j : pending) j->cancel = true;
            if (active) active->cancel = true;
        }
        ready.notify_all();
        if (worker.joinable()) worker.join();
        program.reset();
        frontend.reset();
        model.reset();
    }

    GenerationResult execute(const std::shared_ptr<Job>& job) {
        const auto started = Clock::now();
        GenerationResult result;
        result.engine_request_id       = job->id;
        result.prompt                  = job->summary;
        result.timings.prepare_seconds = job->preparation.seconds;
        result.engine_timing.queue_wait_seconds =
            std::chrono::duration<double>(started - job->submitted).count();
        result.speculative.backend = options.speculative.backend;
        result.speculative.enabled = options.speculative.backend == SpeculativeBackend::DFlash2;
        result.speculative.draft_window = result.speculative.enabled ? 7 : 0;
        result.speculative.accepted_per_position.assign(result.speculative.enabled ? 7 : 0, 0);
        program->reset();
        auto before    = program->stats();
        TokenId anchor = 0;
        for (auto token : job->prompt) {
            if (job->cancel) {
                result.finish_reason = FinishReason::Cancelled;
                break;
            }
            anchor = program->target(token);
            ++result.computed_prefill_tokens;
        }
        result.timings.prefill_seconds     = seconds(started);
        result.timings.first_token_seconds = result.timings.prefill_seconds;
        result.timings.prompt_wall_seconds = result.timings.prefill_seconds;
        const auto decode_start            = Clock::now();
        const auto limit =
            job->options.execution.requested_output_tokens
                ? job->options.execution.requested_output_tokens
                : std::min<std::uint32_t>(128, options.max_context - job->prompt.size());
        auto append = [&](TokenId token) {
            result.generated_token_ids.push_back(token);
            auto output =
                frontend->output(result.generated_token_ids, job->summary.starts_in_reasoning,
                                 job->options.output.preserve_special_tokens);
            result.content          = std::move(output.content);
            result.reasoning        = std::move(output.reasoning);
            result.reasoning_tokens = output.reasoning_tokens;
            const auto& stop        = job->options.stop;
            if ((stop.include_model_defaults && frontend->default_stop(token)) ||
                std::find(stop.token_ids.begin(), stop.token_ids.end(), token) !=
                    stop.token_ids.end()) {
                result.finish_reason = FinishReason::StopToken;
                return false;
            }
            for (const auto& s : stop.strings) {
                const auto at = result.content.find(s.text);
                if (at != std::string::npos) {
                    result.content.resize(at + (s.include_in_output ? s.text.size() : 0));
                    result.matched_stop_string = s.text;
                    result.finish_reason       = FinishReason::StopString;
                    return false;
                }
            }
            if (result.generated_token_ids.size() >= limit) {
                result.finish_reason = FinishReason::OutputLimit;
                return false;
            }
            return true;
        };
        bool keep = result.finish_reason != FinishReason::Cancelled && limit > 0;
        if (keep) keep = append(anchor);
        while (keep) {
            if (job->cancel) {
                result.finish_reason = FinishReason::Cancelled;
                break;
            }
            const int room = options.max_context - program->frontier();
            if (room < 1) {
                result.finish_reason = FinishReason::ContextCapacity;
                break;
            }
            if (!result.speculative.enabled) {
                anchor = program->target(anchor);
                keep   = append(anchor);
                continue;
            }
            const int width = std::min<int>(
                {8, room, static_cast<int>(limit - result.generated_token_ids.size())});
            const int frontier = program->frontier();
            const auto block   = program->propose(anchor, width);
            std::vector<TokenId> posterior(width);
            bool cancelled = false;
            for (int i = 0; i < width; ++i) {
                if (job->cancel) {
                    cancelled = true;
                    break;
                }
                posterior[i] = program->target(block[i], i + 1);
            }
            if (cancelled) {
                result.finish_reason = FinishReason::Cancelled;
                break;
            }
            int accepted = 0;
            while (accepted + 1 < width && block[accepted + 1] == posterior[accepted]) ++accepted;
            program->restore(frontier + accepted + 1, accepted + 1);
            ++result.speculative.rounds;
            result.speculative.drafted_tokens += width - 1;
            if (width == 1) ++result.speculative.fallback_steps;
            int delivered = 0;
            for (int i = 1; i <= accepted && keep; ++i) {
                keep = append(block[i]);
                ++delivered;
                ++result.speculative.accepted_per_position[i - 1];
            }
            result.speculative.accepted_tokens += delivered;
            if (!keep && delivered) program->restore(frontier + delivered, delivered);
            anchor = posterior[accepted];
            if (keep) keep = append(anchor);
        }
        result.timings.decode_seconds          = seconds(decode_start);
        result.timings.generation_wall_seconds = result.timings.decode_seconds;
        result.timings.total_seconds           = seconds(started);
        if (result.finish_reason == FinishReason::None)
            result.finish_reason = FinishReason::OutputLimit;
        result.timings.kvmem.enabled = options.kvmem_window_pages != 0;
        result.timings.kvmem.selection_calls =
            program->stats().kvmem_selections - before.kvmem_selections;
        return result;
    }

    void run() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock lock(mutex);
                ready.wait(lock, [&] { return stopping || !pending.empty(); });
                if (stopping && pending.empty()) return;
                job = pending.front();
                pending.pop_front();
                active = job;
            }
            bool executing = false;
            try {
                if (!available)
                    throw std::runtime_error(
                        "DTK Engine unavailable after a previous execution failure");
                if (job->deadline != Clock::time_point{} && Clock::now() >= job->deadline)
                    throw std::runtime_error("DTK request pending deadline exceeded");
                executing   = true;
                auto result = execute(job);
                {
                    std::lock_guard lock(mutex);
                    runtime.prompt_tokens += result.prompt.prompt_tokens;
                    runtime.generated_tokens += result.generated_token_ids.size();
                    runtime.computed_prefill_tokens += result.computed_prefill_tokens;
                    runtime.speculative_rounds += result.speculative.rounds;
                    runtime.speculative_draft_tokens += result.speculative.drafted_tokens;
                    runtime.speculative_accepted_tokens += result.speculative.accepted_tokens;
                    runtime.speculative_fallback_steps += result.speculative.fallback_steps;
                    memory.workspace.peak_used_bytes = 0;
                    for (auto peak : program->stats().workspace_peak)
                        memory.workspace.peak_used_bytes += peak;
                    memory.workspace_logical_peak_bytes = memory.workspace.peak_used_bytes;
                }
                job->result.set_value(std::move(result));
            } catch (...) {
                if (executing) available = false;
                job->result.set_exception(std::current_exception());
            }
            {
                std::lock_guard lock(mutex);
                active.reset();
            }
        }
    }
};

Engine::Engine(EngineOptions options) : impl_(std::make_shared<Impl>(std::move(options))) {}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> tokens, bool) const {
    if (!impl_) throw std::logic_error("moved Engine");
    if (tokens.empty() || tokens.size() > impl_->options.max_context)
        throw std::invalid_argument("GLM raw prompt is empty or exceeds context");
    for (auto t : tokens)
        if (t < 0 || t >= 154880)
            throw std::invalid_argument("GLM prompt token outside vocabulary");
    auto p                         = std::make_unique<PreparedPrompt::Impl>();
    p->summary.prompt_tokens       = tokens.size();
    p->summary.starts_in_reasoning = impl_->frontend->starts_reasoning(tokens);
    p->tokens                      = std::move(tokens);
    return PreparedPrompt(std::move(p));
}

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    if (!impl_) throw std::logic_error("moved Engine");
    const auto start         = Clock::now();
    auto p                   = prepare_tokens(impl_->frontend->prepare(input, control));
    p.impl_->prepare.seconds = seconds(start);
    return p;
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (!impl_) throw std::logic_error("moved Engine");
    return impl_->frontend->encode(text);
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    return prepare(std::move(input), control).summary().prompt_tokens;
}

ModelSamplingDefaults Engine::sampling_defaults() const { return {}; }

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode mode, GenerationObservationOptions observation,
                                Clock::time_point deadline) {
    if (!impl_ || !prompt.impl_)
        throw std::invalid_argument("DTK Engine requires a prepared prompt");
    if (mode != OutputConsumerMode::Aggregate || observation.phase_timings ||
        observation.live_timings || observation.prompt_progress || observation.scheduling ||
        observation.first_token)
        throw std::invalid_argument(
            "DTK qualification Engine supports Aggregate generation without live observations");
    auto sampling = runtime::resolve_sampling(sampling_defaults(), SamplingMode::NonThinking,
                                              options.execution.sampling);
    if (sampling.temperature != 0 || sampling.presence_penalty != 0 ||
        sampling.frequency_penalty != 0 || options.constraint || options.execution.thinking.budget)
        throw std::invalid_argument(
            "DTK GLM qualification profile supports unconstrained greedy generation");
    if (options.execution.requested_output_tokens >
        impl_->options.max_context - prompt.impl_->tokens.size())
        throw std::invalid_argument("DTK request output exceeds resident context");
    for (auto t : options.stop.token_ids)
        if (t < 0 || t >= 154880) throw std::invalid_argument("DTK stop token outside vocabulary");
    for (const auto& stop : options.stop.strings)
        if (stop.text.empty() || stop.channel != OutputChannel::Content)
            throw std::invalid_argument("DTK stop strings require nonempty content text");
    auto job         = std::make_shared<Job>();
    job->prompt      = std::move(prompt.impl_->tokens);
    job->summary     = prompt.impl_->summary;
    job->preparation = prompt.impl_->prepare;
    job->options     = std::move(options);
    job->sampling    = sampling;
    job->submitted   = Clock::now();
    job->deadline =
        deadline == Clock::time_point{} && impl_->options.pending_timeout_ms
            ? job->submitted + std::chrono::milliseconds(impl_->options.pending_timeout_ms)
            : deadline;
    auto handle    = std::make_unique<GenerationHandle::Impl>();
    handle->owner  = impl_;
    handle->job    = job;
    handle->future = job->result.get_future();
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->stopping || !impl_->available ||
            impl_->pending.size() >= impl_->options.max_pending_requests)
            throw std::runtime_error("DTK Engine queue unavailable or full");
        job->id = impl_->next_id++;
        impl_->pending.push_back(job);
    }
    impl_->ready.notify_one();
    return GenerationHandle(std::move(handle));
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    if (sink) throw std::invalid_argument("DTK qualification Engine requires Aggregate output");
    return submit(std::move(prompt), std::move(options)).wait(nullptr, cancellation);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId>, std::uint32_t) {
    throw std::invalid_argument("DTK GLM CausalScoring is not implemented");
}

const EngineOptions& Engine::options() const {
    if (!impl_) throw std::logic_error("moved Engine");
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (!impl_) throw std::logic_error("moved Engine");
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (!impl_) throw std::logic_error("moved Engine");
    std::lock_guard lock(impl_->mutex);
    return impl_->memory;
}

RuntimeStats Engine::runtime_stats() const {
    if (!impl_) throw std::logic_error("moved Engine");
    std::lock_guard lock(impl_->mutex);
    return impl_->runtime;
}

MediaCacheSummary Engine::media_cache_summary() const { return {}; }

bool Engine::is_available() const { return impl_ && impl_->available; }

void Engine::reset_memory_peaks() noexcept {
    if (impl_) {
        std::lock_guard lock(impl_->mutex);
        impl_->memory.workspace.peak_used_bytes = 0;
    }
}
} // namespace ninfer
