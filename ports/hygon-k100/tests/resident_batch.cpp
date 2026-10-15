
#include "core/decode_graph.h"
#include "core/indexed_transfer.h"
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sparse_latent_attention.h"
#include "kvmem/block_retrieval.h"
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace ninfer;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

unsigned short bf(float x) {
    auto b = std::bit_cast<unsigned>(x);
    return (b + 0x7fff + ((b >> 16) & 1)) >> 16;
}

float ub(unsigned short x) { return std::bit_cast<float>(static_cast<unsigned>(x) << 16); }

void attention_transition(DeviceContext& device, bool temporal = false, int mask_mode = 0, bool physical_bf16 = false) {
    constexpr int cap = 129, lanes = 3, pools_count = (cap + 3) / 4, count = 2051;
    const std::array<int, lanes> slots = temporal ? std::array{2, 2, 2} : std::array{2, 0, 1};
    const std::array<int, lanes> positions = temporal ? std::array{126, 127, 128} : std::array{3, 6, 128};
    constexpr int pages = (cap + 63) / 64;
    std::vector<unsigned char> allowed(lanes * pages);
    if (mask_mode == 1)
        for (int row = 0; row < lanes; ++row) allowed[row * pages] = 1;
    std::vector<unsigned short> keys(lanes * cap * 128), gates(keys.size()),
        pools(lanes * pools_count * 128), latent(lanes * cap * 512);
    std::vector<float> bias(512), query(lanes * 4096), weights(lanes * 32), q(lanes * 64 * 512),
        newkey(lanes * 128), newgate(newkey.size()), newlatent(lanes * 512);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i]  = bf(std::sin(i * .134) * .2);
        gates[i] = bf(std::cos(i * .119) * .7);
    }
    for (std::size_t i = 0; i < latent.size(); ++i) latent[i] = bf(std::sin(i * .077) * .25);
    for (int i = 0; i < 512; ++i) bias[i] = std::cos(i * .137) * .3;
    for (std::size_t i = 0; i < query.size(); ++i) query[i] = ub(bf(std::sin(i * .031) * .3));
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = ub(bf(std::cos(i * .037) * .2));
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin(i * .271) * .3;
    // Full naive FP64 pool oracle, with the explicit intermediate BF16 casts.
    for (int slot = 0; slot < lanes; ++slot)
        for (int pool = 0; pool < cap / 4; ++pool)
            for (int c = 0; c < 128; ++c) {
                double a[4], maximum = -INFINITY, total = 0, value = 0;
                for (int j = 0; j < 4; ++j) {
                    a[j]    = double(ub(gates[(slot * cap + pool * 4 + j) * 128 + c])) +
                              bias[j * 128 + c];
                    maximum = std::max(maximum, a[j]);
                }
                for (auto& v : a) {
                    v = std::exp(v - maximum);
                    total += v;
                }
                for (int j = 0; j < 4; ++j)
                    value += ub(
                        bf(ub(bf(a[j] / total)) * ub(keys[(slot * cap + pool * 4 + j) * 128 + c])));
                pools[(slot * pools_count + pool) * 128 + c] = bf(value);
            }
    for (int row = 0; row < lanes; ++row) {
        for (int c = 0; c < 128; ++c) {
            newkey[row * 128 + c]  = ub(keys[(slots[row] * cap + positions[row]) * 128 + c]);
            newgate[row * 128 + c] = ub(gates[(slots[row] * cap + positions[row]) * 128 + c]);
        }
        for (int c = 0; c < 512; ++c)
            newlatent[row * 512 + c] = ub(latent[(slots[row] * cap + positions[row]) * 512 + c]);
    }
    auto expected_pools = pools;
    auto expected_latent = latent;
    if (temporal) {
        auto expected_keys = keys, expected_gates = gates;
        for (int row = 0; row < lanes; ++row) {
            for (int c = 0; c < 128; ++c) {
                newkey[row * 128 + c] = ub(bf(std::cos((row * 128 + c) * .117) * .4));
                newgate[row * 128 + c] = ub(bf(std::sin((row * 128 + c) * .131) * .8));
                expected_keys[(slots[row] * cap + positions[row]) * 128 + c] = bf(newkey[row * 128 + c]);
                expected_gates[(slots[row] * cap + positions[row]) * 128 + c] = bf(newgate[row * 128 + c]);
            }
            for (int c = 0; c < 512; ++c) {
                newlatent[row * 512 + c] = ub(bf(std::cos((row * 512 + c) * .063) * .3));
                expected_latent[(slots[row] * cap + positions[row]) * 512 + c] = bf(newlatent[row * 512 + c]);
            }
        }
        // Position 127 completes pool31 using the newly written position126.
        // This detects a missing inter-kernel dependency between append and pool.
        for (int c = 0; c < 128; ++c) {
            double a[4], maximum = -INFINITY, total = 0, value = 0;
            for (int j = 0; j < 4; ++j) {
                a[j] = ub(expected_gates[(slots[0] * cap + 124 + j) * 128 + c]) + double(bias[j * 128 + c]);
                maximum = std::max(maximum, a[j]);
            }
            for (auto& x : a) { x = std::exp(x - maximum); total += x; }
            for (int j = 0; j < 4; ++j)
                value += ub(bf(ub(bf(a[j] / total)) * ub(expected_keys[(slots[0] * cap + 124 + j) * 128 + c])));
            expected_pools[(slots[0] * pools_count + 31) * 128 + c] = bf(value);
        }
    }
    DeviceBuffer dk(keys.size() * 2), dg(gates.size() * 2), dp(pools.size() * 2),
        dl(latent.size() * 2), db(bias.size() * 4), dq(query.size() * 4), dw(weights.size() * 4),
        dquery(q.size() * 4), dnk(newkey.size() * 4), dng(newgate.size() * 4),
        dnl(newlatent.size() * 4), ds(lanes * 4), dpos(lanes * 4), scores(lanes * 8192 * 4),
        ids(lanes * count * 4), prob(lanes * 64 * count * 4), out(q.size() * 4),
        allowed_device(allowed.size());
    dk.copy_from_host(keys.data(), dk.bytes);
    dg.copy_from_host(gates.data(), dg.bytes);
    dp.copy_from_host(pools.data(), dp.bytes);
    dl.copy_from_host(latent.data(), dl.bytes);
    db.copy_from_host(bias.data(), db.bytes);
    dq.copy_from_host(query.data(), dq.bytes);
    dw.copy_from_host(weights.data(), dw.bytes);
    dquery.copy_from_host(q.data(), dquery.bytes);
    dnk.copy_from_host(newkey.data(), dnk.bytes);
    dng.copy_from_host(newgate.data(), dng.bytes);
    dnl.copy_from_host(newlatent.data(), dnl.bytes);
    ds.copy_from_host(slots.data(), ds.bytes);
    dpos.copy_from_host(positions.data(), dpos.bytes);
    allowed_device.copy_from_host(allowed.data(), allowed.size());
    if (physical_bf16) {
        const auto upload_words = [&](DeviceBuffer& dst, const std::vector<float>& values) {
            std::vector<unsigned short> words; for (auto v : values) words.push_back(bf(v));
            dst.copy_from_host(words.data(), words.size() * 2);
        };
        upload_words(dq, query); upload_words(dnk, newkey);
        upload_words(dng, newgate); upload_words(dnl, newlatent);
    }
    DeviceBuffer selection_counts(lanes * 2 * sizeof(int) + 32); selection_counts.fill(0xa5);
    auto* counts = reinterpret_cast<int*>(static_cast<std::byte*>(selection_counts.p) + 16);
    DecodeGraphDefinition definition;
    definition.capture(device.stream, [&] {
        const auto index = [&]<class T>() {
        ops::resident_index_append(
            static_cast<T*>(dnl.p), static_cast<T*>(dnk.p), static_cast<T*>(dng.p),
            static_cast<float*>(db.p), static_cast<unsigned short*>(dl.p),
            static_cast<unsigned short*>(dk.p), static_cast<unsigned short*>(dg.p),
            static_cast<unsigned short*>(dp.p), static_cast<int*>(ds.p), static_cast<int*>(dpos.p),
            cap, lanes, device.stream);
        ops::resident_pool_select(static_cast<T*>(dq.p), static_cast<float*>(dw.p),
                                  static_cast<unsigned short*>(dp.p),
                                  mask_mode ? static_cast<unsigned char*>(allowed_device.p) : nullptr,
                                  static_cast<float*>(scores.p), static_cast<int*>(ids.p),
                                  static_cast<int*>(ds.p), static_cast<int*>(dpos.p), cap, 256,
                                  lanes, device.stream, counts);
        };
        if (physical_bf16) index.template operator()<unsigned short>();
        else index.template operator()<float>();
        ops::resident_sparse_attention(static_cast<float*>(dquery.p),
                                       static_cast<unsigned short*>(dl.p), static_cast<int*>(ids.p),
                                       static_cast<float*>(prob.p), static_cast<float*>(out.p),
                                       static_cast<int*>(ds.p), cap, lanes, device.stream, 64, counts);
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    graph.launch(device.stream);
    device.synchronize();
    std::vector<int> count_words(lanes * 2 + 8);
    selection_counts.copy_to_host(count_words.data(), selection_counts.bytes);
    for (int i = 0; i < 4; ++i)
        require(count_words[i] == static_cast<int>(0xa5a5a5a5u) && count_words[count_words.size()-1-i] == static_cast<int>(0xa5a5a5a5u), "selection count guards");
    std::vector<float> actual_probability(lanes * 64 * count);
    prob.copy_to_host(actual_probability.data(), prob.bytes);
    std::vector<int> actual_ids(lanes * count);
    ids.copy_to_host(actual_ids.data(), ids.bytes);
    std::vector<unsigned short> actual_pools(pools.size());
    dp.copy_to_host(actual_pools.data(), dp.bytes);
    const int pool_index = temporal ? (slots[1] * pools_count + 31) : slots[0] * pools_count;
    for (int c = 0; c < 128; ++c)
        require(std::abs(ub(actual_pools[pool_index * 128 + c]) -
                         ub(expected_pools[pool_index * 128 + c])) <=
                    .008 * (1 + std::abs(ub(expected_pools[pool_index * 128 + c]))),
                "batched pool append FP64 cast oracle");
    std::vector<float> actual(q.size());
    out.copy_to_host(actual.data(), out.bytes);
    // Ranking oracle consumes the represented pool operand actually published;
    // append was independently checked above, score/attention remain unfused FP64.
    for (int row = 0; row < lanes; ++row) {
        const int visible = positions[row] + 1, complete = visible / 4;
        std::vector<double> rank(complete);
        std::vector<int> order(complete);
        std::iota(order.begin(), order.end(), 0);
        if (mask_mode)
            std::erase_if(order, [&](int pool) { return !allowed[row * pages + pool / 16]; });
        for (int pool = 0; pool < complete; ++pool)
            for (int h = 0; h < 32; ++h) {
                double dot = 0;
                for (int c = 0; c < 128; ++c)
                    dot += double(query[row * 4096 + h * 128 + c]) *
                           ub(actual_pools[(slots[row] * pools_count + pool) * 128 + c]);
                rank[pool] += std::max(dot * double(.08838834764831844f), 0.) *
                              weights[row * 32 + h] * double(.17677669529663688f);
            }
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return rank[a] > rank[b]; });
        std::vector<int> selected(count, -1);
        for (int j = 0; j < static_cast<int>(order.size()); ++j)
            for (int c = 0; c < 4; ++c) selected[j * 4 + c] = order[j] * 4 + c;
        for (int c = 0; c < visible % 4; ++c) selected[2048 + c] = complete * 4 + c;
        const auto prefix = std::count_if(selected.begin(), selected.begin() + 2048, [](int id) { return id >= 0; });
        require(count_words[4 + row * 2] == prefix && count_words[5 + row * 2] == visible % 4,
                "device selection extents disagree with independent visible/masked candidates");
        for (int h = 0; h < 64; ++h) for (int j = 0; j < count; ++j)
            if (selected[j] < 0) require(actual_probability[(row * 64 + h) * count + j] == 0.f,
                                        "inactive probability retained stale storage");
        for (int j = 0; j < count; ++j)
            require(actual_ids[row * count + j] == selected[j],
                    "batched pool ranking/tails FP64 oracle");
        for (int h = 0; h < 64; ++h) {
            std::vector<double> probability(count);
            double maximum = -INFINITY, total = 0;
            for (int j = 0; j < count; ++j)
                if (selected[j] >= 0) {
                    double dot = 0;
                    for (int c = 0; c < 512; ++c)
                        dot += double(q[(row * 64 + h) * 512 + c]) *
                               ub(expected_latent[(slots[row] * cap + selected[j]) * 512 + c]);
                    probability[j] = dot / 16.;
                    maximum        = std::max(maximum, probability[j]);
                }
            for (int j = 0; j < count; ++j)
                if (selected[j] >= 0) {
                    probability[j] = std::exp(probability[j] - maximum);
                    total += probability[j];
                }
            for (int c = 0; c < 512; ++c) {
                double ideal = 0, absolute = 0;
                for (int j = 0; j < count; ++j)
                    if (selected[j] >= 0) {
                        double term = probability[j] / total *
                                      ub(expected_latent[(slots[row] * cap + selected[j]) * 512 + c]);
                        ideal += term;
                        absolute += std::abs(term);
                    }
                require(std::abs(actual[(row * 64 + h) * 512 + c] - ideal) <=
                            2.e-6 + 5.e-5 * absolute,
                        "ragged batched sparse attention complete FP64 oracle");
            }
        }
    }
}

int main() {
    try {
        DeviceContext device(0);
        attention_transition(device);
        attention_transition(device, true, 1, true);
        attention_transition(device, false, 0, true);
        attention_transition(device, true);
        attention_transition(device, true, 1);
        attention_transition(device, true, 2);
        constexpr int capacity = 513, lanes = 3, blocks = (capacity + 127) / 128,
                      pages = (capacity + 63) / 64;
        std::vector<unsigned short> keys(lanes * capacity * 128);
        for (std::size_t i = 0; i < keys.size(); ++i)
            keys[i] = bf(std::sin(float(i % 10007) * .123f) * .25f);
        DeviceBuffer dk(keys.size() * 2), dm(lanes * blocks * 128 * 4), dq(lanes * 4096 * 4),
            da(lanes * pages), ds(lanes * 4), dp(lanes * 4);
        dk.copy_from_host(keys.data(), dk.bytes);
        std::array<int, lanes> slots{2, 0, 1};
        ds.copy_from_host(slots.data(), ds.bytes);
        for (bool physical : {false, true}) {
        DecodeGraphDefinition definition;
        definition.capture(device.stream, [&] {
            const auto retrieve = [&]<class T>() {
            ops::resident_retrieval(static_cast<T*>(dq.p), static_cast<unsigned short*>(dk.p),
                                    static_cast<float*>(dm.p), static_cast<unsigned char*>(da.p),
                                    static_cast<int*>(ds.p), static_cast<int*>(dp.p), capacity,
                                    lanes, 3, device.stream);
            };
            if (physical) retrieve.template operator()<unsigned short>();
            else retrieve.template operator()<float>();
        });
        DecodeGraphExecutable graph;
        graph.instantiate(definition);
        graph.upload(device.stream);
        device.synchronize();
        for (int trial = 0; trial < 5; ++trial) {
            std::array<int, lanes> positions{127 + trial * 64, 256 + trial * 32, 511 - trial * 128};
            std::rotate(slots.begin(), slots.begin() + 1, slots.end());
            std::vector<float> means(lanes * blocks * 128), query(lanes * 4096);
            for (int slot = 0; slot < lanes; ++slot)
                for (int b = 0; b < 4; ++b)
                    for (int c = 0; c < 128; ++c) {
                        float sum = 0;
                        for (int j = 0; j < 128; ++j)
                            sum += ub(keys[(slot * capacity + b * 128 + j) * 128 + c]);
                        means[(slot * blocks + b) * 128 + c] = sum / 128.f;
                    }
            for (std::size_t i = 0; i < query.size(); ++i)
                query[i] = trial == 4 ? 0.f : float(std::cos(double(i + trial) * .034) * .125);
            dm.copy_from_host(means.data(), dm.bytes);
            if (physical) {
                std::vector<unsigned short> words;
                for (auto& v : query) { words.push_back(bf(v)); v = ub(words.back()); }
                dq.copy_from_host(words.data(), words.size() * 2);
            } else dq.copy_from_host(query.data(), dq.bytes);
            ds.copy_from_host(slots.data(), ds.bytes);
            dp.copy_from_host(positions.data(), dp.bytes);
            graph.launch(device.stream);
            device.synchronize();
            std::vector<unsigned char> actual(da.bytes);
            da.copy_to_host(actual.data(), actual.size());
            for (int row = 0; row < lanes; ++row) {
                kvmem::RetrievalIndex index(128, 1, 1, 128);
                index.append(positions[row] + 1);
                for (unsigned b = 0; b < index.block_count(); ++b)
                    if (index.block(b).full)
                        index.write_block_mean(
                            b, 0, {means.data() + (slots[row] * blocks + b) * 128, 128});
                std::array<float, 128> q{};
                for (int c = 0; c < 128; ++c)
                    for (int h = 0; h < 32; ++h) q[c] += query[row * 4096 + h * 128 + c];
                std::vector<float> rank(index.block_count(), NAN);
                const std::array<unsigned, 1> count{1};
                for (unsigned b = 0; b < index.block_count(); ++b)
                    if (index.block(b).full)
                        require(index.score(b, q, count, rank[b]), "oracle unscored full block");
                auto selected = kvmem::select_blocks(index, rank, {128, 3, 1, 1, 0});
                std::vector<unsigned char> expected(pages);
                for (auto b : selected.selected)
                    for (int j = 0; j < 2; ++j)
                        if (b * 2 + j < pages) expected[b * 2 + j] = 1;
                for (int p = 0; p < pages; ++p)
                    require(actual[row * pages + p] == expected[p],
                            "device retrieval differs from independent FP64 CPU contract");
            }
            std::vector<float> output(means.size());
            dm.copy_to_host(output.data(), dm.bytes);
            require(output == means, "ordered FP32 block means changed");
        }
        // Exact raw gather/scatter, nonidentity lanes and protected inter-row bytes.
        constexpr int width = 257, stride = 300;
        std::vector<unsigned char> source(lanes * stride, 0xa5), compact(lanes * width);
        for (int l = 0; l < lanes; ++l)
            for (int c = 0; c < width; ++c) source[l * stride + c] = (l * 31 + c) % 251;
        DeviceBuffer src(source.size()), dst(compact.size());
        src.copy_from_host(source.data(), src.bytes);
        indexed_transfer(dst.p, src.p, static_cast<int*>(ds.p), width, stride, lanes, false,
                         device.stream);
        device.synchronize();
        dst.copy_to_host(compact.data(), dst.bytes);
        for (int l = 0; l < lanes; ++l)
            for (int c = 0; c < width; ++c)
                require(compact[l * width + c] == source[slots[l] * stride + c], "raw gather");
        for (auto& x : compact) x ^= 0x5a;
        dst.copy_from_host(compact.data(), dst.bytes);
        indexed_transfer(dst.p, src.p, static_cast<int*>(ds.p), width, stride, lanes, true,
                         device.stream);
        device.synchronize();
        src.copy_to_host(source.data(), src.bytes);
        for (int l = 0; l < lanes; ++l) {
            for (int c = 0; c < width; ++c)
                require(source[slots[l] * stride + c] == compact[l * width + c], "raw scatter");
            for (int c = width; c < stride; ++c)
                require(source[l * stride + c] == 0xa5, "raw transfer guard");
        }
        }
        std::cout << "PASS dynamic Graph retrieval FP64 selection, FP32 means, ragged frontiers, "
                     "lane remapping, zero-score ties, exact indexed transfer\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
