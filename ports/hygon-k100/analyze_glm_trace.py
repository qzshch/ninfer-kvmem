"""Summarize a native GLM hipprof SQLite trace without exporting host identity."""
import argparse
import bisect
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def union_ns(intervals):
    total = 0
    previous_end = 0
    for begin, end in sorted(intervals):
        total += max(0, end - max(begin, previous_end))
        previous_end = max(previous_end, end)
    return total


def category(name):
    if "router_kernel" in name or "router_serial_kernel" in name:
        return "expert_router"
    if "norm_kernel" in name:
        return "normalization"
    if "convert_kernel" in name:
        return "rounding_and_pack"
    if "indexed_kernel" in name or "indexed_wave64" in name:
        return "indexed_projection"
    if "projection<" in name or "projection_wave64<" in name:
        return "dense_projection"
    if "kimi_delta_attention" in name:
        return "kda_recurrence"
    if any(x in name for x in ["coefficients_kernel", "collapse_kernel", "expand_kernel"]):
        return "hyper_connection"
    if "embedding_kernel" in name:
        return "embedding"
    if any(x in name for x in ["pool_append", "pool_scores", "pool_select", "attention_scores", "attention_values"]):
        return "attention_index_and_values"
    return "other"


def load(db_path, api_path):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    names = {str(r[0]): r[1] for r in db.execute("SELECT STR_ID,STR_NAME FROM STR_TABLE WHERE TYPE=6")}
    if api_path:
        api_names = json.loads(Path(api_path).read_text())
    else:
        import ctypes
        hip = ctypes.CDLL("libamdhip64.so")
        function = hip.hipApiName
        function.argtypes = [ctypes.c_uint32]
        function.restype = ctypes.c_char_p
        api_names = {str(i): function(i).decode() for i in range(500) if function(i)}
    result = {}
    for prefix, label in [("HIPOPS_", "kernel"), ("HIPCOPY_", "copy"), ("HIP_", "api")]:
        selected = [x for x in tables if x.startswith(prefix)]
        assert len(selected) == 1, (prefix, selected)
        # Table names come from SQLite's own schema, never from a SQL argument.
        rows = [dict(x) for x in db.execute('SELECT * FROM "' + selected[0].replace('"', '""') + '" ORDER BY BeginNs')]
        for row in rows:
            if label == "kernel":
                row["name"] = names[str(row["Name"])]
            elif label == "api":
                row["name"] = api_names.get(str(row["Name"]), "api_" + str(row["Name"]))
        result[label] = rows
    return result


def frames(kernels):
    device0 = [k for k in kernels if k["dev_id"] == 0]
    starts = [k for i, k in enumerate(device0[:-1]) if "embedding_kernel" in k["name"] and "streams_kernel<false>" in device0[i + 1]["name"]]
    ends = [k for k in device0 if "argmax_kernel" in k["name"]]
    assert len(starts) == len(ends), (len(starts), len(ends))
    result = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        assert start["BeginNs"] < end["EndNs"]
        if i:
            assert result[-1][1] < start["BeginNs"]
        result.append((start["BeginNs"], end["EndNs"]))
    return result


def aggregate(data, begin, end, target_frames):
    def within(rows):
        return [r for r in rows if begin <= r["BeginNs"] and r["EndNs"] <= end]

    kernels, copies, apis = (within(data[x]) for x in ["kernel", "copy", "api"])
    wall = end - begin
    categories, names, api_times, copy_times, copy_bytes = (Counter() for _ in range(5))
    api_calls, kernel_calls, copy_calls = (Counter() for _ in range(3))
    starts = [a for a, b in target_frames]

    def in_target(row):
        i = bisect.bisect_right(starts, row["BeginNs"]) - 1
        return i >= 0 and row["EndNs"] <= target_frames[i][1]

    for r in kernels:
        duration = r["EndNs"] - r["BeginNs"]
        categories[category(r["name"])] += duration
        names[r["name"]] += duration
        kernel_calls[r["name"]] += 1
    for r in apis:
        api_times[r["name"]] += r["EndNs"] - r["BeginNs"]
        api_calls[r["name"]] += 1
    for r in copies:
        key = str(r["Kind"])
        copy_times[key] += r["EndNs"] - r["BeginNs"]
        copy_bytes[key] += r["Bytes"]
        copy_calls[key] += 1
    kernel_sum = sum(names.values())
    gpu_work = kernels + copies
    per_device = []
    for device in range(3):
        device_kernels = [r for r in kernels if r["dev_id"] == device]
        per_device.append({"device": device, "kernel_seconds": sum(r["DurationNs"] for r in device_kernels) / 1e9,
                           "kernel_busy_fraction": union_ns((r["BeginNs"], r["EndNs"]) for r in device_kernels) / wall})
    worker_apis = defaultdict(list)
    for r in apis:
        worker_apis[r["tid"]].append(r)
    metadata_gaps = []
    # A query's 128 FP32 values return to CPU, then the 512-byte page mask
    # goes to GPU; no HIP call occurs during CPU block selection in between.
    copy_by_index = {r["_Index"]: r for r in copies}
    for rows in worker_apis.values():
        for previous, following in zip(rows, rows[1:]):
            left, right = copy_by_index.get(previous["_Index"]), copy_by_index.get(following["_Index"])
            if left and right and left["Bytes"] == right["Bytes"] == 512 and left["Kind"] == 4595 and right["Kind"] == 4596:
                metadata_gaps.append(max(0, following["BeginNs"] - previous["EndNs"]))
    return {
        "kernel_defined_window_seconds": wall / 1e9,
        "target_steps": len(target_frames),
        "target_frame_wall_seconds": sum(b - a for a, b in target_frames) / 1e9,
        "target_frame_median_ms": statistics.median((b - a) / 1e6 for a, b in target_frames),
        "kernel_calls": len(kernels), "gpu_kernel_seconds_sum": kernel_sum / 1e9,
        "gpu_kernel_busy_union_seconds": union_ns((r["BeginNs"], r["EndNs"]) for r in kernels) / 1e9,
        "gpu_kernel_or_copy_busy_union_seconds": union_ns((r["BeginNs"], r["EndNs"]) for r in gpu_work) / 1e9,
        "kernel_seconds_inside_target_frames": sum(r["DurationNs"] for r in kernels if in_target(r)) / 1e9,
        "kernel_seconds_outside_target_frames": sum(r["DurationNs"] for r in kernels if not in_target(r)) / 1e9,
        "per_device": per_device,
        "categories": [{"category": k, "seconds": v / 1e9, "fraction_of_kernel_time": v / kernel_sum} for k, v in categories.most_common()],
        "kernels": [{"name": k, "calls": kernel_calls[k], "seconds": v / 1e9, "fraction_of_kernel_time": v / kernel_sum} for k, v in names.most_common()],
        "host_apis_inclusive": [{"name": k, "calls": api_calls[k], "seconds": v / 1e9} for k, v in api_times.most_common()],
        "copies": [{"kind": k, "calls": copy_calls[k], "bytes": copy_bytes[k], "gpu_seconds": v / 1e9} for k, v in copy_times.most_common()],
        "kvmem_cpu_query_to_mask_gaps": {"calls": len(metadata_gaps), "seconds_sum": sum(metadata_gaps) / 1e9,
                                         "max_ms": max(metadata_gaps, default=0) / 1e6},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", required=True)
    p.add_argument("--api-names", help="Optional HIP API-name map for analysis away from DTK")
    p.add_argument("--run", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    data = load(args.database, args.api_names)
    all_frames = frames(data["kernel"])
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["measurements"]
    result = {"method": "Unmodified native GLM Engine under DTK hipprof --hip-trace --no-export. Windows delimited by target embedding and argmax kernels; host API time includes GPU waits and overlaps GPU duration.", "modes": {}}
    cursor = 0
    if "ordinary" not in run:
        assert run["speculative"] == "ordinary", "Expected an ordinary run or native comparison"
        run = {"ordinary": run}
    for mode in ("ordinary", "dflash2"):
        if mode not in run:
            continue
        current = run[mode]
        prompt = current["result"]["prompt_tokens"]
        assert current["result"]["generated_ids"] == baseline[mode]["result"]["generated_ids"]
        count = prompt + (len(current["result"]["generated_ids"]) - 1 if mode == "ordinary"
                          else current["result"]["drafted_tokens"] + current["result"]["draft_rounds"])
        assert count > prompt, "Attribution requires a request with decode steps"
        if current["kvmem_window_pages"]:
            assert current["result"]["kvmem_selection_calls"] == count * 11
        selected = all_frames[cursor:cursor + count]
        assert len(selected) == count
        cursor += count
        spans = {"prefill": (selected[0][0], selected[prompt - 1][1], selected[:prompt]),
                 "decode": (selected[prompt - 1][1], selected[-1][1], selected[prompt:])}
        result["modes"][mode] = {
            "generated_ids_equal_to_unprofiled": True,
            "traced_prefill_seconds": current["result"]["prefill_seconds"],
            "traced_decode_seconds": current["result"]["decode_seconds"],
            "unprofiled_prefill_seconds": baseline[mode]["result"]["prefill_seconds"],
            "unprofiled_decode_seconds": baseline[mode]["result"]["decode_seconds"],
            "phases": {name: aggregate(data, *span) for name, span in spans.items()},
        }
        print(mode, "steps", count, "decode seconds", current["result"]["decode_seconds"])
    assert cursor == len(all_frames), (cursor, len(all_frames))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
