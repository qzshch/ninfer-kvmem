#!/usr/bin/env python3
"""Versioned performance evidence gates. Observers never control inference services."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

VERSION = 1
VALIDATOR_REVISION = 2
SHA_LENGTH = 64
PHASES = {"P", "D", "PP", "DD", "PD"}
REQUIRED = {
    "operator_latency": {"cuda_event_ms", "fp64_oracle", "graph_replay", "nsight.kernel", "nsight.api"},
    "host_allocation": {"allocation_ms", "checkpoint_event", "lease_integrity", "host_memory_bytes",
                        "nsight.api", "nsight.kernel", "nsight.nvtx", "clock_witness"},
    "engine_performance": {"request_join", "uncached_prefill_tokens", "uncached_prefill_s", "decode_tokens",
                           "decode_rounds", "decode_s", "ttft_s", "queue_s", "sse_gap_s", "cache_tokens",
                           "actual_phase", "context_frontier", "quality", "tool_schema", "host_memory_bytes",
                           "gpu_memory_bytes", "nsight.api", "nsight.kernel", "nsight.memcpy",
                           "nsight.graph", "nsight.nvtx", "clock_witness"},
}

def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

def sha_valid(value) -> bool:
    return isinstance(value, str) and len(value) == SHA_LENGTH and all(c in "0123456789abcdef" for c in value)

def preflight(plan: dict) -> list[str]:
    errors = []
    def need(condition, message):
        if not condition: errors.append(message)
    need(plan.get("version") == VERSION, "unsupported plan version")
    need(bool(plan.get("id")), "campaign id required")
    claims = plan.get("claims", [])
    need(bool(claims) and set(claims) <= REQUIRED.keys(), "select explicit supported claims")
    arms = plan.get("arms", [])
    need(len(arms) == 2 and {a.get("arm") for a in arms} == {"A", "B"}, "exactly A/B identities required")
    for arm in arms:
        for field in ("binary_sha256", "source_manifest_sha256", "recipe_sha256", "device_profile_sha256"):
            need(sha_valid(arm.get(field)), f"{arm.get('arm')}: missing {field}")
        need(bool(arm.get("treatment")), "declare treatment (including control)")
    requests = plan.get("requests", [])
    ids = [r.get("id") for r in requests]
    need(bool(ids) and None not in ids and len(ids) == len(set(ids)), "unique request ids required")
    for request in requests:
        need(sha_valid(request.get("body_sha256")), "request body hash required")
        need(bool(request.get("source")), "request source required")
    need(bool(plan.get("cache_protocol")) and bool(plan.get("arrival_protocol")), "freeze cache and arrival protocols")
    need(bool(plan.get("allowed_differences")), "declare the A/B change before running")
    collectors = plan.get("collectors", {})
    capabilities = set().union(*(set(v.get("fields", [])) for v in collectors.values() if v.get("enabled")))
    for claim in claims:
        if claim in REQUIRED:
            need(not (REQUIRED[claim] - capabilities), f"{claim}: missing capabilities {sorted(REQUIRED[claim]-capabilities)}")
    for metric in plan.get("metrics", []):
        key = metric.get("key")
        need(bool(key) and metric.get("direction") in ("higher", "lower"), "metric key and direction required")
        if metric.get("availability") == "unavailable":
            need(bool(metric.get("reason")), f"{key}: unavailable needs a reason")
            need(not metric.get("claim"), f"{key}: unavailable timer cannot support a claim")
        else:
            need(metric.get("availability") == "required", f"{key}: required/unavailable must be explicit")
            need(key in capabilities, f"{key}: timer/counter not implemented by enabled collectors")
    need(bool(plan.get("metrics")), "freeze metrics before running")
    comparison = plan.get("comparison", {})
    need(comparison.get("order") == ["A", "B", "B", "A"], "ordinary measurement uses ABBA")
    need(isinstance(comparison.get("minimum_blocks"), int) and comparison.get("minimum_blocks", 0) >= 1,
         "declare ABBA block count")
    need(isinstance(comparison.get("noise_floor_pct"), (int, float)) and comparison.get("noise_floor_pct", -1) >= 0,
         "freeze A/A noise floor")
    need(isinstance(comparison.get("regression_limit_pct"), (int, float)) and comparison.get("regression_limit_pct", -1) >= 0,
         "freeze regression limit")
    need(sha_valid(comparison.get("aa_evidence_sha256")), "bind A/A evidence; repetitions are not independent workloads")
    roi = plan.get("roi", {})
    need(bool(roi.get("trigger")) and bool(roi.get("target_event")), "declare a logical ROI trigger and target event")
    need(roi.get("minimum_target_events", 0) >= 1, "require observed target events")
    need(collectors.get("nsight", {}).get("enabled") is True, "paired detailed Nsight is required")
    need(plan.get("observer_failure_policy") == "record_only", "observer failure must not control inference")
    if "engine_performance" in claims:
        need(len(requests) >= 30, "engine claim needs at least 30 distinct original requests")
        need(len({r.get("body_sha256") for r in requests}) >= 30, "repeated bodies are not 30 independent requests")
        need(len({r.get("session") for r in requests}) >= 8, "engine claim needs eight source sessions")
        need(all(r.get("source") == "original" for r in requests), "engine claim cannot substitute derived requests")
        pairs = plan.get("independent_pairs", {})
        for phase in ("PP", "DD", "PD"):
            pairs_for_phase = pairs.get(phase, [])
            need(len({tuple(sorted(p)) for p in pairs_for_phase}) >= 3, f"{phase}: need three independent request pairs")
            need(all(len(p) == 2 and p[0] != p[1] and set(p) <= set(ids) for p in pairs_for_phase), f"{phase}: invalid pair")
        need(set(plan.get("required_phases", [])) == PHASES, "engine claim must account for P/D/PP/DD/PD")
    return errors

def seal(plan_path: Path, seal_path: Path) -> dict:
    plan = json.loads(plan_path.read_bytes())
    errors = preflight(plan)
    if errors: raise ValueError("; ".join(errors))
    receipt = {"version": VERSION, "plan_sha256": digest(plan_path), "created_unix": time.time(),
               "validator_sha256": digest(Path(__file__)), "validator_revision": VALIDATOR_REVISION,
               "preflight": "passed", "observer_failure_policy": "record_only"}
    # Exclusive creation makes a changed plan a new campaign, not a silent retry.
    with seal_path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2)
    return receipt

def artifact_errors(artifacts: list[dict], root: Path, require_archive=True) -> list[str]:
    errors = []
    if not artifacts: return ["artifact manifest missing"]
    seen = set()
    root = root.resolve()
    for artifact in artifacts:
        name = artifact.get("path", "")
        target = (root / name).resolve()
        if not name or name in seen or not target.is_relative_to(root):
            errors.append("invalid/duplicate artifact path"); continue
        seen.add(name)
        if not target.is_file() or not sha_valid(artifact.get("sha256")) or digest(target) != artifact["sha256"]:
            errors.append(f"artifact bytes/hash mismatch: {name}")
        if require_archive:
            archive = artifact.get("archive", {})
            if archive.get("readback_sha256") != artifact.get("sha256") or not archive.get("uri"):
                errors.append(f"archive readback missing: {name}")
    return errors

def exclusive_ledger(intervals: list[dict], start: float, end: float, priority: list[str]) -> dict:
    """Partition overlapping scopes once; inclusive Host submit is not GPU idle."""
    if end <= start: raise ValueError("invalid ledger window")
    clipped = [(max(start, v["start"]), min(end, v["end"]), v["scope"]) for v in intervals]
    clipped = [(a,b,s) for a,b,s in clipped if a < b]
    points = sorted({start, end, *(p for a,b,_ in clipped for p in (a,b))})
    result = {name: 0.0 for name in [*priority, "outside"]}
    for a,b in zip(points, points[1:]):
        active = {s for x,y,s in clipped if x <= a and y >= b}
        owner = next((s for s in priority if s in active), "outside")
        result[owner] += b-a
    return result

def evaluate(plan: dict, receipt: dict, root: Path, sealed: dict, plan_sha256: str) -> dict:
    errors = preflight(plan)
    differences = []
    if receipt.get("plan_sha256") != plan_sha256 or sealed.get("plan_sha256") != plan_sha256:
        errors.append("plan binding changed")
    if receipt.get("started_unix", 0) < sealed.get("created_unix", math.inf):
        errors.append("run predates frozen plan")
    if not receipt.get("run_completed"): errors.append("run incomplete")
    identities = {a["arm"]: a for a in plan["arms"]}
    requests = {r["id"]: r for r in plan["requests"]}
    runs = receipt.get("runs", [])
    ordinary = [r for r in runs if not r.get("profiled")]
    sequence = [r.get("arm") for r in ordinary]
    expected = plan["comparison"]["order"] * plan["comparison"]["minimum_blocks"]
    if sequence != expected: errors.append("ordinary ABBA coverage/order incomplete")
    trace_arms = set()
    for run in runs:
        arm = run.get("arm")
        if arm not in identities: errors.append("unknown run arm"); continue
        for field in ("binary_sha256", "source_manifest_sha256", "recipe_sha256", "device_profile_sha256"):
            if run.get(field) != identities[arm][field]: errors.append(f"{arm}: {field} binding mismatch")
        if not run.get("pid_identity") or not run.get("boot_id") or not run.get("command"):
            errors.append(f"{arm}: process/command identity missing")
        if run.get("external_interference") is not False: errors.append(f"{arm}: external interference/unknown")
        if run.get("sampler_errors"): errors.append(f"{arm}: sampler failed; service must remain unaffected")
        required_fields=REQUIRED_FOR(plan)-{"fp64_oracle","graph_replay","lease_integrity","quality","tool_schema"}
        if not run.get("profiled"):
            required_fields={f for f in required_fields if not f.startswith("nsight.")}
        if not required_fields <= set(run.get("observed_fields", [])):
            errors.append(f"{arm}: declared collectors did not produce required fields")
        if run.get("request_hashes") != {k:v["body_sha256"] for k,v in requests.items()}:
            errors.append(f"{arm}: actual request bodies differ")
        if not run.get("cache_history_sha256") or not run.get("arrival_log_sha256"):
            errors.append(f"{arm}: actual cache/arrival history missing")
        if run.get("profiled"):
            trace = run.get("trace", {})
            if trace.get("target_event") != plan["roi"]["target_event"]:
                errors.append(f"{arm}: wrong ROI target event")
            if trace.get("target_event_count", 0) < plan["roi"]["minimum_target_events"]:
                errors.append(f"{arm}: ROI missed target event")
            needed = {f.removeprefix("nsight.") for f in REQUIRED_FOR(plan) if f.startswith("nsight.")}
            if not needed <= set(trace.get("observed_tables", [])): errors.append(f"{arm}: trace tables missing")
            if not trace.get("tail_verified") or ("clock_witness" in REQUIRED_FOR(plan) and not trace.get("clock_witness_verified")):
                errors.append(f"{arm}: trace tail/clock missing")
            if trace.get("loss_status") not in ("none", "qualified"):
                errors.append(f"{arm}: trace loss status unknown")
            if trace.get("loss_status") == "qualified" and not trace.get("loss_limits"):
                errors.append(f"{arm}: profiler loss limits not described")
            if not trace.get("version") or not trace.get("command"):
                errors.append(f"{arm}: profiler command/version missing")
            trace_arms.add(arm)
        elif run.get("profiler_active") is not False:
            errors.append(f"{arm}: ordinary timing contaminated by profiling")
        if "engine_performance" in plan["claims"]:
            rows=run.get("requests", [])
            if {v.get("id") for v in rows} != set(requests) or len(rows)!=len(requests):
                errors.append(f"{arm}: incomplete/duplicate per-request ledger")
            fields=("prompt_tokens","cache_tokens","uncached_prefill_tokens","uncached_prefill_s",
                    "decode_tokens","decode_rounds","decode_s","ttft_s","queue_s","request_s",
                    "sse_gap_p95_s","sse_gap_p99_s","sse_gap_max_s")
            for row in rows:
                if any(not isinstance(row.get(f),(int,float)) or not math.isfinite(row[f]) or row[f]<0 for f in fields):
                    errors.append(f"{arm}: invalid/missing per-request counter or time")
                elif row["cache_tokens"]+row["uncached_prefill_tokens"] != row["prompt_tokens"]:
                    errors.append(f"{arm}: cached tokens counted as computed prefill")
                if row.get("join_verified") is not True or not row.get("finish_reason"):
                    errors.append(f"{arm}: client/native request join or finish missing")
            if not set(plan["required_phases"]) <= set(run.get("observed_phases", [])):
                errors.append(f"{arm}: required concurrency phases not observed")
    if trace_arms != {"A", "B"}: errors.append("paired Nsight evidence missing")
    quality = receipt.get("quality", {})
    quality_qualified = isinstance(quality.get("passed"),bool) and bool(quality.get("assertions")) and quality.get("exact_heads") is True
    quality_ok = quality_qualified and quality["passed"]
    if not quality_qualified: errors.append("quality evidence missing/unqualified")
    errors += artifact_errors(receipt.get("artifacts", []), root)
    measurements = receipt.get("measurements", [])
    comparisons = []
    for metric in plan["metrics"]:
        if metric["availability"] == "unavailable": continue
        name = metric["key"]
        selected = [v for v in measurements if v.get("metric") == name and v.get("profiled") is False]
        if not selected: errors.append(f"{name}: measurements missing"); continue
        grouped = {}
        for row in selected:
            value = row.get("value")
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                errors.append(f"{name}: invalid counter/time"); continue
            key = (row.get("request_id"),row.get("phase"),row.get("context_band"),row.get("cache_class"),row.get("cache_tokens"))
            if row.get("request_id") not in requests or row.get("phase") not in PHASES:
                errors.append(f"{name}: request/phase missing"); continue
            if not row.get("context_band") or row.get("cache_class") not in ("cold","partial","high","none") or \
                not isinstance(row.get("cache_tokens"),(int,float)) or row["cache_tokens"]<0:
                errors.append(f"{name}: cache/context stratum missing"); continue
            grouped.setdefault(key, {}).setdefault(row.get("arm"), []).append(value)
        matched = 0
        for key,arms in grouped.items():
            if set(arms) != {"A","B"}:
                differences.append({"metric":name,"stratum":key,"reason":"unmatched cache/context/phase"}); continue
            if min(map(len,arms.values())) < 2 * plan["comparison"]["minimum_blocks"]:
                errors.append(f"{name}: repeated coverage missing"); continue
            a,b = statistics.median(arms["A"]),statistics.median(arms["B"])
            if min(arms["A"])<=0: errors.append(f"{name}: zero control denominator"); continue
            delta=100*(b/a-1)
            improvement=delta if metric.get("direction") == "higher" else -delta
            # The slowest candidate must beat the fastest control. A drifting
            # control must not become a median-only gain.
            if metric.get("direction") == "higher":
                conservative=100*(min(arms["B"])/max(arms["A"])-1)
            else:
                conservative=100*(1-max(arms["B"])/min(arms["A"]))
            comparisons.append({"metric":name,"stratum":key,"A":a,"B":b,"change_pct":delta,"improvement_pct":improvement,
                                "conservative_improvement_pct":conservative,
                                "control_spread_pct":100*(max(arms["A"])/min(arms["A"])-1)})
            matched += 1
        if not matched: errors.append(f"{name}: no matched strata")
    claimed = [c for c in comparisons if any(m["key"]==c["metric"] and m.get("claim") and
               (not m.get("claim_request_ids") or c["stratum"][0] in m["claim_request_ids"])
               for m in plan["metrics"])]
    regression = any(c["conservative_improvement_pct"] < -plan["comparison"]["regression_limit_pct"] for c in claimed)
    gain = bool(claimed) and all(c["conservative_improvement_pct"] > plan["comparison"]["noise_floor_pct"] for c in claimed)
    complete = not errors and not differences
    status = "evidence_incomplete" if not complete else "quality_failed" if not quality_ok else "regression" if regression else "gain" if gain else "inconclusive"
    return {"version":VERSION,"validator_revision":VALIDATOR_REVISION,"validator_sha256":digest(Path(__file__)),
            "run_completed":receipt.get("run_completed") is True,
            "evidence_complete":complete,"quality_pass":quality_ok,"performance_gain":complete and quality_ok and gain and not regression,
            "status":status,"errors":errors,"unmatched":differences,"comparisons":comparisons,
            "unavailable_metrics":[m for m in plan["metrics"] if m["availability"]=="unavailable"]}

def REQUIRED_FOR(plan):
    return set().union(*(REQUIRED[c] for c in plan["claims"] if c in REQUIRED))

def record_command(plan_path: Path, seal_path: Path, out: Path, command: list[str], timeout: int) -> int:
    """Preflight before launching an owned finite test. No inference health/threshold control."""
    plan=json.loads(plan_path.read_bytes()); sealed=json.loads(seal_path.read_bytes())
    errors=preflight(plan)
    if sealed.get("plan_sha256") != digest(plan_path): errors.append("sealed plan changed")
    if errors: raise ValueError("; ".join(errors))
    out.mkdir(exist_ok=False,parents=True)
    child=None
    owner={"command":command,"started_unix":time.time(),"purpose":plan["id"],"plan_sha256":digest(plan_path)}
    try:
        with (out/"stdout.log").open("wb") as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            owner["pid"]=child.pid
            (out/"owner.json").write_text(json.dumps(owner,indent=2),encoding="utf8")
            return child.wait(timeout=timeout)
    finally:
        if child and child.poll() is None:
            # Only the explicitly owned test driver. Production services are not
            # discovered, signalled or coupled to collector health by this module.
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
        owner.update(ended_unix=time.time(),exit_code=child.returncode if child else None)
        (out/"owner.json").write_text(json.dumps(owner,indent=2),encoding="utf8")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["preflight","seal","evaluate","record-run"])
    parser.add_argument("--plan",required=True,type=Path)
    parser.add_argument("--seal",type=Path)
    parser.add_argument("--receipt",type=Path)
    parser.add_argument("--artifact-root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path)
    parser.add_argument("--timeout",type=int,default=1800)
    args, command=parser.parse_known_args()
    if args.action!="record-run" and command: parser.error("unexpected arguments")
    plan=json.loads(args.plan.read_bytes())
    if args.action=="record-run":
        if not args.seal or not args.output or not command: parser.error("record-run needs seal, output and command")
        command=command[1:] if command[0]=="--" else command
        return record_command(args.plan,args.seal,args.output,command,args.timeout)
    if args.action=="preflight": result={"errors":preflight(plan)}; ok=not result["errors"]
    elif args.action=="seal":
        if not args.seal: parser.error("seal needs --seal")
        result=seal(args.plan,args.seal);ok=True
    else:
        if not args.seal or not args.receipt: parser.error("evaluate needs --seal and --receipt")
        result=evaluate(plan,json.loads(args.receipt.read_bytes()),args.artifact_root,
                        json.loads(args.seal.read_bytes()),digest(args.plan))
        ok=result["evidence_complete"] and result["quality_pass"]
    text=json.dumps(result,indent=2,ensure_ascii=True)
    if args.output: args.output.write_text(text+"\n",encoding="utf8")
    print(text)
    return 0 if ok else 2

if __name__=="__main__":
    raise SystemExit(main())
