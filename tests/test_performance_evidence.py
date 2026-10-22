"""Regression cases for evidence failures that previously forced repeated campaigns."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

spec=importlib.util.spec_from_file_location("evidence",Path(__file__).resolve().parents[1]/"scripts/performance_evidence.py")
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
H="a"*64

def fixture():
    fields=sorted(e.REQUIRED["operator_latency"])
    plan=dict(version=1,id="operator-test",claims=["operator_latency"],
        arms=[dict(arm=a,binary_sha256=H,source_manifest_sha256=H,recipe_sha256=H,
                   device_profile_sha256=H,treatment=a) for a in "AB"],
        requests=[dict(id="shape-251",body_sha256=H,source="synthetic")],
        cache_protocol="cold fragmented; warm5",arrival_protocol="exclusive serial processes",
        allowed_differences=["parallel INT8 prefill only"],
        metrics=[dict(key="cuda_event_ms",availability="required",direction="lower",claim=True)],
        collectors={"ordinary":dict(enabled=True,fields=fields),"nsight":dict(enabled=True,fields=fields)},
        comparison=dict(order=["A","B","B","A"],minimum_blocks=1,noise_floor_pct=2,
                        regression_limit_pct=3,aa_evidence_sha256=H),
        roi=dict(trigger="first measured graph",target_event="INT8 attention",minimum_target_events=1),
        observer_failure_policy="record_only")
    runs=[]
    for arm,profiled in [("A",False),("B",False),("B",False),("A",False),("A",True),("B",True)]:
        runs.append(dict(plan["arms"][0 if arm=="A" else 1],profiled=profiled,profiler_active=profiled,
            pid_identity="owned-pid:start",boot_id="boot",command=["bench"],external_interference=False,
            sampler_errors=[],request_hashes={"shape-251":H},cache_history_sha256=H,arrival_log_sha256=H,
            observed_fields=fields,trace=dict(target_event="INT8 attention",target_event_count=5,
                observed_tables=["api","kernel"],tail_verified=True,clock_witness_verified=True,
                loss_status="none",version="recorded",command=["nsys","profile"])))
    rows=[dict(metric="cuda_event_ms",arm=a,request_id="shape-251",phase="P",context_band="36k",
               cache_class="cold",cache_tokens=0,profiled=False,value=v)
          for a,v in [("A",100),("B",90),("B",90),("A",100)]]
    receipt=dict(plan_sha256=H,started_unix=2,run_completed=True,runs=runs,measurements=rows,
                 quality=dict(passed=True,assertions=["FP64, cache guards, Graph"],exact_heads=True),artifacts=[])
    return plan,receipt

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ.get("KVMEM_TEST_TMP"));self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/"raw").write_bytes(b"raw data")
        self.plan,self.receipt=fixture()
        sha=e.digest(self.root/"raw")
        self.receipt["artifacts"]=[dict(path="raw",sha256=sha,archive=dict(uri="nas:campaign/raw",readback_sha256=sha))]
    def result(self):
        return e.evaluate(self.plan,self.receipt,self.root,dict(plan_sha256=H,created_unix=1),H)
    def test_complete_measurement_is_gain(self):
        r=self.result();self.assertEqual(r["errors"],[]);self.assertTrue(r["performance_gain"])
    def test_missing_prefix_restore_timer_fails_before_launch(self):
        self.plan["metrics"].append(dict(key="prefix_restore_s",availability="required",direction="lower",claim=True))
        self.assertTrue(any("not implemented" in s for s in e.preflight(self.plan)))
    def test_unavailable_timer_cannot_be_claimed(self):
        self.plan["metrics"].append(dict(key="prefix_restore_s",availability="unavailable",reason="timer absent",direction="lower",claim=False))
        self.assertFalse(e.preflight(self.plan));self.plan["metrics"][-1]["claim"]=True
        self.assertTrue(e.preflight(self.plan))
    def test_early_roi_misses_late_allocation(self):
        self.receipt["runs"][-1]["trace"]["target_event_count"]=0
        self.assertTrue(any("ROI missed" in s for s in self.result()["errors"]))
    def test_operator_elapsed_time_does_not_need_host_clock_join(self):
        for run in self.receipt["runs"]:
            run["trace"]["clock_witness_verified"]=False
        self.assertTrue(self.result()["evidence_complete"])
    def test_host_claim_still_requires_clock_witness(self):
        self.plan["claims"]=["host_allocation"]
        fields=sorted(e.REQUIRED["host_allocation"])
        self.plan["collectors"]["nsight"]["fields"]+=fields
        for run in self.receipt["runs"]:
            run["observed_fields"]+=fields
            run["trace"]["observed_tables"].append("nvtx")
            run["trace"]["clock_witness_verified"]=False
        self.assertTrue(any("trace tail/clock missing" in s for s in self.result()["errors"]))
    def test_median_gain_cannot_hide_control_drift(self):
        self.receipt["measurements"][0]["value"]=80
        result=self.result()
        self.assertFalse(result["performance_gain"])
        self.assertGreater(result["comparisons"][0]["control_spread_pct"],20)
    def test_wrong_target_cannot_use_unrelated_trace(self):
        self.receipt["runs"][-1]["trace"]["target_event"]="unrelated allocation"
        self.assertFalse(self.result()["evidence_complete"])
    def test_sampler_failure_is_incomplete_not_quality_failure(self):
        self.receipt["runs"][0]["sampler_errors"]=["PowerShell counter unavailable"]
        r=self.result();self.assertTrue(r["run_completed"]);self.assertTrue(r["quality_pass"])
        self.assertFalse(r["evidence_complete"])
    def test_changed_binary_is_not_inherited_pass(self):
        self.receipt["runs"][0]["binary_sha256"]="b"*64
        self.assertFalse(self.result()["evidence_complete"])
    def test_equal_cache_totals_different_positions_not_matched(self):
        # Aggregate hit fraction can be equal while a request's hit frontier changed.
        for row in self.receipt["measurements"]:
            if row["arm"]=="B": row["cache_tokens"]=100
        r=self.result();self.assertTrue(r["unmatched"]);self.assertFalse(r["performance_gain"])
    def test_profiled_timings_never_replace_ordinary(self):
        self.receipt["runs"][0]["profiler_active"]=True
        self.assertTrue(any("contaminated" in s for s in self.result()["errors"]))
    def test_claimed_collectors_need_actual_data(self):
        self.receipt["runs"][0]["observed_fields"]=[]
        self.assertFalse(self.result()["evidence_complete"])
    def test_archive_readback_required(self):
        del self.receipt["artifacts"][0]["archive"]["readback_sha256"]
        self.assertFalse(self.result()["evidence_complete"])
    def test_modified_artifact_is_rejected(self):
        (self.root/"raw").write_bytes(b"different data")
        self.assertFalse(self.result()["evidence_complete"])
    def test_nan_counter_is_not_fast(self):
        self.receipt["measurements"][0]["value"]=float("nan")
        self.assertFalse(self.result()["performance_gain"])
    def test_trace_loss_needs_explicit_limits(self):
        self.receipt["runs"][-1]["trace"]["loss_status"]="qualified"
        self.assertFalse(self.result()["evidence_complete"])
    def test_quality_failure_cannot_hide_under_throughput(self):
        self.receipt["quality"]["passed"]=False
        r=self.result();self.assertFalse(r["performance_gain"])
        self.assertTrue(r["evidence_complete"]);self.assertEqual(r["status"],"quality_failed")
    def test_nested_scopes_are_not_added_twice(self):
        ledger=e.exclusive_ledger([dict(scope="capture",start=2,end=5),dict(scope="P",start=0,end=8),
                                   dict(scope="PD",start=7,end=9)],0,10,["capture","PD","P"])
        self.assertEqual(ledger,dict(capture=3,PD=2,P=4,outside=1))
        self.assertEqual(sum(ledger.values()),10)
    def test_seal_refuses_rewrite_and_binds_bytes(self):
        path=self.root/"plan.json";path.write_text(json.dumps(self.plan));seal=self.root/"seal.json"
        result=e.seal(path,seal);self.assertEqual(result["plan_sha256"],e.digest(path))
        with self.assertRaises(FileExistsError):e.seal(path,seal)
    def test_documented_cli_actually_preflights(self):
        path=self.root/"plan.json";path.write_text(json.dumps(self.plan))
        result=subprocess.run([sys.executable,str(Path(e.__file__)),"preflight","--plan",str(path)],
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),{"errors":[]})
    def test_microbenchmark_cannot_become_engine_acceptance(self):
        self.plan["claims"]=["engine_performance"]
        errors=e.preflight(self.plan)
        self.assertTrue(any("30 distinct" in v for v in errors))
        self.assertTrue(any("three independent" in v for v in errors))

if __name__=="__main__":unittest.main()
