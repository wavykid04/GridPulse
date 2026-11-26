import pandas as pd
from fastapi.testclient import TestClient

import api
from conftest import write_uci_file


client = TestClient(api.app)


def test_summary_unknown_job_404():
    r = client.get("/api/summary/does-not-exist")
    assert r.status_code == 404


def test_summary_unfinished_job_409():
    api.jobs["pending"] = {"status": "processing", "progress": 10}
    try:
        r = client.get("/api/summary/pending")
        assert r.status_code == 409
    finally:
        api.jobs.pop("pending", None)


def test_summary_returns_computed_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_uci_file(tmp_path / "data.txt", negative_rows=range(500, 560))
    job_id = "job1"
    api.jobs[job_id] = {"status": "queued"}
    try:
        api.process_pipeline(job_id, str(path), 4, 0.05, 0.5)
        assert api.jobs[job_id]["status"] == "done", api.jobs[job_id]
        r = client.get(f"/api/summary/{job_id}")
        assert r.status_code == 200
        summary = r.json()

        result = pd.read_csv(tmp_path / f"results_{job_id}.csv", index_col=0)
        assert summary["total_records"] == len(result)
        assert sum(b["count"] for b in summary["score_distribution"]) == len(result)
        assert len(summary["score_distribution"]) == 10

        expected = result.groupby("Meter_ID")["Final_Risk_Score"].max()
        table = summary["meter_risk_table"]
        assert {row["Meter_ID"] for row in table} == set(expected.index)
        for row in table:
            assert abs(row["max_score"] - expected[row["Meter_ID"]]) < 1e-4
        assert set(summary["rule_contributions"]) == {f"R{i}" for i in range(1, 8)}
        assert summary["rule_contributions"]["R4"] > 0
    finally:
        api.jobs.pop(job_id, None)
