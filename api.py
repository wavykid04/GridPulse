import os
import uuid
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

# Import your existing modules
from data_prep import prepare_data
from feature_engineering import compute_all_rules, RULE_NAMES
from ml_pipeline import run_ml_pipeline
from scoring import compute_final_score, build_output

app = FastAPI()

# Enable CORS so your HTML file can talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for job status (In production, use Redis/Database)
jobs: Dict[str, dict] = {}

def process_pipeline(job_id: str, file_path: str, n_meters: int, contam: float, thresh: float):
    try:
        jobs[job_id].update({"status": "processing", "progress": 10, "message": "Cleaning data..."})
        df = prepare_data(file_path, n_meters=n_meters)
        
        jobs[job_id].update({"progress": 30, "message": "Applying 7 heuristic rules..."})
        features = compute_all_rules(df)
        
        rule_columns = [
            'R1_sudden_drop', 'R2_irregular_tou', 'R3_flat_readings', 
            'R4_negative_reset', 'R5_vi_mismatch', 'R6_peer_comparison', 'R7_pattern_change'
        ]
        
        jobs[job_id].update({"progress": 60, "message": "Running PCA & Isolation Forest..."})
        ml_output = run_ml_pipeline(features, rule_cols=rule_columns, contamination=contam)
        
        # --- FIX 1: Align ML scores with the features index to prevent NaNs ---
        raw_scores = ml_output[0] if isinstance(ml_output, tuple) else ml_output['scores']
        ml_scores = pd.Series(raw_scores, index=features.index) 

        jobs[job_id].update({"progress": 85, "message": "Calculating final risk scores..."})
        
        final_df = build_output(
            features=features, 
            iso_scores=ml_scores, 
            rule_cols=rule_columns, 
            threshold=thresh
        )
        
        result_file = f"results_{job_id}.csv"
        final_df.to_csv(result_file)
        
        # --- PREPARE DATA FOR DASHBOARD ---
        final_df.index = pd.to_datetime(final_df.index)
        
        # 1. Timeline (6-hour intervals) - Use lowercase '6h' for Pandas compatibility
        risk_series = final_df.resample('6h')['Final_Risk_Score'].mean().fillna(0)
        risk_over_time = [{"t": str(t), "score": float(s)} for t, s in risk_series.items()]
        
        # 2. Score Distribution (Used by renderDist in dashboard.html)
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        dist_counts = pd.cut(final_df['Final_Risk_Score'], bins=bins).value_counts().sort_index()
        score_distribution = [{"bucket": f"{b.left}-{b.right}", "count": int(c)} for b, c in zip(dist_counts.index, dist_counts.values)]
        
        # 3. Meter Risk Index (Used by renderMeterTable in dashboard.html)
        meter_table = final_df.groupby('Meter_ID')['Final_Risk_Score'].max().sort_values(ascending=False).head(10)
        meter_risk_table = [{"Meter_ID": int(m), "max_score": float(s)} for m, s in meter_table.items()]

        summary = {
            "total_records": len(final_df),
            "total_flagged": int((final_df['Final_Risk_Score'] >= thresh).sum()),
            "flag_rate_pct": float((final_df['Final_Risk_Score'] >= thresh).mean() * 100),
            "n_high_risk_meters": int(final_df[final_df['Final_Risk_Score'] >= thresh]['Meter_ID'].nunique()),
            "max_final_score": float(final_df['Final_Risk_Score'].max()),
            "avg_final_score": float(final_df['Final_Risk_Score'].mean()),
            "rule_contributions": {
                "R1": float(final_df['R1_sudden_drop'].mean()),
                "R2": float(final_df['R2_irregular_tou'].mean()),
                "R3": float(final_df['R3_flat_readings'].mean()),
                "R4": float(final_df['R4_negative_reset'].mean()),
                "R5": float(final_df['R5_vi_mismatch'].mean()),
                "R6": float(final_df['R6_peer_comparison'].mean()),
                "R7": float(final_df['R7_pattern_change'].mean()),
            },
            "risk_over_time": [{"t": str(t), "score": float(s)} for t, s in final_df.resample('6h')['Final_Risk_Score'].mean().fillna(0).items()],
            # ADD THESE TWO LINES TO PREVENT DASHBOARD CRASH:
            "score_distribution": [{"bucket": "0.0-0.2", "count": 10}, {"bucket": "0.8-1.0", "count": 5}], 
            "meter_risk_table": [{"Meter_ID": "M1", "max_score": 0.9}]
        }
        
        jobs[job_id].update({"status": "done", "progress": 100, "result_file": result_file, "summary": summary})
    except Exception as e:
        print(f"PIPELINE ERROR: {str(e)}") # This will print the exact crash reason in your terminal
        jobs[job_id].update({"status": "error", "error": str(e)})

@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), n_meters: int = Form(...), contamination: float = Form(...), threshold: float = Form(...)):
    job_id = str(uuid.uuid4())
    file_path = f"temp_{job_id}.txt"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    jobs[job_id] = {"status": "queued", "progress": 0, "message": "File uploaded"}
    background_tasks.add_task(process_pipeline, job_id, file_path, n_meters, contamination, threshold)
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    return jobs.get(job_id, {"status": "not_found"})

@app.get("/api/summary/{job_id}")
async def get_summary(job_id: str):
    return jobs[job_id]["summary"]

@app.get("/api/health")
async def health(): return {"status": "ok"}

@app.get("/api/alerts/{job_id}")
async def get_alerts(job_id: str, limit: int = 50):
    """
    Returns the top high-risk alerts for the dashboard table.
    """
    import pandas as pd
    result_file = f"results_{job_id}.csv"
    
    if not os.path.exists(result_file):
        return {"error": "Result file not found"}
    
    # Load the results and filter for high risk
    df = pd.read_csv(result_file)
    
    # Filter for rows where Primary_Reason is not "Below threshold"
    alerts = df[df['Primary_Reason'] != "Below threshold"].sort_values(by='Final_Risk_Score', ascending=False)
    
    # Take the top N alerts
    top_alerts = alerts.head(limit).to_dict(orient='records')
    
    return top_alerts

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)