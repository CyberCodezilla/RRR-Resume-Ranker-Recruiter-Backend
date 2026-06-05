from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any

app = FastAPI(title="RRR Resume Ranker Backend (stub)")


class RankRequest(BaseModel):
    job_description: str
    candidates: List[Any]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/rank")
async def rank(req: RankRequest):
    # Minimal stub implementation: return up to 100 candidates with a deterministic placeholder score
    if not req.candidates:
        raise HTTPException(status_code=400, detail="candidates array is required")

    results = []
    for i, c in enumerate(req.candidates[:100]):
        cid = c.get("candidate_id") or c.get("id") or f"CAND_{i:07d}"
        # Simple deterministic score based on years_of_experience if available
        yrs = 0
        try:
            yrs = float(c.get("profile", {}).get("years_of_experience", 0) or 0)
        except Exception:
            yrs = 0
        score = min(1.0, yrs / 15.0)
        results.append({"candidate_id": cid, "rank": i + 1, "score": round(score, 4), "reasoning": "stub score based on experience"})

    return {"results": results}
