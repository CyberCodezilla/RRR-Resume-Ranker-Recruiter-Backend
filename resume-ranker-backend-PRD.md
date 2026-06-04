# Resume Ranker — Backend PRD

**Repo:** `resume-ranker-backend`  
**Stack:** Python, FastAPI, sentence-transformers  
**Deployment:** Render (free tier)  
**Version:** 1.0 (Hackathon)

---

## 1. Overview

A REST API that accepts a Job Description and a list of candidate profiles (matching the hackathon dataset schema), runs a multi-signal scoring pipeline, and returns a ranked shortlist with explainable scores.

---

## 2. Goals

- Rank candidates intelligently — semantic fit, not just keyword match
- Return explainable scores (breakdown per signal)
- Fast enough for a live demo (target: < 5s for 50 candidates)
- Simple to deploy on Render free tier

---

## 3. Out of Scope

- Authentication / API keys
- Persistent storage / database
- Batch async processing
- Fine-tuned domain-specific models

---

## 4. Dataset Schema (Key Fields Used)

Based on the provided JSON dataset:

| Field | Path | Used For |
|-------|------|----------|
| Headline | `profile.headline` | Semantic similarity |
| Summary | `profile.summary` | Semantic similarity |
| Years of experience | `profile.years_of_experience` | Experience score |
| Skills | `skills[].name`, `.proficiency`, `.endorsements` | Skill match score |
| Career history | `career_history[].description` | Semantic similarity |
| Assessment scores | `redrob_signals.skill_assessment_scores` | Verified skill signal |
| Open to work | `redrob_signals.open_to_work_flag` | Activity signal |
| Last active | `redrob_signals.last_active_date` | Activity signal |
| GitHub activity | `redrob_signals.github_activity_score` | Activity signal |
| Profile completeness | `redrob_signals.profile_completeness_score` | Activity signal |

---

## 5. Scoring Pipeline

### 5.1 Signal Weights (Default)

| Signal | Weight | Description |
|--------|--------|-------------|
| Semantic Fit | 40% | Cosine similarity between JD embedding and candidate text |
| Skill Match | 30% | Overlap of required skills vs candidate skills + proficiency |
| Experience | 20% | Years of experience normalized against JD requirements |
| Activity Signal | 10% | Open to work, last active, GitHub score, profile completeness |

### 5.2 Semantic Fit

- Concatenate `headline + summary + career descriptions`
- Embed using `sentence-transformers/all-MiniLM-L6-v2`
- Compute cosine similarity with embedded JD
- Normalize to 0–100

### 5.3 Skill Match

- Extract skill keywords from JD (simple NLP / keyword extraction)
- Score each candidate by:
  - Matched skills count
  - Weighted by `proficiency` (beginner=1, intermediate=2, advanced=3)
  - Boosted by `endorsements` count
  - Boosted further if `skill_assessment_scores` exist for that skill

### 5.4 Experience Score

- Extract required years from JD (regex: "X+ years", "X years of experience")
- Score = `min(candidate_years / required_years, 1.0) * 100`
- If JD has no explicit requirement, normalize across candidate pool

### 5.5 Activity Signal

- `open_to_work_flag`: +20 pts if true
- `last_active_date`: full score if active within 30 days, decays linearly to 0 at 180 days
- `github_activity_score`: normalized 0–100 (already scored in dataset, max observed ~10)
- `profile_completeness_score`: normalized 0–100 (already in dataset)

---

## 6. API Endpoints

### `POST /rank`

**Request:**
```json
{
  "job_description": "We are looking for a Senior ML Engineer with 5+ years...",
  "candidates": [ ...array of candidate objects from dataset... ]
}
```

**Response:**
```json
{
  "ranked_candidates": [
    {
      "rank": 1,
      "candidate_id": "CAND_0000001",
      "name": "Ira Vora",
      "headline": "Backend Engineer | SQL, Spark, Cloud",
      "overall_score": 82.4,
      "score_breakdown": {
        "semantic_fit": 88.1,
        "skill_match": 74.2,
        "experience": 91.0,
        "activity_signal": 63.5
      },
      "top_matching_skills": ["NLP", "Fine-tuning LLMs", "Milvus"],
      "why_ranked": "Strong semantic fit with JD. Advanced NLP and LLM skills with high endorsements. 6.9 years experience exceeds requirement."
    }
  ],
  "total_candidates": 50,
  "processing_time_ms": 1240
}
```

### `GET /health`

Returns `{ "status": "ok" }` — used by Vercel frontend to check if backend is alive.

---

## 7. Folder Structure

```
resume-ranker-backend/
├── app/
│   ├── main.py              # FastAPI app, routes
│   ├── ranker.py            # Core scoring pipeline
│   ├── embedder.py          # Sentence transformer wrapper
│   ├── skill_extractor.py   # JD keyword extraction
│   └── models.py            # Pydantic request/response models
├── requirements.txt
├── render.yaml
└── README.md
```

---

## 8. Dependencies

```
fastapi
uvicorn
sentence-transformers
numpy
pandas
scikit-learn
pydantic
python-dateutil
```

---

## 9. Deployment (Render)

**`render.yaml`:**
```yaml
services:
  - type: web
    name: resume-ranker-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

1. Push repo to GitHub
2. New Web Service on [render.com](https://render.com)
3. Connect GitHub repo
4. Render auto-detects `render.yaml`
5. First deploy downloads the model (~90MB) — subsequent deploys are faster
6. Copy the live URL → paste into frontend `VITE_API_URL`

**CORS config** — allow Vercel frontend domain in FastAPI:
```python
app.add_middleware(CORSMiddleware, allow_origins=["https://your-app.vercel.app"], allow_methods=["*"], allow_headers=["*"])
```

---

## 10. Acceptance Criteria

- [ ] `POST /rank` returns ranked list with score breakdowns
- [ ] Semantic similarity working (not keyword-only)
- [ ] All 4 signals contributing to final score
- [ ] Response time < 5s for 50 candidates
- [ ] CORS configured for Vercel frontend
- [ ] Deployed live on Render free tier
