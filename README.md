# Resume Ranker — Backend

> FastAPI ranking engine that scores and ranks candidates against a Job Description using semantic similarity + multi-signal scoring.

Built for a hackathon. Deployed on Render.

---

## Tech Stack

- Python + FastAPI
- sentence-transformers (`all-MiniLM-L6-v2`)
- scikit-learn (cosine similarity)
- Deployed on Render (free tier)

---

## Getting Started

```bash
git clone https://github.com/your-username/resume-ranker-backend
cd resume-ranker-backend
pip install -r requirements.txt
```

Run locally:

```bash
uvicorn app.main:app --reload
```

API docs available at: `http://localhost:8000/docs`

---

## API

### `POST /rank`

Ranks a list of candidates against a job description.

**Request body:**
```json
{
  "job_description": "Looking for a Senior ML Engineer with 5+ years...",
  "candidates": [ ...array of candidate JSON objects... ]
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
      "overall_score": 82.4,
      "score_breakdown": {
        "semantic_fit": 88.1,
        "skill_match": 74.2,
        "experience": 91.0,
        "activity_signal": 63.5
      },
      "top_matching_skills": ["NLP", "Fine-tuning LLMs", "Milvus"],
      "why_ranked": "Strong semantic fit with JD..."
    }
  ],
  "total_candidates": 50,
  "processing_time_ms": 1240
}
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## Scoring Model

Final score is a weighted combination of 4 signals:

| Signal | Weight | Method |
|--------|--------|--------|
| Semantic Fit | 40% | Cosine similarity via sentence-transformers |
| Skill Match | 30% | Keyword overlap + proficiency + endorsements |
| Experience | 20% | Years of experience vs JD requirement |
| Activity Signal | 10% | Open to work, last active, GitHub score |

---

## Project Structure

```
app/
├── main.py              # FastAPI app + routes
├── ranker.py            # Scoring pipeline
├── embedder.py          # Sentence transformer wrapper
├── skill_extractor.py   # JD keyword extraction
└── models.py            # Pydantic schemas
```

---

## Deployment (Render)

1. Push to GitHub
2. New Web Service on [render.com](https://render.com)
3. Connect this repo — Render picks up `render.yaml` automatically
4. First deploy downloads the model (~90MB), takes ~2 min
5. Copy the live URL → set as `VITE_API_URL` in the frontend

**Note:** Render free tier spins down after 15 min of inactivity. First request after sleep takes ~30s to wake up.

---

## Dependencies

```
fastapi
uvicorn
sentence-transformers
numpy
scikit-learn
pydantic
python-dateutil
```

Install: `pip install -r requirements.txt`

---

## Related

- [resume-ranker-frontend](https://github.com/your-username/resume-ranker-frontend) — React UI on Vercel
- 
