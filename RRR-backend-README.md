# RRR — Resume Ranker Recruiter (Backend)

> **Hack2Skill Hackathon** — Ranking engine for the Intelligent Candidate Discovery Challenge.

Offline Python pipeline that scores ~5000 candidates against a job description using 5 weighted signals and outputs a ranked `submission.csv`.

---

## Tech Stack

- Python 3.11
- `sentence-transformers` (`all-MiniLM-L6-v2`) — semantic skill matching
- `scikit-learn` — cosine similarity
- `pandas` / `numpy` — data processing
- FastAPI — demo sandbox API (HuggingFace Spaces)

---

## Getting Started

```bash
git clone https://github.com/your-username/RRR-Resume-Ranker-Recruiter-Backend
cd RRR-Resume-Ranker-Recruiter-Backend
pip install -r requirements.txt
```

---

## Run the Ranker

```bash
python rank.py --candidates ./data/sample_candidates.json --out ./submission.csv
```

First run downloads `all-MiniLM-L6-v2` (~80MB) and caches candidate embeddings (~2 min).  
Subsequent runs use the cache — completes in under 30 seconds.

| Argument | Default | Description |
|---|---|---|
| `--candidates` | required | Path to candidates JSON |
| `--jd` | `data/job_description.docx` | Path to job description |
| `--out` | `submission.csv` | Output file path |

---

## Scoring Model

Final score (0.0–1.0) is a weighted combination of 5 signals:

| Component | Weight | Method |
|---|---|---|
| Skill Match | 35% | Cosine similarity via sentence-transformers |
| Career Fit | 25% | Recency-decayed title + industry match |
| Signal Modifier | 15% | GitHub, response rate, interview rate, assessments |
| Education | 15% | Institution tier × field-of-study match |
| Availability | 10% | Open to work, notice period, relocation |

Career fit uses exponential decay (`e^(-0.3 × years_ago)`) — a role from 5 years ago contributes ~22% of a current role.

---

## Output Format

```csv
candidate_id,rank,score,reasoning
CAND_0004989,1,0.9920,"HR Manager with 6.1 yrs; 9 AI core skills; response rate 0.76."
CAND_0001195,2,0.9840,"HR Manager with 8.7 yrs; 9 AI core skills; response rate 0.20."
```

---

## Validate Before Submitting

```bash
python validate_submission.py --submission submission.csv --candidates data/sample_candidates.json
```

---

## Project Structure

```
├── rank.py                    ← CLI entrypoint (graded submission)
├── ranker/
│   ├── jd_parser.py           ← Parses job_description.docx
│   ├── candidate_scorer.py    ← Orchestrates all 5 scoring components
│   ├── embedding_utils.py     ← Loads model, caches embeddings
│   └── signal_scorer.py       ← Normalizes redrob_signals to 0–1
├── app/
│   └── main.py                ← FastAPI server (demo sandbox only)
├── data/
│   ├── sample_candidates.json
│   └── job_description.docx
├── requirements.txt
├── submission.csv
├── submission_metadata.yaml
└── README.md
```

---

## Submission Metadata

Fill in `submission_metadata.yaml` before final submission:

```yaml
team_name: "RRR-Team"
github_repo: "https://github.com/your-username/RRR-Resume-Ranker-Recruiter-Backend"
sandbox_link: "https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker"
reproduce_command: "python rank.py --candidates ./data/sample_candidates.json --out ./submission.csv"
compute:
  uses_gpu_for_inference: false
  has_network_during_ranking: false
```

---

## Deployment (HuggingFace Spaces — Demo Sandbox)

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** runtime
3. Push this repo — Space auto-builds and runs FastAPI on port 7860
4. Copy the Space URL → paste as `sandbox_link` in metadata

---

## Compute Constraints

- ✅ CPU only — no GPU required
- ✅ No network calls during ranking
- ✅ Runs within 5 minutes on 16GB RAM

---

## Related

- [RRR-Resume-Ranker-Recruiter-Frontend](https://github.com/your-username/RRR-Resume-Ranker-Recruiter-Frontend) — Next.js dashboard on Vercel
