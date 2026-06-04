# RRR — Resume Ranker Recruiter

> **Hack2Skill Hackathon** | Intelligent Candidate Discovery & Ranking Challenge

Build an AI system that ranks candidates the way a great recruiter would — not by matching keywords, but by actually understanding who fits the role.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Overview](#dataset-overview)
- [Ranking Strategy](#ranking-strategy)
- [Project Structure](#project-structure)
- [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
- [Frontend Design](#frontend-design)
- [Submission Requirements](#submission-requirements)
- [Tech Stack](#tech-stack)

---

## Problem Statement

Recruiters go through hundreds of profiles and still miss the right person — not because the talent isn't there, but because keyword filters can't see what actually matters.

**Your system must:**
- Read and semantically understand a job description (not just extract words)
- Look at the full picture — career history, skills, behavioral signals, platform activity
- Deliver a shortlist that a recruiter can trust

Architecture is your choice: semantic search, LLM ranking, vector embeddings, hybrid scoring. The outcome is what gets judged.

---

## Dataset Overview

The dataset (`sample_candidates.json`) contains ~5000 candidate profiles conforming to the **Redrob Candidate Profile Schema**. Each candidate has:

| Layer | Key Fields |
|---|---|
| **Profile** | `headline`, `summary`, `years_of_experience`, `current_title`, `current_industry` |
| **Career History** | Up to 10 roles — `title`, `duration_months`, `company_size`, `description` |
| **Skills** | `name`, `proficiency` (beginner→expert), `endorsements`, `duration_months` |
| **Education** | `degree`, `field_of_study`, `institution`, `tier` (tier_1–tier_4) |
| **Redrob Signals** | `github_activity_score`, `recruiter_response_rate`, `skill_assessment_scores`, `open_to_work_flag`, `notice_period_days`, `expected_salary_range_inr_lpa` |

### Candidate ID Format
```
CAND_XXXXXXX  (7-digit zero-padded number)
```

### Key Schema Notes
- `github_activity_score`: 0–100 or `-1` if no GitHub linked
- `offer_acceptance_rate`: 0–1 or `-1` if no offer history
- `skill_assessment_scores`: dict of `skill_name → score (0–100)` from Redrob's own platform
- `institution tier`: `tier_1` (IIT/IIM-level) → `tier_4`/`unknown`

---

## Ranking Strategy

### Recommended: Weighted Multi-Factor Scoring

Combine five scoring components into a final `score` between 0 and 1:

```
final_score =
  0.35 × skill_match_score
+ 0.25 × career_fit_score
+ 0.15 × signal_modifier
+ 0.15 × education_score
+ 0.10 × availability_score
```

### Component Breakdown

#### 1. Skill Match Score (weight: 0.35)
The most important signal. Use `sentence-transformers/all-MiniLM-L6-v2` to embed:
- The job description's required + preferred skills as a single text block
- Each candidate's skill list (name + proficiency + duration)

Compute cosine similarity between the two embeddings. This beats keyword matching because "MLOps" and "model deployment" will score high even without exact token overlap.

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

jd_embedding = model.encode(jd_skills_text)
candidate_embedding = model.encode(candidate_skills_text)
score = float(util.cos_sim(jd_embedding, candidate_embedding))
```

#### 2. Career Fit Score (weight: 0.25)
Weight recent roles more heavily using exponential decay:

```python
import math

def career_fit(career_history, target_title, target_industry):
    score = 0
    for role in career_history:
        years_ago = (today - role["end_date"]).days / 365
        decay = math.exp(-0.3 * years_ago)
        title_match = fuzzy_match(role["title"], target_title)
        industry_match = 1.0 if role["industry"] == target_industry else 0.3
        score += decay * (0.6 * title_match + 0.4 * industry_match)
    return normalize(score)
```

A role from 5 years ago contributes `e^(-1.5) ≈ 0.22x` of a current role — stopping outdated jobs from dominating the score.

#### 3. Signal Modifier (weight: 0.15)
Normalize Redrob platform signals to 0–1:

```python
def signal_score(signals):
    github = max(signals["github_activity_score"], 0) / 100  # -1 → 0
    response = signals["recruiter_response_rate"]             # already 0–1
    interview = signals["interview_completion_rate"]          # already 0–1
    assessment_avg = mean(signals["skill_assessment_scores"].values()) / 100
    return mean([github, response, interview, assessment_avg])
```

**Important:** Candidates with `github_activity_score = -1` are penalized slightly (treated as 0), not eliminated — GitHub is optional.

#### 4. Education Score (weight: 0.15)

```python
TIER_WEIGHTS = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.2}

def education_score(education, required_field):
    scores = []
    for edu in education:
        tier = TIER_WEIGHTS.get(edu.get("tier", "unknown"), 0.2)
        field_match = 1.0 if required_field.lower() in edu["field_of_study"].lower() else 0.4
        scores.append(tier * field_match)
    return max(scores) if scores else 0.0
```

#### 5. Availability Score (weight: 0.10)

```python
def availability_score(signals, job_location):
    open_to_work = 1.0 if signals["open_to_work_flag"] else 0.5
    notice_norm = 1 - (signals["notice_period_days"] / 180)  # shorter = better
    reloc = 1.0 if signals["willing_to_relocate"] else 0.6
    return mean([open_to_work, notice_norm, reloc])
```

---

## Project Structure

```
RRR-Resume-Ranker-Recruiter/
├── rank.py                        ← Single entrypoint: reads candidates, writes submission.csv
├── ranker/
│   ├── __init__.py
│   ├── jd_parser.py               ← Extracts required/preferred skills, title, experience from JD
│   ├── candidate_scorer.py        ← Orchestrates all 5 scoring components per candidate
│   ├── embedding_utils.py         ← Loads all-MiniLM-L6-v2, caches embeddings to avoid recompute
│   └── signal_scorer.py           ← Normalizes redrob_signals fields to 0–1
├── data/
│   ├── sample_candidates.json     ← Hackathon-provided dataset
│   └── job_description.docx       ← Hackathon-provided JD
├── frontend/                      ← Next.js recruiter dashboard (see Frontend Design section)
├── requirements.txt
├── submission.csv                 ← Auto-generated output
├── submission_metadata.yaml       ← Fill in team details + sandbox link
└── README.md                      ← This file
```

### `requirements.txt`
```
sentence-transformers==2.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
python-docx>=1.1.0
pyyaml>=6.0
```

---

## Step-by-Step Implementation Guide

### Step 1 — Parse the Job Description

Read `job_description.docx` using `python-docx` and extract:
- Required skills (hard requirements)
- Preferred/nice-to-have skills
- Target title / seniority level
- Years of experience required
- Industry and location preferences

```python
from docx import Document

def parse_jd(path):
    doc = Document(path)
    full_text = "\n".join([p.text for p in doc.paragraphs])
    # Use regex or keyword extraction to parse sections
    return {"required_skills": [...], "preferred_skills": [...], "title": "...", "min_exp": 3}
```

### Step 2 — Set Up Embedding Pipeline

```python
# ranker/embedding_utils.py
import pickle, os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_FILE = ".embedding_cache.pkl"

def load_model():
    return SentenceTransformer(MODEL_NAME)

def get_or_compute_embeddings(texts, model, cache_key):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        if cache_key in cache:
            return cache[cache_key]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    cache = {cache_key: embeddings}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    return embeddings
```

Caching is essential — encoding 5000 candidates takes ~2 minutes on first run, ~0.1s on subsequent runs.

### Step 3 — Score Each Candidate

```python
# ranker/candidate_scorer.py
def score_candidate(candidate, jd, jd_embedding, skill_model):
    s_skill    = skill_match(candidate["skills"], jd_embedding, skill_model)
    s_career   = career_fit(candidate["career_history"], jd["title"], jd["industry"])
    s_signal   = signal_score(candidate["redrob_signals"])
    s_edu      = education_score(candidate["education"], jd["field"])
    s_avail    = availability_score(candidate["redrob_signals"], jd["location"])

    score = (0.35 * s_skill + 0.25 * s_career + 0.15 * s_signal
             + 0.15 * s_edu + 0.10 * s_avail)

    reasoning = (
        f"{candidate['profile']['current_title']} with "
        f"{candidate['profile']['years_of_experience']} yrs; "
        f"skill match {s_skill:.2f}; response rate "
        f"{candidate['redrob_signals']['recruiter_response_rate']:.2f}."
    )
    return {"candidate_id": candidate["candidate_id"], "score": round(score, 4), "reasoning": reasoning}
```

### Step 4 — Build `rank.py` Entrypoint

```python
# rank.py
import argparse, json, csv
from ranker.jd_parser import parse_jd
from ranker.embedding_utils import load_model, get_or_compute_embeddings
from ranker.candidate_scorer import score_candidate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--jd", default="data/job_description.docx")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    with open(args.candidates) as f:
        candidates = json.load(f)

    jd = parse_jd(args.jd)
    model = load_model()

    # Precompute JD embedding once
    jd_embedding = model.encode(jd["skills_text"])

    # Score all candidates
    results = [score_candidate(c, jd, jd_embedding, model) for c in candidates]

    # Sort by score descending and assign ranks
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Write submission.csv
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. Wrote {len(results)} candidates to {args.out}")

if __name__ == "__main__":
    main()
```

Run with:
```bash
python rank.py --candidates ./data/sample_candidates.json --out ./submission.csv
```

### Step 5 — Validate Your Submission

The hackathon provides `validate_submission.py`. Run it before submitting:
```bash
python validate_submission.py --submission submission.csv --candidates data/sample_candidates.json
```

### Step 6 — Fill Submission Metadata

Edit `submission_metadata.yaml`:
```yaml
team_name: "RRR-Team"
github_repo: "https://github.com/CyberCodezilla/RRR-Resume-Ranker-Recruiter-Frontend"
sandbox_link: "https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker"
reproduce_command: "python rank.py --candidates ./data/sample_candidates.json --out ./submission.csv"
compute:
  uses_gpu_for_inference: false       # MUST be false
  has_network_during_ranking: false   # MUST be false — no API calls
```

---

## Frontend Design

The frontend is a **recruiter-facing dashboard** built with Next.js 14 (App Router).

### Pages / Views

| View | Purpose |
|---|---|
| **Upload / Configure** | Paste or upload a job description; select candidate pool |
| **Ranked Results Table** | Sortable table — rank, score bar, candidate name, title, exp, skill chips |
| **Candidate Detail Drawer** | Right-side panel with full profile, career timeline, skill match breakdown |
| **Score Breakdown Panel** | Radar chart or stacked bar showing all 5 scoring components per candidate |

### Component Architecture

```
app/
├── page.tsx                  ← Upload + Configure view
├── results/
│   └── page.tsx              ← Ranked Results Table
├── components/
│   ├── CandidateTable.tsx    ← Sortable results table
│   ├── ScoreBar.tsx          ← Visual 0–1 score as progress bar
│   ├── SkillChips.tsx        ← Top matching skills as color chips
│   ├── CandidateDrawer.tsx   ← Right-side detail panel
│   ├── ScoreBreakdown.tsx    ← Radar chart (Recharts RadarChart)
│   └── ReasoningTag.tsx      ← Parses reasoning string into structured badges
└── lib/
    ├── mockData.ts           ← Loads sample_submission.csv for dev
    └── api.ts                ← Calls Python backend in production
```

### Reasoning Display

The `reasoning` field from `submission.csv` (e.g., `"HR Manager with 6.1 yrs; 9 AI core skills; response rate 0.76"`) should be parsed into structured inline tags — not rendered as raw text:

```tsx
// ReasoningTag.tsx
const parts = reasoning.split(";").map(s => s.trim());
// Renders: [HR Manager] [6.1 yrs exp] [9 AI skills] [76% response rate]
```

### Development Workflow

1. Use `sample_submission.csv` + `sample_candidates.json` as mock data during frontend dev
2. Build all components against mock data first
3. Add a `/api/rank` Next.js API route that shells out to `python rank.py` (or calls a FastAPI backend)
4. Deploy to Vercel for the sandbox link required in `submission_metadata.yaml`

---

## Submission Requirements

You need to submit three things:

1. **GitHub repo** — clean, complete, working code (this repo)
2. **PDF deck** — explaining your approach, why you built it this way, and how it works
3. **`submission.csv`** — ranked output file in the exact format:

```csv
candidate_id,rank,score,reasoning
CAND_0004989,1,0.9920,"HR Manager with 6.1 yrs; 9 AI core skills; response rate 0.76."
CAND_0001195,2,0.9840,"HR Manager with 8.7 yrs; 9 AI core skills; response rate 0.20."
...
```

### Compute Constraints
- ✅ Must run on CPU only (`uses_gpu_for_inference: false`)
- ✅ No network calls during ranking (`has_network_during_ranking: false`)
- ✅ Must complete within 5 minutes on 16GB RAM CPU

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Ranking Engine** | Python 3.11, `sentence-transformers`, `scikit-learn`, `pandas` |
| **Embeddings Model** | `all-MiniLM-L6-v2` (CPU-friendly, ~80MB, fast) |
| **Frontend** | Next.js 14 (App Router), Tailwind CSS |
| **Charts** | Recharts (RadarChart for score breakdown) |
| **Deployment** | Vercel (frontend), HuggingFace Spaces (ranker sandbox) |
| **Validation** | `validate_submission.py` (provided by hackathon) |

---

*Built for the Hack2Skill — Intelligent Candidate Discovery & Ranking Challenge.*
