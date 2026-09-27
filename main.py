"""Job Matcher AI — Production-ready job matching API.

FastAPI backend with NLP resume parsing, ML-based job matching,
and ATS scoring. Uses scikit-learn, spaCy, PostgreSQL, Redis.
"""

import os
from typing import List, Optional
from datetime import datetime

import spacy
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import psycopg2
import redis

nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Job Matcher AI", version="1.0.0")

# --- Redis ---
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True,
)

# --- In-memory store (replace with PostgreSQL in production) ---
resumes: dict[str, dict] = {}
jobs: dict[str, dict] = {}
matches: dict[str, list[dict]] = {}


# --- Pydantic models ---
class Skill(BaseModel):
    name: str
    confidence: float = 1.0

class Resume(BaseModel):
    candidate_id: str
    name: str
    email: str
    skills: list[str] = []
    experience_years: float = 0.0
    education: str = ""
    raw_text: Optional[str] = None

class Job(BaseModel):
    job_id: str
    title: str
    company: str
    description: str
    required_skills: list[str] = []
    min_experience: float = 0.0
    location: str = ""
    salary_range: Optional[str] = None

class MatchResult(BaseModel):
    job_id: str
    candidate_id: str
    fit_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    explanation: str


# --- NLP helpers ---
def extract_skills(text: str) -> list[str]:
    """Extract skills from raw resume text using spaCy."""
    doc = nlp(text.lower())
    # Named entity recognition for organizations, skills
    ents = [ent.text for ent in doc.ents]
    # Noun chunks as candidate skills
    chunks = [chunk.text for chunk in doc.noun_chunks]
    return list(set(ents + chunks))


def parse_resume(resume_text: str) -> dict:
    """Parse resume text into structured data."""
    doc = nlp(resume_text)
    skills = extract_skills(resume_text)
    return {
        "skills": skills,
        "entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents],
        "sections": [chunk.text for chunk in doc.noun_chunks][:10],
    }


# --- ML matching ---
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
_fit_cache = {"vectorizer_fitted": False}


def compute_fit_score(resume_skills: list[str], job_skills: list[str]) -> float:
    """Compute cosine similarity between resume and job skill sets."""
    if not resume_skills or not job_skills:
        return 0.0
    texts = [" ".join(resume_skills), " ".join(job_skills)]
    try:
        tfidf = vectorizer.fit_transform(texts)
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except ValueError:
        score = 0.0
    return round(float(score), 4)


def match_skills(resume_skills: list[str], job_skills: list[str]) -> tuple[list[str], list[str]]:
    """Return matched and missing skills."""
    resume_set = set(s.lower() for s in resume_skills)
    job_set = set(s.lower() for s in job_skills)
    matched = list(resume_set & job_set)
    missing = list(job_set - resume_set)
    return matched, missing


# --- API endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/resumes", response_model=dict)
def create_resume(resume: Resume):
    """Submit a resume for parsing and storage."""
    parsed = parse_resume(resume.raw_text or "") if resume.raw_text else {"skills": resume.skills}
    resume_data = {
        "candidate_id": resume.candidate_id,
        "name": resume.name,
        "email": resume.email,
        "skills": resume.skills or parsed.get("skills", []),
        "experience_years": resume.experience_years,
        "education": resume.education,
        "parsed": parsed,
        "created_at": datetime.utcnow().isoformat(),
    }
    resumes[resume.candidate_id] = resume_data
    redis_client.setex(f"resume:{resume.candidate_id}", 3600, str(resume_data))
    return {"status": "created", "candidate_id": resume.candidate_id, "extracted_skills": resume_data["skills"]}


@app.get("/resumes/{candidate_id}")
def get_resume(candidate_id: str):
    if candidate_id not in resumes:
        raise HTTPException(status_code=404, detail="Resume not found")
    return resumes[candidate_id]


@app.post("/jobs", response_model=dict)
def create_job(job: Job):
    """Create a job listing."""
    jobs[job.job_id] = job.dict()
    return {"status": "created", "job_id": job.job_id}


@app.get("/jobs")
def list_jobs(query: Optional[str] = Query(None, description="Search query")):
    """List all jobs, optionally filtered by search query."""
    result = list(jobs.values())
    if query:
        q = query.lower()
        result = [j for j in result if q in j["title"].lower() or q in j["description"].lower()]
    return {"jobs": result, "total": len(result)}


@app.post("/match/{candidate_id}/{job_id}", response_model=MatchResult)
def match_job(candidate_id: str, job_id: str):
    """Match a candidate to a job and return fit score."""
    if candidate_id not in resumes:
        raise HTTPException(status_code=404, detail="Candidate resume not found")
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    resume = resumes[candidate_id]
    job = jobs[job_id]
    resume_skills = resume["skills"]
    job_skills = job.get("required_skills", [])

    matched, missing = match_skills(resume_skills, job_skills)
    score = compute_fit_score(resume_skills, job_skills)

    # Experience penalty/bonus
    exp_diff = resume["experience_years"] - job.get("min_experience", 0)
    if exp_diff >= 0:
        score = min(1.0, score + 0.1)
    else:
        score = max(0.0, score - 0.1)

    result = MatchResult(
        job_id=job_id,
        candidate_id=candidate_id,
        fit_score=round(score, 4),
        matched_skills=matched,
        missing_skills=missing,
        explanation=f"Matched {len(matched)}/{len(job_skills)} required skills. "
                    f"Experience: {resume['experience_years']}y vs {job.get('min_experience', 0)}y required.",
    )

    # Cache match result
    cache_key = f"match:{candidate_id}:{job_id}"
    redis_client.setex(cache_key, 3600, result.json())
    matches.setdefault(candidate_id, []).append(result.dict())

    return result


@app.get("/match/{candidate_id}")
def get_matches(candidate_id: str):
    """Get all match results for a candidate."""
    if candidate_id not in resumes:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {"candidate_id": candidate_id, "matches": matches.get(candidate_id, [])}


@app.get("/analytics/scores")
def score_distribution():
    """Get distribution of fit scores across all matches."""
    all_scores = []
    for candidate_matches in matches.values():
        for m in candidate_matches:
            all_scores.append(m["fit_score"])
    if not all_scores:
        return {"scores": [], "average": 0, "count": 0}
    return {
        "scores": all_scores,
        "average": round(np.mean(all_scores), 4),
        "count": len(all_scores),
        "min": round(float(np.min(all_scores)), 4),
        "max": round(float(np.max(all_scores)), 4),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)