# Job Matcher AI

AI-powered job matching platform that intelligently connects candidates with relevant opportunities using NLP and machine learning.

## Features

- **Resume Parsing**: Extract skills, experience, and education using NLP
- **Smart Matching**: ML-based recommendation engine ranks jobs by fit score
- **Job Alerts**: Real-time notifications for matching positions
- **ATS Scoring**: Resume optimization with ATS compatibility analysis
- **Analytics Dashboard**: Track applications, match rates, and market trends

## Tech Stack

- **Backend**: Python, FastAPI, scikit-learn, spaCy
- **Database**: PostgreSQL, Redis (caching)
- **ML**: TF-IDF, cosine similarity, sentence transformers
- **Frontend**: Next.js, Tailwind CSS, Recharts
- **Infrastructure**: Docker, PostgreSQL, Redis

## Quick Start

```bash
git clone https://github.com/FlourishP/job-matcher-ai
cd job-matcher-ai
docker-compose up -d
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

## License

MIT — see [LICENSE](LICENSE)
