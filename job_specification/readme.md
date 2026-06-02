# Job Recommendation System 

This repository contains scripts for fetching jobs, embedding them, storing them in Pinecone, retrieving them, computing job fit, and generating a personalised job recommendation output.

---

### 📁 Project Structure

```
src/
├── adzuna_client.py        # Fetch jobs from Adzuna API with retry + pagination
├── config.py               # Loads .env, keys & configuration variables
├── embeddings.py           # Embedding generation using OpenAI embeddings
├── job_fit.py              # Calculates job_fit_score & classifies jobs
├── job_retriever.py        # Retrieves jobs from Pinecone DB (profile / resume based)
├── personalize_agent.py    # Generates personalised recommendation summary
├── pipeline.py             # Fetch jobs -> save CSV -> embed -> upsert to Pinecone
├── role_generator.py       # Generates job role keywords using LLM
├── skills_utils.py         # Basic keyword skill extraction logic
└── vector_db.py            # Pinecone index initialization + connection
```

---

### 🔧 What each module does

| File | Responsibility |
|------|---------------|
| adzuna_client.py | Calls Adzuna API and returns raw job listings |
| config.py | Loads environment variables (.env) and defines settings |
| embeddings.py | Generates vector embeddings for text |
| job_fit.py | Compares user skills vs job skills + calculates job_fit_score + classification |
| job_retriever.py | Fetches best matched jobs stored in Pinecone |
| personalize_agent.py | Combines jobs + job_fit results → returns human readable recommendation |
| pipeline.py | Full job ingestion pipeline: fetch → CSV → embed → upsert |
| role_generator.py | LLM based role keyword generator |
| skills_utils.py | Extracts skills from text using keyword scan |
| vector_db.py | Creates/loads Pinecone index |

---

### 🛠 Run Job Fetch + Storage Pipeline

```
python src/pipeline.py
```

---

### 🛠 Generate Personalised Recommendation for a Resume

1. Put resume_id and profile_text inside personalize_agent.py  
2. Run:

```
python src/personalize_agent.py
```

Output → Eligible + Near-Eligible job list + courses (if missing skills)

---

### 📝 Environment Variables Required (.env)

```
ADZUNA_APP_ID=
ADZUNA_APP_KEY=
ADZUNA_COUNTRY=in

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

PINECONE_API_KEY=
PINECONE_INDEX_NAME=job-embeddings-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

RESUME_INDEX_NAME=resumes-index
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

---

### 💡 Notes

- Install dependencies using: `pip install -r requirements.txt`

