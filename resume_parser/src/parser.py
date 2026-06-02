import argparse
import json
import logging
import os
import sys
import threading
import uuid
import time
from copy import deepcopy
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# ---------------- LangChain ----------------
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_extraction_chain_pydantic
import google.generativeai as gemini_genai

# ---------------- Gemini (for embeddings) ----------------
from google import genai
from google.genai import types

# ---------------- Pinecone ----------------
from pinecone import Pinecone, ServerlessSpec

from src.pydantic_models_prompts import (
    Education,
    WorkExperience,
    basic_details_prompt,
    skills_prompt,
    fallback_education_prompt,
    companies_prompt,
    work_experience_prompt,
)
from src.utils import extract_emails, extract_github_and_linkedin_urls, output_template

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "live-assistant-index-v2")

# Gemini embedding model (3072 dimensions)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIMENSION = 3072  # IMPORTANT


RESUME_SUMMARY_PROMPT = """
You are given a parsed resume JSON.

Create a concise professional summary (6–8 lines) that captures:
- Current role / profile
- Key technical skills
- Experience level (student / fresher / experienced)
- Career direction

Write in third person.
Do NOT add recommendations, advice, or teaching.

Resume JSON:
{resume_json}
"""


# =========================================================
# Resume Manager
# =========================================================

class ResumeManager:
    def __init__(self, resume_file, model_name, extension=None):
        self.output = deepcopy(output_template)
        self.resume = get_resume_content(resume_file, extension)

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            request_timeout=30,
            max_retries=2,
        )
    
    def process_file(self):
        threads = []
        for fn in [
            self.extract_basic_info,
            self.extract_skills,
            self.extract_education,
            self.extract_work_experience,
        ]:
            t = threading.Thread(target=fn)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
            

    def query_model(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content

    def query_gemini_model(self, prompt: str) -> str:
        """Fallback: call Gemini flash when OpenAI fails."""
        gemini_genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = gemini_genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text

    def query_model_with_fallback(self, prompt: str) -> str:
        """Try OpenAI first; fall back to Gemini on any error."""
        try:
            result = self.query_model(prompt)
            logger.info("[LLM] ✅ OpenAI responded successfully.")
            return result
        except Exception as e:
            logger.warning("=" * 60)
            logger.warning("[FALLBACK TRIGGERED] OpenAI failed.")
            logger.warning(f"[FALLBACK REASON] {type(e).__name__}: {e}")
            logger.warning("[FALLBACK ACTION] Switching to Gemini...")
            logger.warning("=" * 60)
            result = self.query_gemini_model(prompt)
            logger.info("[LLM] ✅ Gemini responded successfully as fallback.")
            return result

    # ---------------- Extractors ----------------

    def extract_basic_info(self):
        try:
            result = json.loads(
                self.query_model_with_fallback(basic_details_prompt.format(resume=self.resume))
            )
            self.output["candidate_name"] = result.get("name", "")
            self.output["job_title"] = result.get("job_title", "")
            self.output["bio"] = result.get("bio", "")
            self.output["contact_info"]["location"] = result.get("location", "")
            self.output["contact_info"]["phone_number"] = result.get("phone", "")
        except Exception as e:
            logger.warning(f"Basic info extraction failed: {e}")

        self.output["contact_info"]["email_address"] = extract_emails(self.resume)
        self.output["contact_info"]["personal_urls"] = extract_github_and_linkedin_urls(
            self.resume
        )

    def extract_skills(self):
        try:
            result = json.loads(
                self.query_model_with_fallback(skills_prompt.format(resume=self.resume))
            )
            self.output["skills"] = result.get("skills", [])
            self.output["professional_development"] = result.get(
                "professional_development", []
            )
            self.output["other_info"] = result.get("other", [])
        except Exception as e:
            logger.warning(f"Skills extraction failed: {e}")

    def extract_education(self):
        try:
            chain = create_extraction_chain_pydantic(Education, self.llm)
            result = chain.invoke({"input": self.resume})
            self.output["education"] = [json.loads(x.json()) for x in result]
        except Exception:
            self.output["education"] = self.query_model_with_fallback(
                fallback_education_prompt.format(resume=self.resume)
            )

    def extract_work_experience(self):
        try:
            chain = create_extraction_chain_pydantic(WorkExperience, self.llm)
            result = chain.invoke({"input": self.resume})
            self.output["work_output"] = [json.loads(x.json()) for x in result]
        except Exception:
            self.fallback_extract_work_experience()

    def fallback_extract_work_experience(self):
        output = self.query_model_with_fallback(companies_prompt.format(resume=self.resume))
        self.output["work_output"] = []

        for line in output.split("\n"):
            if "answer" in line.lower():
                continue

            parts = line.split(",")
            if not parts:
                continue

            company = parts[0].strip()
            role = parts[1].strip() if len(parts) > 1 else ""

            try:
                parsed = json.loads(
                    self.query_model_with_fallback(
                        work_experience_prompt.format(
                            resume=self.resume,
                            role=role,
                            company=company,
                        )
                    )
                )
                self.output["work_output"].append(parsed)
            except Exception:
                pass


# =========================================================
# Resume File Reader
# =========================================================

def get_resume_content(file_path, extension=None):
    if hasattr(file_path, "read"):
        tmp = "/tmp/resume_tmp"
        with open(tmp, "wb") as f:
            f.write(file_path.read())
        file_path = tmp

    if not extension:
        extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join([p.extract_text() or "" for p in reader.pages])

    if extension in [".doc", ".docx"]:
        from docx import Document
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    raise ValueError("Unsupported file format")


# =========================================================
# Pinecone Storage
# =========================================================

def store_embeddings_in_pinecone(parsed: dict, user_id: str):

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        logger.info(f"Creating index '{INDEX_NAME}' with dimension {EMBED_DIMENSION}...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
            ),
        )

        # Safe wait (max 60 seconds)
        for _ in range(60):
            if pc.describe_index(INDEX_NAME).status["ready"]:
                break
            time.sleep(1)
        else:
            raise TimeoutError("Pinecone index not ready after 60 seconds")

        logger.info("Index ready.")

    index = pc.Index(INDEX_NAME)

    # Gemini embed client
    embed_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
    summary = generate_resume_summary(llm, parsed)
    formatted_context = format_resume_for_voice(parsed)

    result = embed_client.models.embed_content(
        model=EMBED_MODEL,
        contents=summary,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )

    summary_vector = result.embeddings[0].values

    # Validate dimension
    if len(summary_vector) != EMBED_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch: got {len(summary_vector)}, expected {EMBED_DIMENSION}"
        )

    index.upsert(
        vectors=[{
            "id": f"{user_id}_resume_summary",
            "values": summary_vector,
            "metadata": {
                "doc_type": "resume_summary",
                "text": summary,
                "formatted_context": formatted_context,
                "structured_data": json.dumps(parsed),
                "source": "resume",
                "timestamp": int(time.time()),
            },
        }],
        namespace=user_id,
    )

    logger.info(f"Resume stored for user_id: {user_id}")


# =========================================================
# Summary Generator
# =========================================================

def generate_resume_summary(llm: ChatOpenAI, parsed: dict) -> str:
    prompt = RESUME_SUMMARY_PROMPT.format(
        resume_json=json.dumps(parsed, ensure_ascii=False)
    )
    try:
        result = llm.invoke(prompt).content.strip()
        logger.info("[SUMMARY] ✅ OpenAI generated resume summary successfully.")
        return result
    except Exception as e:
        logger.warning("=" * 60)
        logger.warning("[SUMMARY FALLBACK TRIGGERED] OpenAI failed for summary.")
        logger.warning(f"[SUMMARY FALLBACK REASON] {type(e).__name__}: {e}")
        logger.warning("[SUMMARY FALLBACK ACTION] Switching to Gemini...")
        logger.warning("=" * 60)
        gemini_genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = gemini_genai.GenerativeModel("gemini-2.0-flash")
        result = model.generate_content(prompt).text.strip()
        logger.info("[SUMMARY] ✅ Gemini generated resume summary successfully as fallback.")
        return result

def format_resume_for_voice(parsed: dict) -> str:
    lines = []
    name = parsed.get("candidate_name", "")
    job_title = parsed.get("job_title", "")

    if name:
        lines.append(f"Name: {name}")
    if job_title:
        lines.append(f"Current Role: {job_title}")

    skills = parsed.get("skills", [])
    if skills:
        lines.append(f"\nTechnical Skills: {', '.join(skills)}")

    return "\n".join(lines)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path to resume PDF/DOCX")
    parser.add_argument("--user_id", help="User ID (optional)")
    args = parser.parse_args()

    user_id = args.user_id or str(uuid.uuid4())

    rm = ResumeManager(args.file_path, "gpt-3.5-turbo-1106")
    rm.process_file()

    os.makedirs("parsed_outputs", exist_ok=True)
    out_path = f"parsed_outputs/{Path(args.file_path).stem}_output.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rm.output, f, indent=2, ensure_ascii=False)

    store_embeddings_in_pinecone(rm.output, user_id)

    print(f"✔ Resume parsed and stored successfully for user_id={user_id}")
