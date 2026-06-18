"""
Course Catalog Mapper — Phase 3.1

Replaces LLM-invented skill names with canonical names from AVAILABLE_COURSES.
Pure functions, no external deps, no Pinecone, no side effects.

Matching: exact -> normalized -> synonym -> substring -> token overlap -> fuzzy.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher


# Synonym map: common LLM skill names → canonical catalog names
_SKILL_SYNONYMS = {
    "rest api": "API Design & Service Layer",
    "api design": "API Design & Service Layer",
    "api": "API Design & Service Layer",
    "microservices": "Microservices Architecture",
    "system design": "Foundations of System Design",
    "scalability": "Scalability & Performance",
    "distributed systems": "Distributed Systems",
    "caching": "Caching & Performance Optimization",
    "cloud deployment": "Cloud & Deployment",
    "concurrency": "Concurrency & Multithreading",
    "load balancing": "Load Balancing & API Gateways",
    "kafka": "Kafka & Event Streaming",
    "message queues": "Message Queues",
    "docker": "Docker & Containerization",
    "kubernetes": "Kubernetes & Orchestration",
    "ci/cd": "CI/CD & DevOps",
    "testing": "Testing & QA",
    "monitoring": "Monitoring & Observability",
    "security": "Security & Compliance",
    "database design": "Database Design & Data Modeling",
    "data modeling": "Database Design & Data Modeling",
    "nlp": "NLP & Sequence Models",
    "deep learning": "Deep Learning Basics",
    "mlops": "MLOps & Monitoring",
    "leadership": "Capstone & Interview Readiness",
    "technical leadership": "Capstone & Interview Readiness",
    "system architecture": "Foundations of System Design",
    "technical design documentation": "Behavioral Design Patterns",
    "architectural decision making": "Advanced Architectures",
    "architectural decisions": "Advanced Architectures",
    "design documentation": "Behavioral Design Patterns",
}

# Inline the same AVAILABLE_COURSES from roadmap_agent to stay self-contained
AVAILABLE_COURSES = {
    "ai_ready": {
        "name": "AI Ready Course",
        "modules": [
            "Python for AI", "Math Refresher for AI",
            "Algorithms & Data Handling", "Data Wrangling",
            "Data Visualization", "Deep Learning Basics",
            "Model Debugging & Explainability",
            "Collaboration & Tools", "Cloud & Deployment",
        ],
        "skills_include": [
            "Variables, Data Types, Operators", "Control Flow", "Data Structures",
            "OOP Basics", "File Handling", "Linear Algebra", "Probability",
            "Bayes Theorem", "Calculus for ML", "Sorting & Searching",
            "Data Cleaning", "NumPy", "Pandas", "Data Visualization",
            "Regression", "Classification", "Clustering", "Model Evaluation",
            "Neural Networks", "Keras", "Loss Functions", "Backpropagation",
            "Overfitting", "Cross-Validation", "Git & GitHub", "Docker",
            "FastAPI", "AWS SageMaker", "HuggingFace Spaces",
        ],
    },
    "machine_learning": {
        "name": "Machine Learning Course",
        "modules": [
            "Python for Data & ML", "Math & Stats Essentials",
            "Supervised Learning", "Unsupervised Learning",
            "Feature Engineering & Pipelines",
            "Neural Networks & Computer Vision",
            "NLP & Sequence Models", "Model Serving & APIs",
            "MLOps & Monitoring", "GenAI & Responsible AI",
        ],
        "skills_include": [
            "Linear Regression", "Logistic Regression", "Ridge & Lasso",
            "Decision Trees", "kNN", "SVM", "Model Evaluation",
            "Cross Validation", "K-Means", "PCA", "t-SNE", "UMAP",
            "Anomaly Detection", "Scikit-learn Pipelines",
            "Feature Engineering", "CNN", "Transfer Learning",
            "RNN", "LSTM", "Transformers", "BERT", "GPT",
            "FastAPI", "Docker", "MLflow", "Weights & Biases",
            "Drift Detection", "Prompt Engineering", "LoRA",
            "Fine-tuning", "RLHF",
        ],
    },
    "generative_ai": {
        "name": "Generative AI Course",
        "modules": [
            "Foundations & Overview", "Core Generative Model Families",
            "Diffusion & Modern Image Generation", "Transformers & Attention",
            "Tokenization & Embeddings", "Prompting & Inference Techniques",
            "Tools & Frameworks", "Fine-Tuning & Safety",
            "Multimodal & Advanced Applications", "NLP", "RAG & Capstone",
        ],
        "skills_include": [
            "VAE", "GANs", "Diffusion Models", "Stable Diffusion",
            "Self-attention", "Multi-head Attention", "Transformer Architecture",
            "Tokenization", "Word Embeddings", "Prompt Engineering",
            "Zero-shot", "Few-shot", "HuggingFace", "LangChain",
            "Vector Databases", "LLM APIs", "Streamlit", "SFT",
            "LoRA", "QLoRA", "RLHF", "Safety", "CLIP", "Whisper",
            "RAG", "Chatbot",
        ],
    },
    "computer_vision": {
        "name": "Computer Vision Course",
        "modules": [
            "OpenCV Basics", "Image Processing", "Feature Detection",
            "Classical Computer Vision", "Deep Learning Basics",
            "Advanced Architectures", "Generative Models",
            "Deployment", "Research Literacy",
        ],
        "skills_include": [
            "Image Filtering", "Edge Detection", "Thresholding",
            "SIFT", "SURF", "ORB", "Feature Matching", "Face Detection",
            "Object Detection", "Image Segmentation", "CNN", "ResNet",
            "YOLO", "Faster R-CNN", "U-Net", "Autoencoders", "GAN",
            "DCGAN", "Neural Style Transfer", "Model Quantization",
        ],
    },
    "high_level_system_design": {
        "name": "High Level System Design",
        "modules": [
            "Foundations of System Design", "Networking & Communication",
            "Databases & Storage", "Distributed Systems",
            "Microservices Architecture", "Scalability & Performance",
            "Security, Reliability & Observability",
            "Case Studies", "Capstone & Career Readiness",
        ],
        "skills_include": [
            "Scalability", "Availability", "Load Balancing", "Caching",
            "Microservices", "SQL vs NoSQL", "Sharding", "Replication",
            "CAP Theorem", "Message Queues", "Kafka",
            "Event-Driven Architecture", "Docker", "CI/CD",
            "Rate Limiting", "API Gateway", "Auth OAuth2", "JWT",
            "Logging & Monitoring", "System Design Twitter",
            "System Design WhatsApp", "System Design Netflix",
            "System Design Uber",
        ],
    },
    "low_level_system_design": {
        "name": "Low Level System Design",
        "modules": [
            "OOP & SOLID Principles", "Creational & Structural Patterns",
            "Behavioral Design Patterns", "Database Design & Data Modeling",
            "API Design & Service Layer", "Concurrency & Multithreading",
            "Caching & Performance Optimization",
            "Testing, Logging & DevOps Foundations",
            "AI-Powered Systems", "Capstone & Interview Readiness",
        ],
        "skills_include": [
            "SOLID Principles", "UML Diagrams", "Singleton", "Factory",
            "Builder", "Decorator", "Adapter", "Composite", "Proxy",
            "Strategy", "Observer", "State Pattern", "Command Pattern",
            "SQL vs NoSQL", "ORM", "ER Diagrams", "REST Principles",
            "Dependency Injection", "JWT", "Concurrency", "Deadlocks",
            "Producer Consumer", "LRU Cache", "Redis", "Unit Testing",
            "RAG", "Vector Databases",
        ],
    },
}


def normalize_skill_name(name: str) -> str:
    """Lowercase, remove punctuation, replace underscores with spaces, collapse whitespace.
    
    Also strips parenthetical content (e.g. '(Kubernetes)' → '')
    and normalizes '&' to 'and'.
    """
    name = name.lower().strip()
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\([^)]*\)", "", name)  # strip (parentheticals)
    name = name.replace("&", " and ")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def _token_stem(words: set[str]) -> set[str]:
    """Produce canonical stems for singular/plural matching."""
    stems = set()
    for w in words:
        stem = w
        if stem.endswith("s") and len(stem) > 3:
            stem = stem[:-1]  # queues → queue, patterns → pattern
        if stem.endswith("ing") and len(stem) > 4:
            stem = stem[:-3]  # caching → cach
            stem_r = stem + "e"  # caching → cache
            stems.add(stem_r)
        stems.add(stem)
    return stems


def build_catalog_index() -> list[dict]:
    """Build a searchable index from AVAILABLE_COURSES (skills + modules)."""
    entries = []
    for course_id, course in AVAILABLE_COURSES.items():
        for skill in course.get("skills_include", []):
            entries.append({
                "type": "skill",
                "course_id": course_id,
                "course_title": course["name"],
                "canonical_name": skill,
                "normalized": normalize_skill_name(skill),
            })
        for module in course.get("modules", []):
            entries.append({
                "type": "module",
                "course_id": course_id,
                "course_title": course["name"],
                "canonical_name": module,
                "normalized": normalize_skill_name(module),
            })
    return entries


def _score_match(norm: str, norm_words: set, en: str) -> float:
    """Score a single catalog entry against a normalized skill name."""
    en = en.strip()
    if not en:
        return 0.0
    # 1. Exact
    if norm == en:
        return 1.0
    # 2. Substring (one contained in the other)
    if norm in en or en in norm:
        return 0.85
    # 3. Token overlap with stem matching
    en_words = set(en.split())
    if norm_words and en_words:
        norm_stems = _token_stem(norm_words)
        en_stems = _token_stem(en_words)
        common_stems = norm_stems & en_stems
        stem_overlap = len(common_stems)
        max_len = max(len(norm_words), len(en_words))
        if stem_overlap > 0:
            overlap_ratio = stem_overlap / max_len
            # Require at least 2 overlapping stems when input has 3+ words
            # to prevent false positives like "Data Consistency Models" -> "Diffusion Models"
            min_stems = 2 if len(norm_words) >= 3 else 1
            if stem_overlap >= min_stems:
                if overlap_ratio >= 0.5:
                    return min(0.6 + overlap_ratio * 0.3, 0.95)
                if overlap_ratio >= 0.3:
                    return 0.45 + overlap_ratio * 0.3
    # 4. Fuzzy
    return SequenceMatcher(None, norm, en).ratio() * 0.8


def find_best_catalog_match(skill_name: str, catalog_index: list[dict]) -> dict:
    """
    Find best catalog match for a skill name.

    Matching order:
      1. exact match (score 1.0)
      2. normalized match (score 1.0)
      3. synonym match (score 0.95)
      4. substring match (score 0.85)
      5. stem-aware token overlap (score 0.60-0.95)
      6. fuzzy similarity (score 0.0-0.80)

    Tries multiple variants of the input name (with/without parentheticals, etc.)

    Returns:
      {"matched": bool, "course_id": str, "course_title": str,
       "canonical_name": str, "confidence": float}
    """
    norm = normalize_skill_name(skill_name)
    if not norm:
        return {"matched": False, "course_id": "", "course_title": "",
                "canonical_name": "", "confidence": 0.0}

    # 0. Synonym check (before index scan)
    norm_lower = norm.lower().strip()
    for syn_key, syn_val in _SKILL_SYNONYMS.items():
        if norm_lower == syn_key or syn_key in norm_lower or norm_lower in syn_key:
            # Found synonym — return canonical match immediately
            syn_norm = normalize_skill_name(syn_val)
            for entry in catalog_index:
                if entry["normalized"] == syn_norm:
                    return {
                        "matched": True,
                        "course_id": entry["course_id"],
                        "course_title": entry["course_title"],
                        "canonical_name": entry["canonical_name"],
                        "confidence": 0.95,
                    }
            # Fallback: return canonical name even without index hit
            return {
                "matched": True,
                "course_id": "",
                "course_title": "",
                "canonical_name": syn_val,
                "confidence": 0.90,
            }

    norm_words = set(norm.split())
    best = None
    best_score = 0.0

    for entry in catalog_index:
        score = _score_match(norm, norm_words, entry["normalized"])
        if score > best_score:
            best_score = score
            best = {
                "matched": score >= 0.60,
                "course_id": entry["course_id"],
                "course_title": entry["course_title"],
                "canonical_name": entry["canonical_name"],
                "confidence": round(min(score, 1.0), 3),
            }

    # If no match found with full name, try a shorter variant (strip parentheticals from raw)
    if best is None or not best["matched"]:
        stripped = re.sub(r"\s*\([^)]*\)", "", skill_name).strip()
        if stripped and stripped != skill_name:
            norm2 = normalize_skill_name(stripped)
            if norm2 and norm2 != norm:
                # Try synonym on stripped version too
                norm2_lower = norm2.lower().strip()
                for syn_key, syn_val in _SKILL_SYNONYMS.items():
                    if norm2_lower == syn_key or syn_key in norm2_lower or norm2_lower in syn_key:
                        syn_norm = normalize_skill_name(syn_val)
                        for entry in catalog_index:
                            if entry["normalized"] == syn_norm:
                                return {
                                    "matched": True,
                                    "course_id": entry["course_id"],
                                    "course_title": entry["course_title"],
                                    "canonical_name": entry["canonical_name"],
                                    "confidence": 0.95,
                                }
                norm2_words = set(norm2.split())
                for entry in catalog_index:
                    score = _score_match(norm2, norm2_words, entry["normalized"])
                    if score > best_score:
                        best_score = score
                        best = {
                            "matched": score >= 0.60,
                            "course_id": entry["course_id"],
                            "course_title": entry["course_title"],
                            "canonical_name": entry["canonical_name"],
                            "confidence": round(min(score, 1.0), 3),
                        }

    if best is None:
        return {"matched": False, "course_id": "", "course_title": "",
                "canonical_name": "", "confidence": 0.0}
    return best


def repair_course_catalog_alignment(roadmap_data: dict) -> dict:
    """
    Run AFTER roadmap generation, BEFORE validators.

    Two-pass mapping:
      Pass 1: map skills with confidence >= 0.75 (replace title + store metadata)
      Pass 2: for unmapped skills, force-map with any candidate >= 0.70

    Returns coverage stats dict with keys:
      coverage, mapped, total, missing_skills
    """
    catalog_index = build_catalog_index()
    total = 0
    mapped = 0
    missing_skills = []

    # Pass 1: map confidently matched skills
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                total += 1
                name = skill.get("title") or skill.get("n", "")
                if not name:
                    continue
                result = find_best_catalog_match(name, catalog_index)
                if result["matched"] and result["confidence"] >= 0.75:
                    skill["title"] = result["canonical_name"]
                    skill["catalog_course_id"] = result["course_id"]
                    skill["catalog_match_confidence"] = result["confidence"]
                    skill["catalog_status"] = "mapped"
                    mapped += 1
                else:
                    skill["catalog_status"] = "missing"
                    missing_skills.append(name)

    # Pass 2: force-repair unmapped skills with any candidate >= 0.70
    repaired_in_pass2 = 0
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                if skill.get("catalog_status") != "missing":
                    continue
                name = skill.get("title") or skill.get("n", "")
                if not name:
                    continue
                result = find_best_catalog_match(name, catalog_index)
                if result["matched"] and result["confidence"] >= 0.70:
                    skill["title"] = result["canonical_name"]
                    skill["catalog_course_id"] = result["course_id"]
                    skill["catalog_match_confidence"] = result["confidence"]
                    skill["catalog_status"] = "mapped"
                    mapped += 1
                    repaired_in_pass2 += 1
                    if name in missing_skills:
                        missing_skills.remove(name)

    if repaired_in_pass2 > 0:
        print(f"[CATALOG REPAIR] Pass 2: force-mapped {repaired_in_pass2} skill(s)")

    coverage = round(mapped / max(total, 1), 3)
    return {
        "coverage": coverage,
        "mapped": mapped,
        "total": total,
        "missing": missing_skills,
    }


def _build_flat_skill_set() -> set[str]:
    """Return set of all canonical skill names (lowercase)."""
    result = set()
    for course in AVAILABLE_COURSES.values():
        for s in course.get("skills_include", []):
            result.add(s.lower())
    return result


ALL_CATALOG_SKILLS = _build_flat_skill_set()
