"""
Generate 8 MCQs per module from a local roadmap_example.json
Uses LangChain OpenAI wrapper. Expects OPENAI_API_KEY in environment or .env.
Outputs: generated_mcqs.json
"""

import os
import json
from dotenv import load_dotenv

from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY in environment or .env")

# initialize LLM (adjust model_name if needed)
llm = OpenAI(temperature=0.4, api_key=OPENAI_API_KEY, model="gpt-4o-mini")

prompt_template = """
You are an AI MCQ generator.

Generate 8 to 10 high-quality multiple-choice questions (MCQs) for the module below.
Each question must have exactly 4 options labeled in-line as "A. ...", "B. ...", "C. ...", "D. ...".
Return STRICT JSON only, with this structure:

{{
  "ModuleName": "<module name>",
  "MCQs": [
    {{
      "question": "<text>",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct_answer": "A"
    }},
    ...
  ]
}}

Module: {module_name}
Topics:
{topics_text}
"""

prompt = PromptTemplate(input_variables=["module_name", "topics_text"], template=prompt_template)
chain = prompt | llm

def generate_mcqs_for_module(module: dict) -> dict:
    module_name = module.get("ModuleName", "Module")
    topics_text = "\n".join([f"- {t}" for t in module.get("Topics", [])])
    resp = chain.invoke({"module_name": module_name, "topics_text": topics_text})
    text = resp.strip() if isinstance(resp, str) else str(resp)

    # try parse JSON directly, else try code fence extraction
    try:
        return json.loads(text)
    except Exception:
        if "```json" in text:
            try:
                json_str = text.split("```json", 1)[1].split("```", 1)[0].strip()
                return json.loads(json_str)
            except Exception:
                pass
    # fallback - return raw output for debugging
    return {"ModuleName": module_name, "raw_output": text}

def main():
    # load roadmap
    with open("roadmap_example.json", "r", encoding="utf-8") as f:
        roadmap = json.load(f)["roadmap"]

    results = []
    for module in roadmap.get("Modules", []):
        print(f"Generating MCQs for: {module['ModuleName']}")
        mcq_block = generate_mcqs_for_module(module)
        results.append(mcq_block)

    output = {"roadmap_title": roadmap.get("CourseTitle"), "mcqs": results}
    with open("generated_mcqs.json", "w", encoding="utf-8") as out:
        json.dump(output, out, indent=2, ensure_ascii=False)

    print("Saved generated_mcqs.json")

if __name__ == "__main__":
    main()
