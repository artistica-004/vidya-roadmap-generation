import json
import logging
import base64
from pathlib import Path
from src.models import CourseRoadmap
from src.pdf_generator import render_pdf

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        # 1. Force the event to be a dictionary
        if isinstance(event, str):
            event = json.loads(event)
        
        # 2. Get the body (handle cases where body is a string or doesn't exist)
        body = event.get("body", event)
        if isinstance(body, str):
            body = json.loads(body)
            
        # 3. Pull the roadmap data
        roadmap_json = body.get("roadmap")
        if not roadmap_json:
            return {"statusCode": 400, "body": json.dumps({"error": "No roadmap data found"})}

        # 4. Validate with Pydantic
        # Ensure the keys in your test command match your Pydantic model exactly!
        roadmap = CourseRoadmap(**roadmap_json)

        # 5. PDF Generation
        temp_pdf_path = Path("/tmp/generated_notes.pdf")
        render_pdf(roadmap=roadmap, output_path=temp_pdf_path)

        with open(temp_pdf_path, "rb") as f:
            encoded_pdf = base64.b64encode(f.read()).decode("utf-8")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="notes.pdf"'
            },
            "body": encoded_pdf,
            "isBase64Encoded": True
        }

    except Exception as e:
        logger.exception("Global Handler Error")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }