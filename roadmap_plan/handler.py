import os
import json
import traceback
import uuid
import requests
import boto3  
from datetime import datetime
from dotenv import load_dotenv

load_dotenv() 
from src.roadmap_agent import run_pipeline
from src.pinecone_utils import retrieve_session_id


def _response(status_code: int, body: dict):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def generate_unique_id(prefix: str) -> str:
    """Generate unique ID with timestamp and UUID"""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique_part = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_part}"


def submit_to_backend(roadmap_data: dict, auth_token: str) -> dict:
    """Submit complete roadmap (with MCQs) to SQS"""

    SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]
    
    sqs = boto3.client("sqs")

    print(f"[SQS] Submitting roadmap to queue: {SQS_QUEUE_URL}")
    
    try:
        message_body = {
            **roadmap_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        response = sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(message_body)
        )

        print(f"[SQS] ✓ Successfully sent message")
        print(f"[SQS] MessageId: {response.get('MessageId')}")


        return {"success": True, "message_id": response.get("MessageId")}

    except Exception as e:
        print(f"[SQS] ✗ Submission failed: {e}")
        return {"success": False, "error": str(e)}


def lambda_handler(event, context):
    try:
        # ---------- 1. Parse incoming event ----------
        payload = {}

        # Case A: SQS event (batch of records)
        if isinstance(event, dict) and "Records" in event:
            record = event["Records"][0]
            if "body" in record:
                try:
                    payload = json.loads(record["body"])
                except json.JSONDecodeError:
                    payload = {}
        
        # Case B: API Gateway proxy with JSON body
        elif isinstance(event, dict) and "body" in event:
            raw_body = event.get("body")
            if raw_body:
                try:
                    payload = json.loads(raw_body)
                except json.JSONDecodeError:
                    payload = {}

        # Case C: Direct Lambda invoke
        elif isinstance(event, dict) and "user_id" in event:
            payload = event

        # Case D: Query parameters
        elif isinstance(event, dict) and event.get("queryStringParameters"):
            payload = event["queryStringParameters"] or {}

        if not payload:
            payload = {}

        # ---------- 2. Extract and validate required fields ----------
        user_id = payload.get("user_id")
        if not user_id:
            return _response(400, {
                "status": "error",
                "message": "Missing 'user_id' in request"
            })

        # Get auth token
        auth_token = payload.get("auth_token") or os.environ.get("AUTH_TOKEN")
        
        print(f"\n[LAMBDA] ==========================================")
        print(f"[LAMBDA] Processing roadmap generation")
        print(f"[LAMBDA] ==========================================")
        print(f"  - user_id: {user_id}")
        print(f"  - auth_token: {'✓ Present' if auth_token else '✗ Missing'}")

        # ---------- 3. RETRIEVE SESSION ID FROM PINECONE ----------
        print(f"\n[LAMBDA] Retrieving ai_session_id from Pinecone...")
        
        ai_session_id = retrieve_session_id(str(user_id))
        
        if not ai_session_id:
            print(f"[LAMBDA] ⚠ No session ID found in Pinecone - generating new one")
            ai_session_id = generate_unique_id("ai_sess")
        else:
            print(f"[LAMBDA] ✓ Retrieved session ID: {ai_session_id}")
        
        # Generate roadmap ID
        ai_roadmap_id = payload.get("ai_roadmap_id") or generate_unique_id("ai_roadmap")
        
        print(f"  - ai_session_id: {ai_session_id} (from Pinecone)")
        print(f"  - ai_roadmap_id: {ai_roadmap_id}")

        # ---------- 4. Generate complete roadmap (with MCQs) ----------
        print(f"\n[LAMBDA] Starting roadmap generation...")
        
        roadmap_data = run_pipeline(
            user_id=str(user_id),
            ai_session_id=ai_session_id,
            ai_roadmap_id=ai_roadmap_id
        )
        
        # Check for errors
        if "error" in roadmap_data:
            return _response(500, {
                "status": "error",
                "message": roadmap_data["error"],
                "user_id": user_id
            })
        
        print(f"\n[LAMBDA] ✓ Roadmap generated successfully!")

        # ---------- 5. Submit to SQS ----------
        print(f"\n[LAMBDA] Submitting to SQS...")
        submission_result = submit_to_backend(roadmap_data, auth_token)

        if submission_result["success"]:
            print(f"[LAMBDA] ✓ Successfully sent to SQS")
        else:
            print(f"[LAMBDA] ✗ SQS submission failed: {submission_result['error']}")

        # ---------- 6. Return response ----------
        return _response(200, {
            "status": "success",
            "user_id": user_id,
            "ai_session_id": ai_session_id,
            "ai_roadmap_id": ai_roadmap_id,
            "roadmap": roadmap_data,
            "backend_submission": submission_result,
            "metadata": roadmap_data.get("roadmap_structure", {}).get("metadata", {})
        })

    except Exception as exc:
        print(f"\n[LAMBDA] ✗ Fatal error: {exc}")
        print(traceback.format_exc())
        
        return _response(500, {
            "status": "error",
            "message": str(exc),
            "traceback": traceback.format_exc()
        })
