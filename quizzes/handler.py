"""
AWS Lambda handler that processes roadmaps from SQS and generates MCQs.
This Lambda is triggered automatically when a message arrives in the SQS queue.
"""

import os
import json
import traceback
from typing import Dict, Any

import boto3
from src.mcq_generator import generate_mcqs_for_module

s3 = boto3.client("s3")

# S3 bucket for storing MCQs (optional, set via environment variable)
OUTPUT_BUCKET = os.getenv("MCQ_OUTPUT_BUCKET")


def _save_to_s3(bucket: str, key: str, content: Dict[str, Any]):
    """Save JSON content to S3."""
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(content, indent=2, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"[S3] ✅ Saved to s3://{bucket}/{key}")
        return True
    except Exception as e:
        print(f"[S3] ❌ Failed to save: {e}")
        return False


def process_roadmap_message(message_body: dict) -> dict:
    """
    Process a single roadmap message and generate MCQs.
    
    Args:
        message_body: Dict containing 'roadmap' and 'user_id'
    
    Returns:
        dict: Result with status and generated MCQs
    """
    user_id = message_body.get("user_id", "unknown")
    roadmap_data = message_body.get("roadmap")
    
    if not roadmap_data:
        raise ValueError("No roadmap data in message")
    
    # Extract roadmap details
    roadmap_title = roadmap_data.get("CourseTitle", "Untitled Roadmap")
    modules = roadmap_data.get("Modules", [])
    
    print(f"[MCQ GEN] Processing: {roadmap_title}")
    print(f"[MCQ GEN] User: {user_id}")
    print(f"[MCQ GEN] Modules: {len(modules)}")
    
    # Generate MCQs for each module
    all_mcqs = []
    total_questions = 0
    
    for idx, module in enumerate(modules, 1):
        module_name = module.get("ModuleName", f"Module {idx}")
        print(f"[MCQ GEN] Generating MCQs for: {module_name}")
        
        try:
            mcq_block = generate_mcqs_for_module(module)
            
            # Count questions
            num_questions = len(mcq_block.get("MCQs", []))
            total_questions += num_questions
            
            all_mcqs.append(mcq_block)
            print(f"[MCQ GEN]  Generated {num_questions} questions for {module_name}")
            
        except Exception as e:
            print(f"[MCQ GEN]   Error generating MCQs for {module_name}: {e}")
            # Add error placeholder
            all_mcqs.append({
                "ModuleName": module_name,
                "error": str(e),
                "MCQs": []
            })
    
    # Prepare output
    output = {
        "user_id": user_id,
        "roadmap_title": roadmap_title,
        "total_modules": len(modules),
        "total_questions": total_questions,
        "mcqs": all_mcqs,
        "metadata": {
            "difficulty": roadmap_data.get("DifficultyLevel"),
            "weeks": roadmap_data.get("Weeks"),
            "learning_style": roadmap_data.get("LearningStyle")
        }
    }
    
    # Save to S3 if bucket is configured
    if OUTPUT_BUCKET:
        s3_key = f"mcqs/{user_id}/roadmap_mcqs.json"
        _save_to_s3(OUTPUT_BUCKET, s3_key, output)
    
    return output


def lambda_handler(event, context):
    """
    AWS Lambda handler for SQS-triggered MCQ generation.
    
    Expected SQS message format:
    {
        "roadmap": {...},  # The complete roadmap JSON
        "user_id": "user123"
    }
    """
    print(f"[LAMBDA] Received event with {len(event.get('Records', []))} messages")
    
    results = []
    errors = []
    
    try:
        # Process each SQS message
        for record in event.get("Records", []):
            try:
                # Parse message body
                message_body = json.loads(record["body"])
                
                # Extract message attributes
                user_id = message_body.get("user_id")
                print(f"\n[LAMBDA] Processing message for user: {user_id}")
                
                # Generate MCQs
                result = process_roadmap_message(message_body)
                
                results.append({
                    "user_id": user_id,
                    "status": "success",
                    "questions_generated": result["total_questions"]
                })
                
                print(f"[LAMBDA]  Successfully processed {user_id}")
                
            except Exception as msg_error:
                error_info = {
                    "status": "error",
                    "error": str(msg_error),
                    "traceback": traceback.format_exc()
                }
                errors.append(error_info)
                print(f"[LAMBDA]  Error processing message: {msg_error}")
                print(traceback.format_exc())
        
        # Return summary
        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "completed",
                "processed": len(results),
                "successful": len(results),
                "failed": len(errors),
                "results": results,
                "errors": errors if errors else None
            })
        }
    
    except Exception as e:
        print(f"[LAMBDA]  Fatal error: {e}")
        print(traceback.format_exc())
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            })
        }