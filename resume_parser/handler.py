import os
import json
import base64
import boto3
from io import BytesIO
import traceback

# Import your existing code (do not change parser.py, etc.)
# Ensure parser.py, pydantic_models_prompts.py, utils.py are in the image
from src.parser import ResumeManager, get_resume_content, store_embeddings_in_pinecone

s3 = boto3.client("s3")


def _load_file_from_s3(bucket, key):
    tmp = BytesIO()
    s3.download_fileobj(bucket, key, tmp)
    tmp.seek(0)
    return tmp


def _save_json_to_s3(bucket, key, content):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(content, indent=2, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def _response(status_code: int, body: dict):
    """Format response for API Gateway and still readable in Lambda console."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    try:
        # ---------- 1. Normalize incoming event ----------
        payload = {}

        # Case A: SQS event
        if isinstance(event, dict) and "Records" in event:
            try:
                record = event["Records"][0]
                payload = json.loads(record.get("body", "{}"))
                print("[EVENT] Source: SQS")
            except Exception:
                payload = {}

        # Case B: API Gateway proxy
        elif isinstance(event, dict) and "body" in event:
            raw_body = event.get("body")
            if raw_body:
                try:
                    payload = json.loads(raw_body)
                    print("[EVENT] Source: API Gateway (body)")
                except json.JSONDecodeError:
                    return _response(
                        400,
                        {"status": "error", "message": "Invalid JSON in request body"},
                    )

        # Case C: Direct Lambda invoke
        elif isinstance(event, dict) and "user_id" in event:
            payload = event
            print("[EVENT] Source: Direct Invoke")

        # Case D: Query parameters
        elif isinstance(event, dict) and event.get("queryStringParameters"):
            payload = event.get("queryStringParameters") or {}
            print("[EVENT] Source: Query Params")

        # Default fallback
        if not payload:
            payload = {}
            print("[EVENT] Source: Empty / Unknown")

        # ---------- 2. Read common fields ----------
        model_name = payload.get(
            "model_name", os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo-1106")
        )

        user_id = payload.get("user_id")
        if not user_id:
            return _response(
                400,
                {
                    "status": "error",
                    "message": "user_id is required in request payload",
                },
            )

        print(f"Processing resume for user_id={user_id}")

        # ---------- 3. Handle S3-based input ----------
        if payload.get("s3_bucket") and payload.get("s3_key"):
            bucket = payload["s3_bucket"]
            key = payload["s3_key"]

            try:
                fileobj = _load_file_from_s3(bucket, key)
            except Exception as e:
                return _response(
                    400,
                    {
                        "status": "error",
                        "message": f"Failed to load file from S3: {str(e)}",
                        "traceback": traceback.format_exc(),
                    },
                )

            _, ext = os.path.splitext(key)
            rm = ResumeManager(fileobj, model_name, extension=ext)
            rm.process_file()

            print("Resume parsing completed")

            output_bucket = os.getenv("OUTPUT_BUCKET")
            if output_bucket:
                out_key = (
                    f"parsed_outputs/"
                    f"{os.path.splitext(os.path.basename(key))[0]}_output.json"
                )
                _save_json_to_s3(output_bucket, out_key, rm.output)

            try:
                print("Storing resume summary embedding in Pinecone")

                store_embeddings_in_pinecone(rm.output, user_id)

                print("Resume parsed successful")

                return _response(
                    200,
                    {
                        "status": "success",
                        "parsed_output": rm.output,
                    },
                )
            except Exception as e:
                return _response(
                    200,
                    {
                        "status": "partial_success",
                        "parsed_output": rm.output,
                        "pinecone_error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )

        # ---------- 4. Handle base64 file input ----------
        elif payload.get("content_base64") and payload.get("filename"):
            b64 = payload["content_base64"]
            filename = payload["filename"]

            try:
                data = base64.b64decode(b64)
            except Exception as e:
                return _response(
                    400,
                    {
                        "status": "error",
                        "message": f"Invalid base64 content: {str(e)}",
                        "traceback": traceback.format_exc(),
                    },
                )

            tmp = BytesIO(data)
            _, ext = os.path.splitext(filename)
            rm = ResumeManager(tmp, model_name, extension=ext)
            rm.process_file()

            # ✅ LOG 2
            print("Resume parsing completed")

            try:
                print("Storing resume summary embedding in Pinecone")

                store_embeddings_in_pinecone(rm.output, user_id)

                print("Resume parsed successful")

                return _response(
                    200,
                    {
                        "status": "success",
                        "parsed_output": rm.output,
                    },
                )
            except Exception as e:
                return _response(
                    200,
                    {
                        "status": "partial_success",
                        "parsed_output": rm.output,
                        "pinecone_error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )

        else:
            return _response(
                400,
                {
                    "status": "error",
                    "message": "Invalid request. Provide either "
                    "'s3_bucket' + 's3_key' or 'filename' + 'content_base64'.",
                },
            )

    except Exception as exc:
        return _response(
            500,
            {
                "status": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
