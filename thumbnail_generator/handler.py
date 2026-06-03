import json
import os
from src.thumbnail_generator import generate_thumbnail, upload_thumbnail

COURSE_API_BASE_URL = os.environ["BACKEND_API_URL"]

def lambda_handler(event, context):
    batch_item_failures = []

    try:
        for record in event.get("Records", []):
            message_id = record.get("messageId")

            try:
                body = json.loads(record.get("body", "{}"))
            except json.JSONDecodeError:
                print(json.dumps({
                    "level": "ERROR",
                    "message": "Invalid JSON format",
                    "messageId": message_id
                }))
                batch_item_failures.append({"itemIdentifier": message_id})
                continue

            course_title = body.get("course_title")
            course_id = body.get("course_id")

            auth_token = os.getenv("AUTH_TOKEN")

            if not course_title or not course_id:
                print(json.dumps({
                    "level": "ERROR",
                    "message": "Missing required fields",
                    "course_id": course_id,
                    "messageId": message_id
                }))
                batch_item_failures.append({"itemIdentifier": message_id})
                continue

            try:
                base64_image = generate_thumbnail(course_title)

                if not base64_image:
                    raise Exception("Thumbnail generation failed")

                upload_success = upload_thumbnail(
                    base64_image=base64_image,
                    course_id=course_id,
                    base_url=COURSE_API_BASE_URL,
                    auth_token=auth_token
                )

                if not upload_success:
                    raise Exception("Upload failed")

                print(json.dumps({
                    "level": "INFO",
                    "message": "Thumbnail uploaded successfully",
                    "course_id": course_id
                }))

            except Exception as e:
                print(json.dumps({
                    "level": "ERROR",
                    "message": str(e),
                    "course_id": course_id,
                    "messageId": message_id
                }))
                batch_item_failures.append({"itemIdentifier": message_id})

    except Exception as e:
        print(json.dumps({
            "level": "CRITICAL",
            "message": "Lambda level failure",
            "error": str(e)
        }))
        raise e

    return {
        "batchItemFailures": batch_item_failures
    }
