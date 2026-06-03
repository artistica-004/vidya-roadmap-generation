import json
import importlib.util
import os
from dotenv import load_dotenv

load_dotenv()
# Load handler
spec = importlib.util.spec_from_file_location("handler_module", "handler.py")
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)

event = {
    "Records": [
        {
            "body": json.dumps({
                "course_title": "AI for Beginners",
                "course_id": 123
            })
        }
    ]
}

handler.lambda_handler(event, None)