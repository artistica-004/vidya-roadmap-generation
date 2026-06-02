import os
import base64
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

# Get API key from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    print(f"[GEMINI] Configured successfully")
else:
    print(f"[GEMINI] WARNING: GEMINI_API_KEY not found in .env")

def generate_thumbnail(course_title: str) -> str:
    """
    Generates a thumbnail image using Gemini 2.5 Flash Image (Nano Banana).
    Uses config parameter with ImageConfig for 16:9 aspect ratio.
    Returns the image as a Base64-encoded PNG string.
    """
    print(f"[THUMBNAIL] Generating thumbnail for: '{course_title}'")
    
    try:
        # Create the image generation prompt
        prompt = f"""
Create a professional, modern course thumbnail for the course titled: "{course_title}"

Requirements:
- Modern, educational design
- Professional color scheme
- Include the course title text prominently  
- Clean, minimalist aesthetic
- Suitable for an online learning platform
- No harsh colors, use complementary color schemes
"""
        
        print(f"[GEMINI] Generating image with Nano Banana...")
        
        # Use Gemini 2.5 Flash Image with config for 16:9 aspect ratio
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9"  # Configure 16:9 aspect ratio
                )
            )
        )
        
        # Extract image data from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_bytes = part.inline_data.data
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    print(f"[THUMBNAIL] ✓ Generated successfully (Base64 size: {len(base64_image)} chars)")
                    return base64_image
        
        print(f"[THUMBNAIL] ✗ No image data in response")
        return None
        
    except Exception as e:
        print(f"[THUMBNAIL] ✗ Error generating image: {e}")
        return None

def upload_thumbnail(base64_image: str, course_id: int, base_url: str, auth_token: str) -> bool:
    """
    Uploads the Base64-encoded thumbnail to the backend API.
    
    Args:
        base64_image: Base64-encoded PNG image string
        course_id: The ID of the course
        base_url: The backend base URL (e.g., https://0t3p5fhzah.execute-api.ap-south-1.amazonaws.com)
        auth_token: Authorization bearer token
    
    Returns:
        True if upload successful, False otherwise
    """
    if not base64_image or not course_id:
        print("[UPLOAD] ✗ Missing base64_image or course_id")
        return False
    
    upload_url = f"{base_url}/courses/image/upload"
    
    try:
        payload = {
            "file": base64_image,
            "file_name": "course_thumbnail.png",
            "course_id": int(course_id)
        }
        
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
        
        print(f"[UPLOAD] Uploading to {upload_url} for Course ID: {course_id}")
        
        response = requests.post(upload_url, json=payload, headers=headers, timeout=30)
        
        if response.status_code in [200, 201]:
            print(f"[UPLOAD] ✓ Success! Status: {response.status_code}")
            return True
        else:
            print(f"[UPLOAD] ✗ Failed with status {response.status_code}")
            try:
                print(f"[UPLOAD] Response: {response.json()}")
            except:
                print(f"[UPLOAD] Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"[UPLOAD] ✗ Error: {e}")
        return False
