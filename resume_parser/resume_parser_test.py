"""
Test script for resume parser Lambda handler
Tests both S3 and base64 input methods
"""

import json
import base64
import os
from pathlib import Path

# Import your Lambda handler
from handler import lambda_handler


def test_with_base64(resume_path: str, user_id: str = "test_user_1"):
    """
    Test Lambda handler with base64-encoded resume file
    
    Args:
        resume_path: Path to your resume PDF/DOCX file
        user_id: User ID to associate with the resume
    """
    print("=" * 60)
    print("TEST 1: Base64 File Upload")
    print("=" * 60)
    
    # Read the file
    resume_file = Path(resume_path)
    
    if not resume_file.exists():
        print(f"❌ ERROR: File not found: {resume_path}")
        return
    
    print(f" Reading file: {resume_file.name}")
    
    with open(resume_file, "rb") as f:
        file_content = f.read()
    
    # Encode to base64
    b64_content = base64.b64encode(file_content).decode("utf-8")
    
    print(f" File encoded: {len(b64_content)} chars (base64)")
    
    # Construct Lambda event (simulating API Gateway)
    event = {
        "body": json.dumps({
            "user_id": user_id,
            "filename": resume_file.name,
            "content_base64": b64_content,
            "model_name": "gpt-3.5-turbo-1106"  # Optional
        })
    }
    
    print(f" Invoking Lambda handler for user: {user_id}")
    print("-" * 60)
    
    # Call the handler
    try:
        response = lambda_handler(event, None)
        
        # Parse response
        status_code = response.get("statusCode")
        body = json.loads(response.get("body", "{}"))
        
        print(f"\n RESPONSE:")
        print(f"Status Code: {status_code}")
        print(f"Status: {body.get('status')}")
        
        if status_code == 200:
            print(" SUCCESS!")
            
            # Show parsed data summary
            parsed = body.get("parsed_output", {})
            print(f"\n Parsed Resume Summary:")
            print(f"  Name: {parsed.get('candidate_name', 'N/A')}")
            print(f"  Job Title: {parsed.get('job_title', 'N/A')}")
            print(f"  Skills: {len(parsed.get('skills', []))} found")
            print(f"  Work Experience: {len(parsed.get('work_output', []))} entries")
            print(f"  Education: {len(parsed.get('education', []))} entries")
            
            # Save full output to file
            output_file = f"test_output_{user_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            print(f"\n Full output saved to: {output_file}")
            
        else:
            print(f"❌ FAILED with status {status_code}")
            print(f"Message: {body.get('message')}")
            
            if "traceback" in body:
                print(f"\n🔍 Traceback:")
                print(body["traceback"])
        
        return response
        
    except Exception as e:
        print(f"❌ ERROR calling handler: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_s3(bucket: str, key: str, user_id: str = "test_user_456"):
    """
    Test Lambda handler with S3 file reference
    
    Args:
        bucket: S3 bucket name
        key: S3 object key (path to resume)
        user_id: User ID to associate with the resume
    """
    print("\n" + "=" * 60)
    print("TEST 2: S3 File Reference")
    print("=" * 60)
    
    # Construct Lambda event
    event = {
        "body": json.dumps({
            "user_id": user_id,
            "s3_bucket": bucket,
            "s3_key": key,
            "model_name": "gpt-3.5-turbo-1106"  # Optional
        })
    }
    
    print(f"📦 S3 Bucket: {bucket}")
    print(f"🔑 S3 Key: {key}")
    print(f"👤 User ID: {user_id}")
    print(f"🚀 Invoking Lambda handler...")
    print("-" * 60)
    
    try:
        response = lambda_handler(event, None)
        
        status_code = response.get("statusCode")
        body = json.loads(response.get("body", "{}"))
        
        print(f"\n RESPONSE:")
        print(f"Status Code: {status_code}")
        print(f"Status: {body.get('status')}")
        
        if status_code == 200:
            print(" SUCCESS!")
            
            parsed = body.get("parsed_output", {})
            print(f"\n Parsed Resume Summary:")
            print(f"  Name: {parsed.get('candidate_name', 'N/A')}")
            print(f"  Job Title: {parsed.get('job_title', 'N/A')}")
            print(f"  Skills: {len(parsed.get('skills', []))} found")
            
            output_file = f"test_output_s3_{user_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            print(f"\n Full output saved to: {output_file}")
            
        else:
            print(f" FAILED with status {status_code}")
            print(f"Message: {body.get('message')}")
            
            if "traceback" in body:
                print(f"\n🔍 Traceback:")
                print(body["traceback"])
        
        return response
        
    except Exception as e:
        print(f" ERROR calling handler: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_direct_call(resume_path: str, user_id: str = "test_user_789"):
    """
    Test Lambda handler with direct event (no API Gateway wrapper)
    
    Args:
        resume_path: Path to your resume PDF/DOCX file
        user_id: User ID to associate with the resume
    """
    print("\n" + "=" * 60)
    print("TEST 3: Direct Event (No API Gateway)")
    print("=" * 60)
    
    resume_file = Path(resume_path)
    
    if not resume_file.exists():
        print(f" ERROR: File not found: {resume_path}")
        return
    
    with open(resume_file, "rb") as f:
        file_content = f.read()
    
    b64_content = base64.b64encode(file_content).decode("utf-8")
    
    # Direct event (no "body" wrapper)
    event = {
        "user_id": user_id,
        "filename": resume_file.name,
        "content_base64": b64_content,
    }
    
    print(f" File: {resume_file.name}")
    print(f" User ID: {user_id}")
    print(f" Invoking handler directly...")
    print("-" * 60)
    
    try:
        response = lambda_handler(event, None)
        
        status_code = response.get("statusCode")
        body = json.loads(response.get("body", "{}"))
        
        print(f"\n RESPONSE:")
        print(f"Status Code: {status_code}")
        
        if status_code == 200:
            print(" SUCCESS!")
        else:
            print(f" FAILED: {body.get('message')}")
        
        return response
        
    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pinecone_retrieval(user_id: str):
    """
    Test if resume was successfully stored in Pinecone
    
    Args:
        user_id: User ID to check
    """
    print("\n" + "=" * 60)
    print("TEST 4: Pinecone Storage Verification")
    print("=" * 60)
    
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "live-assistant-index")
        
        print(f" Checking index: {index_name}")
        print(f" User ID: {user_id}")
        
        index = pc.Index(index_name)
        
        # Try to fetch the resume summary
        vector_id = f"{user_id}_resume_summary"
        
        print(f" Looking for vector ID: {vector_id}")
        
        result = index.fetch(
            ids=[vector_id],
            namespace=user_id
        )
        
        if result.get("vectors") and vector_id in result["vectors"]:
            vector_data = result["vectors"][vector_id]
            metadata = vector_data.get("metadata", {})
            
            print("\n RESUME FOUND IN PINECONE!")
            print(f"\n Metadata:")
            print(f"  Doc Type: {metadata.get('doc_type')}")
            print(f"  Source: {metadata.get('source')}")
            print(f"  Timestamp: {metadata.get('timestamp')}")
            
            # Show summary
            summary = metadata.get("text", "")
            if summary:
                print(f"\n Summary (first 200 chars):")
                print(f"  {summary[:200]}...")
            
            # Show formatted context
            formatted = metadata.get("formatted_context", "")
            if formatted:
                print(f"\n Formatted Context (first 300 chars):")
                print(f"  {formatted[:300]}...")
            
            return True
        else:
            print(f"\n Resume NOT found in Pinecone")
            print(f"Vector ID '{vector_id}' does not exist in namespace '{user_id}'")
            return False
            
    except Exception as e:
        print(f" ERROR checking Pinecone: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(resume_path: str):
    """
    Run all test scenarios
    
    Args:
        resume_path: Path to test resume file
    """
    print("\n" + "🧪" * 30)
    print("RESUME PARSER LAMBDA - FULL TEST SUITE")
    print("🧪" * 30)
    
    user_id = "test_user_full_suite"
    
    # Test 1: Base64 upload
    print("\n")
    response1 = test_with_base64(resume_path, user_id)
    
    # Test 2: Direct call (alternative format)
    if response1 and response1.get("statusCode") == 200:
        print("\n")
        test_direct_call(resume_path, user_id + "_direct")
    
    # Test 3: Verify Pinecone storage
    if response1 and response1.get("statusCode") == 200:
        import time
        print("\n⏳ Waiting 2 seconds for Pinecone indexing...")
        time.sleep(2)
        test_pinecone_retrieval(user_id)
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 60)


# ==================================================
# MAIN - Run Tests
# ==================================================
if __name__ == "__main__":
    import sys
    
    # Check for resume file argument
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test.py <path_to_resume.pdf>")
        print("\nExamples:")
        print("  python test.py sample_resume.pdf")
        print("  python test.py /path/to/John_Doe_Resume.docx")
        print("\nOr edit this file and set RESUME_PATH directly")
        
        # Default test file (change this to your resume path)
        RESUME_PATH = "sample_resume.pdf"
        
        if not Path(RESUME_PATH).exists():
            print(f"\n❌ Default resume not found: {RESUME_PATH}")
            print("Please provide a resume file path as argument")
            sys.exit(1)
    else:
        RESUME_PATH = sys.argv[1]
    
    # Verify file exists
    if not Path(RESUME_PATH).exists():
        print(f"❌ ERROR: File not found: {RESUME_PATH}")
        sys.exit(1)
    
    print(f"\n📂 Using resume file: {RESUME_PATH}")
    
    # Choose test mode
    print("\nSelect test mode:")
    print("1. Base64 upload test (recommended)")
    print("2. S3 reference test (requires S3 setup)")
    print("3. Direct event test")
    print("4. Pinecone verification only")
    print("5. Run ALL tests")
    
    choice = input("\nEnter choice (1-5) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        test_with_base64(RESUME_PATH)
    
    elif choice == "2":
        bucket = input("Enter S3 bucket name: ").strip()
        key = input("Enter S3 key (path to resume): ").strip()
        test_with_s3(bucket, key)
    
    elif choice == "3":
        test_direct_call(RESUME_PATH)
    
    elif choice == "4":
        user_id = input("Enter user_id to check: ").strip()
        test_pinecone_retrieval(user_id)
    
    elif choice == "5":
        run_all_tests(RESUME_PATH)
    
    else:
        print(f"❌ Invalid choice: {choice}")
        sys.exit(1)