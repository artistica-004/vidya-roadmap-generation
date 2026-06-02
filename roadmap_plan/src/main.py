

from src.roadmap_agent import run_pipeline
import json
import os


def generate_roadmap(user_id: str, auto_trigger_mcq: bool = True):
    """
    Generate roadmap for a given user_id and optionally trigger MCQ generation.
    
    Args:
        user_id: User identifier (used as Pinecone namespace)
        auto_trigger_mcq: If True, sends roadmap to SQS for MCQ generation
    
    Returns:
        dict: Generated roadmap or error
    """
    # Ensure Pinecone index exists
   

    # Generate roadmap (this will also trigger MCQ generation if enabled)
    print(f"[MAIN] Generating roadmap for user: {user_id}")
    result = run_pipeline(
        user_id=user_id,
        trigger_mcq=auto_trigger_mcq
    )

    return result


def main():
    print("\n" + "="*60)
    print("🎓 PERSONALIZED ROADMAP GENERATOR")
    print("="*60 + "\n")
    
    user_id = input("Enter your user_id: ").strip()
    
    if not user_id:
        print(" User ID cannot be empty!")
        return
    
    print(f"\n Generating roadmap for: {user_id}")
    print(" This may take 10-30 seconds...\n")
    
    # Generate roadmap (MCQs will be queued automatically)
    roadmap = generate_roadmap(
        user_id=user_id,
        auto_trigger_mcq=True  # Set to False to skip MCQ generation
    )
    
    # Display results
    print("\n" + "="*60)
    print(" GENERATED ROADMAP")
    print("="*60 + "\n")
    
    if "error" in roadmap:
        print(f" Error: {roadmap['error']}")
        if "raw_output" in roadmap:
            print(f"\nRaw output (first 500 chars):")
            print(roadmap["raw_output"][:500])
    else:
        # Pretty print roadmap
        print(json.dumps(roadmap, indent=2, ensure_ascii=False))
        
        print("\n" + "="*60)
        print(" ROADMAP GENERATED SUCCESSFULLY!")
        print("="*60)
        
        # Show roadmap summary
        print(f"\n Course: {roadmap.get('CourseTitle')}")
        print(f" Difficulty: {roadmap.get('DifficultyLevel')}")
        print(f" Duration: {roadmap.get('Weeks')} weeks")
        print(f" Modules: {len(roadmap.get('Modules', []))}")
        
        # MCQ status
        mcq_status = roadmap.get('_mcq_status', 'unknown')
        if mcq_status == 'queued':
            print(f"\n MCQ Generation: QUEUED")
            print(f" Message ID: {roadmap.get('_mcq_message_id')}")
            print("\n MCQs are being generated asynchronously via AWS Lambda.")
            print(f"   They will be saved to S3 at: mcqs/{user_id}/roadmap_mcqs.json")
        elif mcq_status == 'not_queued':
            print(f"\n  MCQ Generation: NOT QUEUED")
            print("   (MCQ_QUEUE_URL not configured in .env)")
        
        # Save locally
        output_file = f"roadmap_{user_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(roadmap, f, indent=2, ensure_ascii=False)
        print(f"\n Roadmap saved locally: {output_file}")


if __name__ == "__main__":
    main()