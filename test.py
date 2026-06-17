"""
Local test script for Lambda handler
Run this to test roadmap generation without deploying to AWS
"""

import json
import sys
import os
from datetime import datetime

# Add parent directory to path so we can import the lambda function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your lambda handler
# Adjust the import path based on your file structure
# If your lambda is in handler.py, use:
from handler import lambda_handler

# If it's in a different file, adjust accordingly:
# from your_lambda_file import lambda_handler


def test_roadmap_generation(user_id: str):
    """
    Test the Lambda handler locally with a user_id

    Args:
        user_id: User ID to generate roadmap for
    """

    print("\n" + "=" * 70)
    print("🧪 TESTING LAMBDA HANDLER LOCALLY")
    print("=" * 70)
    print(f"User ID: {user_id}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    # Create a mock Lambda event (simulating direct invocation)
    event = {
        "user_id": user_id,

        # Optional: You can add these if you want to test with specific IDs
        # "ai_session_id": "test_session_123",
        # "ai_roadmap_id": "test_roadmap_456",
        # "auth_token": "your_test_token_here"  # Add if you want to test backend submission
    }

    # Mock Lambda context
    class MockContext:
        def __init__(self):
            self.function_name = "test-roadmap-generator"
            self.memory_limit_in_mb = 2048
            self.invoked_function_arn = "arn:aws:lambda:local:test"
            self.aws_request_id = "test-request-id-123"

    context = MockContext()

    print("📤 Invoking Lambda handler...\n")

    try:
        # Call the handler
        response = lambda_handler(event, context)

        # Parse the response
        status_code = response.get("statusCode", 500)
        body = json.loads(response.get("body", "{}"))

        print("\n" + "=" * 70)
        print("📥 LAMBDA RESPONSE")
        print("=" * 70)
        print(f"Status Code: {status_code}")
        print("=" * 70 + "\n")

        if status_code == 200:
            print("✅ SUCCESS!\n")

            # Display key information
            print(f"User ID: {body.get('user_id')}")
            print(f"Session ID: {body.get('ai_session_id')}")
            print(f"Roadmap ID: {body.get('ai_roadmap_id')}")
            print(f"Status: {body.get('status')}")

            roadmap = body.get("roadmap", body)

            # Display metadata
            metadata = body.get("metadata") or roadmap.get(
                "roadmap_structure", {}
            ).get("metadata", {})

            if metadata:
                print("\n📊 Roadmap Metadata:")
                print(f"  - Total Courses: {metadata.get('total_courses', 0)}")
                print(f"  - Total Chapters: {metadata.get('total_chapters', 0)}")
                print(f"  - Total Topics: {metadata.get('total_topics', 0)}")
                print(
                    f"  - Total Quiz Questions: "
                    f"{metadata.get('total_quiz_questions', 0)}"
                )

            # Display roadmap info
            if roadmap:
                print("\n📚 Roadmap Details:")
                print(f"  - Title: {roadmap.get('title')}")
                print(f"  - Description: {roadmap.get('description')}")
                print(
                    f"  - Duration: "
                    f"{roadmap.get('estimated_duration_weeks')} weeks"
                )
                print(f"  - Difficulty: {roadmap.get('difficulty_level')}")

            # Print the full JSON for inspection
            print("\n" + "=" * 70)
            print("FULL ROADMAP JSON")
            print("=" * 70)
            print(json.dumps(body, indent=2, ensure_ascii=False))

            # ==========================================================
            # Display milestone information
            # ==========================================================

            milestones = roadmap.get("milestones", [])

            if milestones:
                print("\n🎯 Career Milestones:")
                print("=" * 70)

                for milestone in milestones:
                    milestone_id = milestone.get("milestone_id")
                    if isinstance(milestone_id, int):
                        milestone_label = f"M{milestone_id:02d}"
                    else:
                        milestone_label = f"M{milestone_id}"

                    print(
                        f"\n{milestone_label} | "
                        f"{milestone.get('title')}"
                    )

                    print(f"Role: {milestone.get('role')}")
                    print(
                        f"Estimated Days: "
                        f"{milestone.get('estimated_days')}"
                    )
                    print(
                        f"Market Value: "
                        f"{milestone.get('market_value')}"
                    )
                    print(f"Quote: {milestone.get('quote')}")

                    skills = milestone.get("skills", [])
                    if isinstance(skills, list):
                        print(f"Skills: {', '.join(skills)}")
                    else:
                        print(f"Skills: {skills}")

                    gaps = milestone.get("gaps", [])
                    if isinstance(gaps, list):
                        print(f"Gaps: {', '.join(gaps)}")
                    else:
                        print(f"Gaps: {gaps}")

                    career_progression = milestone.get("career_progression", [])
                    if isinstance(career_progression, list):
                        print(
                            "Career Progression: "
                            f"{', '.join(career_progression)}"
                        )
                    else:
                        print(
                            f"Career Progression: {career_progression}"
                        )

                    new_opportunities = milestone.get("new_opportunities", [])
                    if isinstance(new_opportunities, list):
                        print(
                            "New Opportunities: "
                            f"{', '.join(new_opportunities)}"
                        )
                    else:
                        print(
                            f"New Opportunities: {new_opportunities}"
                        )

                    modules = milestone.get("modules", {})

                    if isinstance(modules, dict) and modules:
                        print("\nModule Plan:")
                        print(
                            f"  Module ID: {modules.get('module_id')}"
                        )

                        week_range = modules.get("week_range", {})
                        if isinstance(week_range, dict) and week_range:
                            print(
                                "  Week Range: "
                                f"{week_range.get('start')}"
                                f"-{week_range.get('end')}"
                            )

                        print(
                            f"  Mastery: {modules.get('mastery')}"
                        )

                        weeks = modules.get("weeks", [])
                        if isinstance(weeks, list) and weeks:
                            print(
                                f"  Week Breakdown ({len(weeks)}):"
                            )
                            for week in weeks:
                                print(
                                    f"   - W{week.get('week')}: "
                                    f"{week.get('focus')} "
                                    f"[{week.get('status')}]"
                                )

                print("\n" + "=" * 70)

            # Backend submission status
            backend = body.get("backend_submission", {})

            if backend:
                print("\n🌐 Backend Submission:")

                if backend.get("success"):
                    print("  ✅ Successfully submitted to backend")
                else:
                    print(
                        f"  ❌ Failed: "
                        f"{backend.get('error', 'Unknown error')}"
                    )

            # Save full response to file
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "test_outputs"
            )
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir,
                f"user_{user_id}_roadmap.json"
            )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(body, f, indent=2, ensure_ascii=False)

            print(f"\n💾 Full response saved to: {output_file}")

            if milestones:
                milestones_file = os.path.join(
                    output_dir,
                    f"user_{user_id}_milestones.json"
                )

                with open(milestones_file, "w", encoding="utf-8") as f:
                    json.dump(milestones, f, indent=2, ensure_ascii=False)

                print(
                    f"💾 Milestones saved to: {milestones_file}"
                )

        else:
            print("❌ ERROR!\n")
            print(f"Message: {body.get('message', 'Unknown error')}")

            if "traceback" in body:
                print("\n📋 Traceback:")
                print(body["traceback"])

        print("\n" + "=" * 70)

        return response

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")

        import traceback
        traceback.print_exc()

        return None


def main():
    """Interactive mode - ask for user ID"""

    print("\n" + "=" * 70)
    print("🚀 LOCAL LAMBDA HANDLER TEST")
    print("=" * 70)

    # Get user ID from command line or prompt
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = input("\nEnter User ID to test: ").strip()

    if not user_id:
        print("❌ User ID cannot be empty!")
        return

    # Run the test
    test_roadmap_generation(user_id)


if __name__ == "__main__":
    main()