import json
from src.pinecone_utils import fetch_poc_record

user_id = "test_user_1781682923803"

records = {
    "ONBOARDING_CONVERSATION": f"{user_id}_onboarding_conversation",
    "ROADMAP_CONVERSATION": f"{user_id}_roadmap_conversation",
    "ROADMAP_OUTPUT": f"{user_id}_roadmap_output",
}

for name, record_id in records.items():

    print("\n" + "=" * 100)
    print(f"{name}")
    print("=" * 100)

    data = fetch_poc_record(
        user_id=user_id,
        record_id=record_id
    )

    if not data:
        print("❌ NOT FOUND IN PINECONE")
        continue

    print(f"✅ FOUND IN PINECONE")
    print(f"Length: {len(data)} characters")

    try:
        parsed = json.loads(data)

        print("\nJSON CONTENT:")
        print(
            json.dumps(
                parsed,
                indent=2,
                ensure_ascii=False
            )[:5000]
        )

    except Exception:
        print("\nRAW CONTENT:")
        print(data[:5000])