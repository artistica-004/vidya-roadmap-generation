import json

with open(
    "synthetic_test_data/onboarding_inputs/user_021.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

save_poc_record(
    user_id="user_021",
    namespace="user_021",
    memory_type="onboarding",
    content=json.dumps(data)
)