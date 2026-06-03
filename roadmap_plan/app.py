import streamlit as st
import json
import sys
import os
import traceback

from dotenv import load_dotenv

# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()

# =====================================================
# FIX IMPORT PATH
# =====================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

# =====================================================
# IMPORTS
# =====================================================

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.roadmap_agent import run_pipeline
from src.pinecone_utils import retrieve_raw_context

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Vidya V3 — AI Roadmap Lab",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================

if "generated_onboarding" not in st.session_state:
    st.session_state.generated_onboarding = ""

if "generated_onboarding_json" not in st.session_state:
    st.session_state.generated_onboarding_json = {}

if "generated_context" not in st.session_state:
    st.session_state.generated_context = ""

if "roadmap_data" not in st.session_state:
    st.session_state.roadmap_data = None

# =====================================================
# GEMINI CONFIG
# =====================================================

gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not gemini_key:
    st.error("❌ Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.7,
    convert_system_message_to_human=True
)

# =====================================================
# PROMPT 1 — HUMAN READABLE PERSONA SUMMARY
# =====================================================

persona_prompt = PromptTemplate(
    input_variables=["icp_type", "learner_state", "weekly_hours", "goal"],
    template="""
You are generating realistic onboarding summaries for Vidya V3.

Generate a believable learner onboarding summary.

RULES

If ICP is HIGH (high_wage = working professional):
- Already employed
- Wants promotion, switch, or upskilling
- Engineering / product / analyst background

If ICP is LOW (low_wage = student / fresher):
- Student or fresher
- Placement focused
- May be from Tier 2/3 college
- Limited resources

The onboarding summary MUST include:
- current background
- current struggles
- learning goals
- skill gaps
- confidence level
- career target
- urgency
- available weekly time

Tone: emotionally believable, realistic, India-specific, simple language (no jargon).

ICP TYPE: {icp_type}
LEARNER STATE: {learner_state}
WEEKLY HOURS: {weekly_hours}
GOAL: {goal}

OUTPUT: Return ONLY the onboarding summary text. No headings, no JSON, no markdown.
"""
)

# =====================================================
# PROMPT 2 — STRUCTURED ONBOARDING JSON
# =====================================================

onboarding_json_prompt = PromptTemplate(
    input_variables=["icp_type", "learner_state", "weekly_hours", "goal", "persona_summary"],
    template="""
You are a data extraction engine for Vidya V3.

Given the learner persona summary below, extract and return a structured onboarding JSON object.

PERSONA SUMMARY:
{persona_summary}

ADDITIONAL INPUTS:
ICP TYPE: {icp_type}
LEARNER STATE: {learner_state}
WEEKLY HOURS: {weekly_hours}
GOAL: {goal}

REQUIRED JSON FIELDS:

{{
  "full_name": "realistic Indian name",
  "age": integer,
  "location": "Indian city",
  "education": "highest qualification",
  "current_role": "current job title or student",
  "years_experience": integer,
  "current_salary_monthly": integer (0 if student),
  "target_role": "what they want to become",
  "target_salary_monthly": integer,
  "primary_goal": "one sentence goal",
  "skill_gaps": ["gap1", "gap2", "gap3"],
  "known_skills": ["skill1", "skill2"],
  "learning_style": "visual | reading | video | practice",
  "weekly_hours_available": integer,
  "device_access": "mobile | laptop | both",
  "language_preference": "en | hi | ta",
  "confidence_level": "low | medium | high",
  "urgency": "low | medium | high",
  "motivation": "one sentence about why they want this",
  "biggest_fear": "one sentence about their biggest fear",
  "icp_type": "{icp_type}"
}}

RULES:
- Return ONLY valid raw JSON. No markdown, no code fences, no explanation.
- All values must be realistic and India-specific.
"""
)

persona_chain = persona_prompt | llm | StrOutputParser()
onboarding_json_chain = onboarding_json_prompt | llm | StrOutputParser()

# =====================================================
# HEADER
# =====================================================

st.title("🚀 Vidya V3 — AI Roadmap Testing Lab")
st.markdown("""
Internal testing dashboard for AI persona generation, roadmap generation,
ICP testing, learner-state testing, and roadmap visualization.
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("🧠 Persona Configuration")

icp_type = st.sidebar.selectbox(
    "ICP Type",
    ["high", "low"],
    help="high = working professional | low = student / fresher"
)

learner_state = st.sidebar.selectbox(
    "Learner State",
    ["beginner", "intermediate", "advanced"]
)

weekly_hours = st.sidebar.slider("Weekly Hours Available", 2, 20, 5)

# =====================================================
# MODE SELECTION
# =====================================================

mode = st.radio("Select Mode", ["AI Generated Persona", "Real User ID"])

user_id = None
goal = ""

# =====================================================
# REAL USER MODE
# =====================================================

if mode == "Real User ID":

    st.subheader("👤 Real User Testing")

    user_id = st.text_input("Enter User ID", placeholder="Example: user_001")

    if user_id:
        if st.button("🔍 Preview Pinecone Data"):
            with st.spinner("Fetching from Pinecone..."):
                raw_context = retrieve_raw_context(user_id)

            if raw_context:
                st.success("✅ Data found in Pinecone")
                st.text_area("Raw Context", value=raw_context, height=250)
            else:
                st.error(
                    "❌ No data found for this user ID. "
                    "Please check that the user has completed onboarding."
                )

# =====================================================
# AI PERSONA MODE
# =====================================================

else:

    st.subheader("🤖 AI Persona Generator")

    goal = st.text_input(
        "Learner Goal",
        placeholder="Example: Become a backend developer in 6 months"
    )

    if st.button("✨ Generate AI Persona"):

        if not goal.strip():
            st.error("Please enter a learner goal")
            st.stop()

        with st.spinner("Generating learner profile..."):

            # Step 1 — human readable summary
            generated_onboarding = persona_chain.invoke({
                "icp_type":      icp_type,
                "learner_state": learner_state,
                "weekly_hours":  weekly_hours,
                "goal":          goal
            })
            st.session_state.generated_onboarding = generated_onboarding

            # Step 2 — structured JSON from the same summary
            raw_json_output = onboarding_json_chain.invoke({
                "icp_type":        icp_type,
                "learner_state":   learner_state,
                "weekly_hours":    weekly_hours,
                "goal":            goal,
                "persona_summary": generated_onboarding
            })

            clean_json = raw_json_output.strip()
            # Strip markdown code fences if present
            if "```" in clean_json:
                clean_json = clean_json.replace("```json", "").replace("```", "").strip()

            try:
                st.session_state.generated_onboarding_json = json.loads(clean_json)
            except json.JSONDecodeError:
                st.session_state.generated_onboarding_json = {}
                st.warning("⚠ Could not parse structured JSON — only text summary available.")

        st.success("✅ AI Persona Generated")

# =====================================================
# SHOW GENERATED PERSONA (AI MODE ONLY)
# =====================================================

if mode == "AI Generated Persona" and st.session_state.generated_onboarding:

    st.divider()
    st.subheader("📋 Learner Profile")

    tab1, tab2 = st.tabs(["👤 Readable Summary", "🗂 Structured Data"])

    with tab1:
        st.info(st.session_state.generated_onboarding)

    with tab2:
        onboarding_json = st.session_state.generated_onboarding_json

        if onboarding_json:

            col_a, col_b = st.columns(2)

            field_labels = {
                "full_name":              "👤 Name",
                "age":                    "🎂 Age",
                "location":               "📍 Location",
                "education":              "🎓 Education",
                "current_role":           "💼 Current Role",
                "years_experience":       "📅 Experience (yrs)",
                "current_salary_monthly": "💰 Current Salary / month",
                "target_role":            "🎯 Target Role",
                "target_salary_monthly":  "💸 Target Salary / month",
                "primary_goal":           "🚀 Primary Goal",
                "learning_style":         "📚 Learning Style",
                "weekly_hours_available": "⏰ Weekly Hours",
                "device_access":          "💻 Device Access",
                "language_preference":    "🗣 Language",
                "confidence_level":       "💪 Confidence",
                "urgency":                "⚡ Urgency",
                "motivation":             "🔥 Motivation",
                "biggest_fear":           "😰 Biggest Fear",
                "icp_type":               "🏷 ICP Type",
            }

            list_fields = {"skill_gaps", "known_skills"}
            simple_items = [(k, v) for k, v in onboarding_json.items() if k not in list_fields]

            for i, (key, value) in enumerate(simple_items):
                label = field_labels.get(key, key.replace("_", " ").title())
                col = col_a if i % 2 == 0 else col_b
                if isinstance(value, (int, float)) and "salary" in key:
                    col.metric(label, f"₹ {value:,}")
                else:
                    col.metric(label, str(value))

            st.markdown("---")
            col_c, col_d = st.columns(2)

            with col_c:
                st.markdown("**🚧 Skill Gaps**")
                for gap in onboarding_json.get("skill_gaps", []):
                    st.write(f"• {gap}")

            with col_d:
                st.markdown("**✅ Known Skills**")
                for skill in onboarding_json.get("known_skills", []):
                    st.write(f"• {skill}")

            with st.expander("🔧 Raw JSON (developers)"):
                st.json(onboarding_json)
        else:
            st.warning("Structured JSON not available.")

# =====================================================
# GENERATE ROADMAP BUTTON
# =====================================================

st.divider()
generate_btn = st.button("🚀 Generate Personalized Roadmap")

if generate_btn:

    try:

        # ── AI PERSONA MODE ────────────────────────────────────
        if mode == "AI Generated Persona":

            if not st.session_state.generated_onboarding:
                st.error("Please generate AI persona first")
                st.stop()

            onboarding_json = st.session_state.generated_onboarding_json

            json_section = (
                json.dumps(onboarding_json, indent=2) if onboarding_json
                else "Not available"
            )

            context = f"""USER PROFILE

ICP TYPE: {icp_type}
LEARNER STATE: {learner_state}
WEEKLY HOURS: {weekly_hours}

ONBOARDING SUMMARY:
{st.session_state.generated_onboarding}

STRUCTURED ONBOARDING DATA:
{json_section}
"""

            roadmap_input = {
                "goal": goal or onboarding_json.get("primary_goal", ""),
                "current_role": (
                    "student"
                    if icp_type == "low"
                    else onboarding_json.get("current_role", "working professional")
                ),
                "years_experience": onboarding_json.get(
                    "years_experience",
                    0 if learner_state == "beginner" else 2
                ),
                "current_salary_annual":  onboarding_json.get("current_salary_monthly", 0) * 12,
                "current_salary_monthly": onboarding_json.get("current_salary_monthly", 0),
                "language_preference":    onboarding_json.get("language_preference", "en"),
                "weekly_hours_available": weekly_hours,
                "urgency":                onboarding_json.get("urgency", "high"),
                "self_efficacy":          onboarding_json.get(
                    "confidence_level",
                    "low" if learner_state == "beginner"
                    else ("medium" if learner_state == "intermediate" else "high")
                ),
                "goal_context": context
            }

        # ── REAL USER MODE ─────────────────────────────────────
        else:

            if not user_id:
                st.error("Please enter a User ID")
                st.stop()

            with st.spinner("Fetching user context from Pinecone..."):
                raw_context = retrieve_raw_context(user_id)

            if not raw_context:
                st.error(
                    "❌ No data found for this user ID. "
                    "Please make sure the user has completed onboarding."
                )
                st.stop()

            context = raw_context
            roadmap_input = user_id

        st.session_state.generated_context = context

        with st.expander("🧠 Context sent to Roadmap AI"):
            st.text_area("Context", value=context, height=250)

        with st.spinner("🤖 Generating personalized roadmap — this may take 30-60 seconds..."):
            roadmap_data = run_pipeline(roadmap_input)
            st.session_state.roadmap_data = roadmap_data

        if "error" in roadmap_data:
            st.error(f"❌ {roadmap_data['error']}")
            st.stop()

        st.success("✅ Roadmap Generated Successfully!")

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.code(traceback.format_exc(), language="python")

# =====================================================
# RENDER ROADMAP — V3 STRUCTURE
# =====================================================

if st.session_state.roadmap_data:

    roadmap_data = st.session_state.roadmap_data

    if "error" in roadmap_data:
        st.stop()

    st.divider()
    st.header("📚 Your Personalized Learning Roadmap")

    milestones = roadmap_data.get("milestones", [])

    total_modules = sum(len(m.get("modules", [])) for m in milestones)
    total_skills  = sum(
        len(mod.get("skills", []))
        for m in milestones
        for mod in m.get("modules", [])
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏁 Milestones",   len(milestones))
    c2.metric("📦 Modules",      total_modules)
    c3.metric("🎯 Skills",       total_skills)
    c4.metric("🏷 ICP Type",     roadmap_data.get("icp_type", "—").upper())
    c5.metric("🌐 Language",     roadmap_data.get("language", "en").upper())

    vision = roadmap_data.get("vision_profile", {})

    if vision:
        st.divider()
        st.subheader("🔭 Vision Profile")

        cv1, cv2 = st.columns(2)
        with cv1:
            st.info(f"**📍 Current State**\n\n{vision.get('current_state', '—')}")
            st.warning(f"**🚧 Main Blocker**\n\n{vision.get('main_blocker', '—')}")
        with cv2:
            st.success(f"**🌟 12-Month Vision**\n\n{vision.get('vision_12mo', '—')}")
            st.info(f"**🔥 Top Motivation**\n\n{vision.get('top_motivation', '—')}")

    st.divider()
    st.subheader(f"🎯 Target Role: {roadmap_data.get('target_role', '—')}")
    st.caption(
        f"Starting at **{roadmap_data.get('starting_milestone', '—')}** · "
        f"Currently active: **{roadmap_data.get('current_active_milestone', '—')}**"
    )

    if milestones:
        st.divider()
        st.header("🏆 Career Milestones")

        for milestone in milestones:

            m_id    = milestone.get("milestone_id", "?")
            m_label = milestone.get("identity_label", "")
            m_stmt  = milestone.get("identity_statement", "")
            m_value = milestone.get("market_value_display", "")
            m_seq   = milestone.get("sequence_order", "?")
            modules = milestone.get("modules", [])
            checkpoint = milestone.get("checkpoint_rule", {})

            with st.container(border=True):

                col_hdr, col_val = st.columns([3, 1])
                with col_hdr:
                    st.subheader(f"{m_id} — {m_label}")
                    st.caption(f"Sequence {m_seq}  ·  {m_stmt}")
                with col_val:
                    st.metric("💰 Market Value", m_value)
                    st.caption(
                        f"✅ Complete when all skills ≥ "
                        f"{int(checkpoint.get('required_mastery', 0.9) * 100)}% mastery"
                    )

                st.markdown("---")

                for mod in modules:
                    mod_id    = mod.get("module_id", "?")
                    mod_title = mod.get("title", "")
                    mod_desc  = mod.get("description", "")
                    mod_seq   = mod.get("sequence_order", "?")
                    skills    = mod.get("skills", [])

                    with st.expander(
                        f"📦 Module {mod_seq}: {mod_title}  [{mod_id}]",
                        expanded=(mod_seq == 1)
                    ):
                        st.caption(mod_desc)

                        for skill in skills:
                            s_id    = skill.get("skill_id", "?")
                            s_title = skill.get("title", "")
                            s_desc  = skill.get("description", "")
                            s_diff  = skill.get("difficulty", 0)
                            s_hours = skill.get("estimated_hours", 0)

                            mastery_state = skill.get("mastery_state", {})
                            unlock_rules  = skill.get("unlock_rules", {})
                            content_flow  = skill.get("content_flow", {})

                            with st.container(border=True):

                                sk1, sk2, sk3 = st.columns([3, 1, 1])
                                with sk1:
                                    st.markdown(f"**🎯 {s_title}**")
                                    st.caption(f"`{s_id}` — {s_desc}")
                                with sk2:
                                    st.metric("Difficulty", f"{int(s_diff * 100)}%")
                                with sk3:
                                    st.metric("Est. Hours", f"{s_hours}h")

                                current_mastery = mastery_state.get("current_mastery", 0.0)
                                target_mastery  = mastery_state.get("target_mastery", 0.9)
                                st.progress(
                                    current_mastery,
                                    text=f"Mastery: {int(current_mastery * 100)}% / Target: {int(target_mastery * 100)}%"
                                )

                                requires    = unlock_rules.get("requires", [])
                                min_mastery = unlock_rules.get("minimum_mastery", 0.0)

                                if requires:
                                    st.caption(
                                        f"🔒 Unlocks after: `{'`, `'.join(requires)}` "
                                        f"(min mastery: {int(min_mastery * 100)}%)"
                                    )
                                else:
                                    st.caption("🟢 Unlocked immediately (no prerequisites)")

                                st.markdown("**📂 Content Flow**")
                                cf1, cf2, cf3, cf4 = st.columns(4)

                                video    = content_flow.get("video", {})
                                scenario = content_flow.get("scenario", {})
                                mock     = content_flow.get("mock", {})
                                review   = content_flow.get("review", {})

                                with cf1:
                                    st.markdown("🎬 **Video**")
                                    st.write(video.get("title", "—"))
                                    st.caption(
                                        f"{video.get('duration_minutes', '?')} min · "
                                        f"`{video.get('content_id', '?')}`"
                                    )

                                with cf2:
                                    st.markdown("🧩 **Scenario**")
                                    st.write(scenario.get("title", "—"))
                                    st.caption(
                                        f"Difficulty {int(scenario.get('difficulty', 0) * 100)}% · "
                                        f"`{scenario.get('content_id', '?')}`"
                                    )

                                with cf3:
                                    st.markdown("📝 **Mock**")
                                    st.caption(
                                        f"Unlocks at {int(mock.get('unlock_mastery', 0.75) * 100)}% mastery"
                                    )
                                    st.caption(f"`{mock.get('content_id', '?')}`")

                                with cf4:
                                    st.markdown("🔁 **Review**")
                                    st.caption(review.get("review_type", "spaced_repetition"))
                                    next_review = review.get("next_review_at")
                                    st.caption(
                                        f"Next: {next_review}" if next_review else "Not scheduled yet"
                                    )

                                bkt = mastery_state.get("bkt", {})
                                if bkt:
                                    with st.expander("🔬 BKT Parameters"):
                                        b1, b2, b3, b4 = st.columns(4)
                                        b1.metric("Prior",      bkt.get("prior", "—"))
                                        b2.metric("Learn Rate", bkt.get("learn_rate", "—"))
                                        b3.metric("Guess",      bkt.get("guess", "—"))
                                        b4.metric("Slip",       bkt.get("slip", "—"))

    stored = roadmap_data.get("pinecone_stored", None)
    if stored is True:
        st.success("✅ Roadmap saved to Pinecone successfully")
    elif stored is False:
        st.warning("⚠ Roadmap generated but NOT saved to Pinecone — check logs")

    st.divider()

    with st.expander("🧾 View Full Raw JSON"):
        st.json(roadmap_data)

    st.download_button(
        label="⬇ Download Roadmap JSON",
        data=json.dumps(roadmap_data, indent=2),
        file_name="vidya_roadmap_v3.json",
        mime="application/json"
    )

    st.download_button(
    label="⬇ Download Roadmap JSON",
    data=json.dumps(roadmap_data, indent=2),
    file_name="vidya_roadmap_v3.json",
    mime="application/json"
)

# ============================================================
# Pinecone Debug Section
# ============================================================

st.divider()
st.subheader("🔎 Debug: Verify Pinecone Storage")

check_uid = st.text_input(
    "User ID to check",
    value="ai_generated_user",
    key="debug_uid"
)

if st.button("🔍 Check Pinecone"):
    from src.pinecone_utils import pc, INDEX_NAME

    index = pc.Index(INDEX_NAME)

    results = index.query(
        vector=[0.0] * 768,
        top_k=20,
        namespace=check_uid,
        include_metadata=True
    )

    matches = results.get("matches", [])

    roadmap_matches = [
        m for m in matches
        if m["metadata"].get("doc_type")
        in ("roadmap", "roadmap_summary")
    ]

    if not roadmap_matches:
        st.error(f"❌ No roadmap vectors found for '{check_uid}'")
    else:
        st.success(f"✅ Found {len(roadmap_matches)} roadmap(s)")

        for m in roadmap_matches:
            meta = m["metadata"]

            st.write(f"### {m['id']}")

            st.json({
                "doc_type": meta.get("doc_type"),
                "target_role": meta.get("target_role"),
                "icp_type": meta.get("icp_type"),
                "generated_at": meta.get("generated_at"),
                "full_roadmap_stored": meta.get("full_roadmap_stored"),
                "milestone_labels": meta.get("milestone_labels")
            })