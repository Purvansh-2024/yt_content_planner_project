import json
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI YouTube Content Planner",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 AI YouTube Content Planner")
st.caption(
    "LangChain + Gemini agent that generates video ideas, SEO titles, "
    "thumbnail concepts, and a weekly upload schedule for your channel."
)

# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    # Priority: Streamlit secrets (for deployed app) -> env var -> sidebar input
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    if os.environ.get("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]
    return None


with st.sidebar:
    st.header("⚙️ Settings")
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get a free key at https://aistudio.google.com/apikey",
        )
    else:
        st.success("API key loaded from secrets.")

    st.divider()
    num_ideas = st.slider("Number of video ideas", min_value=3, max_value=15, value=8)
    schedule_days_per_week = st.slider("Upload days per week", 1, 7, 3)
    model_name = st.selectbox(
        "Gemini model",
        ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
    )

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("planner_form"):
    col1, col2 = st.columns(2)
    with col1:
        niche = st.text_input(
            "Channel niche / topic*",
            placeholder="e.g. Personal finance for college students",
        )
        audience = st.text_input(
            "Target audience",
            placeholder="e.g. 18-24 year olds new to budgeting",
        )
    with col2:
        tone = st.selectbox(
            "Tone / style",
            ["Casual & fun", "Professional & informative", "Motivational",
             "Witty & sarcastic", "Calm & educational", "Bold & controversial"],
        )
        extra_notes = st.text_input(
            "Anything else? (optional)",
            placeholder="e.g. focus on short-form Shorts content",
        )

    submitted = st.form_submit_button("✨ Generate Content Plan", use_container_width=True)

# ---------------------------------------------------------------------------
# LangChain setup
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are an expert YouTube content strategist and SEO specialist.

Channel niche: {niche}
Target audience: {audience}
Tone/style: {tone}
Extra notes: {extra_notes}

Generate a YouTube content plan with exactly {num_ideas} video ideas.

Return ONLY valid JSON (no markdown fences, no commentary) matching this schema:

{{
  "channel_summary": "1-2 sentence summary of the content strategy for this channel",
  "video_ideas": [
    {{
      "video_idea": "short description of the video concept",
      "seo_title": "an SEO-optimized, clickable YouTube title (under 70 characters)",
      "seo_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
      "thumbnail_concept": "a concrete visual description of the thumbnail (colors, text overlay, imagery)",
      "video_format": "one of: Long-form, Short, Tutorial, Vlog, Listicle, Interview, Reaction"
    }}
  ]
}}

Make each idea distinct, specific to the niche, and genuinely useful — avoid generic filler ideas.
"""


def build_chain(key: str, model: str):
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=0.9,
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    parser = JsonOutputParser()
    return prompt | llm | parser


def build_schedule(video_ideas: list[dict], days_per_week: int) -> pd.DataFrame:
    upload_dow = {
        1: [0],
        2: [0, 3],
        3: [0, 2, 4],
        4: [0, 1, 3, 5],
        5: [0, 1, 2, 3, 4],
        6: [0, 1, 2, 3, 4, 5],
        7: [0, 1, 2, 3, 4, 5, 6],
    }[days_per_week]

    rows = []
    today = date.today()
    # find the next Monday to start a clean week
    start = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
    idea_cycle = list(video_ideas)
    idx = 0
    d = start
    weeks_needed = -(-len(idea_cycle) // days_per_week)  # ceil division
    for _ in range(weeks_needed):
        for dow in upload_dow:
            if idx >= len(idea_cycle):
                break
            upload_date = d + timedelta(days=dow)
            rows.append({
                "Upload Date": upload_date.strftime("%a, %b %d"),
                "Video Title": idea_cycle[idx]["seo_title"],
                "Format": idea_cycle[idx].get("video_format", ""),
            })
            idx += 1
        d += timedelta(days=7)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
if submitted:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar first.")
    elif not niche.strip():
        st.error("Please enter a channel niche/topic.")
    else:
        with st.spinner("Generating your content plan..."):
            try:
                chain = build_chain(api_key, model_name)
                result = chain.invoke({
                    "niche": niche,
                    "audience": audience or "general audience interested in this niche",
                    "tone": tone,
                    "extra_notes": extra_notes or "none",
                    "num_ideas": num_ideas,
                })
                st.session_state["plan"] = result
            except Exception as e:
                st.error(f"Something went wrong generating the plan: {e}")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "plan" in st.session_state:
    plan = st.session_state["plan"]

    st.subheader("📋 Strategy Summary")
    st.info(plan.get("channel_summary", ""))

    ideas = plan.get("video_ideas", [])

    st.subheader("💡 Video Ideas")
    for i, idea in enumerate(ideas, start=1):
        with st.expander(f"{i}. {idea.get('seo_title', idea.get('video_idea', 'Untitled'))}"):
            st.markdown(f"**Concept:** {idea.get('video_idea', '')}")
            st.markdown(f"**Format:** {idea.get('video_format', '')}")
            st.markdown(f"**SEO Title:** {idea.get('seo_title', '')}")
            st.markdown(f"**SEO Tags:** {', '.join(idea.get('seo_tags', []))}")
            st.markdown(f"**Thumbnail Concept:** {idea.get('thumbnail_concept', '')}")

    st.subheader("🗓️ Suggested Upload Schedule")
    schedule_df = build_schedule(ideas, schedule_days_per_week)
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    st.subheader("⬇️ Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Plan (JSON)",
            data=json.dumps(plan, indent=2),
            file_name="youtube_content_plan.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        ideas_df = pd.DataFrame(ideas)
        st.download_button(
            "Download Ideas (CSV)",
            data=ideas_df.to_csv(index=False),
            file_name="youtube_video_ideas.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.markdown(
        "👈 Fill in your channel details in the form above and click "
        "**Generate Content Plan** to get started."
    )
