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

# ---------------------------------------------------------------------------
# Custom background & graphics (CSS)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* App-wide background — YouTube's own dark-theme page color */
.stApp {
    background-color: #0f0f0f;
}

/* Make default text readable on dark background */
.stApp, .stApp p, .stApp li, .stApp label, .stApp span {
    color: #f1f1f1;
}

/* Hero banner — YouTube header bar styling with a play-button mark */
.hero-banner {
    background-color: #0f0f0f;
    border-bottom: 1px solid #272727;
    padding: 1.4rem 0.2rem 1.6rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 14px;
}
.hero-banner .play-mark {
    width: 46px;
    height: 32px;
    background-color: #ff0000;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.hero-banner .play-mark svg {
    width: 16px;
    height: 16px;
}
.hero-banner h1 {
    color: #ffffff;
    font-size: 1.6rem;
    margin: 0 0 2px 0;
    font-weight: 600;
}
.hero-banner p {
    color: #aaaaaa;
    font-size: 0.95rem;
    margin: 0;
}

/* Content cards — YouTube's card gray (#212121) */
.glass-card {
    background-color: #212121;
    border: 1px solid #303030;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.glass-card h3 {
    margin-top: 0;
    color: #ff4d4d;
}

/* Comparison table styling */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.95rem;
}
.compare-table th {
    background-color: #272727;
    color: #ff4d4d;
    text-align: left;
    padding: 10px 14px;
}
.compare-table td {
    padding: 10px 14px;
    border-top: 1px solid #303030;
    background-color: #181818;
}
.compare-table tr:nth-child(even) td {
    background-color: #1c1c1c;
}
.compare-table td.ai-col {
    color: #6fdc8c;
    font-weight: 500;
}
.compare-table td.manual-col {
    color: #ff8a8a;
}

/* Sidebar tint — YouTube's slightly darker chrome */
section[data-testid="stSidebar"] {
    background-color: #0f0f0f;
    border-right: 1px solid #272727;
}
section[data-testid="stSidebar"] * {
    color: #f1f1f1 !important;
}

/* Form container */
div[data-testid="stForm"] {
    background-color: #212121;
    border: 1px solid #303030;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="play-mark">
            <svg viewBox="0 0 24 24" fill="#ffffff" xmlns="http://www.w3.org/2000/svg">
                <path d="M8 5v14l11-7z"/>
            </svg>
        </div>
        <div>
            <h1>AI YouTube Content Planner</h1>
            <p>LangChain + Gemini agent that generates video ideas, SEO titles,
            thumbnail concepts, and a weekly upload schedule for your channel.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# What problem it solves
# ---------------------------------------------------------------------------
with st.expander("💡 What problem does this tool solve?", expanded=False):
    st.markdown(
        """
        <div class="glass-card">
        <h3>The problem</h3>
        <p>Planning a YouTube channel manually means juggling video ideas,
        SEO research, thumbnail concepts, and an upload calendar across
        spreadsheets, notes apps, and scattered chats — a process that can
        eat up hours every week, especially for beginner or solo creators
        who don't have a strategist or an editorial team.</p>
        <p>This app collapses that entire workflow into one form: tell it
        your niche, audience, tone, and upload cadence, and it returns a
        complete, structured content plan in seconds — ideas, SEO-ready
        titles and tags, thumbnail concepts, and a calendar — ready to
        export as JSON or CSV.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <table class="compare-table">
            <tr>
                <th>Step</th>
                <th>Traditional / Manual Planning</th>
                <th>AI YouTube Content Planner</th>
            </tr>
            <tr>
                <td>Idea generation</td>
                <td class="manual-col">Brainstorm alone or in a group; often runs dry</td>
                <td class="ai-col">Niche-specific ideas generated instantly, at any volume</td>
            </tr>
            <tr>
                <td>SEO titles &amp; tags</td>
                <td class="manual-col">Manual keyword research across multiple tools</td>
                <td class="ai-col">SEO-optimized titles and tags generated with each idea</td>
            </tr>
            <tr>
                <td>Thumbnail concepts</td>
                <td class="manual-col">Designed from scratch with no starting direction</td>
                <td class="ai-col">Concrete visual concept suggested for every video</td>
            </tr>
            <tr>
                <td>Upload schedule</td>
                <td class="manual-col">Built by hand in a spreadsheet or calendar app</td>
                <td class="ai-col">Auto-generated calendar based on chosen upload cadence</td>
            </tr>
            <tr>
                <td>Time required</td>
                <td class="manual-col">Hours per planning session</td>
                <td class="ai-col">Under a minute per full content plan</td>
            </tr>
            <tr>
                <td>Cost</td>
                <td class="manual-col">Strategist / premium SEO tool subscriptions</td>
                <td class="ai-col">Free — bring your own Gemini API key</td>
            </tr>
            <tr>
                <td>Output format</td>
                <td class="manual-col">Scattered across docs, sheets, and notes</td>
                <td class="ai-col">One export: downloadable JSON &amp; CSV</td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True,
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
        ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"],
        index=0,
        help="Model names as of the Gemini API's current lineup. "
             "Older names like gemini-1.5-flash and gemini-2.0-flash "
             "have been retired by Google and will 404.",
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
