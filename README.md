# 🎬 AI YouTube Content Planner

A LangChain + Gemini agent that turns a channel niche into a full content plan:
video ideas, SEO-optimized titles, thumbnail concepts, and a weekly upload
schedule — deployed as a Streamlit app.

Built for: **MED-01 · Media & Content Creation** (Agentic AI Project Catalogue)

---

## 1. Get a free Gemini API key

1. Go to https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click **Create API key** → copy it

## 2. Run it locally

```bash
git clone <your-repo-url>
cd youtube_content_planner
pip install -r requirements.txt

# Option A: paste the key directly in the app's sidebar (simplest)
streamlit run app.py

# Option B: store it as a local secret instead
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your key in
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 3. Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repo (make sure `.streamlit/secrets.toml`
   is **not** committed — it's already in `.gitignore`)
2. Go to https://share.streamlit.io → **New app** → point it at your repo, branch,
   and `app.py`
3. In **App settings → Secrets**, paste:
   ```toml
   GOOGLE_API_KEY = "your-gemini-api-key-here"
   ```
4. Deploy. Your app gets a public `*.streamlit.app` URL.

## How it works

- **LangChain flow:** `ChatPromptTemplate → ChatGoogleGenerativeAI → JsonOutputParser`
  (a single structured-output chain, no external tools — matches the
  "Intermediate" difficulty spec: single Chain, 3-4 day build)
- **Inputs:** niche/topic, target audience, tone, optional notes, number of
  ideas, upload cadence
- **Outputs:**
  - Strategy summary
  - N video ideas, each with concept, format, SEO title, SEO tags, thumbnail
    concept
  - An auto-generated weekly upload calendar based on your chosen cadence
  - Downloadable JSON (full plan) and CSV (ideas table)

## Project structure

```
youtube_content_planner/
├── app.py                          # Streamlit app + LangChain logic
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example
├── .gitignore
└── README.md
```

## Ideas for extending it (stretch goals)

- Swap the plain Chain for a **Tavily-enabled Agent** that checks trending
  topics/competitor videos before generating ideas (bumps this up to Advanced)
- Add a **calendar view** (e.g. `streamlit-calendar`) instead of a table
- Cache generated plans per niche using `st.cache_data`
- Add multi-language support for titles/tags
