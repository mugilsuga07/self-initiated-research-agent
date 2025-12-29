import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from src.ui.pipeline import run_full_pipeline

st.set_page_config(
    page_title="Autonomous AI Research & Decision Support Agent",
    page_icon=None,
    layout="centered",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp, .stApp p, .stApp span, .stApp li, .stApp div {
        color: #e8e8e8 !important;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: #ffffff !important;
    }
    .subtitle {
        color: rgba(255,255,255,0.8) !important;
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    .decision-text {
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
        color: #ffffff !important;
    }
    .section-header {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #a78bfa !important;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .tradeoff-pro {
        color: #6ee7b7 !important;
    }
    .tradeoff-con {
        color: #fca5a5 !important;
    }
    .stats-bar {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        padding: 1rem;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6) !important;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .disclaimer {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5) !important;
        text-align: center;
        margin-top: 1.5rem;
        padding: 1rem;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stTextInput > div > div > input {
        background: #ffffff !important;
        color: #1a1a2e !important;
        border-radius: 25px !important;
        border: none !important;
        padding: 0.75rem 1.25rem !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46a1 100%) !important;
    }
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        color: #e8e8e8 !important;
    }
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('''
<div class="main-header">
    <p class="main-title">Autonomous AI Research & Decision Support Agent</p>
    <p class="subtitle">Ask a question and get a researched recommendation</p>
</div>
''', unsafe_allow_html=True)

question = st.text_input(
    "Your question",
    placeholder="Enter your query",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    research_button = st.button("Research", use_container_width=True, type="primary")

if research_button and question:
    if len(question.strip()) < 5:
        st.error("Please provide a valid question (at least 5 characters).")
    else:
        with st.spinner("Researching... This may take a minute."):
            try:
                result = run_full_pipeline(question)
                st.session_state.result = result
            except Exception as e:
                st.error(f"Research failed: {str(e)}")
                st.session_state.result = None

if "result" in st.session_state and st.session_state.result:
    result = st.session_state.result
    rec = result["recommendation"]
    reasoning = result["reasoning"]
    stats = result["stats"]
    
    st.divider()
    
    st.markdown(f'<p class="decision-text">{rec["decision"]}</p>', unsafe_allow_html=True)
    
    if rec["key_reasons"]:
        st.markdown('<p class="section-header">Key Reasons</p>', unsafe_allow_html=True)
        for reason in rec["key_reasons"]:
            st.markdown(f"* {reason}")
    
    if rec["trade_offs"]:
        st.markdown('<p class="section-header">Trade-offs</p>', unsafe_allow_html=True)
        for to in rec["trade_offs"]:
            if isinstance(to, dict):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<span class="tradeoff-pro">+ {to.get("pro", "")}</span>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<span class="tradeoff-con">- {to.get("con", "")}</span>', unsafe_allow_html=True)
    
    if rec["risks"]:
        st.markdown('<p class="section-header">Risks & Limitations</p>', unsafe_allow_html=True)
        for risk in rec["risks"]:
            st.markdown(f"* {risk}")
    
    if rec["next_steps"]:
        st.markdown('<p class="section-header">Next Steps</p>', unsafe_allow_html=True)
        for i, step in enumerate(rec["next_steps"], 1):
            st.markdown(f"{i}. {step}")
    
    with st.expander("Show reasoning & evidence"):
        st.markdown("#### Gap Analysis")
        unknowns = reasoning["gaps"]["unknowns"]
        if unknowns:
            for u in unknowns:
                importance = u.get("importance", "medium").upper()
                st.markdown(f"**[{importance}]** {u['description']}")
        else:
            st.markdown("_No significant unknowns identified._")
        
        st.markdown("---")
        
        st.markdown("#### Top Sources")
        sources = reasoning["top_sources"]
        if sources:
            for s in sources:
                st.markdown(f"* **{s['title']}** - _{s['domain']}_")
        else:
            st.markdown("_No sources available._")
        
        st.markdown("---")
        
        st.markdown("#### Clarifying Questions")
        questions = reasoning["clarifying_questions"]
        if questions:
            for q in questions:
                st.markdown(f"**{q['question']}**")
                st.markdown(f"_{q['why']}_")
                st.markdown("")
        else:
            st.markdown("_No clarifying questions needed._")
    
    st.markdown(
        f'<div class="stats-bar">'
        f'<span>{stats["sources_analyzed"]} sources</span>'
        f'<span>{stats["claims_extracted"]} claims</span>'
        f'<span>{stats["gaps_identified"]} gaps</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    st.markdown(f'<p class="disclaimer">{rec["disclaimer"]}</p>', unsafe_allow_html=True)
