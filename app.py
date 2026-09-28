"""
Trip Planner AI Agent — Main Streamlit Application
Capstone Project: Step 1 — Project Setup & API Integration
"""

import streamlit as st
from api_clients import test_nominatim, test_overpass

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trip Planner AI",
    page_icon="✈️",
    layout="wide",
)

# ── Sidebar: API Key Management ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    # Secure API key input — stored only in session state (never persisted to disk)
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is stored only in this browser session and never saved.",
    )

    if api_key_input:
        st.session_state["openai_api_key"] = api_key_input
        st.success("✅ API key saved for this session")
    elif "openai_api_key" not in st.session_state:
        st.warning("Please enter your OpenAI API key to continue.")

    st.markdown("---")
    st.caption("🔒 Key is held in session state only — cleared when you close the tab.")

# ── Main Header ───────────────────────────────────────────────────────────────
st.title("✈️ Trip Planner AI Agent")
st.markdown(
    "An intelligent trip planner powered by OpenAI function calling, "
    "live OpenStreetMap data, and interactive maps."
)
st.markdown("---")

# ── Connection Status Panel ───────────────────────────────────────────────────
st.subheader("🔌 API Connection Status")

col1, col2, col3 = st.columns(3)

# 1. OpenAI
with col1:
    st.markdown("**OpenAI API**")
    if "openai_api_key" in st.session_state:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=st.session_state["openai_api_key"])
            # Lightweight call — just list models
            client.models.list()
            st.success("✅ Connected")
            st.session_state["openai_client"] = client
        except Exception as e:
            st.error(f"❌ Failed: {e}")
    else:
        st.info("⏳ Awaiting API key")

# 2. Nominatim (geocoding)
with col2:
    st.markdown("**Nominatim (Geocoding)**")
    ok, msg = test_nominatim()
    if ok:
        st.success(f"✅ {msg}")
    else:
        st.error(f"❌ {msg}")

# 3. Overpass (POI search)
with col3:
    st.markdown("**Overpass API (POI)**")
    ok, msg = test_overpass()
    if ok:
        st.success(f"✅ {msg}")
    else:
        st.error(f"❌ {msg}")

st.markdown("---")

# ── App Skeleton: coming in later steps ───────────────────────────────────────
st.subheader("🗺️ Plan Your Trip")

if "openai_api_key" not in st.session_state:
    st.info("👈 Enter your OpenAI API key in the sidebar to get started.")
else:
    destination = st.text_input(
        "Where do you want to go?",
        placeholder="e.g. Paris, France",
    )
    trip_days = st.slider("How many days?", min_value=1, max_value=14, value=3)

    if st.button("🚀 Plan My Trip", type="primary"):
        if not destination.strip():
            st.warning("Please enter a destination first.")
        else:
            st.info(
                f"🛠️ Agent workflow coming in Step 2! "
                f"You asked for a **{trip_days}-day trip to {destination}**."
            )

st.markdown("---")
st.caption("Infosys Springboard Capstone Project · Step 1 Complete")
