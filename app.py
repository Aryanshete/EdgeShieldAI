"""EdgeShield AI — Autonomous Edge Security Operations Platform."""

from __future__ import annotations

import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="EdgeShield AI — Intelligent Edge Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.dashboard import render_dashboard

if __name__ == "__main__":
    render_dashboard()
