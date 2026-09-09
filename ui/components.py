"""Reusable UI components for the EdgeShield AI security dashboard."""

from __future__ import annotations

import streamlit as st
from typing import Any

from src.incidents import Incident, IncidentManager
from src.reasoning import ReasoningOutput
from src.risk import RISK_CRITICAL, RISK_HIGH, RISK_LOW, RISK_MEDIUM, RiskAssessment
from src.runtime import RuntimeStatus


def render_custom_css() -> None:
    """Inject polished enterprise cybersecurity dark theme CSS."""
    st.markdown(
        """
        <style>
        /* Base page styling */
        .stApp {
            background-color: #0b0f19;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Sidebar enterprise dark styling */
        section[data-testid="stSidebar"], [data-testid="stSidebar"] {
            background-color: #0d121f !important;
            border-right: 1px solid #1e293b !important;
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] .stRadio label {
            color: #cbd5e1 !important;
        }

        /* Top Header Container */
        .edgeshield-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.5rem;
            background: linear-gradient(135deg, #111827 0%, #1a2234 100%);
            border: 1px solid #1f293d;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        .edgeshield-logo {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            color: #f8fafc;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .edgeshield-tagline {
            font-size: 0.8rem;
            letter-spacing: 0.2em;
            color: #94a3b8;
            text-transform: uppercase;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .status-online {
            background-color: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .device-badge {
            background-color: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            padding: 0.35rem 0.75rem;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .device-badge-amd {
            background-color: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Metric & Risk Cards */
        .card-container {
            background: #111827;
            border: 1px solid #1f293d;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
        }
        .card-header {
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .risk-score-display {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1;
            margin: 0.5rem 0;
            display: flex;
            align-items: baseline;
            gap: 0.4rem;
        }
        .risk-max-scale {
            font-size: 1rem;
            color: #64748b;
            font-weight: 500;
        }
        .risk-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .risk-critical {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border: 1px solid #ef4444;
        }
        .risk-high {
            background: rgba(249, 115, 22, 0.2);
            color: #f97316;
            border: 1px solid #f97316;
        }
        .risk-medium {
            background: rgba(234, 179, 8, 0.2);
            color: #eab308;
            border: 1px solid #eab308;
        }
        .risk-low {
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid #10b981;
        }
        .risk-summary-text {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-top: 0.5rem;
            line-height: 1.4;
        }

        /* Timeline Items */
        .timeline-scroll {
            max-height: 280px;
            overflow-y: auto;
            padding-right: 0.5rem;
        }
        .timeline-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 0.75rem;
            background: rgba(31, 41, 61, 0.5);
            border-left: 3px solid #3b82f6;
            border-radius: 0 6px 6px 0;
            margin-bottom: 0.4rem;
            font-size: 0.85rem;
        }
        .timeline-row-alert {
            border-left-color: #f97316;
            background: rgba(249, 115, 22, 0.08);
        }
        .timeline-row-critical {
            border-left-color: #ef4444;
            background: rgba(239, 68, 68, 0.12);
        }
        .timeline-time {
            font-family: monospace;
            color: #94a3b8;
            font-weight: 600;
        }
        .timeline-event {
            color: #f1f5f9;
            font-weight: 500;
        }

        /* Reasoning Cards */
        .reasoning-box {
            background: #151d2f;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 0.5rem;
        }
        .action-box {
            background: rgba(239, 68, 68, 0.07);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 0.75rem;
        }
        .evidence-chip {
            display: inline-block;
            background: rgba(148, 163, 184, 0.15);
            color: #cbd5e1;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            font-size: 0.75rem;
            margin: 0.2rem 0.2rem 0.2rem 0;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(runtime: RuntimeStatus) -> None:
    """Render the top banner with system status and truthful backend information."""
    is_amd = runtime.amd_rocm_active
    device_label = f"AMD ROCm ({runtime.device_name or 'GPU'})" if is_amd else f"PyTorch Backend: {runtime.backend} ({runtime.device})"
    badge_class = "device-badge-amd" if is_amd else "device-badge"

    st.markdown(
        f"""
        <div class="edgeshield-header">
            <div>
                <div class="edgeshield-logo">
                    <span>🛡️ EDGESHIELD AI</span>
                </div>
                <div class="edgeshield-tagline">Intelligent Edge Security • See. Understand. Act.</div>
            </div>
            <div style="display: flex; gap: 0.75rem; align-items: center;">
                <span class="status-pill status-online">● System Online</span>
                <span class="device-badge {badge_class}">{device_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_panel(assessment: RiskAssessment) -> None:
    """Render the prominent risk assessment card matching PDF Page 21."""
    level = assessment.level.upper()
    if level == RISK_CRITICAL:
        badge_class = "risk-critical"
    elif level == RISK_HIGH:
        badge_class = "risk-high"
    elif level == RISK_MEDIUM:
        badge_class = "risk-medium"
    else:
        badge_class = "risk-low"

    st.markdown(
        f"""
        <div class="card-container">
            <div class="card-header">
                <span>Threat Assessment</span>
                <span class="risk-badge {badge_class}">{level} RISK</span>
            </div>
            <div class="risk-score-display">
                <span>{assessment.score}</span>
                <span class="risk-max-scale">/ 100</span>
            </div>
            <div class="risk-summary-text">{assessment.summary_label}</div>
            <hr style="border: 0; border-top: 1px solid #1f293d; margin: 0.75rem 0;" />
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.3rem;">CONTRIBUTING FACTORS:</div>
        """,
        unsafe_allow_html=True,
    )

    if assessment.factors:
        cols = st.columns(len(assessment.factors))
        for col, (factor_name, pts) in zip(cols, assessment.factors.items()):
            formatted_name = factor_name.replace("_", " ").title()
            col.metric(label=formatted_name, value=f"+{pts}")
    else:
        st.caption("No risk factors triggered")

    st.markdown("</div>", unsafe_allow_html=True)


def render_event_timeline(events: list[dict[str, Any]]) -> None:
    """Render the chronological event timeline panel."""
    st.markdown(
        """
        <div class="card-container">
            <div class="card-header">
                <span>Event Timeline</span>
                <span style="color: #3b82f6;">● Real-Time Log</span>
            </div>
            <div class="timeline-scroll">
        """,
        unsafe_allow_html=True,
    )

    if not events:
        st.markdown(
            """
            <div style="color: #64748b; font-size: 0.85rem; padding: 1rem; text-align: center;">
                No security events observed yet. Start video or demo mode to stream detections.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for ev in events:
            time_str = ev.get("timestamp", "00:00:00")
            event_type = ev.get("event_type", "event")
            zone_str = f" [{ev.get('zone')}]" if ev.get("zone") else ""
            display_title = event_type.replace("_", " ").title() + zone_str

            is_alert = "entry" in event_type or "approach" in event_type or "after_hours" in event_type
            is_critical = "dwell" in event_type or "critical" in event_type
            row_class = "timeline-row-critical" if is_critical else ("timeline-row-alert" if is_alert else "timeline-row")
            icon = "🔴" if is_critical else ("⚠️" if is_alert else "ℹ️")

            st.markdown(
                f"""
                <div class="{row_class}">
                    <span class="timeline-time">{time_str}</span>
                    <span class="timeline-event">{display_title}</span>
                    <span>{icon}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_reasoning_card(reasoning: ReasoningOutput) -> None:
    """Render the AI contextual reasoning and operator recommendations."""
    st.markdown(
        f"""
        <div class="card-container">
            <div class="card-header">
                <span>AI Contextual Reasoning</span>
                <span style="color: #8b5cf6;">Llama 3.2 Security Agent</span>
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.4rem; font-weight: 600;">
                WHY WAS THIS FLAGGED?
            </div>
            <div class="reasoning-box">
                <div style="font-size: 0.9rem; line-height: 1.5; color: #f1f5f9;">
                    {reasoning.explanation}
                </div>
                <div style="margin-top: 0.6rem;">
        """,
        unsafe_allow_html=True,
    )

    for item in reasoning.evidence:
        st.markdown(f'<span class="evidence-chip">• {item}</span>', unsafe_allow_html=True)

    st.markdown(
        f"""
                </div>
            </div>
            <div class="action-box">
                <div style="font-size: 0.8rem; color: #ef4444; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 0.2rem;">
                    RECOMMENDED OPERATOR ACTION:
                </div>
                <div style="font-size: 0.9rem; color: #f8fafc; font-weight: 500;">
                    {reasoning.recommended_action}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_incidents_view(manager: IncidentManager) -> None:
    """Render the persistent incident management and report export interface."""
    incidents = manager.load_incidents()
    st.subheader("Security Incidents & Audit Records")

    if not incidents:
        st.info("No recorded security incidents in persistent storage.")
        return

    col_select, col_actions = st.columns([2, 1])
    incident_options = [f"{inc.incident_id} - {inc.location} ({inc.risk_level}, Score: {inc.risk_score})" for inc in incidents]
    selected_option = col_select.selectbox("Select Incident to Review", options=incident_options)
    selected_id = selected_option.split(" - ")[0]
    incident = manager.get_incident(selected_id)

    if not incident:
        return

    with col_actions:
        new_status = st.selectbox(
            "Update Lifecycle Status",
            options=["OPEN", "INVESTIGATING", "RESOLVED", "DISMISSED"],
            index=["OPEN", "INVESTIGATING", "RESOLVED", "DISMISSED"].index(incident.status),
            key=f"status_select_{incident.incident_id}",
        )
        if st.button("Apply Status Change", use_container_width=True):
            manager.update_incident_status(incident.incident_id, new_status)
            st.success(f"Status updated to {new_status}")
            st.rerun()

    # Multi-format report export
    st.markdown("**Download Official Incident Audit Report:**")
    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    with dcol1:
        st.download_button(
            label="🌐 Download HTML Report\n(Opens in Browser / PDF)",
            data=manager.generate_html_report(incident),
            file_name=f"{incident.incident_id}_audit_report.html",
            mime="text/html",
            use_container_width=True,
            type="primary",
            help="Opens automatically in any web browser with print-to-PDF formatting.",
        )
    with dcol2:
        st.download_button(
            label="📄 Download Plain Text\n(Opens in Notepad)",
            data=manager.generate_text_report(incident),
            file_name=f"{incident.incident_id}_audit_report.txt",
            mime="text/plain",
            use_container_width=True,
            help="Opens instantly in Windows Notepad or any text editor.",
        )
    with dcol3:
        st.download_button(
            label="📝 Download Markdown\n(Documentation)",
            data=manager.generate_report(incident),
            file_name=f"{incident.incident_id}_report.md",
            mime="text/markdown",
            use_container_width=True,
            help="Standard GitHub-flavored markdown format.",
        )
    with dcol4:
        import json
        st.download_button(
            label="📊 Download JSON\n(SIEM / Machine Data)",
            data=json.dumps(incident.as_dict(), indent=2),
            file_name=f"{incident.incident_id}_data.json",
            mime="application/json",
            use_container_width=True,
            help="Full structured incident schema for SIEM / SOC ingestion.",
        )

    report_text = manager.generate_report(incident)
    with st.expander("📄 View Full In-Dashboard Audit Report Preview", expanded=True):
        st.markdown(report_text)
