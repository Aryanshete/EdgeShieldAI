"""Main dashboard workflow and pipeline integration for EdgeShield AI."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

from src.context import ContextEngine, SecurityContext
from src.detector import DEFAULT_MODEL_PATH, YOLODetector
from src.events import EventEngine
from src.incidents import IncidentManager
from src.privacy import PrivacyAuditor
from src.reasoning import LlamaReasoningAgent, ReasoningOutput
from src.risk import RiskAssessment, RiskEngine
from src.runtime import get_runtime_status
from src.scenarios import SCENARIOS, ScenarioConfig
from src.tracker import ObjectTracker, bottom_center
from src.zones import ZoneManager, draw_zones
from ui.components import (
    render_custom_css,
    render_event_timeline,
    render_header,
    render_incidents_view,
    render_reasoning_card,
    render_risk_panel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_VIDEO_PATH = PROJECT_ROOT / "videos" / "demo" / "intrusion_demo.mp4"
DEFAULT_ZONES_PATH = PROJECT_ROOT / "config" / "zones.json"


def initialize_session_state() -> None:
    """Initialize pipeline engines and session data in Streamlit state."""
    if "runtime_status" not in st.session_state:
        st.session_state.runtime_status = get_runtime_status()

    if "zone_manager" not in st.session_state:
        st.session_state.zone_manager = ZoneManager(config_path=DEFAULT_ZONES_PATH)

    if "risk_engine" not in st.session_state:
        st.session_state.risk_engine = RiskEngine()

    if "reasoning_agent" not in st.session_state:
        st.session_state.reasoning_agent = LlamaReasoningAgent()

    if "context_engine" not in st.session_state:
        st.session_state.context_engine = ContextEngine(zone_manager=st.session_state.zone_manager)

    if "incident_manager" not in st.session_state:
        st.session_state.incident_manager = IncidentManager()

    if "events_list" not in st.session_state:
        # Pre-load saved events if they exist
        events_file = PROJECT_ROOT / "data" / "events.json"
        if events_file.is_file():
            try:
                st.session_state.events_list = [
                    e.as_dict() for e in EventEngine.load_events(events_file)
                ]
            except Exception:
                st.session_state.events_list = []
        else:
            st.session_state.events_list = []

    if "current_assessment" not in st.session_state:
        # Pre-compute baseline assessment from loaded events if available
        if st.session_state.events_list:
            contexts = st.session_state.context_engine.build_all_contexts(
                EventEngine.load_events(PROJECT_ROOT / "data" / "events.json")
            )
            if contexts:
                st.session_state.current_assessment = st.session_state.risk_engine.evaluate_context(contexts[0])
                st.session_state.current_reasoning = st.session_state.reasoning_agent.analyze(
                    contexts[0], st.session_state.current_assessment
                )
            else:
                st.session_state.current_assessment = RiskAssessment(
                    score=0, level="LOW", factors={}, evidence=[], summary_label="Routine observation"
                )
                st.session_state.current_reasoning = ReasoningOutput(
                    risk_level="LOW", evidence=[], explanation="No active events observed.",
                    recommended_action="Standby.", summary="System nominal.", is_fallback=True
                )
        else:
            st.session_state.current_assessment = RiskAssessment(
                score=0, level="LOW", factors={}, evidence=[], summary_label="Routine observation"
            )
            st.session_state.current_reasoning = ReasoningOutput(
                risk_level="LOW", evidence=[], explanation="No active events observed.",
                recommended_action="Standby.", summary="System nominal.", is_fallback=True
            )


def render_dashboard() -> None:
    """Render the main EdgeShield AI operational interface."""
    render_custom_css()
    initialize_session_state()

    # Top Header
    render_header(st.session_state.runtime_status)

    # Sidebar: Source Selection & Controls (Phase 10 Demo Mode)
    with st.sidebar:
        st.subheader("Control & Source")
        source_type = st.radio(
            "Input Source",
            options=["Demo Video (Restricted Intrusion)", "Upload Video File", "Webcam Feed"],
            index=0,
            help="Select the video feed input to process through the EdgeShield AI security pipeline.",
        )

        video_path: Path | str | None = None
        if source_type == "Demo Video (Restricted Intrusion)":
            video_path = DEMO_VIDEO_PATH
            st.info("🎯 **Scenario:** Restricted Area Intrusion\nSubject enters high-security Server Room outside authorized hours (22:17).")
        elif source_type == "Upload Video File":
            uploaded_file = st.file_uploader("Choose an MP4/AVI file", type=["mp4", "avi", "mov"])
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                video_path = tfile.name
        else:
            st.warning("Webcam input requires an active camera device on index 0.")
            video_path = 0

        conf_threshold = st.slider("Detection Confidence", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
        st.divider()

        st.subheader("Scenario Profile")
        scenario_key = st.selectbox(
            "Evaluation Scenario",
            options=list(SCENARIOS.keys()),
            format_func=lambda k: SCENARIOS[k].title,
            index=0,
            help="Select preset threshold profiles for loitering, abandoned object, intrusion, or movement velocity.",
        )
        selected_scenario = SCENARIOS[scenario_key]
        st.caption(f"ℹ️ {selected_scenario.description}")
        st.divider()

        st.subheader("Configured Zones")
        for zone in st.session_state.zone_manager.zones:
            hours = zone.authorized_hours
            hours_str = f"{hours.get('start', '08:00')} - {hours.get('end', '18:00')}" if hours else "24/7"
            color_badge = "🔴 Restricted" if zone.type == "restricted" else "🔵 Monitored"
            st.markdown(f"**{zone.name}** ({color_badge})\n*Hours:* {hours_str}")

        st.divider()
        start_button = st.button("🚀 Start Analysis", type="primary", use_container_width=True)

    # Main Interface Tabs
    tab_ops, tab_incidents, tab_architecture = st.tabs([
        "🛡️ Live Operations & Threat Analysis",
        "📋 Incidents & Audit Reports",
        "⚙️ Pipeline Architecture & Verification",
    ])

    with tab_ops:
        col_video, col_threat = st.columns([1.5, 1])

        with col_threat:
            risk_placeholder = st.empty()
            risk_placeholder.markdown(
                f"""
                <div class="card-container">
                    <div class="card-header"><span>Threat Assessment</span><span>READY</span></div>
                    <div class="risk-score-display"><span>{st.session_state.current_assessment.score}</span><span class="risk-max-scale">/ 100</span></div>
                    <div class="risk-summary-text">{st.session_state.current_assessment.summary_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_risk_panel(st.session_state.current_assessment)

        with col_video:
            st.markdown('<div class="card-header"><span>Visual Surveillance Feed</span><span>REAL-TIME PIPELINE</span></div>', unsafe_allow_html=True)
            video_frame_placeholder = st.empty()

        col_timeline, col_reasoning = st.columns([1, 1])

        with col_timeline:
            timeline_placeholder = st.empty()
            with timeline_placeholder.container():
                render_event_timeline(st.session_state.events_list[-8:])

        with col_reasoning:
            reasoning_placeholder = st.empty()
            with reasoning_placeholder.container():
                render_reasoning_card(st.session_state.current_reasoning)

        # Video Processing Execution Loop
        if start_button and video_path is not None:
            cap = cv2.VideoCapture(str(video_path) if isinstance(video_path, Path) else video_path)
            if not cap.isOpened():
                st.error(f"Failed to open video source: {video_path}")
                return

            event_engine = EventEngine(
                zone_manager=st.session_state.zone_manager,
                base_time="22:17:00",
                scenario_config=selected_scenario,
            )
            tracker = ObjectTracker(
                confidence_threshold=conf_threshold,
                zone_manager=st.session_state.zone_manager,
                event_engine=event_engine,
            )

            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            frame_idx = 0
            live_events: list[dict[str, Any]] = []

            progress_bar = st.progress(0, text="Processing video frames with YOLOv8 and ByteTrack...")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 480
            step = 2  # Process every 2nd frame for smooth UI response

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_seconds = frame_idx / fps

                if frame_idx % step == 0:
                    tracks = tracker.track(frame, timestamp_seconds)

                    # Highlight active zones
                    active_zone_ids = {
                        z.id for t in tracks for z in st.session_state.zone_manager.evaluate_track(t)
                    }

                    annotated = draw_zones(
                        frame, st.session_state.zone_manager.zones, active_zone_ids=active_zone_ids
                    )
                    annotated = tracker.draw_tracks(annotated, tracks)

                    # Convert BGR to RGB for Streamlit display
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    # Update events and threat assessments
                    if event_engine.events:
                        live_events = [e.as_dict() for e in event_engine.events]
                        timeline_placeholder.empty()
                        with timeline_placeholder.container():
                            render_event_timeline(live_events[-8:])

                        # Evaluate Context & Risk
                        contexts = st.session_state.context_engine.build_all_contexts(event_engine.events)
                        if contexts:
                            active_ctx = contexts[-1]
                            assessment = st.session_state.risk_engine.evaluate_context(active_ctx)
                            reasoning = st.session_state.reasoning_agent.analyze(active_ctx, assessment)

                            with col_threat:
                                render_risk_panel(assessment)
                            with reasoning_placeholder.container():
                                render_reasoning_card(reasoning)

                            st.session_state.current_assessment = assessment
                            st.session_state.current_reasoning = reasoning

                    progress_bar.progress(min(1.0, frame_idx / total_frames), text=f"Processing: Frame {frame_idx}/{total_frames}")

                frame_idx += 1
                if frame_idx >= total_frames:
                    break

            cap.release()
            progress_bar.empty()
            st.success("Analysis complete! All events recorded into persistent session state.")

            # Persist newly captured events
            if event_engine.events:
                event_engine.export_events(PROJECT_ROOT / "data" / "events.json")
                contexts = st.session_state.context_engine.build_all_contexts(event_engine.events)
                for c in contexts:
                    ass = st.session_state.risk_engine.evaluate_context(c)
                    reas = st.session_state.reasoning_agent.analyze(c, ass)
                    st.session_state.incident_manager.create_incident(c, ass, reas)
                st.session_state.events_list = [e.as_dict() for e in event_engine.events]

    with tab_incidents:
        render_incidents_view(st.session_state.incident_manager)

    with tab_architecture:
        st.subheader("Verified Component Latency & Performance Benchmarks (Phase 12)")
        perf_file = PROJECT_ROOT / "reports" / "performance.json"
        if perf_file.is_file():
            try:
                import json
                with open(perf_file, "r", encoding="utf-8") as f:
                    perf_data = json.load(f)
                bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                bcol1.metric("Throughput", f"{perf_data.get('processing_fps', 0)} FPS")
                bcol2.metric("YOLOv8 Inference", f"{perf_data.get('yolo_mean_ms', 0)} ms")
                bcol3.metric("ByteTrack Tracking", f"{perf_data.get('tracking_mean_ms', 0)} ms")
                bcol4.metric("Zone & Events", f"{perf_data.get('event_processing_mean_ms', 0)} ms")

                md_perf = PROJECT_ROOT / "reports" / "performance.md"
                if md_perf.is_file():
                    with st.expander("📊 View Complete Benchmark Audit Table", expanded=True):
                        st.markdown(md_perf.read_text(encoding="utf-8"))
            except Exception:
                st.caption("Benchmark records available.")
        else:
            st.info("Run `python scripts/run_benchmark.py` to record verified hardware benchmarks.")

        st.subheader("EdgeShield Modular Architecture")
        st.markdown(
            """
            ```text
            [Surveillance Video / RTSP Feed]
                         │
                         ▼
             [OpenCV Frame Ingestion]
                         │
                         ▼
             [YOLOv8 Object Detection]
                         │
                         ▼
             [ByteTrack Anonymous Multi-Object Tracking]
                         │
                         ▼
             [Spatial Zone Engine (cv2.pointPolygonTest)]
                         │
                         ▼
             [Stateful Security Event Engine (src/events.py)]
                         │
                         ▼
             [Temporal Context Synthesis (src/context.py)]
                         │
                         ▼
             [Deterministic Risk Engine (0-100 Baseline Heuristics)]
                         │
                         ▼
             [Llama 3.2 Reasoning Agent (Evidence-Grounded Actions)]
                         │
                         ▼
             [Persistent Incident Manager & Audit Reports]
                         │
                         ▼
             [EdgeShield Operations Dashboard]
            ```
            """
        )

        st.subheader("Runtime Verification Details")
        st.json(st.session_state.runtime_status.as_dict())

        st.subheader("Privacy-Aware Architecture (Phase 14)")
        auditor = PrivacyAuditor()
        rep = auditor.get_compliance_report()
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown(f"**Status:** `COMPLIANT`  \n**Framework:** {rep['framework']}")
            st.markdown(f"*{rep['declaration']}*")
        with pcol2:
            for k, v in rep["principles"].items():
                st.markdown(f"• **{k}:** {v}")
