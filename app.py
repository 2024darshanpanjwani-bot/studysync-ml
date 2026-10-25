"""StudySync Streamlit application."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from studysync.analytics import subject_weaknesses, summarize_progress
from studysync.data import SUBJECTS, generate_synthetic_dataset
from studysync.database import (
    add_study_session,
    delete_study_session,
    fetch_study_sessions,
)
from studysync.model_store import save_model
from studysync.models import MODEL_NAMES, compare_models, train_model
from studysync.planner import generate_timetable

st.set_page_config(
    page_title="StudySync | Student Performance Studio",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#a78bfa"
CYAN = "#7ccfe8"
BLUE = "#7f9cff"
MUTED = "#9b98aa"
PANEL = "#15141c"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0&display=swap');
    :root {
        --ink: #f3f1f8;
        --muted: #9b98aa;
        --panel: #15141c;
        --line: #2a2935;
        --violet: #a78bfa;
        --cyan: #7ccfe8;
        --blue: #7f9cff;
        --green: #8fe0bb;
        --surface: #111017;
    }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp {
        color: var(--ink);
        background-color: #0b0a10;
        background-image: linear-gradient(rgba(171, 157, 220, .025) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(171, 157, 220, .025) 1px, transparent 1px);
        background-size: 42px 42px;
    }
    [data-testid="stHeader"] { background: rgba(11, 10, 16, .9); }
    [data-testid="stMain"] > div { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1480px; padding-top: 2rem; padding-bottom: 3.5rem; }
    [data-testid="stSidebar"] { background: #0f0e15; border-right: 1px solid #24232e; }
    [data-testid="stSidebar"] > div { padding: 1.5rem 1rem 1rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: .32rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border: 1px solid transparent; border-radius: 9px; padding: .5rem .7rem;
        transition: background .18s ease, border-color .18s ease, transform .18s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,.035); transform: translateX(2px);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(100deg, rgba(167,139,250,.17), rgba(127,156,255,.06));
        border-color: rgba(167,139,250,.2);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color: #f2edff; font-weight: 700; }
    .brand { display: flex; align-items: center; gap: .65rem; font-size: 1.25rem; font-weight: 800; }
    .brand-mark {
        display: grid; place-items: center; width: 2rem; height: 2rem; border-radius: 9px;
        background: linear-gradient(145deg, #b79aff, #718dff); color: #111018; font-size: 1.1rem;
        box-shadow: 0 5px 18px rgba(132,117,255,.28);
    }
    .material-symbols-rounded { font-size: 1.2rem; line-height: 1; vertical-align: middle; font-variation-settings: 'FILL' 0, 'wght' 450, 'GRAD' 0, 'opsz' 24; }
    .eyebrow { color: var(--cyan); font-family: 'DM Mono', monospace; font-size: .69rem; text-transform: uppercase; letter-spacing: .12em; }
    .page-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; margin: .1rem 0 1.35rem; }
    .page-title { font-size: 2.35rem; line-height: 1.12; font-weight: 800; margin: .35rem 0 .5rem; letter-spacing: 0; }
    .page-subtitle { color: var(--muted); max-width: 740px; margin: 0; font-size: .96rem; line-height: 1.65; }
    .heading-status { display: inline-flex; align-items: center; gap: .5rem; white-space: nowrap; margin-top: .5rem; padding: .48rem .7rem; border: 1px solid var(--line); border-radius: 999px; background: rgba(21,20,28,.8); color: #c6c3d0; font: .68rem 'DM Mono', monospace; text-transform: uppercase; }
    .status-dot { width: .42rem; height: .42rem; border-radius: 50%; background: var(--green); box-shadow: 0 0 12px rgba(143,224,187,.6); }
    .notice { background: linear-gradient(105deg, rgba(167,139,250,.12), rgba(127,156,255,.055) 58%, rgba(124,207,232,.035)); border: 1px solid rgba(167,139,250,.22); border-left: 3px solid var(--violet); border-radius: 9px; padding: .85rem 1rem; color: #ded9e8; font-size: .84rem; line-height: 1.55; margin: .65rem 0 1.35rem; }
    .metric-wrap {
        position: relative; overflow: hidden; min-height: 142px; padding: 1.05rem 1.1rem;
        border: 1px solid rgba(181,168,220,.14); border-radius: 13px;
        background: linear-gradient(145deg, rgba(31,29,41,.96), rgba(18,17,24,.97) 62%);
        box-shadow: inset 0 1px rgba(255,255,255,.035), 0 12px 30px rgba(0,0,0,.12);
        transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    }
    .metric-wrap:hover { transform: translateY(-2px); border-color: rgba(167,139,250,.35); box-shadow: inset 0 1px rgba(255,255,255,.05), 0 16px 34px rgba(0,0,0,.2); }
    .metric-wrap:after { content: ''; position: absolute; width: 90px; height: 90px; right: -42px; top: -42px; border: 1px solid rgba(167,139,250,.16); border-radius: 50%; box-shadow: 0 0 0 12px rgba(167,139,250,.025), 0 0 0 25px rgba(127,156,255,.018); }
    .metric-top { display: flex; align-items: center; justify-content: space-between; gap: .5rem; }
    .metric-label { color: #aaa7b7; font: .68rem 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .09em; }
    .metric-icon { display: grid; place-items: center; width: 1.8rem; height: 1.8rem; border: 1px solid rgba(167,139,250,.2); border-radius: 7px; color: var(--violet); background: rgba(167,139,250,.09); }
    .metric-icon .material-symbols-rounded { font-size: 1.12rem; }
    .metric-value { color: #f6f3fc; font-size: 1.8rem; line-height: 1.2; font-weight: 800; margin-top: .5rem; }
    .metric-note { color: var(--muted); font-size: .73rem; line-height: 1.4; margin-top: .22rem; }
    .section-title { font-size: 1.02rem; font-weight: 750; margin: .5rem 0 .2rem; }
    .section-caption { color: var(--muted); font-size: .78rem; line-height: 1.5; margin-bottom: .7rem; }
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 8px; border: 1px solid rgba(167,139,250,.45);
        background: linear-gradient(115deg, #433263, #31335b); color: #fff;
        font-weight: 700; min-height: 2.55rem; transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover { border-color: var(--violet); color: white; box-shadow: 0 5px 18px rgba(115,92,180,.22); transform: translateY(-1px); }
    div[data-baseweb="input"] input, div[data-baseweb="select"] > div, textarea { background: #17161f !important; border-color: #34323f !important; border-radius: 8px !important; }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { box-shadow: 0 0 0 4px rgba(167,139,250,.13); }
    [data-testid="stMetric"] { background: linear-gradient(145deg, #1d1b26, #14131a); border: 1px solid var(--line); border-radius: 11px; padding: .9rem 1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stDataFrame"] { border: 1px solid #302e3a; border-radius: 9px; overflow: hidden; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: #302e3a !important; border-radius: 12px !important; background: linear-gradient(145deg, rgba(24,23,32,.82), rgba(17,16,23,.86)); }
    [data-testid="stProgress"] > div > div { background: linear-gradient(90deg, var(--violet), var(--blue), var(--cyan)); }
    [data-testid="stAlert"] { border-radius: 9px; }
    div[data-testid="stPlotlyChart"] { border: 1px solid rgba(181,168,220,.09); border-radius: 11px; overflow: hidden; }
    hr { border-color: var(--line); margin: 1.15rem 0; }
    .empty-state { border: 1px dashed #383543; border-radius: 12px; padding: 1.25rem; background: rgba(21,20,28,.62); }
    .empty-icon { display: inline-grid; place-items: center; width: 2.2rem; height: 2.2rem; border-radius: 9px; color: var(--cyan); background: rgba(124,207,232,.09); margin-bottom: .6rem; }
    .empty-title { color: var(--ink); font-weight: 700; margin-bottom: .2rem; }
    .empty-copy { color: var(--muted); font-size: .82rem; line-height: 1.55; }
    @media (max-width: 820px) {
        [data-testid="stMainBlockContainer"] { padding: 1.4rem 1rem 2.5rem; }
        .heading-status { display: none; }
        .metric-wrap { min-height: 128px; padding: .9rem; }
    }
    @media (max-width: 520px) {
        .page-title { font-size: 1.8rem; }
        .page-subtitle { font-size: .88rem; }
        .metric-value { font-size: 1.45rem; }
        .metric-wrap { min-height: 120px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Preparing the synthetic demo cohort...")
def get_cohort() -> pd.DataFrame:
    return generate_synthetic_dataset(1200, seed=42)


@st.cache_data(show_spinner="Comparing regression models on a held-out split...")
def get_model_comparison() -> pd.DataFrame:
    return compare_models(get_cohort())


@st.cache_resource(show_spinner="Training the selected demo model...")
def get_trained_model(model_name: str):
    return train_model(get_cohort(), model_name)


def chart_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"family": "Manrope, sans-serif", "color": "#d7d4df", "size": 11},
        margin={"l": 20, "r": 20, "t": 28, "b": 22},
        legend={"bgcolor": "rgba(0,0,0,0)", "orientation": "h", "y": 1.12, "x": 0},
        hoverlabel={"bgcolor": "#211e2c", "bordercolor": "#6d5d96", "font": {"family": "Manrope, sans-serif", "color": "#f5f1ff"}},
        hovermode="closest",
        transition={"duration": 260, "easing": "cubic-in-out"},
    )
    fig.update_xaxes(
        gridcolor="#2b2936", zerolinecolor="#393644", linecolor="#393644",
        ticks="outside", tickcolor="#393644",
    )
    fig.update_yaxes(
        gridcolor="#2b2936", zerolinecolor="#393644", linecolor="#393644",
        ticks="outside", tickcolor="#393644",
    )
    return fig


def page_heading(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        '<div class="page-heading"><div>'
        f'<div class="eyebrow">{eyebrow}</div><div class="page-title">{title}</div>'
        f'<div class="page-subtitle">{subtitle}</div></div>'
        '<div class="heading-status"><span class="status-dot"></span> Personal workspace</div></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str, icon: str = "insights") -> None:
    st.markdown(
        f'<div class="metric-wrap"><div class="metric-top"><div class="metric-label">{label}</div>'
        f'<span class="metric-icon"><span class="material-symbols-rounded">{icon}</span></span></div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="empty-state"><span class="empty-icon"><span class="material-symbols-rounded">{icon}</span></span>'
        f'<div class="empty-title">{title}</div><div class="empty-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )


def section_heading(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def render_dashboard(cohort: pd.DataFrame, sessions: pd.DataFrame) -> None:
    page_heading("OVERVIEW / 01", "Your study, in focus.", "A clear view of performance signals and the work you have logged.")
    st.markdown(
        '<div class="notice"><strong>Demo data:</strong> Cohort charts and model signals below use '
        'synthetic records. They demonstrate the product workflow and are not evidence of real-world accuracy.</div>',
        unsafe_allow_html=True,
    )

    progress = summarize_progress(sessions)
    columns = st.columns(4)
    values = [
        ("Study time", f"{progress['total_minutes'] / 60:.1f}h", "Logged in your study tracker", "schedule"),
        ("Sessions", str(progress["session_count"]), "Personal study records", "library_books"),
        ("Focus average", f"{progress['average_focus']:.1f}/5", "Across logged sessions", "center_focus_strong"),
        ("Cohort mean", f"{cohort['exam_score'].mean():.1f}", "Synthetic exam score / 100", "query_stats"),
    ]
    for column, (label, value, note, icon) in zip(columns, values):
        with column:
            metric_card(label, value, note, icon)

    left, right = st.columns([1.3, 1])
    with left:
        section_heading("Subject performance", "Synthetic cohort averages, grouped by subject")
        subject_scores = cohort.groupby("subject", as_index=False)["exam_score"].mean().sort_values("exam_score")
        fig = px.bar(
            subject_scores,
            x="exam_score",
            y="subject",
            orientation="h",
            color="exam_score",
            color_continuous_scale=[[0, "#594283"], [1, CYAN]],
            labels={"exam_score": "Mean score", "subject": ""},
        )
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(range=[0, 100], title="Mean exam score / 100")
        fig.update_traces(marker_line_width=0, opacity=.94, hovertemplate="%{y}<br>Mean score: %{x:.1f}<extra></extra>")
        st.plotly_chart(chart_layout(fig, 330), width="stretch")
    with right:
        section_heading("Score distribution", "Illustrative synthetic outcomes")
        fig = px.histogram(
            cohort,
            x="exam_score",
            nbins=24,
            color_discrete_sequence=[ACCENT],
            labels={"exam_score": "Exam score", "count": "Students"},
        )
        fig.update_traces(marker_line_color="#0b0a0f", marker_line_width=1)
        fig.update_xaxes(title="Exam score / 100")
        fig.update_yaxes(title="Students")
        st.plotly_chart(chart_layout(fig, 330), width="stretch")

    section_heading("Your recent rhythm", "Personal study time saved to this device")
    if sessions.empty:
        empty_state("monitoring", "Your rhythm starts with one session", "Log a study block in Progress and your daily activity trend will appear here.")
    else:
        activity = sessions.copy()
        activity["session_date"] = pd.to_datetime(activity["session_date"])
        daily = activity.groupby("session_date", as_index=False)["duration_minutes"].sum()
        fig = px.area(
            daily,
            x="session_date",
            y="duration_minutes",
            color_discrete_sequence=[CYAN],
            labels={"session_date": "Date", "duration_minutes": "Study minutes"},
        )
        fig.update_traces(fillcolor="rgba(97,217,232,.14)")
        st.plotly_chart(chart_layout(fig, 250), width="stretch")


def render_prediction() -> None:
    page_heading("PREDICTION / 02", "Explore a score estimate.", "Adjust study inputs and see how a demo regression model responds.")
    st.markdown(
        '<div class="notice"><strong>Illustrative estimate only.</strong> The model is trained on generated '
        'synthetic data and has not been validated for real students. Do not use this estimate for academic decisions.</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        section_heading("Student profile", "Adjust the inputs used by the selected regression model")
        with st.form("prediction_form"):
            left, right = st.columns(2)
            with left:
                subject = st.selectbox("Subject", SUBJECTS)
                study_hours = st.slider("Weekly study hours", 0.5, 20.0, 6.0, 0.5)
                previous_score = st.slider("Most recent score", 0.0, 100.0, 68.0, 1.0)
                attendance_rate = st.slider("Attendance", 0.0, 100.0, 90.0, 1.0)
            with right:
                sleep_hours = st.slider("Average nightly sleep", 3.0, 12.0, 7.5, 0.5)
                extracurricular_hours = st.slider("Weekly extracurricular hours", 0.0, 30.0, 4.0, 0.5)
                model_name = st.selectbox("Regression model", MODEL_NAMES)
                submitted = st.form_submit_button("Generate estimate", icon=":material/auto_awesome:")

    if submitted:
        try:
            model = get_trained_model(model_name)
            input_row = pd.DataFrame(
                [{
                    "subject": subject,
                    "study_hours": study_hours,
                    "attendance_rate": attendance_rate,
                    "previous_score": previous_score,
                    "sleep_hours": sleep_hours,
                    "extracurricular_hours": extracurricular_hours,
                }]
            )
            estimate = float(np.clip(model.predict(input_row)[0], 0, 100))
            st.markdown("#### Model estimate")
            result, context = st.columns([1, 1.25])
            with result:
                st.metric("Synthetic-model estimate", f"{estimate:.1f} / 100")
                st.progress(estimate / 100, text="Illustrative score scale")
            with context:
                st.markdown(
                    '<div class="notice"><strong>Read as a demo signal.</strong> This estimate is generated from '
                    'synthetic training data. It is not a confidence score, promise, or validated forecast.</div>',
                    unsafe_allow_html=True,
                )
        except (ValueError, RuntimeError) as error:
            st.error(f"Could not generate an estimate: {error}")


def render_planner() -> None:
    page_heading("STUDY PLANNER / 03", "Make room for the hard parts.", "Build a weekly plan that gives extra time to subjects below your target.")
    with st.container(border=True):
        section_heading("Plan inputs", "Set a score for each subject and your available study time")
        with st.form("planner_form"):
            columns = st.columns(3)
            scores = {}
            for index, subject in enumerate(SUBJECTS):
                with columns[index % len(columns)]:
                    scores[subject] = st.slider(f"{subject}", 0, 100, 70, key=f"planner_{subject}")
            controls = st.columns(2)
            with controls[0]:
                weekly_hours = st.slider("Available study hours this week", 1, 40, 10)
            with controls[1]:
                days = st.slider("Days to schedule", 1, 7, 5)
            submitted = st.form_submit_button("Build my timetable", icon=":material/calendar_month:")

    if submitted:
        try:
            timetable = generate_timetable(scores, weekly_hours, days)
            st.markdown("#### Your study blocks")
            if timetable.empty:
                empty_state("event_busy", "No study blocks yet", "Increase available study hours and build the timetable again.")
            else:
                summary = st.columns(3)
                plan_cards = [
                    ("Weekly allocation", f"{timetable['duration_minutes'].sum() / 60:.1f}h", "Across all subjects", "schedule"),
                    ("Study blocks", str(len(timetable)), "Focused sessions in this plan", "view_timeline"),
                    ("Planned days", str(timetable["day"].nunique()), f"Of {days} available days", "date_range"),
                ]
                for column, (label, value, note, icon) in zip(summary, plan_cards):
                    with column:
                        metric_card(label, value, note, icon)
                pivot = timetable.pivot_table(
                    index="day", columns="subject", values="duration_minutes", aggfunc="sum", fill_value=0
                )
                section_heading("Weekly distribution", "Session minutes by day and subject")
                chart_data = timetable.groupby(["day", "subject"], as_index=False)["duration_minutes"].sum()
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][:days]
                fig = px.bar(
                    chart_data,
                    x="day",
                    y="duration_minutes",
                    color="subject",
                    barmode="stack",
                    category_orders={"day": day_order, "subject": list(SUBJECTS)},
                    color_discrete_sequence=[ACCENT, BLUE, CYAN, "#c6a0f6", "#72bda8"],
                    labels={"day": "", "duration_minutes": "Minutes", "subject": "Subject"},
                )
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(chart_layout(fig, 300), width="stretch")
                section_heading("Block details", "Use the allocation as a starting point; adjust around your week")
                st.caption(f"Allocated {timetable['duration_minutes'].sum()} minutes across {timetable['day'].nunique()} days.")
                st.dataframe(timetable, width="stretch", hide_index=True)
        except ValueError as error:
            st.error(f"Could not build a timetable: {error}")


def render_subject_insights(cohort: pd.DataFrame) -> None:
    page_heading("SUBJECT INSIGHTS / 04", "Know where to focus.", "Use score gaps as prompts for planning, not labels of ability.")
    st.markdown(
        '<div class="notice"><strong>Synthetic cohort signals:</strong> The averages and suggestions on this page '
        'come from generated demo data and should not be interpreted as an individual diagnosis.</div>',
        unsafe_allow_html=True,
    )
    target = st.slider("Target score", 40, 95, 70, 1)
    insights = subject_weaknesses(cohort, target)
    weakest = insights.iloc[0]
    if weakest["gap_to_target"] > 0:
        st.markdown(
            f'<div class="notice"><strong>Largest cohort gap:</strong> {weakest["subject"]} is '
            f'{weakest["gap_to_target"]:.1f} points below the selected target on average. '
            'Treat this as a cohort-level planning prompt, not an individual prediction.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("All synthetic subject averages meet the selected target. Keep reviewing missed questions.")
    left, right = st.columns([1, 1.2])
    with left:
        section_heading("Subject signals", "Sorted from lowest synthetic mean score")
        fig = px.bar(
            insights,
            x="subject",
            y="mean_score",
            color="gap_to_target",
            color_continuous_scale=[[0, CYAN], [1, "#bb83ec"]],
            labels={"subject": "", "mean_score": "Mean score", "gap_to_target": "Gap to target"},
        )
        fig.add_hline(y=target, line_dash="dot", line_color="#f0b56a", annotation_text="Target")
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(chart_layout(fig, 340), width="stretch")
    with right:
        section_heading("Suggested next steps", "Short, practical prompts by subject")
        st.dataframe(
            insights.rename(
                columns={
                    "subject": "Subject",
                    "mean_score": "Synthetic mean",
                    "observations": "Records",
                    "gap_to_target": "Gap",
                    "recommendation": "Recommendation",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={"Synthetic mean": st.column_config.NumberColumn(format="%.1f")},
        )


def render_progress() -> None:
    page_heading("PROGRESS / 05", "Small sessions add up.", "Log focused study time and review your personal history, stored locally in SQLite.")
    try:
        sessions = fetch_study_sessions()
    except (OSError, RuntimeError) as error:
        st.error(f"Could not open your local progress database: {error}")
        return

    metrics = summarize_progress(sessions)
    columns = st.columns(4)
    cards = [
        ("Sessions logged", str(metrics["session_count"]), "Saved to this device", "library_books"),
        ("Total time", f"{metrics['total_minutes'] / 60:.1f}h", "Across all logged sessions", "schedule"),
        ("Focus average", f"{metrics['average_focus']:.1f}/5", "Self-rated session focus", "center_focus_strong"),
        ("Active days", str(metrics["active_days"]), "Days with study activity", "calendar_month"),
    ]
    for column, (label, value, note, icon) in zip(columns, cards):
        with column:
            metric_card(label, value, note, icon)

    left, right = st.columns([.8, 1.2])
    with left:
        section_heading("Log a study session", "Only stored in your local SQLite database")
        with st.form("study_session_form", clear_on_submit=True):
            session_date = st.date_input("Date", value=date.today())
            subject = st.selectbox("Subject", SUBJECTS + ("Other",))
            duration = st.number_input("Minutes studied", min_value=1, max_value=720, value=45, step=5)
            focus = st.select_slider("Focus", options=[1, 2, 3, 4, 5], value=4)
            notes = st.text_area("Session note (optional)", max_chars=300)
            submitted = st.form_submit_button("Save session", icon=":material/save:")
        if submitted:
            try:
                add_study_session(session_date, subject, int(duration), int(focus), notes)
                st.success("Study session saved.")
                st.rerun()
            except (ValueError, OSError) as error:
                st.error(f"Could not save session: {error}")

    with right:
        section_heading("Session history", "Your recent logged sessions")
        if sessions.empty:
            empty_state("edit_calendar", "Your session history is ready", "Save a study session to start seeing daily trends and your recent activity here.")
        else:
            activity = sessions.copy()
            activity["session_date"] = pd.to_datetime(activity["session_date"])
            daily = activity.groupby("session_date", as_index=False)["duration_minutes"].sum()
            fig = px.area(
                daily,
                x="session_date",
                y="duration_minutes",
                markers=True,
                color_discrete_sequence=[CYAN],
                labels={"session_date": "Date", "duration_minutes": "Minutes"},
            )
            fig.update_traces(
                line={"color": CYAN, "width": 2.5},
                marker={"size": 7, "line": {"color": PANEL, "width": 2}},
                fillcolor="rgba(124,207,232,.12)",
                hovertemplate="%{x|%b %d}<br>%{y} study minutes<extra></extra>",
            )
            st.plotly_chart(chart_layout(fig, 245), width="stretch")
            display = sessions.drop(columns="id").rename(
                columns={
                    "session_date": "Date",
                    "subject": "Subject",
                    "duration_minutes": "Minutes",
                    "focus_rating": "Focus / 5",
                    "notes": "Note",
                }
            )
            st.dataframe(display, width="stretch", hide_index=True)
            delete_columns = st.columns([3, 1])
            with delete_columns[0]:
                selected_id = st.selectbox(
                    "Remove a session",
                    sessions["id"].tolist(),
                    format_func=lambda item: (
                        f"{sessions.loc[sessions['id'] == item, 'session_date'].iloc[0]} · "
                        f"{sessions.loc[sessions['id'] == item, 'subject'].iloc[0]} · #{item}"
                    ),
                    label_visibility="collapsed",
                )
            with delete_columns[1]:
                if st.button("Delete", icon=":material/delete:", width="stretch"):
                    if delete_study_session(int(selected_id)):
                        st.success("Session removed.")
                        st.rerun()


def render_model_insights(cohort: pd.DataFrame) -> None:
    page_heading("MODEL INSIGHTS / 06", "Compare the approaches.", "Three regressors, one shared holdout, and transparent error metrics.")
    st.markdown(
        '<div class="notice"><strong>Evaluation scope:</strong> These metrics describe only a held-out slice of '
        'the synthetic dataset. They must not be read as real-world predictive performance.</div>',
        unsafe_allow_html=True,
    )
    try:
        results = get_model_comparison()
    except (ValueError, RuntimeError) as error:
        st.error(f"Could not compare models: {error}")
        return
    best = results.iloc[0]
    metric_columns = st.columns(3)
    for column, label, value in zip(
        metric_columns,
        ("Lowest synthetic MAE", "Holdout records", "Training records"),
        (f"{best['mae']:.2f}", str(int(len(cohort) * 0.2)), str(int(len(cohort) * 0.8))),
    ):
        with column:
            icon = {"Lowest synthetic MAE": "query_stats", "Holdout records": "fact_check", "Training records": "model_training"}[label]
            note = "Lower is better on this synthetic split" if label == "Lowest synthetic MAE" else "Shared reproducible 80/20 split"
            metric_card(label, value, note, icon)

    left, right = st.columns([1.1, 1])
    with left:
        section_heading("Held-out comparison", "Lower MAE and RMSE indicate smaller errors on this synthetic holdout")
        plot_data = results.melt(id_vars="model", value_vars=["mae", "rmse"], var_name="metric", value_name="error")
        fig = px.bar(
            plot_data,
            x="model",
            y="error",
            color="metric",
            barmode="group",
            color_discrete_sequence=[ACCENT, CYAN],
            labels={"model": "", "error": "Points", "metric": "Metric"},
        )
        st.plotly_chart(chart_layout(fig, 340), width="stretch")
    with right:
        section_heading("Metric detail", "R-squared is shown as a descriptive holdout statistic")
        st.dataframe(
            results.rename(columns={"model": "Model", "mae": "MAE", "rmse": "RMSE", "r2": "R-squared"}),
            width="stretch",
            hide_index=True,
            column_config={
                "MAE": st.column_config.NumberColumn(format="%.3f"),
                "RMSE": st.column_config.NumberColumn(format="%.3f"),
                "R-squared": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption(f"Lowest MAE on this synthetic split: {best['model']}.")

    st.divider()
    section_heading("Save a local demo artifact", "The selected pipeline is refit on all synthetic demo rows")
    st.caption("Joblib files can execute code when loaded. Keep this artifact local and only load files you trust.")
    model_choice = st.selectbox("Model to save", MODEL_NAMES, key="artifact_model")
    if st.button("Train and save model", icon=":material/download:"):
        try:
            artifact = save_model(get_trained_model(model_choice), Path("models") / "studysync_demo.joblib")
            st.success(f"Saved synthetic-data demo model to {artifact}.")
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not save model: {error}")


def main() -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">S</span> StudySync</div>', unsafe_allow_html=True)
        st.caption("STUDENT PERFORMANCE STUDIO")
        st.divider()
        page = st.radio(
            "Workspace",
            ["Dashboard", "Prediction", "Study planner", "Subject insights", "Progress", "Model insights"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("<span class='eyebrow'>DATA MODE</span>", unsafe_allow_html=True)
        st.caption("Synthetic cohort · local-only progress")

    try:
        cohort = get_cohort()
        if page == "Dashboard":
            render_dashboard(cohort, fetch_study_sessions())
        elif page == "Prediction":
            render_prediction()
        elif page == "Study planner":
            render_planner()
        elif page == "Subject insights":
            render_subject_insights(cohort)
        elif page == "Progress":
            render_progress()
        else:
            render_model_insights(cohort)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"StudySync could not load this view: {error}")
        st.info("Check the local database path and try reloading the page.")


if __name__ == "__main__":
    main()