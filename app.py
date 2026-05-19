from __future__ import annotations

from pathlib import Path
from datetime import datetime
import base64
import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.clinicaltrials_client import ClinicalTrialsClient
from src.metrics import (
    ACTIVE_STATUSES,
    add_activity_flags,
    phase_distribution,
    sponsor_activity,
    status_distribution,
    target_metrics,
    target_phase_heatmap,
    target_year_heatmap,
    target_momentum_table,
    future_start_table,
    future_start_summary,
    yearly_trial_momentum,
    yoy_summary,
    add_clinical_intelligence_flags,
    modality_mix,
    combo_intelligence,
    sponsor_conviction,
    line_of_therapy_mix,
    risk_signal_table,
    asset_level_tracking,
    modality_by_target,
    target_exploitation_profile,
    crowding_scores,
    target_evolution_timeline,
    combo_agent_frequency,
)
from src.normalize import normalize_studies
from src.target_registry import load_targets, query_terms_for_target

APP_TITLE = "Invenra ADC Capital Map"
DATA_PATH = Path("data/adc_targets.csv")
EMBLEM_PATH = Path("assets/buildwell_emblem.png")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  --iv-bg: #050710;
  --iv-panel: rgba(13, 18, 33, .82);
  --iv-panel-2: rgba(15, 21, 38, .72);
  --iv-line: rgba(151, 92, 255, .22);
  --iv-line-soft: rgba(255,255,255,.075);
  --iv-purple: #9d4dff;
  --iv-purple-2: #7a35f0;
  --iv-violet-soft: rgba(157,77,255,.12);
  --iv-blue: #4f8cff;
  --iv-teal: #25d6b5;
  --iv-green: #53e36b;
  --iv-orange: #ff7a1a;
  --iv-text: #f8f7ff;
  --iv-muted: rgba(248,247,255,.78);
  --iv-dim: rgba(248,247,255,.62);
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
  background:
    radial-gradient(circle at 78% -12%, rgba(157,77,255,.21), transparent 32%),
    radial-gradient(circle at 8% 12%, rgba(30,79,255,.11), transparent 35%),
    radial-gradient(circle at 70% 92%, rgba(37,214,181,.08), transparent 28%),
    linear-gradient(180deg, #050710 0%, #070b15 50%, #050710 100%);
  color: var(--iv-text);
}

.block-container {
  padding-top: 1.05rem;
  padding-bottom: 3.5rem;
  max-width: 1480px;
}

#MainMenu, header, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

[data-testid="stSidebar"] {
  background:
    radial-gradient(circle at 16% 0%, rgba(157,77,255,.14), transparent 25%),
    linear-gradient(180deg, rgba(4,8,18,.98), rgba(6,10,20,.99));
  border-right: 1px solid rgba(151,92,255,.18);
  min-width: 295px !important;
}
[data-testid="stSidebar"] > div { padding-top: 1.35rem; }
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label { color: var(--iv-text) !important; }
[data-testid="stSidebar"] .stCaption { color: var(--iv-muted) !important; }
[data-testid="stSidebar"] section { color: var(--iv-text); }
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"],
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
  background: linear-gradient(180deg, rgba(18,24,42,.96), rgba(11,15,28,.96)) !important;
  border: 1px solid rgba(157,77,255,.28) !important;
  border-radius: 14px !important;
  color: var(--iv-text) !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
  background: linear-gradient(135deg, #9d4dff, #6e32df) !important;
  border-radius: 9px !important;
  color: #fff !important;
  font-weight: 700 !important;
}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] span { color: var(--iv-text) !important; }

.sidebar-brand {
  display: flex; align-items: center; gap: 12px;
  margin: 0 0 28px 0;
}
.brand-mark {
  width: 34px; height: 34px; border-radius: 12px;
  display: grid; place-items: center;
  color: #fff; font-weight: 900;
  background: radial-gradient(circle at 30% 25%, #bd7cff, #7a35f0 58%, #27114d);
  box-shadow: 0 0 30px rgba(157,77,255,.35);
}
.brand-name { font-size: 1.65rem; font-weight: 800; letter-spacing: -.055em; }
.sidebar-nav-title {
  color: var(--iv-dim); font-size: .72rem; letter-spacing: .18em; text-transform: uppercase;
  margin: 22px 0 8px;
}
.nav-pill {
  border: 1px solid rgba(255,255,255,.07);
  background: linear-gradient(135deg, rgba(157,77,255,.30), rgba(70,40,126,.18));
  border-radius: 13px;
  padding: 10px 12px;
  font-size: .92rem;
  color: var(--iv-text);
  margin-bottom: 7px;
}
.control-panel {
  border: 1px solid rgba(255,255,255,.09);
  background: linear-gradient(180deg, rgba(17,23,40,.92), rgba(9,13,24,.92));
  border-radius: 18px;
  padding: 18px 16px 18px;
  box-shadow: 0 18px 60px rgba(0,0,0,.35);
  margin-top: 18px;
}
.control-title {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(157,77,255,.24);
  background:
    radial-gradient(circle at 18% 15%, rgba(157,77,255,.34), transparent 32%),
    linear-gradient(135deg, rgba(23,31,55,.96), rgba(11,16,31,.92));
  border-radius: 18px;
  padding: 14px 15px 13px;
  font-size: .76rem;
  color: rgba(248,247,255,.84);
  letter-spacing: .16em;
  text-transform: uppercase;
  font-weight: 900;
  margin: 0 0 16px 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 14px 40px rgba(0,0,0,.22);
}
.control-title:before {
  content:"";
  display:inline-block;
  width:8px; height:8px;
  border-radius:99px;
  background:#ff4f62;
  box-shadow:0 0 16px rgba(255,79,98,.75);
  margin-right:9px;
  vertical-align:1px;
}
.control-title:after {
  content:""; position:absolute; right:-22px; top:-24px; width:90px; height:70px;
  background:radial-gradient(circle, rgba(157,77,255,.30), transparent 66%);
  filter: blur(2px);
}
.status-box {
  border: 1px solid rgba(77, 227, 107, .18);
  background: radial-gradient(circle at 0% 0%, rgba(77,227,107,.14), transparent 34%), rgba(255,255,255,.035);
  border-radius: 16px; padding: 14px; margin-top: 14px;
}
.status-dot { display:inline-block; width:9px; height:9px; border-radius:20px; background: var(--iv-green); margin-right:8px; box-shadow:0 0 14px rgba(83,227,107,.65); }
.layer-badge {
  border: 1px solid rgba(157,77,255,.22);
  background: rgba(157,77,255,.08);
  border-radius: 18px;
  padding: 15px;
  margin-top: 24px;
}

.topbar {
  display:flex; justify-content:space-between; align-items:center;
  margin: 4px 0 14px;
}
.top-eyebrow { color: var(--iv-muted); text-transform:uppercase; letter-spacing:.22em; font-size:.76rem; }
.top-dot { display:inline-block; width:10px; height:10px; border-radius:999px; background: var(--iv-purple); box-shadow:0 0 22px rgba(157,77,255,.72); margin-right:10px; }
.top-actions { display:flex; gap:10px; align-items:center; color:var(--iv-muted); }
.action-chip { border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:9px 11px; background:rgba(255,255,255,.035); }

.hero-shell {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(151, 92, 255, .22);
  border-radius: 22px;
  padding: 30px 34px 28px;
  background:
    radial-gradient(circle at 86% 42%, rgba(157,77,255,.28), transparent 24%),
    radial-gradient(circle at 72% 12%, rgba(43,105,255,.14), transparent 30%),
    linear-gradient(135deg, rgba(16, 23, 42, .92), rgba(8, 12, 24, .78));
  box-shadow: 0 24px 85px rgba(0,0,0,.42);
  min-height: 178px;
}
.hero-shell:before {
  content:""; position:absolute; right:38px; top:22px; width:310px; height:180px;
  background:
    radial-gradient(circle at 72% 54%, rgba(255,83,204,.45), transparent 18%),
    radial-gradient(circle at 48% 44%, rgba(157,77,255,.55), transparent 14%),
    radial-gradient(circle at 63% 39%, rgba(79,140,255,.36), transparent 22%);
  filter: blur(7px); opacity:.62;
  border-radius:999px;
}
.hero-shell:after {
  content:"ADC"; position:absolute; right:56px; top:50px;
  font-size:92px; line-height:1; letter-spacing:-.12em; font-weight:900;
  color:rgba(255,255,255,.045);
  transform: rotate(-4deg);
}
.hero-content { position:relative; z-index:2; max-width: 860px; }
.hero-title {
  font-size: clamp(2.45rem, 4.6vw, 4.7rem);
  line-height: .95;
  letter-spacing: -.075em;
  margin: 0;
  font-weight: 900;
}
.hero-subtitle {
  color: var(--iv-muted);
  font-size: 1.03rem;
  line-height: 1.55;
  max-width: 870px;
  margin-top: 18px;
  margin-bottom: 0;
}
.header-anchor, a.header-anchor { display:none !important; }
.top-metric-grid {
  display:grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 18px;
  margin: 18px 0 22px;
}
.top-metric-grid .metric-card {
  min-height: 136px;
  height: 136px;
  border-radius: 20px;
}
@media (max-width: 1100px) { .top-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.pill-row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 26px; }
.pill {
  display:inline-flex; align-items:center; gap:10px;
  border: 1px solid rgba(151,92,255,.16);
  background: rgba(255,255,255,.045);
  color: rgba(248,247,255,.86);
  border-radius: 12px;
  padding: 11px 14px;
  font-size: .87rem;
  backdrop-filter: blur(18px);
}
.pill-icon { width:24px; height:24px; border-radius:9px; display:grid; place-items:center; background:rgba(157,77,255,.16); color:#be8cff; }

.grid-gap { margin-top: 16px; }
.section-card {
  border: 1px solid rgba(255,255,255,.085);
  border-radius: 22px;
  padding: 24px 26px;
  background: linear-gradient(180deg, rgba(15,22,39,.80), rgba(8,12,24,.70));
  box-shadow: 0 18px 65px rgba(0,0,0,.27);
  margin-top: 20px;
}
.registry-card {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(157,77,255,.28);
  border-radius: 26px;
  padding: 30px 32px 28px;
  background:
    radial-gradient(circle at 5% 0%, rgba(157,77,255,.28), transparent 30%),
    radial-gradient(circle at 86% 12%, rgba(37,214,181,.15), transparent 28%),
    linear-gradient(135deg, rgba(25,22,52,.90), rgba(6,18,28,.78));
  box-shadow: 0 28px 90px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.045);
  margin: 24px 0 18px;
}
.registry-card:before {
  content:""; position:absolute; inset:0;
  background: linear-gradient(90deg, rgba(255,255,255,.045), transparent 34%, rgba(157,77,255,.035));
  pointer-events:none;
}
.registry-card:after {
  content:"TARGETS";
  position:absolute; right:30px; top:34px;
  font-size:64px; font-weight:950; letter-spacing:-.08em;
  color:rgba(255,255,255,.038);
}
.registry-content { position:relative; z-index:2; max-width: 940px; }
.registry-title { font-size:1.42rem; font-weight:900; letter-spacing:-.05em; margin-bottom:12px; }
.registry-copy { color:var(--iv-muted); font-size:1.0rem; line-height:1.62; max-width:820px; }
.registry-meta { display:flex; gap:12px; flex-wrap:wrap; margin:18px 0 0; }
.registry-chip { border:1px solid rgba(255,255,255,.10); background:linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.035)); color:rgba(248,247,255,.88); border-radius:999px; padding:9px 13px; font-size:.78rem; font-weight:800; box-shadow: inset 0 1px 0 rgba(255,255,255,.04); }
.section-card:empty { display:none !important; margin:0 !important; padding:0 !important; border:0 !important; }
.section-title { font-size:1.22rem; font-weight:760; letter-spacing:-.035em; margin:0 0 8px; }
.section-heading-card {
  border: 1px solid rgba(255,255,255,.085);
  border-radius: 20px;
  padding: 16px 18px 15px;
  background: linear-gradient(180deg, rgba(15,22,39,.78), rgba(8,12,24,.64));
  box-shadow: 0 12px 38px rgba(0,0,0,.20);
  margin: 18px 0 12px;
}
.section-heading-card .section-title { margin-bottom: 4px; }
.small-note { color: var(--iv-muted); font-size: .9rem; line-height: 1.55; }
.subtle { color: var(--iv-muted); }

.metric-card {
  min-height: 132px;
  height: 132px;
  display:flex; align-items:center;
  border: 1px solid rgba(255,255,255,.085);
  border-radius: 18px;
  padding: 16px 14px 16px;
  background:
    radial-gradient(circle at 16% 22%, var(--metric-glow, rgba(157,77,255,.22)), transparent 28%),
    linear-gradient(180deg, rgba(16,23,42,.82), rgba(9,13,24,.78));
  box-shadow: 0 18px 55px rgba(0,0,0,.28);
}
.metric-inner { display:flex; align-items:center; gap:12px; width:100%; min-width:0; }
.metric-icon {
  flex:0 0 48px; width:48px; height:48px; border-radius:16px;
  display:grid; place-items:center; font-size:1.48rem; line-height:1;
  background: radial-gradient(circle at 30% 20%, var(--metric-color, #9d4dff), rgba(157,77,255,.16));
  box-shadow: inset 0 0 18px rgba(255,255,255,.05), 0 0 35px var(--metric-glow, rgba(157,77,255,.18));
}
.metric-card .value { color: var(--iv-text); font-size: clamp(1.38rem, 1.65vw, 1.78rem); font-weight: 850; line-height:1; letter-spacing:-.045em; white-space:nowrap; }
.metric-card .label { color: var(--iv-text); font-size: .84rem; font-weight:650; margin-top:7px; line-height:1.14; }
.metric-card .note { color: var(--iv-muted); font-size: .74rem; margin-top: 6px; line-height:1.22; }

.metric-card.purple { --metric-color:#9d4dff; --metric-glow:rgba(157,77,255,.25); }
.metric-card.blue { --metric-color:#2d74ff; --metric-glow:rgba(45,116,255,.22); }
.metric-card.teal { --metric-color:#20d4b6; --metric-glow:rgba(32,212,182,.20); }
.metric-card.orange { --metric-color:#ff7a1a; --metric-glow:rgba(255,122,26,.20); }
.metric-card.green { --metric-color:#53e36b; --metric-glow:rgba(83,227,107,.20); }

.quick-card {
  border:1px solid rgba(255,255,255,.075); border-radius:16px; padding:18px;
  background: linear-gradient(180deg, rgba(20,28,48,.72), rgba(11,16,30,.78));
  min-height:92px; height:92px; display:flex; align-items:center;
}
.quick-row { display:flex; align-items:center; gap:15px; }
.quick-icon { flex:0 0 46px; width:46px; height:46px; border-radius:15px; display:grid; place-items:center; background:rgba(157,77,255,.16); font-size:1.28rem; line-height:1; }
.quick-title { font-weight:700; color:var(--iv-text); }
.quick-note { color:var(--iv-muted); font-size:.84rem; margin-top:4px; line-height:1.35; }

[data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,.08); border-radius: 18px; overflow: hidden; background: rgba(255,255,255,.025); }
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
  background: linear-gradient(180deg, rgba(17,24,43,.72), rgba(8,12,24,.64)) !important;
  border: 1px solid rgba(255,255,255,.08) !important;
  border-radius: 16px !important;
  color: var(--iv-text) !important;
  font-weight: 700 !important;
}
[data-testid="stExpander"] { margin-top: 10px; }

.dark-table-wrap {
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 16px;
  overflow: auto;
  background: linear-gradient(180deg, rgba(12,18,34,.88), rgba(7,11,22,.88));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.045);
  max-height: 520px;
}
.dark-table {
  width: 100%;
  border-collapse: collapse;
  font-size: .86rem;
  color: var(--iv-text);
}
.dark-table th {
  position: sticky; top: 0; z-index: 2;
  background: linear-gradient(180deg, rgba(28,36,60,.98), rgba(16,23,42,.98));
  color: rgba(248,247,255,.74);
  text-align: left;
  font-weight: 700;
  letter-spacing: .02em;
  border-bottom: 1px solid rgba(157,77,255,.22);
  padding: 12px 12px;
  white-space: nowrap;
}
.dark-table td {
  border-bottom: 1px solid rgba(255,255,255,.065);
  padding: 12px 12px;
  color: rgba(248,247,255,.90);
  vertical-align: top;
}
.dark-table tr:nth-child(even) td { background: rgba(255,255,255,.022); }
.dark-table tr:hover td { background: rgba(157,77,255,.075); }
.dark-table .num { text-align: right; font-variant-numeric: tabular-nums; color:#ffffff; }
.dark-table .muted-cell { color: rgba(248,247,255,.66); }

.buildwell-link { display:block; margin-top:18px; border-radius:16px; overflow:hidden; border:1px solid rgba(216,178,103,.36); background:#0b121b; box-shadow:0 14px 38px rgba(0,0,0,.30); }
.buildwell-link img { display:block; width:100%; height:auto; margin:0; padding:0; }
.buildwell-caption { color:#d7b66d; font-size:.69rem; letter-spacing:.16em; text-transform:uppercase; margin-top:10px; }

/* More reliable dark styling for BaseWeb inputs in the sidebar */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] div {
  color: var(--iv-text) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] {
  background: linear-gradient(180deg, rgba(18,24,42,.96), rgba(11,15,28,.96)) !important;
  box-shadow: none !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  border-radius: 999px; padding: 8px 16px; background: rgba(255,255,255,.035);
  border: 1px solid rgba(255,255,255,.075); color: var(--iv-muted);
}
.stTabs [aria-selected="true"] { background: rgba(157,77,255,.22) !important; color: var(--iv-text) !important; border-color: rgba(157,77,255,.35) !important; }

.stButton > button, .stDownloadButton > button {
  border-radius: 14px !important;
  border: 1px solid rgba(157,77,255,.38) !important;
  background: linear-gradient(135deg, #9d4dff, #6e32df) !important;
  color: #fff !important;
  font-weight: 760 !important;
  min-height: 48px;
  box-shadow: 0 14px 34px rgba(157,77,255,.22);
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); border-color: rgba(198,150,255,.65) !important; }

[data-baseweb="radio"] div, [data-baseweb="checkbox"] div { color: var(--iv-text) !important; }
.stSlider [data-baseweb="slider"] div { color: #9d4dff !important; }
hr { border-color: rgba(255,255,255,.08); }
.insight-strip { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:14px; margin-top:16px; }
.insight-card { border:1px solid rgba(255,255,255,.08); border-radius:18px; padding:16px 17px; min-height:150px; height:150px; background:linear-gradient(180deg, rgba(17,24,43,.78), rgba(8,12,24,.72)); overflow:visible; display:flex; flex-direction:column; justify-content:center; }
.insight-label { color:var(--iv-muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; line-height:1.45; min-height:32px; }
.insight-value { color:var(--iv-text); font-size:clamp(1.45rem, 1.8vw, 1.9rem); font-weight:850; letter-spacing:-.045em; margin-top:12px; white-space:nowrap; line-height:1; }
.insight-note { color:var(--iv-muted); font-size:.78rem; margin-top:9px; line-height:1.38; }
.future-calendar-wrap { max-height: 300px; }
.future-calendar-wrap .dark-table { table-layout:auto; height:auto !important; }
.future-calendar-wrap .dark-table tr { height:auto !important; }
.future-calendar-wrap .dark-table th { padding: 9px 12px; }
.future-calendar-wrap .dark-table td { padding: 9px 12px; vertical-align: middle; line-height:1.25; white-space: nowrap; height:42px !important; }
.delta-up { color:#53e36b; } .delta-down { color:#ff6b6b; } .delta-flat { color:#f8f7ff; }
@media (max-width: 1100px) { .insight-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

</style>
""",
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, note: str = "", icon: str = "◈", tone: str = "purple") -> None:
    st.markdown(
        f"""
<div class="metric-card {tone}">
  <div class="metric-inner">
    <div class="metric-icon">{icon}</div>
    <div>
      <div class="value">{value}</div>
      <div class="label">{label}</div>
      <div class="note">{note}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )




def metric_card_html(label: str, value: str, note: str = "", icon: str = "◈", tone: str = "purple") -> str:
    return f"""
<div class="metric-card {tone}">
  <div class="metric-inner">
    <div class="metric-icon">{icon}</div>
    <div>
      <div class="value">{value}</div>
      <div class="label">{label}</div>
      <div class="note">{note}</div>
    </div>
  </div>
</div>
"""

def quick_card(icon: str, title: str, note: str) -> None:
    st.markdown(
        f"""
<div class="quick-card">
  <div class="quick-row">
    <div class="quick-icon">{icon}</div>
    <div>
      <div class="quick-title">{title}</div>
      <div class="quick-note">{note}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def insight_card(label: str, value: str, note: str = "", delta_class: str = "") -> None:
    cls = f" insight-card {delta_class}" if delta_class else " insight-card"
    st.markdown(
        f"""
<div class="{cls.strip()}">
  <div class="insight-label">{label}</div>
  <div class="insight-value">{value}</div>
  <div class="insight-note">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def section_heading(title: str, note: str = "") -> None:
    note_html = f"<div class='small-note'>{html.escape(note)}</div>" if note else ""
    st.markdown(
        f"""
<div class="section-heading-card">
  <div class="section-title">{html.escape(title)}</div>
  {note_html}
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_targets() -> pd.DataFrame:
    return load_targets(DATA_PATH)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_target_trials(target_name: str, query_terms: tuple[str, ...], max_records_per_query: int) -> pd.DataFrame:
    client = ClinicalTrialsClient()
    frames = []
    for term in query_terms:
        query = f'("{term}") AND (ADC OR "antibody-drug conjugate" OR "antibody drug conjugate") AND cancer'
        try:
            studies = client.search(query=query, max_records=max_records_per_query)
            if studies:
                frames.append(normalize_studies(studies, target=target_name, query=query))
        except Exception as exc:
            st.warning(f"ClinicalTrials.gov query failed for {target_name} / {term}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["nct_id", "target"])


def build_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def dark_table(df: pd.DataFrame, max_rows: int = 25, extra_class: str = "") -> None:
    if df is None or df.empty:
        st.info("No records to display yet.")
        return
    view = df.head(max_rows).copy()
    numeric_cols = set(view.select_dtypes(include=["number"]).columns)
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    rows = []
    for _, row in view.iterrows():
        cells = []
        for col in view.columns:
            val = row[col]
            if pd.isna(val):
                text = ""
            elif col in numeric_cols:
                try:
                    text = f"{float(val):,.0f}" if float(val).is_integer() else f"{float(val):,.2f}"
                except Exception:
                    text = str(val)
            else:
                text = str(val)
            cls = "num" if col in numeric_cols else ("muted-cell" if text in {"", "UNKNOWN", "NA", "N/A"} else "")
            cells.append(f"<td class='{cls}'>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    more = ""
    if len(df) > max_rows:
        more = f"<div class='small-note' style='margin-top:10px;'>Showing {max_rows:,} of {len(df):,} rows. Use the export or Trial Evidence table for full detail.</div>"
    wrapper_class = f"dark-table-wrap {extra_class}".strip()
    st.markdown(f"<div class='{wrapper_class}'><table class='dark-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>{more}", unsafe_allow_html=True)


def format_countries(series: pd.Series) -> int:
    countries = set()
    for item in series.dropna():
        for country in str(item).split("; "):
            if country.strip():
                countries.add(country.strip())
    return len(countries)


def style_metrics_table(metrics_df: pd.DataFrame):
    display_cols = [
        "target", "heat_score", "total_trials", "active_trials", "recruiting_trials",
        "late_phase_trials", "active_enrollment", "cumulative_enrollment", "sponsor_count", "country_count", "latest_update"
    ]
    return metrics_df[[c for c in display_cols if c in metrics_df.columns]]


# Premium chart palette. Explicit mappings avoid near-duplicate colors in large basket scans
# and make Plotly full-screen views usable for screenshots/decks.
MODALITY_COLOR_MAP = {
    "Confirmed ADC": "#A855F7",
    "Likely ADC": "#00D4FF",
    "Multispecific / biparatopic ADC": "#FF4FD8",
    "Therapeutic radioconjugate": "#FFB000",
    "Radiolabeled imaging / diagnostic": "#7DD3FC",
    "Bispecific / TCE": "#22C55E",
    "CAR-T / cellular therapy": "#F43F5E",
    "Antibody / mAb": "#FDE68A",
    "Drug / small molecule": "#FB923C",
    "Vaccine": "#14B8A6",
    "Possible radioconjugate": "#C084FC",
    "Combination-heavy": "#60A5FA",
    "Unclassified": "#94A3B8",
}

TARGET_COLOR_SEQUENCE = [
    "#A855F7", "#00D4FF", "#22C55E", "#FFB000", "#F43F5E", "#14B8A6",
    "#FB923C", "#E879F9", "#38BDF8", "#84CC16", "#F97316", "#EC4899",
    "#6366F1", "#2DD4BF", "#FACC15", "#F87171", "#C084FC", "#7DD3FC",
    "#34D399", "#FDA4AF", "#93C5FD", "#D8B4FE", "#A3E635", "#FDBA74",
]

CROWDING_COLOR_MAP = {
    "Emerging whitespace": "#22C55E",
    "Healthy competition": "#00D4FF",
    "Crowded": "#FFB000",
    "Highly saturated": "#F43F5E",
}

MODALITY_ORDER = list(MODALITY_COLOR_MAP.keys())

CHART_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "invenra_adc_intelligence_chart",
        "height": 900,
        "width": 1600,
        "scale": 2,
    },
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

def clean_plotly(fig: go.Figure, height: int) -> go.Figure:
    """Apply a consistent executive-grade Plotly theme.

    Important: use solid dark backgrounds instead of transparent paper so charts still
    look polished when opened with Plotly's full-screen/expand control or copied into
    a slide deck.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=14, r=18, t=28, b=18),
        paper_bgcolor="#080D18",
        plot_bgcolor="#080D18",
        font=dict(color="#F8F7FF", family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=13),
        legend=dict(
            bgcolor="rgba(8,13,24,.86)",
            bordercolor="rgba(255,255,255,.10)",
            borderwidth=1,
            font=dict(color="#F8F7FF", size=13),
            title_font=dict(color="#D8C8FF", size=12),
        ),
        hoverlabel=dict(
            bgcolor="#0B1020",
            bordercolor="rgba(168,85,247,.50)",
            font=dict(color="#FFFFFF", size=13),
        ),
        modebar=dict(bgcolor="rgba(8,13,24,.88)", color="#CBD5E1", activecolor="#A855F7"),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,.09)",
        zerolinecolor="rgba(255,255,255,.14)",
        linecolor="rgba(255,255,255,.16)",
        tickfont=dict(color="#E5E7EB", size=12),
        title_font=dict(color="#F8F7FF", size=13),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,.07)",
        zerolinecolor="rgba(255,255,255,.14)",
        linecolor="rgba(255,255,255,.16)",
        tickfont=dict(color="#E5E7EB", size=12),
        title_font=dict(color="#F8F7FF", size=13),
    )
    # Not every Plotly trace type supports marker.line.width (for example heatmap/image
    # traces created by px.imshow). Applying it globally can crash Streamlit Cloud.
    # Keep bar/scatter-style traces clean while leaving heatmaps and other trace types
    # untouched.
    for trace in fig.data:
        if hasattr(trace, "marker") and trace.marker is not None:
            try:
                trace.marker.line.width = 0
            except Exception:
                pass
    return fig


@st.cache_data(show_spinner=False)
def build_clinical_intelligence_bundle(df: pd.DataFrame, active_scope: bool, current_year: int):
    """Cached clinical-intelligence computation.

    The Clinical Intelligence tab does several grouped analyses. Caching this bundle
    removes the perceived 8-10 second re-computation delay when switching back to
    the tab or changing non-data UI controls.
    """
    intel_df = add_clinical_intelligence_flags(df)
    active_intel = intel_df[intel_df["is_active"]] if not intel_df.empty else intel_df
    scope_df = active_intel if active_scope else intel_df
    return {
        "intel_df": intel_df,
        "scope_df": scope_df,
        "combo_df": combo_intelligence(scope_df),
        "modality_target_df": modality_by_target(scope_df, active_only=False),
        "sponsor_conv": sponsor_conviction(scope_df),
        "risk_df": risk_signal_table(df),
        "asset_df": asset_level_tracking(scope_df, limit=120),
        "line_df": line_of_therapy_mix(scope_df),
        "profile_df": target_exploitation_profile(scope_df),
        "crowd_df": crowding_scores(scope_df),
        "evolution_df": target_evolution_timeline(scope_df, current_year=current_year),
        "combo_agent_df": combo_agent_frequency(scope_df),
    }


# Sidebar navigation and controls
CURRENT_YEAR = datetime.now().year
targets = get_targets()

st.sidebar.markdown(
    """
<div class="sidebar-brand">
  <div class="brand-mark">✣</div>
  <div class="brand-name">invenra</div>
</div>
<div class="nav-pill">⌂ &nbsp; Overview</div>
<div class="sidebar-nav-title">Discovery</div>
<div class="small-note">⌗ &nbsp; Target Registry</div>
<div class="small-note" style="margin-top:10px;">⌕ &nbsp; Scan & Search</div>
<div class="sidebar-nav-title">Analytics</div>
<div class="small-note">⌬ &nbsp; Momentum</div>
<div class="small-note" style="margin-top:10px;">♚ &nbsp; Sponsors</div>
<div class="small-note" style="margin-top:10px;">◌ &nbsp; Phases</div>
<div class="small-note" style="margin-top:10px;">◎ &nbsp; Geography</div>
<div class="small-note" style="margin-top:10px;">◇ &nbsp; Clinical Intel</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='control-panel'><div class='control-title'>Scan mode</div>", unsafe_allow_html=True)
mode = st.sidebar.radio("Scan mode", ["Focused target", "Target basket"], horizontal=False, label_visibility="collapsed")
include_assets = st.sidebar.toggle("Include known asset aliases", value=True)
max_records = st.sidebar.slider("Max records per query", min_value=25, max_value=500, value=150, step=25)
selected_tier = st.sidebar.multiselect(
    "Target tier",
    sorted(targets["tier"].unique()),
    default=sorted(targets["tier"].unique()),
)
filtered_targets = targets[targets["tier"].isin(selected_tier)].copy()

if mode == "Focused target":
    selected_target = st.sidebar.selectbox("ADC target", filtered_targets["target"].tolist(), index=0)
    selected_rows = filtered_targets[filtered_targets["target"].eq(selected_target)]
else:
    default_targets = ["HER2", "TROP2", "B7-H3", "B7-H4", "HER3", "CDH6", "CLDN18.2", "c-Met"]
    selected_names = st.sidebar.multiselect(
        "ADC target basket",
        filtered_targets["target"].tolist(),
        default=[x for x in default_targets if x in filtered_targets["target"].tolist()],
    )
    selected_rows = filtered_targets[filtered_targets["target"].isin(selected_names)]
run_scan = st.sidebar.button("⌕  Run Layer 1 Scan", type="primary", width="stretch")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

emblem_uri = image_data_uri(EMBLEM_PATH)
st.sidebar.markdown(
    f"""
<div class="status-box"><span class="status-dot"></span><b>System Status</b><br><span class="small-note">System: Operational<br>Last check: Just now</span></div>
<div class="layer-badge"><b style="color:#be8cff; letter-spacing:.08em; font-size:.8rem;">LAYER 1</b>
<a class="buildwell-link" href="https://builtbybuildwell.com" target="_blank" rel="noopener"><img src="{emblem_uri}" alt="Built by BuildWell emblem" /></a>
<div class="buildwell-caption">Built by BuildWell</div></div>
""",
    unsafe_allow_html=True,
)

# Top bar and hero
st.markdown(
    """
<div class="topbar">
  <div class="top-eyebrow"><span class="top-dot"></span>Built By BuildWell · Invenra Layer 1</div>
  <div class="top-actions"><span class="action-chip">⇩</span><span class="action-chip">☼</span><span class="action-chip">RW ▾</span></div>
</div>
<div class="hero-shell">
  <div class="hero-content">
    <h1 class="hero-title">ADC Capital Map</h1>
    <p class="hero-subtitle">A clinical-trial intelligence layer for seeing where ADC activity is concentrating across target biology, sponsors, phase progression, enrollment scale, trial status, and geography.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

alias_count = int(filtered_targets["alias_list"].apply(len).sum()) if "alias_list" in filtered_targets.columns else 0
top_metrics_html = "".join([
    metric_card_html("Targets", f"{len(filtered_targets):,}", "In registry", "◎", "purple"),
    metric_card_html("Tiers", f"{filtered_targets['tier'].nunique():,}", "Validation groups", "⚗", "blue"),
    metric_card_html("Aliases", f"{alias_count:,}", "Target + asset expansion", "⌬", "teal"),
    metric_card_html("Confidence", f"{filtered_targets['registry_confidence'].nunique():,}" if "registry_confidence" in filtered_targets.columns else "0", "Audit levels", "▥", "orange"),
])
st.markdown(f"<div class='top-metric-grid'>{top_metrics_html}</div>", unsafe_allow_html=True)

# Pre-scan registry
st.markdown("""
<div class='registry-card'>
  <div class='registry-content'>
    <div class='registry-title'>Target registry</div>
    <div class='registry-copy'>Registry v2 hardens the control layer with expanded ADC-relevant targets, gene symbols, aliases, asset-name expansion, indication focus, and confidence tagging for auditability.</div>
    <div class='registry-meta'>
      <span class='registry-chip'>Expanded ADC targets</span>
      <span class='registry-chip'>Gene + alias normalization</span>
      <span class='registry-chip'>Asset expansion</span>
      <span class='registry-chip'>Confidence source</span>
      <span class='registry-chip'>Indication focus</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
with st.expander("View target registry", expanded=False):
    dark_table(filtered_targets[[c for c in ["tier", "target", "gene", "aliases", "target_category", "indication_focus", "registry_confidence", "asset_expansion_status", "notes"] if c in filtered_targets.columns]], max_rows=120)

if run_scan:
    all_frames = []
    scan_box = st.empty()
    progress = st.progress(0)
    total = max(len(selected_rows), 1)
    for i, (_, row) in enumerate(selected_rows.iterrows(), start=1):
        terms = tuple(query_terms_for_target(row, include_assets=include_assets))
        scan_box.markdown(
            f"""
<div class="section-card">
  <div class="top-eyebrow"><span class="top-dot"></span>Scanning target {i} of {total}</div>
  <h3 style="margin:.2rem 0 0;">{row['target']}</h3>
  <p class="small-note">Searching {len(terms)} target, gene, and asset aliases against ADC and cancer trial language.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        df = fetch_target_trials(row["target"], terms, max_records)
        if not df.empty:
            all_frames.append(df)
        progress.progress(i / total)
    scan_box.empty()
    progress.empty()
    st.session_state["trial_df"] = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["nct_id", "target"]) if all_frames else pd.DataFrame()
    st.session_state["scan_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

trial_df = st.session_state.get("trial_df", pd.DataFrame())

if trial_df.empty:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Recent Target Activity</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>Run a focused scan or target basket scan to populate live trial activity. This placeholder keeps the dashboard composed before the first query.</div>", unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    with q1: quick_card("⌕", "Scan & Search", "Run clinical scans for targets or baskets")
    with q2: quick_card("◎", "Target Registry", "Manage targets, aliases and gene symbols")
    with q3: quick_card("▤", "Trial Explorer", "Explore trial records with filters")
    with q4: quick_card("⇩", "Exports", "Download normalized datasets")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

metrics = target_metrics(trial_df)
scan_time = st.session_state.get("scan_timestamp", "")
trial_df = add_clinical_intelligence_flags(trial_df)
active = trial_df[trial_df["overall_status"].isin(ACTIVE_STATUSES)]

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown(f"<div class='section-title'>Layer 1 results <span class='subtle'>· last scan {scan_time}</span></div>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
with m1: metric_card("Trials", f"{trial_df['nct_id'].nunique():,}", "Unique NCT records", "▤", "purple")
with m2: metric_card("Sponsors", f"{trial_df['lead_sponsor'].nunique():,}", "Lead sponsor count", "♚", "blue")
with m3: metric_card("Active Enrollment", f"{int(active['enrollment_count'].sum()):,}", "Active/recruiting only", "◍", "teal")
with m4: metric_card("Active", f"{active['nct_id'].nunique():,}", "Recruiting, active, upcoming", "●", "green")
with m5: metric_card("Countries", f"{format_countries(trial_df['countries']):,}", "Listed trial countries", "◎", "orange")
st.markdown("</div>", unsafe_allow_html=True)

overview_tab, intelligence_tab, targets_tab, trials_tab, architecture_tab = st.tabs(["Capital Momentum", "Clinical Intelligence", "Target Heat", "Trial Evidence", "Architecture"])

with overview_tab:
    summary = yoy_summary(trial_df, current_year=CURRENT_YEAR)
    delta = int(summary["delta"])
    pct = summary["pct"]
    pct_label = "n/a" if pct is None else f"{pct:+.0%}"
    delta_class = "delta-up" if delta > 0 else "delta-down" if delta < 0 else "delta-flat"
    active_enrollment = int(active["enrollment_count"].sum())
    total_enrollment = int(trial_df["enrollment_count"].sum())
    phase2plus_active = int(active[active["phase_bucket"].isin(["PHASE2", "PHASE2_PHASE3", "PHASE3"])] ["nct_id"].nunique())
    future_summary = future_start_summary(trial_df, current_date=datetime.now())

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Capital Momentum</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>Basket scans are now treated as a portfolio. The top cards show the aggregate picture, while the tables and charts below split momentum, enrollment, and sponsors back down by ADC target.</div>", unsafe_allow_html=True)
    c_a, c_b, c_c, c_d, c_e = st.columns(5)
    with c_a:
        insight_card("New trials current YTD", f"{summary['this_year']:,}", str(summary["label"]))
    with c_b:
        insight_card("YoY new trial delta", f"{delta:+,}", f"{pct_label} change", delta_class)
    with c_c:
        insight_card("Active enrollment", f"{active_enrollment:,}", "Active / recruiting / upcoming")
    with c_d:
        insight_card("Active phase 2/3", f"{phase2plus_active:,}", "Maturity proxy")
    with c_e:
        note = "Later this year" if future_summary["later_this_year"] else "Future estimates"
        insight_card("Estimated future starts", f"{future_summary['future_starts']:,}", note)
    st.markdown("</div>", unsafe_allow_html=True)

    target_mom = target_momentum_table(trial_df, current_year=CURRENT_YEAR)
    left, right = st.columns([1.18, .92], gap="medium")
    with left:
        section_heading("New trials by start year", f"Future-dated estimated starts are excluded from YoY math. Current-year comparisons are anchored to {CURRENT_YEAR}, not the latest year returned by the source data.")
        momentum = yearly_trial_momentum(trial_df, current_year=CURRENT_YEAR, include_future=False)
        if not momentum.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=momentum["start_year"], y=momentum["new_trials"], name="New trials"))
            fig.add_trace(go.Scatter(x=momentum["start_year"], y=momentum["active_trials"], mode="lines+markers", name="Still active/upcoming", yaxis="y2"))
            fig.update_layout(
                yaxis=dict(title="New trials"),
                yaxis2=dict(title="Active subset", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(clean_plotly(fig, 410), width="stretch", config=CHART_CONFIG)
        else:
            st.info("No usable start-date data was returned for this scan.")
        st.markdown("</div>", unsafe_allow_html=True)

        section_heading("Target-level momentum table", "This basket view prevents a multi-target scan from collapsing into one unclear aggregate.")
        if not target_mom.empty:
            dark_table(target_mom, max_rows=18)
        else:
            st.info("No target-level momentum could be calculated.")
        st.markdown("</div>", unsafe_allow_html=True)

        section_heading("Forward estimated start calendar", "Only future-dated records marked estimated are counted here. Current-year future starts are labeled as Later this year with the current year shown in parentheses.")
        future_tbl = future_start_table(trial_df, current_date=datetime.now())
        if not future_tbl.empty:
            future_display = future_tbl[["target", "start_period", "estimated_future_starts", "earliest_estimated_start", "total_planned_enrollment"]].rename(columns={
                "target": "Target",
                "start_period": "Start period",
                "estimated_future_starts": "Est. starts",
                "earliest_estimated_start": "Earliest date",
                "total_planned_enrollment": "Planned enrollment",
            })
            dark_table(future_display, max_rows=30, extra_class="future-calendar-wrap")
        else:
            st.info("No future-dated estimated trial starts were returned for this scan.")
        st.markdown("</div>", unsafe_allow_html=True)

        section_heading("Top sponsors by target")
        sponsor_by_target = sponsor_activity(trial_df, active_only=True, limit=5, by_target=True)
        if not sponsor_by_target.empty:
            dark_table(sponsor_by_target, max_rows=30)
        else:
            st.info("No active sponsor data for this scan.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        section_heading("Active enrollment by phase")
        active_phase = phase_distribution(trial_df, active_only=True, include_unspecified=False)
        if not active_phase.empty:
            fig_e = px.bar(active_phase, x="phase_label", y="enrollment", hover_data=["count"], color="enrollment", color_continuous_scale=["#30205f", "#9d4dff"])
            fig_e.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Active listed enrollment")
            st.plotly_chart(clean_plotly(fig_e, 315), width="stretch", config=CHART_CONFIG)
        else:
            st.info("No phase-specified active enrollment data for this scan.")
        st.markdown("<div class='section-title'>All enrollment by phase</div>", unsafe_allow_html=True)
        all_phase = phase_distribution(trial_df, active_only=False, include_unspecified=False)
        if not all_phase.empty:
            fig_all = px.bar(all_phase, x="phase_label", y="enrollment", hover_data=["count"], color="count", color_continuous_scale=["#16335e", "#4f8cff"])
            fig_all.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Total listed enrollment")
            st.plotly_chart(clean_plotly(fig_all, 270), width="stretch", config=CHART_CONFIG)
        else:
            st.info("No phase-specified total enrollment data for this scan.")
        st.markdown("<div class='small-note'>Unspecified or not-applicable phase records are excluded from these phase charts so the signal does not get distorted by N/A buckets. They remain visible in the trial evidence table.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        section_heading("Future starts by year")
        future_tbl_chart = future_start_table(trial_df, current_date=datetime.now())
        if not future_tbl_chart.empty:
            chart_df = future_tbl_chart.copy()
            chart_df["start_period"] = chart_df["start_period"].astype(str)
            period_order = chart_df.sort_values("start_year")["start_period"].drop_duplicates().tolist()
            chart_df = chart_df.groupby(["start_period", "start_year", "target"], as_index=False).agg({
                "estimated_future_starts": "sum",
                "total_planned_enrollment": "sum",
                "earliest_estimated_start": "min",
            })
            chart_df["Start period"] = pd.Categorical(chart_df["start_period"], categories=period_order, ordered=True)
            fig_f = px.bar(
                chart_df.sort_values(["Start period", "target"]),
                x="Start period",
                y="estimated_future_starts",
                color="target",
                hover_data={
                    "target": True,
                    "estimated_future_starts": True,
                    "earliest_estimated_start": True,
                    "total_planned_enrollment": ":,",
                    "start_year": False,
                    "Start period": False,
                },
                barmode="group",
                category_orders={"Start period": period_order},
                color_discrete_sequence=TARGET_COLOR_SEQUENCE,
            )
            fig_f.update_layout(
                xaxis_title="Estimated start period",
                yaxis_title="Estimated future starts",
                xaxis=dict(type="category", categoryorder="array", categoryarray=period_order),
                bargap=.22,
                bargroupgap=.08,
            )
            st.plotly_chart(clean_plotly(fig_f, 305), width="stretch", config=CHART_CONFIG)
        else:
            st.info("No future estimated starts to chart.")
        st.markdown("</div>", unsafe_allow_html=True)

        section_heading("Status mix")
        status_df = status_distribution(trial_df)
        if not status_df.empty:
            fig3 = px.bar(status_df, x="count", y="overall_status", orientation="h", color="count", color_continuous_scale=["#16335e", "#4f8cff"])
            fig3.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(clean_plotly(fig3, 260), width="stretch", config=CHART_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)


with intelligence_tab:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Clinical intelligence decomposition</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>This page stays inside the clinical-trial record. It interprets how each target is being exploited: modality, combo behavior, sponsor conviction, crowding, line of therapy, and stopped-study signals. Modality labels are inference-based unless explicitly confirmed by protocol language. Radiolabeled imaging is separated from therapeutic radioconjugates to avoid overstating radiopharmaceutical activity.</div>", unsafe_allow_html=True)
    active_scope = st.toggle(
        "Focus intelligence charts on active / upcoming trials",
        value=True,
        help="Recommended for follow-the-money analysis. Stopped-study and termination signals still use the full retrieved dataset."
    )
    bundle = build_clinical_intelligence_bundle(trial_df, active_scope, CURRENT_YEAR)
    intel_df = bundle["intel_df"]
    scope_df = bundle["scope_df"]
    combo_df = bundle["combo_df"]
    modality_target_df = bundle["modality_target_df"]
    sponsor_conv = bundle["sponsor_conv"]
    risk_df = bundle["risk_df"]
    asset_df = bundle["asset_df"]
    line_df = bundle["line_df"]
    profile_df = bundle["profile_df"]
    crowd_df = bundle["crowd_df"]
    evolution_df = bundle["evolution_df"]
    combo_agent_df = bundle["combo_agent_df"]
    scope_note = "Active / upcoming trial scope" if active_scope else "All retrieved trial scope"

    adc_modalities = ["Confirmed ADC", "Likely ADC", "Multispecific / biparatopic ADC"]
    inferred_adc = int(scope_df[scope_df["modality_class"].isin(adc_modalities)]["nct_id"].nunique()) if not scope_df.empty else 0
    confirmed_adc = int(scope_df[scope_df["modality_class"].eq("Confirmed ADC")]["nct_id"].nunique()) if not scope_df.empty else 0
    combo_trials = int(scope_df[scope_df["is_combo"]]["nct_id"].nunique()) if not scope_df.empty else 0
    large_pharma_trials = int(scope_df[scope_df["sponsor_is_large_pharma"]]["nct_id"].nunique()) if not scope_df.empty else 0
    therapeutic_radio = int(scope_df[scope_df["modality_class"].eq("Therapeutic radioconjugate")]["nct_id"].nunique()) if not scope_df.empty else 0
    diagnostic_radio = int(scope_df[scope_df["modality_class"].eq("Radiolabeled imaging / diagnostic")]["nct_id"].nunique()) if not scope_df.empty else 0
    terminated_trials = int(intel_df[intel_df["is_terminated_or_withdrawn"]]["nct_id"].nunique()) if not intel_df.empty else 0

    st.markdown(f"<div class='small-note' style='margin-top:10px;'>Current analytics scope: {scope_note}</div>", unsafe_allow_html=True)
    i1, i2, i3, i4, i5, i6 = st.columns(6)
    with i1: insight_card("Confirmed ADC", f"{confirmed_adc:,}", "Explicit ADC language")
    with i2: insight_card("ADC-classified", f"{inferred_adc:,}", "Confirmed + likely ADC")
    with i3: insight_card("Therapeutic radio", f"{therapeutic_radio:,}", "Radiotherapy signal")
    with i4: insight_card("Imaging / diagnostic", f"{diagnostic_radio:,}", "PET/SPECT/tracer signal")
    with i5: insight_card("Combo trials", f"{combo_trials:,}", "Multi-agent protocols")
    with i6: insight_card("Large pharma", f"{large_pharma_trials:,}", "Sponsor-name heuristic")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Target exploitation profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>A compact readout of how each selected target is being pursued clinically. ADC share, combo share, and modality labels are inferred from trial text and intervention fields.</div>", unsafe_allow_html=True)
    if not profile_df.empty:
        display_profile = profile_df.copy()
        display_profile["adc_share"] = (display_profile["adc_share"] * 100).round(1).astype(str) + "%"
        display_profile["combo_share"] = (display_profile["combo_share"] * 100).round(1).astype(str) + "%"
        dark_table(display_profile[["target", "profile_readout", "dominant_modality", "trials", "active_trials", "adc_trials", "adc_share", "combo_share", "modality_diversity", "phase2plus_active", "sponsor_count", "active_enrollment", "stopped_trials"]], max_rows=35)
    else:
        st.info("No exploitation profile available for this scan.")
    st.markdown("</div>", unsafe_allow_html=True)

    lcol, rcol = st.columns([1.05, .95], gap="medium")
    with lcol:
        section_heading("Modality decomposition", "Shows how target activity breaks down into ADC, therapeutic radioconjugate, radiolabeled imaging/diagnostic, bispecific/TCE, CAR-T, antibody, or other strategies.")
        if not modality_target_df.empty:
            chart_df = modality_target_df.copy()
            top_targets = (
                chart_df.groupby("target", dropna=False)["trials"]
                .sum()
                .sort_values(ascending=False)
                .head(24)
                .index
                .tolist()
            )
            chart_df = chart_df[chart_df["target"].isin(top_targets)].copy()
            if modality_target_df["target"].nunique() > len(top_targets):
                st.markdown(f"<div class='small-note'>Chart optimized for performance: showing top {len(top_targets)} targets by trial count. The table below still contains the full basket.</div>", unsafe_allow_html=True)
            fig_m = px.bar(
                chart_df,
                x="trials",
                y="target",
                color="modality_class",
                orientation="h",
                hover_data=["active_trials", "enrollment"],
                color_discrete_map=MODALITY_COLOR_MAP,
                category_orders={"modality_class": MODALITY_ORDER, "target": top_targets[::-1]},
            )
            fig_m.update_layout(xaxis_title="Trials", yaxis_title="", barmode="stack", legend_title_text="Modality", legend_traceorder="normal")
            st.plotly_chart(clean_plotly(fig_m, max(360, 42 * max(1, chart_df["target"].nunique()) + 140)), width="stretch", config=CHART_CONFIG)
            dark_table(modality_target_df, max_rows=35)
        else:
            st.info("No modality data available.")
        st.markdown("</div>", unsafe_allow_html=True)

        section_heading("Crowding and whitespace pressure", "Scores clinical congestion from active studies, sponsors, Phase 2/3 density, modality diversity, and enrollment scale.")
        if not crowd_df.empty:
            fig_c = px.scatter(
                crowd_df,
                x="sponsor_count",
                y="active_trials",
                size="active_enrollment",
                color="crowding_band",
                hover_name="target",
                hover_data=["crowding_score", "phase2plus_active", "modality_diversity", "dominant_modality"],
                color_discrete_map=CROWDING_COLOR_MAP,
            )
            fig_c.update_layout(xaxis_title="Sponsor count", yaxis_title="Active trials")
            st.plotly_chart(clean_plotly(fig_c, 330), width="stretch", config=CHART_CONFIG)
            dark_table(crowd_df, max_rows=25)
        else:
            st.info("No crowding data available.")
        st.markdown("</div>", unsafe_allow_html=True)

        section_heading("Target evolution timeline", "Year-by-year view of new trials, ADC-inferred trials, combo usage, sponsor expansion, and enrollment.")
        if not evolution_df.empty:
            evo_chart = evolution_df.copy()
            top_evo_targets = (
                evo_chart.groupby("target", dropna=False)["new_trials"]
                .sum()
                .sort_values(ascending=False)
                .head(16)
                .index
                .tolist()
            )
            evo_chart = evo_chart[evo_chart["target"].isin(top_evo_targets)].copy()
            if evolution_df["target"].nunique() > len(top_evo_targets):
                st.markdown(f"<div class='small-note'>Timeline chart is limited to the top {len(top_evo_targets)} targets by new-trial volume for readability. Full timeline data remains in the table.</div>", unsafe_allow_html=True)
            fig_e = px.line(
                evo_chart,
                x="start_year",
                y="new_trials",
                color="target",
                markers=True,
                hover_data=["adc_trials", "combo_trials", "new_sponsors", "phase2plus_trials", "enrollment"],
                color_discrete_sequence=TARGET_COLOR_SEQUENCE,
                category_orders={"target": top_evo_targets},
            )
            fig_e.update_layout(xaxis_title="Start year", yaxis_title="New trials")
            st.plotly_chart(clean_plotly(fig_e, 330), width="stretch", config=CHART_CONFIG)
            dark_table(evolution_df.sort_values(["start_year", "new_trials"], ascending=[False, False]), max_rows=35)
        else:
            st.info("No timeline data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with rcol:
        section_heading("Combination strategy", "Detects common combination logic and named combo agents/classes from interventions, arms, titles, and eligibility text.")
        if not combo_agent_df.empty:
            fig_ca = px.bar(
                combo_agent_df,
                x="active_trials",
                y="agent_or_class",
                color="target",
                orientation="h",
                hover_data=["trials", "enrollment"],
                color_discrete_sequence=TARGET_COLOR_SEQUENCE,
            )
            fig_ca.update_layout(xaxis_title="Active trials", yaxis_title="", yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(clean_plotly(fig_ca, 320), width="stretch", config=CHART_CONFIG)
            dark_table(combo_agent_df, max_rows=30)
        elif not combo_df.empty:
            dark_table(combo_df, max_rows=30)
        else:
            st.info("No combination signals were inferred for this scan.")
        st.markdown("</div>", unsafe_allow_html=True)

        section_heading("Sponsor conviction map", "Scores repeat activity, active studies, Phase 2/3 density, listed enrollment, geography, and large-pharma presence.")
        if not sponsor_conv.empty:
            dark_table(sponsor_conv[["target", "lead_sponsor", "conviction_score", "active_trials", "phase2plus_active", "enrollment", "countries", "large_pharma_flag"]], max_rows=30)
        else:
            st.info("No sponsor conviction data available.")
        st.markdown("</div>", unsafe_allow_html=True)

        section_heading("Line-of-therapy signals", "Infers frontline, later-line, refractory, and maintenance/adjuvant language from trial text and eligibility fields.")
        if not line_df.empty:
            line_chart = line_df[line_df["line_of_therapy"].ne("Not specified")].copy()
            if not line_chart.empty:
                fig_l = px.bar(line_chart, x="active_trials", y="target", color="line_of_therapy", orientation="h", hover_data=["trials", "enrollment"], color_discrete_sequence=TARGET_COLOR_SEQUENCE)
                fig_l.update_layout(xaxis_title="Active trials", yaxis_title="")
                st.plotly_chart(clean_plotly(fig_l, 300), width="stretch", config=CHART_CONFIG)
            dark_table(line_df, max_rows=25)
        else:
            st.info("No line-of-therapy signals available.")
        st.markdown("</div>", unsafe_allow_html=True)

        section_heading("Failure / termination signals", "Captures terminated, withdrawn, or suspended studies and any reason-stopped text available in the record.")
        if not risk_df.empty:
            dark_table(risk_df, max_rows=25)
        else:
            st.info("No terminated, withdrawn, or suspended studies were found for this scan.")
        st.markdown("</div>", unsafe_allow_html=True)

        section_heading("Asset-level tracking", "Breaks target scans into visible intervention or asset names. This is the bridge to future company, payload, and deal intelligence.")
        if not asset_df.empty:
            dark_table(asset_df, max_rows=35)
        else:
            st.info("No asset-level intervention data available.")
        st.markdown("</div>", unsafe_allow_html=True)

with targets_tab:
    left, right = st.columns([1.05, .95], gap="medium")
    with left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Target × start-year heatmap</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-note'>Each cell is the count of new trials started for that target in that year. This makes momentum visible instead of showing one undifferentiated purple block.</div>", unsafe_allow_html=True)
        year_map = target_year_heatmap(trial_df, current_year=CURRENT_YEAR)
        if not year_map.empty:
            fig_y = px.imshow(year_map, text_auto=True, aspect="auto", color_continuous_scale=["#070b15", "#2d1b69", "#9d4dff"], labels=dict(x="Start year", y="Target", color="New trials"))
            fig_y.update_layout(coloraxis_colorbar=dict(title="Trials"))
            st.plotly_chart(clean_plotly(fig_y, max(320, 48 * len(year_map) + 140)), width="stretch", config=CHART_CONFIG)
        else:
            st.info("No start-year data available for heatmap.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Money-flow table</div>", unsafe_allow_html=True)
        dark_table(style_metrics_table(metrics), max_rows=30)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Target × active phase heatmap</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-note'>Each cell is active/upcoming trial count by phase. This helps separate crowded early discovery targets from clinically maturing targets.</div>", unsafe_allow_html=True)
        phase_map = target_phase_heatmap(trial_df, active_only=True)
        if not phase_map.empty:
            fig_p = px.imshow(phase_map, text_auto=True, aspect="auto", color_continuous_scale=["#070b15", "#123f50", "#25d6b5"], labels=dict(x="Phase", y="Target", color="Active trials"))
            fig_p.update_layout(coloraxis_colorbar=dict(title="Trials"))
            st.plotly_chart(clean_plotly(fig_p, max(320, 48 * len(phase_map) + 140)), width="stretch", config=CHART_CONFIG)
        else:
            st.info("No active phase data available for heatmap.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Enrollment table by phase</div>", unsafe_allow_html=True)
        phase_table = phase_distribution(trial_df, active_only=False, include_unspecified=True)
        active_phase_table = phase_distribution(trial_df, active_only=True, include_unspecified=True).rename(columns={"count":"active_trials", "enrollment":"active_enrollment"})
        if not phase_table.empty:
            merged = phase_table.merge(active_phase_table[["phase_bucket", "active_trials", "active_enrollment"]], on="phase_bucket", how="left").fillna(0)
            merged = merged.rename(columns={"phase_label":"phase", "count":"total_trials", "enrollment":"total_enrollment"})
            dark_table(merged[["phase", "active_trials", "active_enrollment", "total_trials", "total_enrollment"]], max_rows=20)
        st.markdown("</div>", unsafe_allow_html=True)

with trials_tab:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Trial-level evidence</div>", unsafe_allow_html=True)
    filter_cols = st.columns([1, 1, 1])
    with filter_cols[0]:
        target_filter = st.multiselect("Filter target", sorted(trial_df["target"].dropna().unique()), default=[])
    with filter_cols[1]:
        status_filter = st.multiselect("Filter status", sorted(trial_df["overall_status"].dropna().unique()), default=[])
    with filter_cols[2]:
        phase_filter = st.multiselect("Filter phase", sorted(trial_df["phase_label"].dropna().unique()), default=[])
    view_df = trial_df.copy()
    if target_filter: view_df = view_df[view_df["target"].isin(target_filter)]
    if status_filter: view_df = view_df[view_df["overall_status"].isin(status_filter)]
    if phase_filter: view_df = view_df[view_df["phase_label"].isin(phase_filter)]
    cols = ["target", "nct_id", "brief_title", "overall_status", "phase_label", "modality_class", "modality_confidence", "modality_evidence", "combo_class", "payload_family", "line_of_therapy", "phases", "start_date", "start_date_type", "primary_completion_date", "enrollment_count", "lead_sponsor", "collaborators", "conditions", "interventions", "countries", "site_count", "why_stopped", "primary_outcomes"]
    st.dataframe(view_df[[c for c in cols if c in view_df.columns]], width="stretch", hide_index=True)
    st.download_button("Download normalized trial CSV", data=build_download(view_df), file_name=f"invenra_adc_layer1_trials_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with architecture_tab:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Database-ready design</div>", unsafe_allow_html=True)
    st.markdown(
        """
- `src/clinicaltrials_client.py` is the replaceable API connector.
- `src/normalize.py` is the canonical flattening layer.
- `src/metrics.py` creates derived Layer 1 money-flow metrics.
- `data/adc_targets.csv` is the seed registry and can later become a `targets` table.
- Normalized trial records can later become `trials`, `trial_targets`, `sponsors`, `locations`, `interventions`, and `outcomes` tables.
- Streamlit is only the presentation layer, so the backend can later move to scheduled jobs, Postgres, Supabase, or a containerized API service.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)
