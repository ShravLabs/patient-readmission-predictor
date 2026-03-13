"""
app.py — Patient Readmission Predictor
Power BI-style Modern Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import io
import base64
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
import shap
from fpdf import FPDF

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReadmissionAI · Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── POWER BI DARK THEME CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.main, .block-container {
    background-color: #1a1d2e !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #2a2d3e !important;
    min-width: 240px !important;
}
section[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
section[data-testid="stSidebar"] .stSlider > label { color: #8892a4 !important; font-size:0.8rem !important; }

/* Top nav bar */
.topbar {
    background: linear-gradient(90deg, #0f1117 0%, #1a1d2e 100%);
    border-bottom: 1px solid #2a2d3e;
    padding: 14px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 24px -1rem;
}
.topbar-logo { display:flex; align-items:center; gap:12px; }
.topbar-title { font-size:1.1rem; font-weight:800; color:#fff; letter-spacing:-0.3px; }
.topbar-sub   { font-size:0.72rem; color:#606880; font-weight:500; }
.topbar-pills { display:flex; gap:8px; }
.pill {
    background:#22263a; border:1px solid #2e3450;
    border-radius:20px; padding:4px 14px;
    font-size:0.72rem; font-weight:600; color:#8892b2;
}
.pill.active { background:#0078D4; border-color:#0078D4; color:#fff; }

/* KPI cards */
.kpi-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin-bottom:20px; }
.kpi {
    background: #22263a;
    border: 1px solid #2e3450;
    border-radius: 10px;
    padding: 16px 14px;
    position: relative;
    overflow: hidden;
}
.kpi::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:3px;
    background: var(--accent);
}
.kpi-val  { font-size:1.7rem; font-weight:800; color:#fff; line-height:1; }
.kpi-lbl  { font-size:0.7rem; color:#606880; margin-top:5px; font-weight:600; text-transform:uppercase; letter-spacing:0.6px; }
.kpi-delta{ font-size:0.72rem; margin-top:6px; font-weight:600; }
.kpi-delta.up   { color:#00c48c; }
.kpi-delta.down { color:#ff6b6b; }
.kpi-delta.neu  { color:#8892a4; }

/* Chart panels */
.panel {
    background:#22263a;
    border:1px solid #2e3450;
    border-radius:12px;
    padding:18px 20px;
    margin-bottom:16px;
}
.panel-title {
    font-size:0.82rem; font-weight:700;
    color:#8892b2; text-transform:uppercase;
    letter-spacing:1px; margin-bottom:14px;
    display:flex; align-items:center; gap:8px;
}
.panel-title span { color:#0078D4; font-size:1rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:#0f1117 !important;
    border-bottom:1px solid #2a2d3e !important;
    gap:0 !important; padding:0 8px !important;
}
.stTabs [data-baseweb="tab"] {
    color:#606880 !important;
    font-weight:600 !important; font-size:0.82rem !important;
    padding:12px 20px !important; border-radius:0 !important;
    border-bottom:2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color:#fff !important;
    border-bottom:2px solid #0078D4 !important;
    background:transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background:#1a1d2e !important; padding:20px 0 !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #0078D4, #005a9e) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 10px 0 !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #006cbe, #004f8c) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,120,212,0.4) !important;
}

/* Result cards */
.risk-high {
    background: linear-gradient(135deg, #2d1a1a, #3d1f1f);
    border: 1px solid #ff6b6b44;
    border-left: 4px solid #ff6b6b;
    border-radius: 10px; padding:20px 24px; margin-top:16px;
}
.risk-low {
    background: linear-gradient(135deg, #1a2d1f, #1f3d25);
    border: 1px solid #00c48c44;
    border-left: 4px solid #00c48c;
    border-radius: 10px; padding:20px 24px; margin-top:16px;
}
.risk-title { font-size:1.3rem; font-weight:800; margin-bottom:4px; }
.risk-prob  { font-size:0.88rem; color:#8892a4; }

/* Comparison table */
.comp-table { width:100%; border-collapse:collapse; }
.comp-table th {
    background:#2e3450; color:#8892b2;
    font-size:0.75rem; text-transform:uppercase;
    letter-spacing:0.8px; padding:10px 14px; text-align:left;
}
.comp-table td { padding:10px 14px; border-bottom:1px solid #2a2d3e; color:#c9d1e0; font-size:0.85rem; }
.comp-table tr:hover td { background:#262b40; }
.badge-high { background:#ff6b6b22; color:#ff6b6b; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
.badge-low  { background:#00c48c22; color:#00c48c; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:700; }

/* Sidebar nav items */
.nav-item {
    display:flex; align-items:center; gap:10px;
    padding:9px 12px; border-radius:8px; margin-bottom:4px;
    cursor:pointer; font-size:0.85rem; font-weight:600;
    color:#8892a4; transition:all 0.15s;
}
.nav-item:hover, .nav-item.active { background:#1a1d2e; color:#fff; }
.nav-item .icon { font-size:1rem; width:20px; text-align:center; }

/* File uploader */
[data-testid="stFileUploader"] {
    background:#22263a !important; border:2px dashed #2e3450 !important;
    border-radius:10px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius:8px !important; overflow:hidden !important; }

/* Select boxes, sliders */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background:#22263a !important; border-color:#2e3450 !important; color:#fff !important;
}
label { color:#8892a4 !important; font-size:0.8rem !important; font-weight:600 !important; }

/* Scrollbar */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#1a1d2e; }
::-webkit-scrollbar-thumb { background:#2e3450; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 8px 8px 8px'>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:20px'>
            <div style='background:linear-gradient(135deg,#0078D4,#005a9e);
                        width:36px;height:36px;border-radius:8px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.1rem'>🏥</div>
            <div>
                <div style='font-size:0.95rem;font-weight:800;color:#fff'>ReadmissionAI</div>
                <div style='font-size:0.7rem;color:#606880'>v2.0 · Power BI Edition</div>
            </div>
        </div>
        <div style='font-size:0.7rem;font-weight:700;color:#606880;
                    text-transform:uppercase;letter-spacing:1px;
                    margin-bottom:8px;padding-left:4px'>Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;font-weight:700;color:#606880;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>⚙️ Model Config</div>", unsafe_allow_html=True)
    test_size    = st.slider("Test Split", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("Estimators", 50, 300, 150, 50)
    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;font-weight:700;color:#606880;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>📂 Data Source</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='background:#22263a;border-radius:8px;padding:12px 14px;border:1px solid #2e3450'>
        <div style='font-size:0.72rem;font-weight:700;color:#0078D4;margin-bottom:6px'>TECH STACK</div>
        <div style='display:flex;flex-wrap:wrap;gap:5px'>
            <span style='background:#0078D422;color:#4da6ff;border-radius:4px;padding:2px 8px;font-size:0.7rem;font-weight:600'>Python</span>
            <span style='background:#0078D422;color:#4da6ff;border-radius:4px;padding:2px 8px;font-size:0.7rem;font-weight:600'>Sklearn</span>
            <span style='background:#0078D422;color:#4da6ff;border-radius:4px;padding:2px 8px;font-size:0.7rem;font-weight:600'>XGBoost</span>
            <span style='background:#0078D422;color:#4da6ff;border-radius:4px;padding:2px 8px;font-size:0.7rem;font-weight:600'>SHAP</span>
            <span style='background:#0078D422;color:#4da6ff;border-radius:4px;padding:2px 8px;font-size:0.7rem;font-weight:600'>Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file): return pd.read_csv(file)

if uploaded_file:
    df = load_data(uploaded_file)
    st.toast("✅ Dataset uploaded successfully!", icon="📂")
else:
    try:
        df = pd.read_csv("data/patient_readmission.csv")
    except:
        st.error("❌ No dataset found. Upload a CSV in the sidebar.")
        st.stop()

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
@st.cache_data
def preprocess(df):
    df = df.copy()
    df['target'] = (df['readmitted'] == 'yes').astype(int)
    age_map = {'[0-10)':5,'[10-20)':15,'[20-30)':25,'[30-40)':35,
               '[40-50)':45,'[50-60)':55,'[60-70)':65,'[70-80)':75,
               '[80-90)':85,'[90-100)':95}
    df['age_num'] = df['age'].map(age_map).fillna(65)
    for col in ['change','diabetes_med']:
        df[col] = (df[col] == 'yes').astype(int)
    test_map = {'no':0,'normal':1,'Norm':1,'>7':2,'>8':3}
    df['glucose_test_enc'] = df['glucose_test'].map(test_map).fillna(0)
    df['A1Ctest_enc']      = df['A1Ctest'].map(test_map).fillna(0)
    le = LabelEncoder()
    for col in ['medical_specialty','diag_1','diag_2','diag_3']:
        df[col+'_enc'] = le.fit_transform(df[col].astype(str))
    df['visit_burden']    = df['n_inpatient'] + df['n_emergency']
    df['med_per_day']     = df['n_medications'] / (df['time_in_hospital'] + 1)
    df['is_elderly']      = (df['age_num'] >= 70).astype(int)
    df['high_lab_work']   = (df['n_lab_procedures'] > 50).astype(int)
    df['multi_diagnosis'] = ((df['diag_1'] != 'Other') & (df['diag_2'] != 'Other')).astype(int)
    return df

df_proc = preprocess(df)
FEATURES = [
    'age_num','time_in_hospital','n_lab_procedures','n_procedures',
    'n_medications','n_outpatient','n_inpatient','n_emergency',
    'glucose_test_enc','A1Ctest_enc','change','diabetes_med',
    'medical_specialty_enc','diag_1_enc','diag_2_enc','diag_3_enc',
    'visit_burden','med_per_day','is_elderly','high_lab_work','multi_diagnosis'
]
X = df_proc[FEATURES]; y = df_proc['target']

# ── TRAIN MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(_df, ts, ne):
    X_ = _df[FEATURES]; y_ = _df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X_, y_, test_size=ts, random_state=42, stratify=y_)
    X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
    m = GradientBoostingClassifier(n_estimators=ne, random_state=42)
    m.fit(X_tr, y_tr)
    yp = m.predict_proba(X_te)[:,1]
    yd = m.predict(X_te)
    return m, X_te, y_te, yd, yp, roc_auc_score(y_te, yp)

with st.spinner(""):
    model, X_test, y_test, y_pred, y_prob, auc = train_model(df_proc, test_size, n_estimators)

report = classification_report(y_test, y_pred,
            target_names=['Not Readmitted','Readmitted'], output_dict=True)

# ── CHART HELPER ──────────────────────────────────────────────────────────────
BG   = '#22263a'
GRID = '#2e3450'
FG   = '#c9d1e0'
BLUE = '#0078D4'
ORG  = '#E8750A'
GRN  = '#00c48c'
RED  = '#ff6b6b'

def dark_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w,h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    return fig, ax

# ── TOP BAR ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div class="topbar-logo">
        <div style='background:linear-gradient(135deg,#0078D4,#005a9e);width:32px;height:32px;
                    border-radius:7px;display:flex;align-items:center;
                    justify-content:center;font-size:1rem'>🏥</div>
        <div>
            <div class="topbar-title">Patient Readmission Predictor</div>
            <div class="topbar-sub">Microsoft Data Science Internship · Power BI Dashboard</div>
        </div>
    </div>
    <div class="topbar-pills">
        <span class="pill active">🟢 Model Live</span>
        <span class="pill">📅 {datetime.now().strftime('%d %b %Y')}</span>
        <span class="pill">📊 {df.shape[0]:,} Records</span>
        <span class="pill">🎯 AUC {auc:.3f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
rr   = (df['readmitted']=='yes').mean()
prec = report['Readmitted']['precision']
rec  = report['Readmitted']['recall']
f1   = report['Readmitted']['f1-score']
acc  = report['accuracy']

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi" style="--accent:#0078D4">
        <div class="kpi-val">{df.shape[0]:,}</div>
        <div class="kpi-lbl">Total Patients</div>
        <div class="kpi-delta neu">↔ Dataset size</div>
    </div>
    <div class="kpi" style="--accent:#E8750A">
        <div class="kpi-val">{rr:.1%}</div>
        <div class="kpi-lbl">Readmission Rate</div>
        <div class="kpi-delta down">↑ High risk cohort</div>
    </div>
    <div class="kpi" style="--accent:#00c48c">
        <div class="kpi-val" style="color:#00c48c">{auc:.3f}</div>
        <div class="kpi-lbl">ROC-AUC Score</div>
        <div class="kpi-delta up">↑ Model quality</div>
    </div>
    <div class="kpi" style="--accent:#a78bfa">
        <div class="kpi-val" style="color:#a78bfa">{acc:.1%}</div>
        <div class="kpi-lbl">Accuracy</div>
        <div class="kpi-delta up">↑ Overall correct</div>
    </div>
    <div class="kpi" style="--accent:#fbbf24">
        <div class="kpi-val" style="color:#fbbf24">{prec:.1%}</div>
        <div class="kpi-lbl">Precision</div>
        <div class="kpi-delta neu">↔ Positive cases</div>
    </div>
    <div class="kpi" style="--accent:#f472b6">
        <div class="kpi-val" style="color:#f472b6">{rec:.1%}</div>
        <div class="kpi-lbl">Recall</div>
        <div class="kpi-delta neu">↔ Sensitivity</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📊  Overview & EDA",
    "🤖  Model Performance",
    "🔍  Feature Importance",
    "🧠  SHAP Explainability",
    "🎯  Risk Predictor",
    "📋  Compare Patients"
])

# ═══════════════════════ TAB 1 — EDA ═════════════════════════════════════════
with tab1:
    # Auto-analysis summary
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title"><span>📋</span> Auto Dataset Analysis</div>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px'>
            <div style='background:#1a1d2e;border-radius:8px;padding:12px'>
                <div style='font-size:0.7rem;color:#606880;font-weight:600;text-transform:uppercase'>Rows</div>
                <div style='font-size:1.2rem;font-weight:800;color:#fff'>{df.shape[0]:,}</div>
            </div>
            <div style='background:#1a1d2e;border-radius:8px;padding:12px'>
                <div style='font-size:0.7rem;color:#606880;font-weight:600;text-transform:uppercase'>Columns</div>
                <div style='font-size:1.2rem;font-weight:800;color:#fff'>{df.shape[1]}</div>
            </div>
            <div style='background:#1a1d2e;border-radius:8px;padding:12px'>
                <div style='font-size:0.7rem;color:#606880;font-weight:600;text-transform:uppercase'>Missing Values</div>
                <div style='font-size:1.2rem;font-weight:800;color:#{"ff6b6b" if df.isnull().sum().sum()>0 else "00c48c"}'>{df.isnull().sum().sum()}</div>
            </div>
            <div style='background:#1a1d2e;border-radius:8px;padding:12px'>
                <div style='font-size:0.7rem;color:#606880;font-weight:600;text-transform:uppercase'>Numeric Features</div>
                <div style='font-size:1.2rem;font-weight:800;color:#fff'>{df.select_dtypes(include=np.number).shape[1]}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title"><span>🗃️</span> Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True, height=280)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel"><div class="panel-title"><span>🎯</span> Target Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 3.5)
        counts = df['readmitted'].value_counts()
        bars = ax.bar(['Not Readmitted','Readmitted'], counts.values,
                      color=[BLUE, ORG], edgecolor='none', width=0.45)
        ax.set_ylabel('Count', color=FG, fontsize=8)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+60,
                    f'{v:,}', ha='center', color=FG, fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><div class="panel-title"><span>📈</span> Readmission Rate by Age</div>', unsafe_allow_html=True)
        age_order = ['[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
        ar = df[df['age'].isin(age_order)].groupby('age')['readmitted'].apply(
            lambda x: (x=='yes').mean()*100).reindex(age_order)
        fig, ax = dark_fig(6, 3.5)
        ax.fill_between(range(len(ar)), ar.values, alpha=0.15, color=BLUE)
        ax.plot(range(len(ar)), ar.values, marker='o', color=BLUE, lw=2.5,
                markersize=7, markerfacecolor=ORG, markeredgecolor='white', markeredgewidth=1.5)
        ax.set_xticks(range(len(ar))); ax.set_xticklabels(age_order, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('Rate (%)', color=FG, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="panel"><div class="panel-title"><span>🏥</span> Readmission by Diagnosis</div>', unsafe_allow_html=True)
        dr = df.groupby('diag_1')['readmitted'].apply(lambda x: (x=='yes').mean()*100).sort_values()
        fig, ax = dark_fig(6, 3.5)
        colors = [BLUE if v < dr.max()*0.85 else ORG for v in dr.values]
        ax.barh(dr.index, dr.values, color=colors, edgecolor='none', height=0.55)
        ax.set_xlabel('Rate (%)', color=FG, fontsize=8)
        for i, v in enumerate(dr.values):
            ax.text(v+0.3, i, f'{v:.1f}%', va='center', fontsize=8, color=FG)
        plt.tight_layout()
        st.pyplot(fig); st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="panel"><div class="panel-title"><span>💊</span> Medications Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 3.5)
        for val, color, label in zip(['no','yes'],[BLUE,ORG],['Not Readmitted','Readmitted']):
            ax.hist(df[df['readmitted']==val]['n_medications'],
                    bins=20, alpha=0.65, color=color, label=label, edgecolor='none')
        ax.set_xlabel('No. of Medications', color=FG, fontsize=8)
        ax.set_ylabel('Count', color=FG, fontsize=8)
        ax.legend(facecolor=GRID, labelcolor=FG, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); st.markdown('</div>', unsafe_allow_html=True)

    # Correlation heatmap full width
    st.markdown('<div class="panel"><div class="panel-title"><span>🔗</span> Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = ['age_num','time_in_hospital','n_lab_procedures','n_procedures',
                'n_medications','n_inpatient','n_emergency','visit_burden','target']
    corr = df_proc[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=BG)
    ax.set_facecolor(BG)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, ax=ax, cmap=cmap, center=0, annot=True, fmt='.2f',
                linewidths=0.5, linecolor=GRID, annot_kws={'size':8,'color':FG})
    ax.tick_params(colors=FG, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════ TAB 2 — MODEL PERFORMANCE ═══════════════════════════
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel"><div class="panel-title"><span>📊</span> Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
        ax.set_facecolor(BG)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not Readmitted','Readmitted'],
                    yticklabels=['Not Readmitted','Readmitted'],
                    linewidths=0.5, linecolor=GRID,
                    annot_kws={'size':14,'weight':'bold'})
        ax.set_ylabel('Actual', color=FG, fontsize=9)
        ax.set_xlabel('Predicted', color=FG, fontsize=9)
        ax.tick_params(colors=FG)
        plt.tight_layout()
        st.pyplot(fig); st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><div class="panel-title"><span>📈</span> ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = dark_fig(6, 4)
        ax.fill_between(fpr, tpr, alpha=0.12, color=BLUE)
        ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f'AUC = {auc:.3f}')
        ax.plot([0,1],[0,1],'--', color=GRID, lw=1.5)
        ax.set_xlabel('False Positive Rate', color=FG, fontsize=8)
        ax.set_ylabel('True Positive Rate', color=FG, fontsize=8)
        ax.legend(facecolor=GRID, labelcolor=FG)
        plt.tight_layout()
        st.pyplot(fig); st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="panel"><div class="panel-title"><span>⚖️</span> Precision-Recall Curve</div>', unsafe_allow_html=True)
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        fig, ax = dark_fig(6, 4)
        ax.fill_between(rec_c, prec_c, alpha=0.12, color=ORG)
        ax.plot(rec_c, prec_c, color=ORG, lw=2.5, label=f'AP = {ap:.3f}')
        ax.axhline(y_test.mean(), color=GRID, linestyle='--', lw=1.5)
        ax.set_xlabel('Recall', color=FG, fontsize=8)
        ax.set_ylabel('Precision', color=FG, fontsize=8)
        ax.legend(facecolor=GRID, labelcolor=FG)
        plt.tight_layout()
        st.pyplot(fig); st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="panel"><div class="panel-title"><span>🔔</span> Score Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 4)
        ax.hist(y_prob[y_test==0], bins=30, alpha=0.65, color=BLUE, label='Not Readmitted', edgecolor='none')
        ax.hist(y_prob[y_test==1], bins=30, alpha=0.65, color=ORG, label='Readmitted', edgecolor='none')
        ax.axvline(0.5, color=RED, lw=1.5, linestyle='--', label='Threshold')
        ax.set_xlabel('Predicted Probability', color=FG, fontsize=8)
        ax.legend(facecolor=GRID, labelcolor=FG, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title"><span>📋</span> Classification Report</div>', unsafe_allow_html=True)
    rdf = pd.DataFrame(report).T.iloc[:2].round(3)
    st.dataframe(rdf[['precision','recall','f1-score','support']], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════ TAB 3 — FEATURE IMPORTANCE ══════════════════════════
with tab3:
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(15)
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown('<div class="panel"><div class="panel-title"><span>🏆</span> Top 15 Features</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(8, 5)
        norm = fi.values / fi.max()
        colors = [BLUE if v < 0.5 else ORG if v < 0.8 else GRN for v in norm]
        bars = ax.barh(fi.index, fi.values, color=colors, edgecolor='none', height=0.6)
        ax.set_xlabel('Importance Score', color=FG, fontsize=8)
        for bar, v in zip(bars, fi.values):
            ax.text(bar.get_width()+0.0003, bar.get_y()+bar.get_height()/2,
                    f'{v:.4f}', va='center', fontsize=7.5, color=FG)
        plt.tight_layout()
        st.pyplot(fig); st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><div class="panel-title"><span>📊</span> Importance Table</div>', unsafe_allow_html=True)
        fi_df = fi.reset_index()[::-1]; fi_df.columns = ['Feature','Score']
        fi_df['Score'] = fi_df['Score'].round(5)
        fi_df.insert(0,'Rank', range(1, len(fi_df)+1))
        st.dataframe(fi_df, use_container_width=True, height=380)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
            <div class="panel-title"><span>💡</span> Key Insights</div>
            <div style='font-size:0.82rem;color:#8892a4;line-height:1.7'>
            🔴 <b style='color:#ff6b6b'>visit_burden</b> is the top predictor<br>
            🟠 <b style='color:#E8750A'>Elderly patients</b> carry highest risk<br>
            🔵 <b style='color:#4da6ff'>Lab procedures</b> reflect case complexity<br>
            🟢 <b style='color:#00c48c'>Prior inpatient</b> visits signal fragility
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════ TAB 4 — SHAP ════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="panel">
        <div class="panel-title"><span>🧠</span> SHAP — Model Explainability</div>
        <div style='font-size:0.83rem;color:#8892a4'>
        SHAP reveals <b style='color:#fff'>why</b> the model makes each prediction.
        In healthcare AI, interpretability is critical — clinicians need to trust the reasoning behind every ML recommendation.
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.spinner("Computing SHAP values..."):
        try:
            exp = shap.TreeExplainer(model)
            sv  = exp.shap_values(X_test.iloc[:300])
            if isinstance(sv, list): sv = sv[1]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="panel"><div class="panel-title"><span>📊</span> Global Feature Importance</div>', unsafe_allow_html=True)
                fig, _ = plt.subplots(figsize=(7,5), facecolor=BG)
                shap.summary_plot(sv, X_test.iloc[:300], plot_type='bar', feature_names=FEATURES, show=False, color=BLUE)
                plt.gcf().patch.set_facecolor(BG)
                plt.tight_layout()
                st.pyplot(plt.gcf()); plt.close(); st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="panel"><div class="panel-title"><span>🐝</span> Beeswarm Plot</div>', unsafe_allow_html=True)
                fig, _ = plt.subplots(figsize=(7,5), facecolor=BG)
                shap.summary_plot(sv, X_test.iloc[:300], feature_names=FEATURES, show=False)
                plt.gcf().patch.set_facecolor(BG)
                plt.tight_layout()
                st.pyplot(plt.gcf()); plt.close(); st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"SHAP error: {e}")

# ═══════════════════════ TAB 5 — RISK PREDICTOR ═══════════════════════════════
with tab5:
    st.markdown('<div class="panel"><div class="panel-title"><span>🎯</span> Predict Readmission Risk for a New Patient</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**👤 Patient Demographics**")
        age_i  = st.selectbox("Age Group", ['[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)'], index=2)
        diag1  = st.selectbox("Primary Diagnosis", ['Circulatory','Diabetes','Respiratory','Digestive','Injury','Other'])
        med_sp = st.selectbox("Medical Specialty", ['Missing','InternalMedicine','Emergency/Trauma','Other','Cardiology'])
    with c2:
        st.markdown("**🏥 Hospital Stay**")
        t_hosp = st.slider("Days in Hospital", 1, 14, 4)
        n_lab  = st.slider("Lab Procedures", 1, 100, 45)
        n_proc = st.slider("Procedures", 0, 10, 1)
        n_meds = st.slider("Medications", 1, 25, 12)
    with c3:
        st.markdown("**📋 Visit History & Labs**")
        n_inp  = st.slider("Prior Inpatient Visits", 0, 10, 0)
        n_emer = st.slider("Prior Emergency Visits", 0, 5, 0)
        n_out  = st.slider("Outpatient Visits", 0, 20, 1)
        gluc   = st.selectbox("Glucose Test", ['no','normal','>7','>8'])
        a1c    = st.selectbox("A1C Test", ['no','normal','>7','>8'])
        ch_med = st.selectbox("Medication Change", ['no','yes'])
        d_med  = st.selectbox("Diabetes Medication", ['yes','no'])

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡ Predict Readmission Risk", use_container_width=True):
        am   = {'[40-50)':45,'[50-60)':55,'[60-70)':65,'[70-80)':75,'[80-90)':85,'[90-100)':95}
        tm   = {'no':0,'normal':1,'Norm':1,'>7':2,'>8':3}
        inp  = pd.DataFrame([[
            am[age_i], t_hosp, n_lab, n_proc, n_meds, n_out, n_inp, n_emer,
            tm.get(gluc,0), tm.get(a1c,0), int(ch_med=='yes'), int(d_med=='yes'),
            df_proc['medical_specialty_enc'].median(), df_proc['diag_1_enc'].median(),
            df_proc['diag_2_enc'].median(), df_proc['diag_3_enc'].median(),
            n_inp+n_emer, n_meds/(t_hosp+1), int(am[age_i]>=70),
            int(n_lab>50), int(diag1!='Other')
        ]], columns=FEATURES)
        prob = model.predict_proba(inp)[0][1]
        risk = "HIGH" if prob >= 0.5 else "LOW"

        col_r, col_g = st.columns(2)
        with col_r:
            if risk == "HIGH":
                st.markdown(f"""
                <div class="risk-high">
                    <div class="risk-title" style="color:#ff6b6b">⚠️ HIGH READMISSION RISK</div>
                    <div class="risk-prob">Predicted probability: <b>{prob:.1%}</b></div>
                    <div style='margin-top:10px;font-size:0.82rem;color:#8892a4'>
                    Consider: Follow-up appointment, medication review, discharge plan assessment.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div class="risk-title" style="color:#00c48c">✅ LOW READMISSION RISK</div>
                    <div class="risk-prob">Predicted probability: <b>{prob:.1%}</b></div>
                    <div style='margin-top:10px;font-size:0.82rem;color:#8892a4'>
                    Standard discharge procedures recommended.
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_g:
            fig, ax = plt.subplots(figsize=(5, 2.5), facecolor=BG)
            ax.set_facecolor(BG)
            ax.barh(0, 1, color=GRID, height=0.35, edgecolor='none')
            ax.barh(0, prob, color=RED if prob>=0.5 else GRN, height=0.35, edgecolor='none')
            ax.axvline(0.5, color='white', lw=1.5, linestyle='--', alpha=0.5)
            ax.text(prob, 0.22, f'{prob:.1%}', ha='center', color='white', fontweight='bold', fontsize=11)
            ax.text(0, -0.28, '0%', ha='center', color=FG, fontsize=8)
            ax.text(0.5, -0.28, '50%', ha='center', color=FG, fontsize=8)
            ax.text(1.0, -0.28, '100%', ha='center', color=FG, fontsize=8)
            ax.set_xlim(0,1); ax.set_ylim(-0.5,0.5)
            ax.set_yticks([]); ax.set_xticks([])
            ax.set_title('Risk Score Gauge', color=FG, fontsize=10, fontweight='bold')
            for s in ax.spines.values(): s.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        # PDF Report
        st.markdown("---")
        st.markdown("**📥 Download Patient Report**")
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 20)
            pdf.set_text_color(0, 120, 212)
            pdf.cell(0, 12, "Patient Readmission Risk Report", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(100,100,100)
            pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", ln=True)
            pdf.ln(6)
            pdf.set_font("Helvetica","B",13)
            pdf.set_text_color(0,0,0)
            pdf.cell(0,8,"Patient Details", ln=True)
            pdf.set_font("Helvetica","",10)
            details = [
                ("Age Group", age_i), ("Diagnosis", diag1),
                ("Days in Hospital", str(t_hosp)), ("Medications", str(n_meds)),
                ("Prior Inpatient Visits", str(n_inp)), ("Prior Emergency Visits", str(n_emer)),
                ("Glucose Test", gluc), ("A1C Test", a1c),
            ]
            for k,v in details:
                pdf.cell(70,7,k+":"); pdf.cell(0,7,v,ln=True)
            pdf.ln(4)
            pdf.set_font("Helvetica","B",13)
            pdf.cell(0,8,"Prediction Result", ln=True)
            pdf.set_font("Helvetica","B",14)
            pdf.set_text_color(211,52,56) if risk=="HIGH" else pdf.set_text_color(16,124,16)
            pdf.cell(0,10,f"Risk Level: {risk}  |  Probability: {prob:.1%}", ln=True)
            pdf.set_text_color(0,0,0)
            pdf.set_font("Helvetica","",10)
            if risk=="HIGH":
                pdf.multi_cell(0,6,"Recommendation: Patient shows elevated readmission risk. Schedule follow-up within 7 days, review discharge medications, and ensure post-discharge care plan is in place.")
            else:
                pdf.multi_cell(0,6,"Recommendation: Patient shows low readmission risk. Standard discharge procedures recommended. Routine follow-up at 30 days.")
            pdf.ln(4)
            pdf.set_font("Helvetica","I",8)
            pdf.set_text_color(150,150,150)
            pdf.cell(0,5,"Generated by ReadmissionAI · Microsoft Data Science Internship Project", ln=True)
            pdf_bytes = pdf.output()
            b64 = base64.b64encode(bytes(pdf_bytes)).decode()
            st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="patient_risk_report.pdf"><button style="background:linear-gradient(135deg,#0078D4,#005a9e);color:white;border:none;padding:10px 24px;border-radius:8px;font-weight:700;font-size:0.9rem;cursor:pointer;width:100%">📄 Download PDF Report</button></a>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"PDF generation skipped: {e}")

# ═══════════════════════ TAB 6 — COMPARE PATIENTS ════════════════════════════
with tab6:
    st.markdown('<div class="panel"><div class="panel-title"><span>📋</span> Compare Multiple Patients Side by Side</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.83rem;color:#8892a4;margin-bottom:16px'>Add up to 5 patients and compare their readmission risk scores instantly.</div>", unsafe_allow_html=True)

    if 'patients' not in st.session_state:
        st.session_state.patients = []

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p_age  = st.selectbox("Age", ['[40-50)','[50-60)','[60-70)','[70-80)','[80-90)'],
                               key="p_age", index=1)
        p_diag = st.selectbox("Diagnosis", ['Circulatory','Diabetes','Respiratory','Other'], key="p_diag")
    with c2:
        p_hosp = st.slider("Days in Hospital", 1, 14, 3, key="p_hosp")
        p_meds = st.slider("Medications", 1, 25, 10, key="p_meds")
    with c3:
        p_inp  = st.slider("Inpatient Visits", 0, 10, 0, key="p_inp")
        p_emer = st.slider("Emergency Visits", 0, 5, 0, key="p_emer")
    with c4:
        p_name = st.text_input("Patient Label", value=f"Patient {len(st.session_state.patients)+1}", key="p_name")
        p_lab  = st.slider("Lab Procedures", 1, 100, 40, key="p_lab")

    col_add, col_clear = st.columns([1,1])
    with col_add:
        if st.button("➕ Add Patient", use_container_width=True):
            if len(st.session_state.patients) < 5:
                am = {'[40-50)':45,'[50-60)':55,'[60-70)':65,'[70-80)':75,'[80-90)':85}
                inp = pd.DataFrame([[
                    am.get(p_age,65), p_hosp, p_lab, 1, p_meds, 0, p_inp, p_emer,
                    0, 0, 0, 1,
                    df_proc['medical_specialty_enc'].median(), df_proc['diag_1_enc'].median(),
                    df_proc['diag_2_enc'].median(), df_proc['diag_3_enc'].median(),
                    p_inp+p_emer, p_meds/(p_hosp+1), int(am.get(p_age,65)>=70),
                    int(p_lab>50), int(p_diag!='Other')
                ]], columns=FEATURES)
                prob = model.predict_proba(inp)[0][1]
                st.session_state.patients.append({
                    'Name':p_name,'Age':p_age,'Diagnosis':p_diag,
                    'Days':p_hosp,'Meds':p_meds,'Inpatient':p_inp,
                    'Emergency':p_emer,'Risk %':f'{prob:.1%}',
                    'Risk':('🔴 HIGH' if prob>=0.5 else '🟢 LOW'),
                    '_prob': prob
                })
                st.toast(f"✅ {p_name} added!", icon="👤")
    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.patients = []

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.patients:
        st.markdown('<div class="panel"><div class="panel-title"><span>👥</span> Patient Comparison Table</div>', unsafe_allow_html=True)
        comp_df = pd.DataFrame(st.session_state.patients).drop(columns=['_prob'])
        st.dataframe(comp_df, use_container_width=True)

        # Bar chart comparison
        probs  = [p['_prob'] for p in st.session_state.patients]
        names  = [p['Name']  for p in st.session_state.patients]
        fig, ax = dark_fig(8, 3.5)
        colors  = [RED if p>=0.5 else GRN for p in probs]
        bars = ax.bar(names, [p*100 for p in probs], color=colors, edgecolor='none', width=0.4)
        ax.axhline(50, color='white', lw=1.5, linestyle='--', alpha=0.4, label='Risk Threshold (50%)')
        ax.set_ylabel('Readmission Risk %', color=FG, fontsize=8)
        ax.set_title('Risk Score Comparison', color=FG, fontsize=10, fontweight='bold')
        for bar, v in zip(bars, probs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{v:.1%}', ha='center', color=FG, fontsize=9, fontweight='bold')
        ax.legend(facecolor=GRID, labelcolor=FG, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Add patients above to start comparing their risk scores.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;color:#606880;font-size:0.78rem;
            margin-top:32px;padding:16px;border-top:1px solid #2a2d3e'>
    🏥 ReadmissionAI &nbsp;·&nbsp; Power BI Dashboard &nbsp;·&nbsp;
    Gradient Boosting + SHAP &nbsp;·&nbsp;
    Microsoft Data Science Internship &nbsp;·&nbsp;
    Built with Streamlit
</div>
""", unsafe_allow_html=True)