"""
EnergyAI - Tableau de bord de prédiction de la demande énergétique
Boston City Hall - ENSA Kénitra - 2025-2026
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EnergyAI - Boston City Hall",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data"
PATH_MAIN = os.path.join(DATA_DIR, "dataset_final_powerbi.csv")
PATH_TEST = os.path.join(DATA_DIR, "data_test.csv")
PATH_MODEL = os.path.join(DATA_DIR, "lgbm_optimise.pkl")
TARGET = "Total_Demand_KW"
DEFAULT_FEATURES = [
    "hour",
    "dayofweek",
    "month",
    "year",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_4",
    "lag_96",
]

DARK_THEME = {
    "bg": "#060B18",
    "surface": "#0F1829",
    "surface2": "#162035",
    "border": "#1E2E4A",
    "cyan": "#00E5FF",
    "violet": "#7B2FBE",
    "green": "#00C853",
    "orange": "#FF6B35",
    "red": "#FF3D71",
    "text": "#E8EDF5",
    "muted": "#9AA8C7",
    "subtle": "#2A3A5A",
}

LIGHT_THEME = {
    "bg": "#F4F7FB",
    "surface": "#FFFFFF",
    "surface2": "#EAF0F7",
    "border": "#C8D4E3",
    "cyan": "#007C91",
    "violet": "#5B35A5",
    "green": "#138A43",
    "orange": "#C05621",
    "red": "#D12F54",
    "text": "#152033",
    "muted": "#54657D",
    "subtle": "#D8E1EE",
}

C = LIGHT_THEME if st.session_state.get("light_mode", False) else DARK_THEME
IS_LIGHT = st.session_state.get("light_mode", False)

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, Inter, sans-serif", color=C["muted"], size=12),
    title_font=dict(
        family="Space Grotesk, DM Sans, sans-serif", color=C["text"], size=14
    ),
    xaxis=dict(
        gridcolor="rgba(30,46,74,0.8)",
        zerolinecolor="rgba(30,46,74,0.8)",
        tickfont=dict(size=11, color=C["muted"]),
    ),
    yaxis=dict(
        gridcolor="rgba(30,46,74,0.8)",
        zerolinecolor="rgba(30,46,74,0.8)",
        tickfont=dict(size=11, color=C["muted"]),
    ),
    legend=dict(
        bgcolor="rgba(15,24,41,0.9)",
        bordercolor=C["border"],
        borderwidth=1,
        font=dict(size=11, color=C["muted"]),
    ),
    margin=dict(l=18, r=18, t=48, b=18),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor=C["surface2"],
        bordercolor=C["border"],
        font=dict(color=C["text"], size=12),
    ),
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
*, html, body, [class*="css"] {{
    font-family: 'DM Sans', 'Inter', sans-serif;
    box-sizing: border-box;
}}
.stApp {{ background-color: {C["bg"]}; color: {C["text"]}; }}
.block-container {{ padding: 1.75rem 2rem 3rem 2rem !important; max-width: 100% !important; }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: {C["bg"]} !important; border-bottom: 1px solid {C["border"]} !important; }}
[data-testid="stSidebar"] {{
    background: {C["surface"]};
    border-right: 1px solid {C["border"]};
    color: {C["text"]} !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label * {{
    color: {'#152033' if IS_LIGHT else '#FFFFFF'} !important;
}}
[data-testid="stSidebar"] .block-container {{ padding: 1.5rem 1.1rem !important; }}
[data-testid="stSidebar"] hr {{ border-color: {C["border"]}; margin: 1rem 0; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    color: {'#152033' if IS_LIGHT else '#FFFFFF'} !important;
    font-weight: 650 !important;
    padding: 0.68rem 0.8rem !important;
    border-radius: 9px !important;
    border-left: 3px solid transparent !important;
    margin-bottom: 0.2rem !important;
    transition: background 0.16s ease, color 0.16s ease, border-left-color 0.16s ease;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {{
    color: inherit !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background: {C["surface2"]} !important;
    border-left-color: {C["cyan"]} !important;
    color: {C["text"]} !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radio"][aria-checked="true"] {{
    background: {'rgba(0,124,145,0.12)' if IS_LIGHT else 'rgba(0,229,255,0.14)'} !important;
    border-left-color: {C["cyan"]} !important;
    color: {C["cyan"]} !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radio"] {{
    border-radius: 9px !important;
}}
.kpi-card {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 14px;
    padding: 1.35rem 1.5rem 1.2rem;
    position: relative;
    overflow: hidden;
    min-height: 128px;
}}
.kpi-accent-bar {{
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 14px 14px 0 0;
}}
.kpi-label {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: {C["muted"]}; margin-bottom: 0.55rem;
}}
.kpi-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem; font-weight: 700;
    line-height: 1; margin-bottom: 0.45rem;
}}
.kpi-sub {{ font-size: 0.75rem; color: {C["muted"]}; }}
.section-header {{
    display: flex; align-items: center; gap: 0.6rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem; font-weight: 600; color: {C["text"]};
    margin: 0 0 1.1rem 0;
}}
.section-icon {{
    width: 1.05rem;
    height: 1.05rem;
    display: inline-flex;
    color: {C["cyan"]};
    flex: 0 0 auto;
}}
.section-icon svg {{
    width: 100%;
    height: 100%;
    stroke: currentColor;
    stroke-width: 2;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}}
.section-header::after {{
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, {C["border"]}, transparent);
    margin-left: 0.6rem;
}}
.stat-pill {{
    background: {C["surface2"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.5rem;
}}
.stat-pill-label {{
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: {C["muted"]}; margin-bottom: 0.2rem;
}}
.stat-pill-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem; font-weight: 600; color: {C["cyan"]};
}}
.pred-card {{
    background: linear-gradient(135deg, rgba(0,229,255,0.07) 0%, rgba(123,47,190,0.07) 100%);
    border: 1px solid rgba(0,229,255,0.28);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
}}
.pred-number {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem; font-weight: 700; line-height: 1;
    color: {C["cyan"]};
}}
.pred-badge {{
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.75rem;
}}
.model-row {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.45rem;
    display: flex; align-items: center; gap: 1rem;
    font-size: 0.84rem;
}}
.model-row.best {{ background: rgba(0,229,255,0.06); border-color: rgba(0,229,255,0.3); }}
.about-card {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
}}
.about-card h3 {{ font-family: 'Space Grotesk', sans-serif; font-size: 1rem; color: {C["cyan"]}; }}
.about-card p, .about-card li {{ color: {C["muted"]}; font-size: 0.86rem; line-height: 1.7; }}
.stTabs [data-baseweb="tab-list"] {{
    background: {C["surface"]}; border-radius: 10px; padding: 4px;
    border: 1px solid {C["border"]}; gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent; border-radius: 8px;
    color: {C["muted"]}; font-weight: 500; font-size: 0.84rem;
    padding: 0.48rem 1rem;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(0,229,255,0.12) !important;
    color: {C["cyan"]} !important;
}}
.stButton > button {{
    background: linear-gradient(135deg, {C["cyan"]}, {C["violet"]}) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    width: 100%;
}}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
.stNumberInput [data-baseweb="input"],
.stDateInput [data-baseweb="input"] {{
    background-color: {C["surface2"]} !important;
    border-color: {C["border"]} !important;
    color: {C["text"]} !important;
    border-radius: 9px !important;
}}
[data-baseweb="input"] input,
.stNumberInput input,
.stDateInput input,
[data-baseweb="select"] span,
[data-baseweb="select"] div {{
    color: {C["text"]} !important;
    -webkit-text-fill-color: {C["text"]} !important;
}}
[data-baseweb="select"] svg {{
    fill: {C["text"]} !important;
}}
label, .stSlider label, .stSelectbox label, .stNumberInput label, .stDateInput label {{
    color: {C["text"]} !important;
}}
.stAlert {{
    background: {C["surface"]} !important;
    border-color: {C["border"]} !important;
    border-radius: 10px !important;
    color: {C["muted"]} !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


ICONS = {
    "trend": '<svg viewBox="0 0 24 24"><path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/></svg>',
    "calendar": '<svg viewBox="0 0 24 24"><path d="M8 2v4M16 2v4M3 10h18"/><rect x="3" y="4" width="18" height="18" rx="2"/></svg>',
    "bars": '<svg viewBox="0 0 24 24"><path d="M4 20V10M12 20V4M20 20v-7"/></svg>',
    "grid": '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    "stats": '<svg viewBox="0 0 24 24"><path d="M4 19h16"/><path d="M7 16V8M12 16V5M17 16v-3"/></svg>',
    "target": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>',
    "model": '<svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 9h6M9 15h6M9 12h6"/></svg>',
    "form": '<svg viewBox="0 0 24 24"><path d="M4 5h16M4 12h16M4 19h10"/><circle cx="18" cy="19" r="2"/></svg>',
    "bolt": '<svg viewBox="0 0 24 24"><path d="M13 2L4 14h7l-1 8 10-13h-7l0-7z"/></svg>',
}


def section(title: str, icon: str = ""):
    icon_html = (
        f'<span class="section-icon">{ICONS[icon]}</span>' if icon in ICONS else ""
    )
    st.markdown(
        f'<div class="section-header">{icon_html}{title}</div>', unsafe_allow_html=True
    )


def fmt_kw(value: float) -> str:
    if pd.isna(value):
        return "-"
    value = abs(float(value))
    return f"{value / 1000:.2f} MW" if value >= 1000 else f"{value:.1f} kW"


def kpi_card(label: str, value: str, sub: str, color: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar" style="background:linear-gradient(90deg,{color},{color}55);"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def show_fig(fig: go.Figure, height: int = 320):
    fig.update_layout(**PLOTLY_BASE, height=height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def model_features(model, df: pd.DataFrame | None = None) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = getattr(model, "feature_name_", None)
    if names is not None:
        return list(names)
    if df is not None:
        return [c for c in df.columns if c != TARGET]
    return DEFAULT_FEATURES.copy()


def scale_features(
    X: pd.DataFrame, df_reference: pd.DataFrame | None, features: list[str]
) -> pd.DataFrame:
    if df_reference is None:
        raise ValueError(
            "Le dataset principal est requis pour standardiser les entrées."
        )
    missing = [c for c in features if c not in df_reference.columns]
    if missing:
        raise ValueError(
            "Colonnes absentes du dataset principal : " + ", ".join(missing)
        )
    means = df_reference[features].mean()
    scales = df_reference[features].std(ddof=0).replace(0, 1)
    return (X[features] - means) / scales


@st.cache_data(show_spinner=False)
def compute_model_outputs(df_test: pd.DataFrame, _model, features: list[str]):
    missing = [c for c in features + [TARGET] if c not in df_test.columns]
    if missing:
        raise ValueError("Colonnes absentes du set de test : " + ", ".join(missing))
    X_test = df_test[features].copy()
    y_true = df_test[TARGET].astype(float)
    y_pred = pd.Series(_model.predict(X_test), index=df_test.index, name="prediction")
    residuals = y_true - y_pred
    non_zero = y_true != 0
    mape = np.nan
    if non_zero.any():
        mape = float(
            np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]))
            * 100
        )
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape,
        "Bias": float(residuals.mean()),
    }
    return y_true, y_pred, residuals, metrics


def default_lag_values(df: pd.DataFrame | None) -> dict[str, float]:
    values = {"lag_1": 350.0, "lag_4": 340.0, "lag_96": 330.0}
    if df is None:
        return values
    for col in values:
        if col in df.columns and df[col].dropna().size:
            values[col] = float(df[col].dropna().iloc[-1])
    return values


def latest_dataset_timestamp(df: pd.DataFrame | None) -> pd.Timestamp | None:
    if df is None or TARGET not in df.columns:
        return None
    clean = df[TARGET].dropna()
    return pd.Timestamp(clean.index.max()) if not clean.empty else None


def prediction_context_from_history(df: pd.DataFrame | None, pred_date, pred_hour: int):
    defaults = default_lag_values(df)
    if df is None or df.empty:
        return defaults, None, False
    available_lags = [c for c in defaults if c in df.columns]
    if not available_lags:
        return defaults, None, False
    idx = pd.to_datetime(df.index, errors="coerce")
    mask = (idx.date == pred_date) & (idx.hour == pred_hour)
    candidates = df.loc[mask, available_lags].dropna(how="any").sort_index()
    if candidates.empty:
        return defaults, None, False
    selected = candidates.iloc[-1]
    values = defaults.copy()
    for col in available_lags:
        values[col] = float(selected[col])
    return values, pd.Timestamp(candidates.index[-1]), True


def demand_thresholds(series: pd.Series | None) -> tuple[float, float, float]:
    if series is None or series.dropna().empty:
        return 300.0, 500.0, 800.0
    clean = series.dropna().astype(float)
    low = float(clean.quantile(0.33))
    high = float(clean.quantile(0.75))
    gauge_max = float(max(clean.quantile(0.99), high * 1.15, 1))
    return low, high, gauge_max


with st.spinner("Chargement des données..."):
    df_main = load_csv(PATH_MAIN)
    df_test = load_csv(PATH_TEST)
    try:
        model = load_model(PATH_MODEL)
        model_error = None
    except Exception as exc:
        model = None
        model_error = str(exc)

MONTH_FR = [
    "Janvier",
    "Février",
    "Mars",
    "Avril",
    "Mai",
    "Juin",
    "Juillet",
    "Août",
    "Septembre",
    "Octobre",
    "Novembre",
    "Décembre",
]
MONTH_SHORT = [
    "Jan",
    "Fév",
    "Mar",
    "Avr",
    "Mai",
    "Juin",
    "Jul",
    "Août",
    "Sep",
    "Oct",
    "Nov",
    "Déc",
]
DOW_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
DOW_SHORT = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

with st.sidebar:
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.45rem;font-weight:700;color:{C['cyan']};line-height:1.1;">
                ⚡ EnergyAI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Analyse des données",
            "Visualisations",
            "Prédiction",
        ],
        label_visibility="collapsed",
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.toggle("Mode clair", key="light_mode")


def page_title(title: str, subtitle: str):
    st.markdown(
        f"""
        <h1 style="font-family:'Space Grotesk',sans-serif;font-size:1.9rem;font-weight:700;color:{C['text']};margin:0 0 0.2rem 0;">
            {title}
        </h1>
        <p style="color:{C['muted']};font-size:0.875rem;margin:0 0 1.75rem 0;">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


if page == "Dashboard":
    page_title(
        "Tableau de bord", "Consommation électrique - Boston City Hall - Vue exécutive"
    )
    if df_main is None or TARGET not in df_main.columns:
        st.error(
            "Fichier `data/dataset_final_powerbi.csv` introuvable ou colonne cible absente."
        )
    else:
        series = df_main[TARGET].dropna()
        years = f"{series.index.min().year} - {series.index.max().year}"
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        cards = [
            (c1, "Observations", f"{len(series):,}", years, C["cyan"]),
            (
                c2,
                "Consommation moy.",
                fmt_kw(series.mean()),
                "Moyenne globale",
                C["violet"],
            ),
            (
                c3,
                "Pic de consomm.",
                fmt_kw(series.max()),
                "Maximum enregistré",
                C["orange"],
            ),
            (
                c4,
                "Heure creuse",
                fmt_kw(series.min()),
                "Minimum enregistré",
                C["green"],
            ),
        ]
        for col, label, value, sub, color in cards:
            with col:
                st.markdown(kpi_card(label, value, sub, color), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Série temporelle - Demande horaire", "trend")
        disp = series.resample("1h").mean().dropna()
        ma7 = disp.rolling(24 * 7, min_periods=24).mean()
        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=disp.index,
                y=disp.values,
                mode="lines",
                name="Demande (kW)",
                line=dict(color=C["cyan"], width=1.1),
                fill="tozeroy",
                fillcolor="rgba(0,229,255,0.04)",
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=ma7.index,
                y=ma7.values,
                mode="lines",
                name="Moyenne mobile 7 j",
                line=dict(color=C["violet"], width=2, dash="dot"),
            )
        )
        fig_ts.update_layout(title="Demande horaire avec moyenne mobile 7 jours")
        show_fig(fig_ts, 310)

        col_l, col_r = st.columns(2, gap="medium")
        with col_l:
            section("Profil mensuel", "calendar")
            monthly = df_main[[TARGET]].copy()
            monthly["month"] = monthly.index.month
            mm = monthly.groupby("month")[TARGET].mean()
            fig_m = go.Figure(
                go.Bar(
                    x=[MONTH_SHORT[i - 1] for i in mm.index],
                    y=mm.values,
                    marker_color=C["cyan"],
                    hovertemplate="%{x} : %{y:.1f} kW<extra></extra>",
                )
            )
            fig_m.update_layout(title="Demande moyenne par mois")
            show_fig(fig_m, 280)
        with col_r:
            section("Profil hebdomadaire", "calendar")
            weekly = df_main[[TARGET]].copy()
            weekly["dow"] = weekly.index.dayofweek
            dw = weekly.groupby("dow")[TARGET].mean().reindex(range(7))
            colors = [C["red"] if i >= 5 else C["cyan"] for i in range(7)]
            fig_w = go.Figure(
                go.Bar(
                    x=DOW_SHORT,
                    y=dw.values,
                    marker_color=colors,
                    hovertemplate="%{x} : %{y:.1f} kW<extra></extra>",
                )
            )
            fig_w.update_layout(title="Demande moyenne par jour")
            show_fig(fig_w, 280)

        section("Heatmap de consommation - Heure x Jour", "grid")
        dc = df_main[[TARGET]].copy()
        dc["hour"] = dc.index.hour
        dc["dow"] = dc.index.dayofweek
        pivot = (
            dc.groupby(["dow", "hour"])[TARGET]
            .mean()
            .unstack()
            .reindex(index=range(7), columns=range(24))
        )
        fig_hm = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=[f"{h}h" for h in pivot.columns],
                y=DOW_SHORT,
                colorscale=[
                    [0, C["bg"]],
                    [0.35, C["violet"]],
                    [0.7, C["cyan"]],
                    [1, C["green"]],
                ],
                hovertemplate="Jour: %{y}<br>Heure: %{x}<br>Demande: %{z:.1f} kW<extra></extra>",
            )
        )
        fig_hm.update_layout(
            title="Intensité de la demande par heure et jour de semaine"
        )
        show_fig(fig_hm, 300)

elif page == "Analyse des données":
    page_title(
        "Analyse exploratoire",
        "Statistiques descriptives, distributions et profils temporels",
    )
    if df_main is None or TARGET not in df_main.columns:
        st.error(
            "Fichier `data/dataset_final_powerbi.csv` introuvable ou colonne cible absente."
        )
    else:
        s = df_main[TARGET].dropna()
        if s.empty:
            st.error("La colonne cible ne contient aucune valeur exploitable.")
        else:
            section("Statistiques descriptives", "stats")
            stats = {
                "Moyenne": f"{s.mean():.2f} kW",
                "Médiane": f"{s.median():.2f} kW",
                "Écart-type": f"{s.std():.2f} kW",
                "Min": f"{s.min():.2f} kW",
                "Max": f"{s.max():.2f} kW",
                "Skewness": f"{s.skew():.4f}",
                "Kurtosis": f"{s.kurt():.4f}",
                "Observations": f"{len(s):,}",
            }
            cols = st.columns(4, gap="small")
            for i, (k, v) in enumerate(stats.items()):
                with cols[i % 4]:
                    st.markdown(
                        f'<div class="stat-pill"><div class="stat-pill-label">{k}</div><div class="stat-pill-value">{v}</div></div>',
                        unsafe_allow_html=True,
                    )

            col_a, col_b = st.columns(2, gap="medium")
            with col_a:
                section("Distribution de la demande", "bars")
                fig_hist = go.Figure(
                    go.Histogram(
                        x=s, nbinsx=80, marker=dict(color=C["cyan"], opacity=0.75)
                    )
                )
                fig_hist.add_vline(x=s.median(), line_color=C["violet"], line_width=1.5)
                fig_hist.update_layout(
                    title="Histogramme - Total_Demand_KW", bargap=0.03
                )
                show_fig(fig_hist, 310)
            with col_b:
                section("Distribution mensuelle", "calendar")
                df_box = df_main[[TARGET]].dropna().copy()
                df_box["month"] = df_box.index.month
                fig_box = go.Figure()
                for m in sorted(df_box["month"].unique()):
                    vals = df_box.loc[df_box["month"] == m, TARGET]
                    fig_box.add_trace(
                        go.Box(
                            y=vals,
                            name=MONTH_SHORT[int(m) - 1],
                            marker_color=C["violet"],
                            line=dict(color=C["cyan"]),
                            boxmean=True,
                            showlegend=False,
                        )
                    )
                fig_box.update_layout(title="Boîtes à moustaches par mois")
                show_fig(fig_box, 310)

            section("Profil horaire moyen (+/- 1 sigma)", "trend")
            dc2 = df_main[[TARGET]].dropna().copy()
            dc2["hour"] = dc2.index.hour
            hourly = (
                dc2.groupby("hour")[TARGET]
                .agg(["mean", "std"])
                .reindex(range(24))
                .fillna(0)
            )
            fig_h = go.Figure()
            fig_h.add_trace(
                go.Scatter(
                    x=list(hourly.index) + list(hourly.index[::-1]),
                    y=list(hourly["mean"] + hourly["std"])
                    + list((hourly["mean"] - hourly["std"])[::-1]),
                    fill="toself",
                    fillcolor="rgba(0,229,255,0.06)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    name="+/- 1 sigma",
                )
            )
            fig_h.add_trace(
                go.Scatter(
                    x=hourly.index,
                    y=hourly["mean"],
                    mode="lines+markers",
                    name="Moyenne horaire",
                    line=dict(color=C["cyan"], width=2),
                )
            )
            fig_h.update_layout(
                title="Demande moyenne par heure",
                xaxis=dict(
                    tickvals=list(range(0, 24, 3)),
                    ticktext=[f"{h}h" for h in range(0, 24, 3)],
                    **PLOTLY_BASE["xaxis"],
                ),
            )
            show_fig(fig_h, 300)

            section("Analyse saisonnière", "grid")
            df_sea = df_main[[TARGET]].dropna().copy()
            df_sea["season"] = df_sea.index.month.map(
                lambda m: (
                    "Hiver"
                    if m in [12, 1, 2]
                    else (
                        "Printemps"
                        if m in [3, 4, 5]
                        else "Été" if m in [6, 7, 8] else "Automne"
                    )
                )
            )
            fig_sea = go.Figure()
            for name, color in zip(
                ["Printemps", "Été", "Automne", "Hiver"],
                [C["green"], C["orange"], C["violet"], C["cyan"]],
            ):
                vals = df_sea.loc[df_sea["season"] == name, TARGET]
                if not vals.empty:
                    fig_sea.add_trace(
                        go.Violin(
                            y=vals,
                            name=name,
                            line_color=color,
                            fillcolor=color,
                            opacity=0.45,
                            box_visible=True,
                            meanline_visible=True,
                        )
                    )
            fig_sea.update_layout(
                title="Distribution de la demande par saison", violinmode="group"
            )
            show_fig(fig_sea, 330)

elif page == "Visualisations":
    page_title(
        "Visualisations & Modèle",
        "Corrélations, comparaison du modèle et résultats de prédiction",
    )
    tab1, tab2 = st.tabs(["Corrélations & Features", "modèle"])
    with tab1:
        if df_main is None or TARGET not in df_main.columns:
            st.error(
                "Fichier `data/dataset_final_powerbi.csv` introuvable ou colonne cible absente."
            )
        else:
            numeric_cols = df_main.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) <= 1:
                st.warning(
                    "Pas assez de colonnes numériques pour calculer les corrélations."
                )
            else:
                corr = df_main[numeric_cols].corr()
                section("Matrice de corrélation", "grid")
                fig_heat = go.Figure(
                    go.Heatmap(
                        z=corr.values,
                        x=corr.columns.tolist(),
                        y=corr.columns.tolist(),
                        colorscale=[
                            [0, C["violet"]],
                            [0.5, C["surface"]],
                            [1, C["cyan"]],
                        ],
                        zmid=0,
                        text=np.round(corr.values, 2),
                        texttemplate="%{text}",
                        textfont=dict(size=9, color=C["text"]),
                    )
                )
                fig_heat.update_layout(title="Corrélation entre toutes les variables")
                show_fig(fig_heat, 470)
                if TARGET in corr.columns:
                    section("Corrélation avec la cible", "target")
                    ct = corr[TARGET].drop(TARGET).sort_values()
                    fig_ct = go.Figure(
                        go.Bar(
                            x=ct.values,
                            y=ct.index,
                            orientation="h",
                            marker_color=[
                                C["red"] if v < 0 else C["cyan"] for v in ct.values
                            ],
                            text=[f"{v:+.3f}" for v in ct.values],
                            textposition="outside",
                        )
                    )
                    fig_ct.add_vline(x=0, line_color=C["border"], line_width=1)
                    fig_ct.update_layout(
                        title=f"Corrélation de chaque variable avec {TARGET}"
                    )
                    show_fig(fig_ct, 340)
            if "lag_96" in df_main.columns:
                section("Autocorrélation - Demande vs Demande J-1", "trend")
                sample = df_main[[TARGET, "lag_96"]].dropna()
                if len(sample) > 6000:
                    sample = sample.sample(6000, random_state=42)
                fig_sc = go.Figure(
                    go.Scattergl(
                        x=sample["lag_96"],
                        y=sample[TARGET],
                        mode="markers",
                        marker=dict(color=C["violet"], size=2.5, opacity=0.35),
                    )
                )
                fig_sc.update_layout(
                    title="Demande actuelle vs Demande J-1 (lag_96)",
                    xaxis_title="Demande J-1 (kW)",
                    yaxis_title="Demande actuelle (kW)",
                )
                show_fig(fig_sc, 320)

    with tab2:
        section("Classement du modèle", "model")
        model_eval = None
        if df_test is not None and model is not None and TARGET in df_test.columns:
            try:
                features = model_features(model, df_test)
                y_true_eval, y_pred_eval, residuals_eval, real_metrics = (
                    compute_model_outputs(df_test, model, features)
                )
                model_eval = (
                    features,
                    y_true_eval,
                    y_pred_eval,
                    residuals_eval,
                    real_metrics,
                )
            except Exception as exc:
                st.error(f"Impossible de calculer les scores réels du modèle : {exc}")
        else:
            st.warning(
                "Scores indisponibles : vérifiez `data_test.csv` et `lgbm_optimise.pkl`."
            )

        if model_eval is not None:
            _, y_true_eval, y_pred_eval, residuals_eval, real_metrics = model_eval
            st.markdown(
                f"""
                <div class="model-row best">
                    <strong style="color:{C['cyan']};width:24%;">LightGBM optimisé</strong>
                    <span>RMSE <strong>{real_metrics['RMSE']:.4f}</strong></span>
                    <span>MAE <strong>{real_metrics['MAE']:.4f}</strong></span>
                    <span>R² <strong>{real_metrics['R2']:.4f}</strong></span>
                    <span>MAPE <strong>{real_metrics['MAPE']:.2f}%</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_l, col_r = st.columns(2, gap="medium")
            with col_l:
                section("Réel vs prédit", "trend")
                n = min(500, len(y_true_eval))
                fig_pv = go.Figure()
                fig_pv.add_trace(
                    go.Scatter(
                        x=y_true_eval.index[:n],
                        y=y_true_eval.iloc[:n],
                        mode="lines",
                        name="Valeurs réelles",
                        line=dict(color=C["text"], width=1.3),
                    )
                )
                fig_pv.add_trace(
                    go.Scatter(
                        x=y_pred_eval.index[:n],
                        y=y_pred_eval.iloc[:n],
                        mode="lines",
                        name="LightGBM prédit",
                        line=dict(color=C["cyan"], width=1.3, dash="dot"),
                    )
                )
                fig_pv.update_layout(title="500 premiers points - Réel vs Prédit")
                show_fig(fig_pv, 330)
            with col_r:
                section("Distribution des résidus", "bars")
                fig_res = go.Figure(
                    go.Histogram(
                        x=residuals_eval,
                        nbinsx=80,
                        marker=dict(color=C["cyan"], opacity=0.72),
                    )
                )
                fig_res.add_vline(
                    x=0, line_color=C["red"], line_dash="dash", line_width=1.5
                )
                fig_res.update_layout(
                    title="Distribution des erreurs de prédiction",
                    xaxis_title="Erreur (kW)",
                    yaxis_title="Fréquence",
                )
                show_fig(fig_res, 330)

            section("Scatter réel vs prédit", "target")
            y_true = y_true_eval.values
            y_pred = y_pred_eval.values
            mn = float(min(np.nanmin(y_true), np.nanmin(y_pred)))
            mx = float(max(np.nanmax(y_true), np.nanmax(y_pred)))
            fig_scatter = go.Figure()
            fig_scatter.add_trace(
                go.Scattergl(
                    x=y_true,
                    y=y_pred,
                    mode="markers",
                    marker=dict(color=C["violet"], size=2.5, opacity=0.3),
                    name="Points",
                )
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=[mn, mx],
                    y=[mn, mx],
                    mode="lines",
                    line=dict(color=C["red"], dash="dash", width=1.5),
                    name="Ligne parfaite",
                )
            )
            fig_scatter.update_layout(
                title="Réel vs Prédit",
                xaxis_title="Réel (kW)",
                yaxis_title="Prédit (kW)",
            )
            show_fig(fig_scatter, 360)

elif page == "Prédiction":
    page_title(
        "Prédiction en temps réel",
        "Estimez la demande énergétique avec le modèle LightGBM",
    )
    latest_ts = latest_dataset_timestamp(df_main)
    default_dt = latest_ts if latest_ts is not None else pd.Timestamp(datetime.now())
    min_pred_date = (
        pd.Timestamp(df_main.index.min()).date()
        if df_main is not None and not df_main.empty
        else None
    )
    max_pred_date = (
        pd.Timestamp(df_main.index.max()).date()
        if df_main is not None and not df_main.empty
        else None
    )

    col_form, col_out = st.columns([1, 1], gap="large")
    with col_form:
        section("Contexte temporel", "form")
        pred_date = st.date_input(
            "Date",
            value=default_dt.date(),
            help="Sélectionnez une date pour la prédiction (hors historique possible).",
        )
        pred_hour = st.slider("Heure", 0, 23, int(default_dt.hour), format="%dh")
        default_month = pred_date.month
        default_dow = pred_date.weekday()
        c_month, c_dow = st.columns(2)
        with c_month:
            pred_month = st.selectbox(
                "Mois",
                range(1, 13),
                format_func=lambda x: MONTH_FR[x - 1],
                index=default_month - 1,
            )
        with c_dow:
            pred_dow = st.selectbox(
                "Jour",
                range(7),
                format_func=lambda x: DOW_FR[x],
                index=default_dow,
            )

        section("Valeurs de lag (kW)", "bolt")
        lag_defaults, _, _ = prediction_context_from_history(
            df_main, pred_date, pred_hour
        )
        lag1 = st.number_input(
            "lag_1 - Demande il y a 15 min",
            value=lag_defaults["lag_1"],
            min_value=0.0,
            step=5.0,
            key=f"lag1_{pred_date}_{pred_hour}",
        )
        lag4 = st.number_input(
            "lag_4 - Demande il y a 1h",
            value=lag_defaults["lag_4"],
            min_value=0.0,
            step=5.0,
            key=f"lag4_{pred_date}_{pred_hour}",
        )
        lag96 = st.number_input(
            "lag_96 - Demande hier même heure",
            value=lag_defaults["lag_96"],
            min_value=0.0,
            step=5.0,
            key=f"lag96_{pred_date}_{pred_hour}",
        )
        predict_btn = st.button("Lancer la prédiction", use_container_width=True)

    with col_out:
        pred_val = None
        is_weekend = 1 if pred_dow >= 5 else 0
        pred_quarter = (pred_month - 1) // 3 + 1
        pred_year = pred_date.year

        if model is None:
            st.error("Modèle non chargé. Vérifiez `data/lgbm_optimise.pkl`.")
            if model_error:
                st.caption(model_error)
        elif not predict_btn:
            st.info("Configurez les paramètres puis cliquez sur Lancer la prédiction.")
        else:
            feature_values = {
                "hour": pred_hour,
                "dayofweek": pred_dow,
                "month": pred_month,
                "year": pred_year,
                "quarter": pred_quarter,
                "is_weekend": is_weekend,
                "lag_1": lag1,
                "lag_4": lag4,
                "lag_96": lag96,
            }
            try:
                features = model_features(
                    model, df_test if df_test is not None else df_main
                )
                X_input = pd.DataFrame([feature_values])
                missing_input = [c for c in features if c not in X_input.columns]
                if missing_input:
                    raise ValueError(
                        "Features manquantes pour la prédiction : "
                        + ", ".join(missing_input)
                    )
                X_model = scale_features(X_input, df_main, features)
                pred_val = float(model.predict(X_model)[0])
                hist_series = (
                    df_main[TARGET]
                    if df_main is not None and TARGET in df_main.columns
                    else None
                )
                low_thr, high_thr, gauge_max = demand_thresholds(hist_series)
                if pred_val < low_thr:
                    level, badge_col = "Charge faible", C["green"]
                elif pred_val < high_thr:
                    level, badge_col = "Charge modérée", C["orange"]
                else:
                    level, badge_col = "Charge élevée", C["red"]
                st.markdown(
                    f"""
                    <div class="pred-card">
                        <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{C['muted']};margin-bottom:0.6rem;">Demande prédite</div>
                        <div class="pred-number">{fmt_kw(pred_val)}</div>
                        <div style="font-size:0.8rem;color:{C['muted']};margin-top:0.5rem;">{DOW_FR[pred_dow]} à {pred_hour}h - {'Week-end' if is_weekend else 'Semaine'} - {MONTH_FR[pred_month - 1]}</div>
                        <span class="pred-badge" style="background:{badge_col}22;color:{badge_col};">{level}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_g = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=abs(pred_val),
                        number=dict(font=dict(color=C["cyan"], size=26), suffix=" kW"),
                        gauge=dict(
                            axis=dict(
                                range=[0, gauge_max],
                                tickfont=dict(color=C["muted"], size=10),
                            ),
                            bar=dict(color=C["cyan"], thickness=0.22),
                            bgcolor="rgba(0,0,0,0)",
                            borderwidth=0,
                            steps=[
                                dict(range=[0, low_thr], color="rgba(0,200,83,0.12)"),
                                dict(
                                    range=[low_thr, high_thr],
                                    color="rgba(255,107,53,0.12)",
                                ),
                                dict(
                                    range=[high_thr, gauge_max],
                                    color="rgba(255,61,113,0.12)",
                                ),
                            ],
                            threshold=dict(
                                line=dict(color=C["red"], width=2),
                                thickness=0.75,
                                value=high_thr,
                            ),
                        ),
                        title=dict(
                            text="Niveau de demande",
                            font=dict(color=C["muted"], size=12),
                        ),
                    )
                )
                fig_g.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=C["muted"]),
                    height=240,
                    margin=dict(l=20, r=20, t=40, b=10),
                )
                st.plotly_chart(
                    fig_g, use_container_width=True, config={"displayModeBar": False}
                )
            except Exception as exc:
                st.error(f"Erreur de prédiction : {exc}")
                st.info(
                    "Vérifiez que le modèle correspond bien aux features du projet."
                )

        if df_main is not None and TARGET in df_main.columns:
            section("Référence - Profil horaire historique", "trend")
            dc = df_main[[TARGET]].dropna().copy()
            dc["hour"] = dc.index.hour
            href = dc.groupby("hour")[TARGET].mean().reindex(range(24))
            fig_ref = go.Figure()
            fig_ref.add_trace(
                go.Scatter(
                    x=href.index,
                    y=href.values,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=C["subtle"], width=1.5),
                    fillcolor="rgba(42,58,90,0.25)",
                    name="Moyenne historique",
                )
            )
            if pred_val is not None:
                fig_ref.add_trace(
                    go.Scatter(
                        x=[pred_hour],
                        y=[abs(pred_val)],
                        mode="markers",
                        name="Votre prédiction",
                        marker=dict(
                            color=C["cyan"],
                            size=13,
                            symbol="star",
                            line=dict(color=C["bg"], width=2),
                        ),
                    )
                )
            fig_ref.update_layout(
                title="Votre prédiction dans le contexte historique",
                xaxis=dict(
                    tickvals=list(range(0, 24, 3)),
                    ticktext=[f"{h}h" for h in range(0, 24, 3)],
                    **PLOTLY_BASE["xaxis"],
                ),
            )
            show_fig(fig_ref, 230)
