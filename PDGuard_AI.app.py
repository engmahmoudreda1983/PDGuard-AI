import hmac
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pdguard_core as pc

st.set_page_config(page_title="PDGuard AI", page_icon="⚡", layout="wide")

STRETCH = "stretch"
NAVY, BLUE = "#0b2e59", "#4da6ff"
GREEN, AMBER, RED = "#1a9e5c", "#d99a00", "#d63a3a"
STATUS_COLOR = {'EXCELLENT': GREEN, 'WATCH': AMBER, 'ACTION': RED}
STATUS_LABEL = {'EXCELLENT': 'NORMAL (Low PD Risk)', 'WATCH': 'WATCH (Elevated PD Risk)',
                'ACTION': 'ACTION REQUIRED (High PD Risk)'}
CLASS_TITLE = {'Transformer': 'Transformers', 'GIS': 'GIS', 'Reactor': 'Reactors', 'Cable': 'Cables', 'Motor': 'Motors'}

st.markdown("""
<style>
.pg-header {background:#0b2e59;padding:16px 22px;border-radius:10px;margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;gap:16px;}
.pg-header img {height:64px;background:#fff;border-radius:8px;padding:4px 8px;}
.pg-header h1 {color:#fff;margin:0;font-size:1.9rem;}
.pg-header p {color:#9fd0ff;margin:4px 0 0 0;font-size:1rem;}
.pg-banner {border:2px solid;border-radius:8px;padding:10px 14px;text-align:center;margin:6px 0 12px 0;}
.pg-banner h2 {margin:0;font-size:1.5rem;}
.pg-banner p {margin:2px 0 0 0;font-size:.95rem;}
.pg-note {background:rgba(77,166,255,.10);border-left:4px solid #4da6ff;padding:8px 12px;border-radius:4px;font-size:.92rem;margin-bottom:10px;}
.pg-card {border:1px solid #d5dce6;border-radius:10px;padding:12px 14px;text-align:center;}
.pg-card h4 {margin:0;color:#0b2e59;font-size:1.05rem;}
.pg-card .big {font-size:1.7rem;font-weight:700;margin:4px 0;}
.pg-card .sub {font-size:.85rem;color:#5a6575;}
.pg-foot {text-align:center;color:#888;font-size:12px;padding:10px;}
</style>
""", unsafe_allow_html=True)

FOOTER = """
<div class='pg-foot'><strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
<em>Confidential and Proprietary. The predictive algorithms and expert system overrides within this application are protected intellectual property.</em></div>
"""


def banner(title, sub, color):
    st.markdown(f"<div class='pg-banner' style='border-color:{color}'><h2 style='color:{color}'>{title}</h2>"
                f"<p>{sub}</p></div>", unsafe_allow_html=True)


def secrets_ok():
    try:
        return "APP_USERNAME" in st.secrets and "APP_PASSWORD" in st.secrets
    except Exception:
        return False


# ==========================================
# --- 1. Login ---
# ==========================================
for key, default in [('logged_in', False), ('register', []), ('hist', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state['logged_in']:
    st.markdown("<br><br><h1 style='text-align:center;color:#4da6ff;'>⚡ PDGuard AI Login</h1>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.warning("🔒 Restricted Access. Authorized Personnel Only.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Secure Login", width=STRETCH):
                if not secrets_ok():
                    st.error("Login is not configured: add APP_USERNAME and APP_PASSWORD to the app's Secrets.")
                elif hmac.compare_digest(username, st.secrets["APP_USERNAME"]) and \
                        hmac.compare_digest(password, st.secrets["APP_PASSWORD"]):
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password")
    st.markdown(FOOTER, unsafe_allow_html=True)
    st.stop()

# ==========================================
# --- 2. Header and model ---
# ==========================================
_, col_r = st.columns([10, 1])
with col_r:
    if st.button("Logout 🚪"):
        st.session_state['logged_in'] = False
        st.rerun()

@st.cache_data
def logo_b64():
    try:
        import base64, io
        from PIL import Image
        im = Image.open('logo.png')
        im.thumbnail((360, 200))
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


_logo = logo_b64()
_logo_html = f"<img src='data:image/png;base64,{_logo}' alt='PowerGuard AI'>" if _logo else ""
st.markdown(f"""
<div class='pg-header'>
  <div>
    <h1>⚡ PDGuard AI: Partial Discharge Risk Monitor</h1>
    <p>GIS, transformers, reactors, cables and motors: PD pattern, trend and confirmation, fused into one risk</p>
  </div>
  {_logo_html}
</div>""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load('pd_classifier.pkl')


try:
    MODEL = load_model()
except Exception as e:
    st.error(f"Model file 'pd_classifier.pkl' could not be loaded: {e}")
    st.stop()


@st.cache_data(show_spinner=False)
def demo_current():
    """Demo substation as it stands on 1 March 2026 (three defects active)."""
    return pc.demo_data(end='2026-03-01')


@st.cache_data(show_spinner=False)
def demo_full():
    """Full demo history with the recorded failures, for the historical study."""
    return pc.demo_data()


@st.cache_data(show_spinner="Assessing every reading…")
def run_assessment(online, sensors, dga, acoustic, sf6):
    return pc.assess(online, sensors, MODEL, dga, acoustic, sf6)


# ==========================================
# --- 3. Shared charts and blocks ---
# ==========================================
def gauge(value, color, title='PD Risk (%)', height=290):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), number={'suffix': '%', 'font': {'size': 34}, 'valueformat': '.0f'},
        title={'text': title, 'font': {'size': 15}}, domain={'x': [0, 1], 'y': [0, 0.85]},
        gauge={'axis': {'range': [0, 100], 'tickvals': [0, 35, 70, 100], 'tickfont': {'size': 12}},
               'bar': {'color': color, 'thickness': 0.3},
               'steps': [{'range': [0, 35], 'color': 'rgba(26,158,92,.25)'},
                         {'range': [35, 70], 'color': 'rgba(217,154,0,.25)'},
                         {'range': [70, 100], 'color': 'rgba(214,58,58,.25)'}]}))
    fig.update_layout(height=height, margin=dict(l=35, r=35, t=40, b=10))
    return fig


def phase_plot(counts, mags, height=330):
    h, m = np.asarray(counts, float), np.asarray(mags, float)
    fig = go.Figure()
    fig.add_bar(x=pc.PHASE_DEG, y=h, width=9, marker=dict(color=m, colorscale='Turbo', cmin=0, cmax=1,
                colorbar=dict(title='Relative<br>magnitude', thickness=12)),
                hovertemplate='%{x}°: %{y:.0f} pulses<extra></extra>')
    x = np.linspace(0, 360, 181)
    top = max(h.max(), 1)
    fig.add_scatter(x=x, y=top / 2 + top / 2 * np.sin(np.radians(x)), mode='lines', line=dict(color='#555', dash='dash'),
                    hoverinfo='skip')
    fig.update_layout(title='PRPD phase profile (dashed line = voltage)', height=height,
                      xaxis=dict(title='Phase angle (°)', tickvals=[0, 90, 180, 270, 360], range=[0, 360]),
                      yaxis_title='Pulses', showlegend=False, margin=dict(l=40, r=10, t=50, b=40))
    return fig


def prob_plot(probs, height=250):
    items = sorted(probs.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v * 100 for _, v in items], y=[k for k, _ in items], orientation='h', marker_color=BLUE,
                           hovertemplate='%{y}: %{x:.0f}%<extra></extra>'))
    fig.update_layout(title='Pattern match by defect type', height=height, xaxis=dict(range=[0, 100], title='%'),
                      margin=dict(l=10, r=10, t=40, b=30))
    return fig


def contrib_plot(row, height=300):
    c = pc.contributions(row)
    names = list(c)[::-1]
    vals = [c[n] for n in names]
    colors = [RED if n == 'Safety rules' and v > 0 else '#9aa7b8' if n == 'Safety rules'
              else AMBER if 'confirmation' in n else NAVY for n, v in zip(names, vals)]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation='h', marker_color=colors,
                           text=[f"{v:+.0f}" if abs(v) >= 0.5 else "0" for v in vals], textposition='outside',
                           cliponaxis=False, hovertemplate='%{y}: %{x:.1f} points<extra></extra>'))
    lo = min(0, min(vals)) - 5
    fig.update_layout(title=f"What drives this PD risk ({row['PD Risk (%)']:.0f}% in total)", height=height,
                      xaxis=dict(title='Risk points', range=[lo, max(max(vals), 10) * 1.25], zeroline=True),
                      margin=dict(l=10, r=20, t=45, b=40))
    return fig


def rpn_plot(row, height=230):
    rpn = int(row['RPN'])
    col = GREEN if rpn < 40 else AMBER if rpn < 100 else RED
    fig = go.Figure()
    for x0, x1, c, t in [(0, 40, 'rgba(26,158,92,.18)', 'Low'), (40, 100, 'rgba(217,154,0,.18)', 'Medium'),
                         (100, 200, 'rgba(214,58,58,.18)', 'High')]:
        fig.add_shape(type='rect', x0=x0, x1=x1, y0=-0.5, y1=0.5, fillcolor=c, line_width=0, layer='below')
        fig.add_annotation(x=(x0 + x1) / 2, y=0.62, text=t, showarrow=False, font=dict(size=11, color='#555'))
    fig.add_bar(x=[rpn], y=[0], orientation='h', width=0.45, marker_color=col,
                text=[f"RPN {rpn} = S {int(row['S'])} × O {int(row['O'])} × D {int(row['D'])}"],
                textposition='inside' if rpn > 110 else 'outside', insidetextanchor='middle', cliponaxis=False,
                hoverinfo='skip', textfont=dict(color='white' if rpn > 110 else '#222'))
    fig.update_layout(title='Risk Priority Number (max 200)', height=height, showlegend=False,
                      xaxis=dict(range=[0, 200], tickvals=[0, 40, 100, 200], title='RPN'),
                      yaxis=dict(visible=False, range=[-0.6, 0.8]), margin=dict(l=10, r=10, t=45, b=40))
    return fig


def show_result(row, probs, counts, mags, k):
    color = STATUS_COLOR[row['Status']]
    banner(STATUS_LABEL[row['Status']], f"{row['Asset ID']}  ·  {pc.DEFECT_TEXT[row['Defect']]}", color)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Above background", f"{row['dB Above Baseline']:.1f} dB")
    m2.metric("Trend", f"{row['Trend (dB/30d)']:+.1f} dB / month")
    m3.metric("Pattern confidence", f"{row['Confidence (%)']:.0f} %" if row['Defect'] != pc.NO_PD else "—")
    m4.metric("Confirmations", f"{int(row['Confirmations'])} of 3")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.plotly_chart(gauge(row['PD Risk (%)'], color), width=STRETCH, key=f"{k}_g")
    with c2:
        st.plotly_chart(contrib_plot(row), width=STRETCH, key=f"{k}_c")
    c3, c4 = st.columns([1.3, 1])
    with c3:
        st.plotly_chart(phase_plot(counts, mags, 290), width=STRETCH, key=f"{k}_ph")
    with c4:
        st.plotly_chart(prob_plot(probs, 290), width=STRETCH, key=f"{k}_pr")
    st.markdown("#### Why this result")
    for t in pc.reasons(row):
        st.markdown(f"- {t}")
    st.warning(f"**Action:** {row['Action']}")
    st.markdown("#### 🧮 FMEA")
    f1, f2 = st.columns([1.2, 1])
    f1.dataframe(pd.DataFrame([{'Failure Mode': f"{row['Defect']} ({row['Asset Class']})", 'S': int(row['S']),
                                'O': int(row['O']), 'D': int(row['D']), 'RPN': int(row['RPN']),
                                'RPN Level': row['RPN Level']}]), hide_index=True, width=STRETCH)
    f1.caption("RPN = Severity × Occurrence × Detection. S = severity of this defect at this equipment; "
               "O = PD risk ÷ 10; D = 2 for online monitoring. Low < 40, Medium 40-99, High ≥ 100.")
    if not pd.isna(row['Days to Critical']):
        f1.metric("Estimated days to critical level", "Already critical" if row['Days to Critical'] == 0
                  else f"{row['Days to Critical']:.0f} days")
    with f2:
        st.plotly_chart(rpn_plot(row), width=STRETCH, key=f"{k}_rpn")


REC_KEYS = ['Asset ID', 'Asset Type', 'Asset Class', 'Location', 'Sensor ID', 'Defect', 'dB Above Baseline',
            'Level (dB, 7d median)', 'Trend (dB/30d)', 'Pulse Rate (pps)', 'Confidence (%)', 'Confirmations',
            'Magnitude Score', 'Trend Score', 'Rate Score',
            'PD Risk (%)', 'Status', 'S', 'O', 'D', 'RPN', 'RPN Level', 'Days to Critical', 'Action',
            'Acoustic', 'SF6 By-products (ppmv)', 'H2 (ppm)', 'C2H2 (ppm)', 'PD Days (14d)', 'Isolated',
            'Low Confidence', 'Noise Suppressed', 'Noise Reference', 'Noise Ref (dB)', 'Single']


def to_record(row, source):
    rec = {k: row.get(k) for k in REC_KEYS}
    rec['Source'] = source
    rec['Assessed'] = datetime.now().strftime('%Y-%m-%d %H:%M') if source == 'Equipment page' else \
        pd.Timestamp(row['Timestamp']).strftime('%Y-%m-%d')
    rec['Findings'] = pc.reasons(row)
    return rec


@st.cache_data(show_spinner="Analyzing the demo substation…")
def demo_records():
    d = demo_current()
    A = pc.assess(d['online'], d['sensors'], MODEL, d['dga'], d['acoustic'], d['sf6'])
    L = pc.latest(A)
    L = L[~L['Noise Reference']]
    worst = L.loc[L.groupby('Asset ID')['PD Risk (%)'].idxmax()]
    return [to_record(r.to_dict(), 'Demo substation') for _, r in worst.iterrows()]


SRC_PAGES, SRC_DEMO = "Results from the equipment pages", "Demo substation (synthetic)"


def current_records():
    if st.session_state.get('ov_src', SRC_PAGES) == SRC_DEMO:
        return demo_records()
    return list(st.session_state['register'])


def ranking_plot(R, height=None, top=20):
    t = R.sort_values('PD Risk (%)', ascending=False).head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=t['PD Risk (%)'], y=t['Asset ID'], orientation='h',
                           marker_color=[STATUS_COLOR[s] for s in t['Status']],
                           text=[f"{r:.0f}%  ·  {d}" for r, d in zip(t['PD Risk (%)'], t['Defect'])],
                           textposition='outside', cliponaxis=False, hovertemplate='%{y}: %{x:.1f}%<extra></extra>'))
    for x, tx in [(35, 'Watch'), (70, 'Action')]:
        fig.add_vline(x=x, line_dash='dash', line_color='#888', annotation_text=tx, annotation_position='top')
    fig.update_layout(title=f"Asset PD risk ranking (top {min(top, len(R))})", height=height or (140 + 30 * len(t)),
                      xaxis=dict(range=[0, 118], title='PD Risk (%)'), yaxis=dict(type='category'),
                      margin=dict(l=10, r=30, t=60, b=40))
    return fig


def type_plot(R, height=320):
    classes = [c for c in ['Transformer', 'GIS', 'Reactor', 'Cable', 'Motor'] if c in set(R['Asset Class'])]
    fig = go.Figure()
    for stt, name in [('EXCELLENT', 'Normal'), ('WATCH', 'Watch'), ('ACTION', 'Action')]:
        fig.add_bar(x=[CLASS_TITLE[c] for c in classes],
                    y=[int(((R['Asset Class'] == c) & (R['Status'] == stt)).sum()) for c in classes],
                    name=name, marker_color=STATUS_COLOR[stt], texttemplate='%{y}', textposition='inside', insidetextanchor='middle', textangle=0)
    fig.update_layout(barmode='stack', title='Assets by equipment type and status', height=height,
                      yaxis=dict(title='Assets', dtick=1 if len(R) <= 12 else None), legend=dict(orientation='h', y=-0.15), margin=dict(l=40, r=10, t=50, b=40))
    return fig


# ==========================================
# --- 4. Equipment page (one per type) ---
# ==========================================
EQUIP = {
    'Transformer': dict(asset_type='Power transformer', asset_id='T1',
                        methods={'UHF probe in the drain valve (dBm)': ('dBm', -66.0, -52.0),
                                 'Bushing tap, IEC 60270 (pC)': ('pC', 100.0, 600.0)},
                        locations=['HV winding / tank', 'HV bushing', 'Tap changer', 'Unknown'], dga=True, sf6=False, ref=True),
    'GIS': dict(asset_type='GIS compartment', asset_id='Q1-CB1',
                methods={'UHF coupler (dBm)': ('dBm', -72.0, -60.0)},
                locations=['Circuit breaker', 'Disconnector / earth switch', 'Busbar', 'CT / VT', 'Cable box', 'Spacer / barrier insulator'],
                dga=False, sf6=True, ref=True),
    'Reactor': dict(asset_type='Shunt reactor', asset_id='RB1',
                    methods={'UHF probe in the drain valve (dBm)': ('dBm', -66.0, -54.0),
                             'Bushing tap, IEC 60270 (pC)': ('pC', 100.0, 500.0)},
                    locations=['Winding / tank', 'Bushing', 'Unknown'], dga=True, sf6=False, ref=True),
    'Cable': dict(asset_type='Cable', asset_id='CABLE-1',
                  methods={'HFCT on the earth link (pC)': ('pC', 20.0, 150.0)},
                  locations=['Sealing end / termination', 'Joint', 'Cable body'], dga=False, sf6=False, ref=False),
    'Motor': dict(asset_type='HV motor', asset_id='MOTOR-1',
                  methods={'Coupling capacitors (pC)': ('pC', 50.0, 400.0)},
                  locations=['Stator slot', 'End winding', 'Terminal box'], dga=False, sf6=False, ref=False),
}


def equipment_page(cls):
    cfg = EQUIP[cls]
    k = cls.lower()
    left, right = st.columns([1, 1.8])
    with left:
        st.markdown(f"#### 📊 {CLASS_TITLE[cls]}: PD Reading")
        aid = st.text_input("Asset ID", cfg['asset_id'], key=f"{k}_id")
        loc = st.selectbox("Location", cfg['locations'], key=f"{k}_loc")
        method = st.selectbox("Measurement", list(cfg['methods']), key=f"{k}_m")
        unit, base_d, peak_d = cfg['methods'][method]
        c1, c2 = st.columns(2)
        baseline = c1.number_input(f"Sensor background ({unit})", value=base_d, key=f"{k}_b_{unit}",
                                   help="Reading of this sensor when the equipment is healthy (noise floor).")
        peak = c2.number_input(f"Today's peak ({unit})", value=peak_d, key=f"{k}_p_{unit}")
        c3, c4 = st.columns(2)
        pps = c3.number_input("Pulse rate (pulses/s)", min_value=0.0, value=150.0, step=10.0, key=f"{k}_pps")
        pd14 = c4.number_input("Days with PD in the last 14", 0, 14, 14, key=f"{k}_pd14",
                               help="How many of the last 14 days showed this pattern. Fewer than 3 = isolated reading.")
        c5, c6 = st.columns(2)
        prev = c5.number_input(f"Previous peak ({unit}, optional)", value=0.0 if unit == 'pC' else peak_d - 4,
                               key=f"{k}_prev_{unit}", help="Leave 0 if not known (pC). Used for the trend.")
        days = c6.number_input("Days since previous", min_value=1, value=30, key=f"{k}_days")
        src = st.radio("PRPD pattern", ["Shape seen on the monitoring screen", "Upload PRPD export (CSV)"],
                       key=f"{k}_src", horizontal=True)
        counts = mags = None
        if src.startswith("Shape"):
            shape = st.selectbox("Pattern shape", list(pc.PATTERN_SHAPES), key=f"{k}_shape")
        else:
            f = st.file_uploader("CSV with columns H00…H35 and M00…M35 (the last row is used)", type='csv', key=f"{k}_csv")
            if f is not None:
                try:
                    dfp = pd.read_csv(f)
                    miss = [c for c in pc.H_COLS + pc.M_COLS if c not in dfp.columns]
                    if miss:
                        st.error(f"Missing columns: {', '.join(miss[:6])} …")
                    else:
                        last = dfp.iloc[-1]
                        counts, mags = last[pc.H_COLS].to_numpy(float), last[pc.M_COLS].to_numpy(float)
                except Exception as e:
                    st.error(f"Cannot read the file: {e}")
        st.markdown("**Confirmation by other methods**")
        ac = st.selectbox("Acoustic measurement (last 90 days)", ["Not tested", "Detected", "Not detected"], key=f"{k}_ac")
        sf6 = h2 = c2h2 = ref = None
        if cfg['sf6']:
            sf6v = st.number_input("SF6 test: SO2 + SOF2 (ppmv, 0 = not tested)", min_value=0.0, value=0.0, step=0.5, key=f"{k}_sf6")
            sf6 = sf6v if sf6v > 0 else None
        if cfg['dga']:
            d1, d2 = st.columns(2)
            h2v = d1.number_input("Online DGA: H2 (ppm)", min_value=0.0, value=40.0, key=f"{k}_h2")
            c2v = d2.number_input("Online DGA: C2H2 (ppm)", min_value=0.0, value=0.0, step=0.5, key=f"{k}_c2h2")
            h2, c2h2 = h2v, c2v
        if cfg['ref'] and unit == 'dBm':
            refv = st.number_input("Noise reference antenna (dB above its background, 0 = none)", min_value=0.0, value=0.0,
                                   key=f"{k}_ref", help="If the outside antenna sees the same signal, it is interference.")
            ref = refv if refv > 0 else None
        go_btn = st.button("🔍 Analyze PD", width=STRETCH, key=f"{k}_go", type='primary')
    with right:
        if go_btn:
            if counts is None:
                if src.startswith("Shape"):
                    counts, mags = pc.example_profile(pc.PATTERN_SHAPES[shape], pps)
                else:
                    st.error("Upload the PRPD export first, or pick the pattern shape.")
                    return
            prev_peak = None if (unit == 'pC' and prev <= 0) else prev
            row, probs = pc.assess_single(MODEL, cfg['asset_type'], peak, baseline, unit, pps, counts, mags,
                                          prev_peak=prev_peak, days_since_prev=days, pd_days_14=pd14, noise_ref_db=ref,
                                          acoustic={'Detected': 'Yes', 'Not detected': 'No'}.get(ac), sf6_byproducts=sf6,
                                          h2=h2, c2h2=c2h2, asset_id=aid.strip() or cfg['asset_id'], location=loc)
            st.session_state[f"{k}_last"] = (row, probs, counts, mags)
            st.session_state['register'] = [r for r in st.session_state['register'] if r['Asset ID'] != row['Asset ID']]
            st.session_state['register'].append(to_record(row, 'Equipment page'))
            st.rerun()  # the overview and the report are drawn earlier in the page: refresh them
        last = st.session_state.get(f"{k}_last")
        if last is None:
            st.markdown("<div class='pg-note'>Enter the latest PD reading from the monitoring system on the left and press "
                        "<b>Analyze PD</b>.</div>", unsafe_allow_html=True)
        else:
            show_result(*last, k)
            st.caption("Added to the Substation Overview and the Combined Report.")


# ==========================================
# --- 5. Tabs ---
# ==========================================
tabs = st.tabs(["🏭 Substation Overview", "⚡ Transformers", "🔌 GIS", "🌀 Reactors", "〰️ Cables", "⚙️ Motors",
                "📄 Combined Report", "📈 Historical Study"])

NEED = {'sensors': ['Sensor ID', 'Asset ID', 'Asset Type', 'Location', 'Method', 'Unit', 'Baseline'],
        'online': ['Timestamp', 'Sensor ID', 'Peak', 'Pulse Rate (pps)'] + pc.H_COLS + pc.M_COLS,
        'dga': ['Asset ID', 'Timestamp', 'Hydrogen', 'Methane', 'Acethylene'],
        'acoustic': ['Date', 'Asset ID', 'Detected'], 'sf6': ['Date', 'Asset ID', 'SO2 (ppmv)', 'SOF2 (ppmv)'],
        'events': ['Asset ID', 'Date', 'Event']}
FILE_LABEL = {'sensors': 'Sensor register (required)', 'online': 'Online PD readings (required)',
              'dga': 'Online DGA (optional)', 'acoustic': 'Acoustic tests (optional)',
              'sf6': 'SF6 decomposition tests (optional)', 'events': 'Events: failures and findings (for the historical study)'}


def read_uploads(prefix):
    files, errors = {}, []
    cols = st.columns(2)
    for i, k in enumerate(NEED):
        f = cols[i % 2].file_uploader(FILE_LABEL[k], type='csv', key=f"{prefix}_{k}")
        if f is None:
            files[k] = None
            continue
        try:
            df_ = pd.read_csv(f)
        except Exception as e:
            errors.append(f"{k}: cannot read the file ({e})")
            continue
        miss = [c for c in NEED[k] if c not in df_.columns]
        if miss:
            errors.append(f"{k}: missing columns {', '.join(miss[:8])}{' …' if len(miss) > 8 else ''}")
        files[k] = df_
    return files, errors


# ---------------- Overview
with tabs[0]:
    st.radio("Show", [SRC_PAGES, SRC_DEMO], horizontal=True, key='ov_src')
    R0 = current_records()
    if st.session_state['ov_src'] == SRC_PAGES:
        st.markdown("<div class='pg-note'>Every reading analyzed on the equipment pages (Transformers, GIS, Reactors, "
                    "Cables, Motors) is collected here. The latest result per asset is kept.</div>", unsafe_allow_html=True)
        if R0 and st.button("🗑 Clear results"):
            st.session_state['register'] = []
            st.rerun()
    else:
        st.info("Synthetic demo substation (400/220 kV GIS, 3 transformers, 4 reactors, 3 cables, 2 motors), "
                "readings up to 2026-03-01. Not real measurements.")
    if not R0:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:30px'>💡 No results yet: analyze a reading on one "
                    "of the equipment pages, or show the demo substation.</h3>", unsafe_allow_html=True)
    else:
        R = pd.DataFrame(R0).sort_values('PD Risk (%)', ascending=False).reset_index(drop=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Assets", len(R))
        k2.metric("🔴 Action required", int((R['Status'] == 'ACTION').sum()))
        k3.metric("🟠 Watch", int((R['Status'] == 'WATCH').sum()))
        k4.metric("🟢 Normal", int((R['Status'] == 'EXCELLENT').sum()))
        cards = st.columns(5)
        for col, cls in zip(cards, ['Transformer', 'GIS', 'Reactor', 'Cable', 'Motor']):
            sub = R[R['Asset Class'] == cls]
            if sub.empty:
                col.markdown(f"<div class='pg-card'><h4>{CLASS_TITLE[cls]}</h4><div class='big' style='color:#999'>—</div>"
                             f"<div class='sub'>no results</div></div>", unsafe_allow_html=True)
                continue
            w = sub.iloc[0]
            c = STATUS_COLOR[w['Status']]
            col.markdown(f"<div class='pg-card' style='border-top:5px solid {c}'><h4>{CLASS_TITLE[cls]}</h4>"
                         f"<div class='big' style='color:{c}'>{w['PD Risk (%)']:.0f}%</div>"
                         f"<div class='sub'>worst: {w['Asset ID']} · {w['Defect']}<br>{len(sub)} assets · "
                         f"{int((sub['Status'] == 'ACTION').sum())} action · {int((sub['Status'] == 'WATCH').sum())} watch</div></div>",
                         unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([1.25, 1])
        with left:
            st.plotly_chart(ranking_plot(R, top=15), width=STRETCH, key='ov_rank')
        with right:
            st.plotly_chart(type_plot(R), width=STRETCH, key='ov_type')
            gis = R[R['Asset Class'] == 'GIS'].set_index('Asset ID')
            names = [f"Q{q}-CB{c}" for q in range(1, 5) for c in range(1, 4)]
            if any(n in gis.index for n in names):
                st.markdown("#### GIS diameters (breaker-and-a-half)")
                html = "<table style='width:100%;border-collapse:separate;border-spacing:6px'><tr><th></th><th>CB1</th><th>CB2 (tie)</th><th>CB3</th></tr>"
                for q in range(1, 5):
                    html += f"<tr><td><b>Q{q}</b></td>"
                    for cb in range(1, 4):
                        n = f"Q{q}-CB{cb}"
                        if n not in gis.index:
                            html += "<td style='background:#eee;text-align:center'>—</td>"
                            continue
                        r = gis.loc[n]
                        c = STATUS_COLOR[r['Status']]
                        html += (f"<td style='background:{c}22;border:2px solid {c};border-radius:6px;text-align:center;padding:6px'>"
                                 f"<b style='color:{c}'>{r['PD Risk (%)']:.0f}%</b><br><span style='font-size:12px'>{r['Defect']}</span></td>")
                    html += "</tr>"
                st.markdown(html + "</table>", unsafe_allow_html=True)
        st.markdown("#### All assets")
        show_cols = ['Asset ID', 'Asset Class', 'Location', 'Defect', 'PD Risk (%)', 'Status', 'RPN', 'RPN Level',
                     'Days to Critical', 'Source', 'Assessed', 'Action']
        st.dataframe(R[show_cols], hide_index=True, width=STRETCH)
        st.download_button("📥 Download asset ranking (CSV)", R[show_cols].to_csv(index=False).encode('utf-8-sig'),
                           f"PDGuard_overview_{datetime.now():%Y%m%d}.csv", 'text/csv')

# ---------------- Equipment pages
for tab, cls in zip(tabs[1:6], ['Transformer', 'GIS', 'Reactor', 'Cable', 'Motor']):
    with tab:
        equipment_page(cls)

# ---------------- Combined report
with tabs[6]:
    recs = current_records()
    if not recs:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:40px'>💡 Analyze readings on the equipment pages "
                    "(or show the demo substation on the Overview page) first.</h3>", unsafe_allow_html=True)
    else:
        R = pd.DataFrame(recs).sort_values('PD Risk (%)', ascending=False).reset_index(drop=True)
        n_a, n_w = int((R['Status'] == 'ACTION').sum()), int((R['Status'] == 'WATCH').sum())
        st.markdown(f"<div class='pg-note'><b>📄 Combined PD Diagnostic Report</b><br>{len(R)} assets · source: "
                    f"{st.session_state.get('ov_src', SRC_PAGES).lower()} · generated {datetime.now():%Y-%m-%d %H:%M}</div>",
                    unsafe_allow_html=True)
        if n_a:
            summary = (f"{n_a} asset(s) need action and {n_w} are on watch. Highest risk: {R.iloc[0]['Asset ID']} "
                       f"({R.iloc[0]['Defect']}, {R.iloc[0]['PD Risk (%)']:.0f}%). {R.iloc[0]['Action']}")
        elif n_w:
            summary = f"No asset needs immediate action; {n_w} asset(s) are on watch. Highest: {R.iloc[0]['Asset ID']} ({R.iloc[0]['Defect']})."
        else:
            summary = "No significant partial discharge activity on any assessed asset."
        st.markdown(f"**Executive summary:** {summary}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Assets", len(R))
        k2.metric("🔴 Action", n_a)
        k3.metric("🟠 Watch", n_w)
        k4.metric("🟢 Normal", len(R) - n_a - n_w)
        c1, c2 = st.columns([1.25, 1])
        with c1:
            st.plotly_chart(ranking_plot(R, top=15), width=STRETCH, key='rep_rank')
        with c2:
            st.plotly_chart(type_plot(R), width=STRETCH, key='rep_type')
        cols = ['Asset ID', 'Asset Class', 'Defect', 'PD Risk (%)', 'Status', 'RPN', 'RPN Level', 'Days to Critical', 'Source', 'Assessed', 'Action']
        st.markdown("#### All assets")
        st.dataframe(R[cols], hide_index=True, width=STRETCH)
        st.markdown("#### Assets above the normal band")
        above = R[R['Status'] != 'EXCELLENT']
        if above.empty:
            st.success("No asset above the normal band.")
        for i, (_, r) in enumerate(above.iterrows()):
            with st.expander(f"{r['Asset ID']}  ·  {r['Defect']}  ·  {r['PD Risk (%)']:.0f}% {r['Status']}", expanded=i < 3):
                e1, e2 = st.columns([1, 1.2])
                with e1:
                    for t in r['Findings']:
                        st.markdown(f"- {t}")
                    st.markdown(f"**RPN** = {r['S']} × {r['O']} × {r['D']} = **{r['RPN']}** ({r['RPN Level']})")
                with e2:
                    st.plotly_chart(contrib_plot(r, 280), width=STRETCH, key=f"rep_c_{i}")
                st.warning(f"**Action:** {r['Action']}")
        lines = ["=========================================", "PDGUARD AI - COMBINED PD DIAGNOSTIC REPORT",
                 "=========================================", f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
                 f"Assets: {len(R)}  (action {n_a}, watch {n_w})", "", "EXECUTIVE SUMMARY:", summary, ""]
        for _, r in R.iterrows():
            lines += ["-----------------------------------------",
                      f"{r['Asset ID']} ({CLASS_TITLE.get(r['Asset Class'], r['Asset Class'])}) - {r['Status']} - PD risk {r['PD Risk (%)']:.0f}%",
                      f"Defect: {r['Defect']}  |  RPN = {r['S']} x {r['O']} x {r['D']} = {r['RPN']} ({r['RPN Level']})",
                      f"Source: {r['Source']} ({r['Assessed']})"] + [f"* {t}" for t in r['Findings']] + [f"Action: {r['Action']}"]
        lines += ["=========================================", "Generated by PDGuard AI Engine"]
        d1, d2 = st.columns(2)
        d1.download_button("📥 Download Full Text Report", "\n".join(lines).encode('utf-8'),
                           f"PDGuard_Report_{datetime.now():%Y%m%d}.txt", 'text/plain')
        d2.download_button("📥 Download Table (CSV)", R[cols].to_csv(index=False).encode('utf-8-sig'),
                           f"PDGuard_Report_{datetime.now():%Y%m%d}.csv", 'text/csv')

# ---------------- Historical study
with tabs[7]:
    st.markdown("<div class='pg-note'>Replays every reading in date order. Each assessment uses only the data that "
                "existed on that date, then the alerts are compared with the recorded failures and major maintenance "
                "findings. The result shows how early the system would have warned.</div>", unsafe_allow_html=True)
    hsrc = st.radio("Data", ["Demo substation (synthetic, Jan 2025 - Aug 2026)", "Upload client files"], horizontal=True, key='h_src')
    hfiles, herr = (None, [])
    if hsrc.startswith("Upload"):
        hfiles, herr = read_uploads('h')
        with st.expander("📄 File formats and demo files"):
            st.markdown("""
- **Sensor register:** `Sensor ID, Asset ID, Asset Type, Location, Method, Unit, Baseline, Noise Reference`.
  `Asset Type` starting with *GIS*, or containing *Transformer*, *Reactor*, *Cable* or *Motor*. `Unit` = `dBm` (UHF) or `pC`.
  `Baseline` = sensor background in the same unit. `Noise Reference` = Yes for an antenna that only sees outside interference.
- **Online PD readings:** `Timestamp, Sensor ID, Peak, Pulse Rate (pps)`, then `H00…H35` = pulses in each 10° window of the
  voltage cycle (PRPD phase profile) and `M00…M35` = mean pulse magnitude in that window (0-1).
- **Online DGA:** `Asset ID, Timestamp, Hydrogen, Methane, Acethylene` (same names as TransformerGuard).
- **Acoustic tests:** `Date, Asset ID, Detected (Yes/No), Amplitude (mV)`.  **SF6 tests:** `Date, Asset ID, SO2 (ppmv), SOF2 (ppmv), HF (ppmv)`.
- **Events:** `Asset ID, Date, Event`.
""")
            demo = demo_full()
            dc = st.columns(6)
            for col, kk in zip(dc, NEED):
                col.download_button(kk.capitalize(), demo[kk].to_csv(index=False).encode('utf-8-sig'), f"demo_{kk}.csv",
                                    'text/csv', key=f"dl_{kk}")
    h1, h2 = st.columns(2)
    thr = h1.number_input("Alert threshold (%)", 50, 95, 70, 5)
    lb = h2.number_input("Look-back window (days)", 30, 1095, 365, 30)
    if st.button("▶️ Run Historical Study", type='primary', key='h_go'):
        if hsrc.startswith("Demo"):
            data = demo_full()
        else:
            for e in herr:
                st.error(e)
            data = hfiles if not herr else None
            if data is not None and (data['sensors'] is None or data['online'] is None):
                st.error("Load at least the sensor register and the online PD readings, plus the events file.")
                data = None
        if data is not None:
            if 'Noise Reference' not in data['sensors'].columns:
                data['sensors'] = data['sensors'].assign(**{'Noise Reference': 'No'})
            A = run_assessment(data['online'], data['sensors'], data.get('dga'), data.get('acoustic'), data.get('sf6'))
            tl, evs, fa, n_ep = pc.historical_study(A, data.get('events'), thr, lb)
            st.session_state['hist'] = dict(tl=tl, evs=evs, fa=fa, n_ep=n_ep, thr=thr, events=data.get('events'),
                                            demo=hsrc.startswith("Demo"))
    H = st.session_state['hist']
    if H is not None:
        tl, evs = H['tl'], H['evs']
        if H['demo']:
            st.info("Synthetic demo data. Not real measurements.")
        caught = int((evs['Caught'] == 'Yes').sum()) if len(evs) else 0
        lt = evs['Lead Time (days)'].dropna() if len(evs) else pd.Series(dtype=float)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Assets", tl['Asset ID'].nunique())
        k2.metric("Events caught", f"{caught} / {len(evs)}" if len(evs) else "no events")
        k3.metric("Median warning before event", f"{int(lt.median())} days" if len(lt) else "—")
        k4.metric("Alert episodes with no event", f"{H['fa']} of {H['n_ep']}")
        show = tl.groupby('Asset ID')['PD Risk (%)'].max()
        show = show[show > pc.EXCELLENT_MAX].index.tolist()
        fig = go.Figure()
        pal = ['#2a78d6', '#eb6834', '#1baf7a', '#8e5cd6', '#e87ba4', '#555555', '#eda100', '#008300']
        for i, aid in enumerate(show):
            g = tl[tl['Asset ID'] == aid]
            fig.add_scatter(x=g['Date'], y=g['PD Risk (%)'], mode='lines', name=aid, line=dict(color=pal[i % len(pal)]))
        fig.add_hline(y=H['thr'], line_dash='dash', line_color=RED, annotation_text=f"Alert threshold {H['thr']}%",
                      annotation_position='top left')
        if H['events'] is not None:
            for _, e in H['events'].iterrows():
                d_ = pd.to_datetime(e['Date'], errors='coerce')
                if pd.notna(d_):
                    fig.add_vline(x=d_, line_dash='dot', line_color='#555')
                    fig.add_annotation(x=d_, y=104, text=f"⚠ {e['Asset ID']}", showarrow=False)
        fig.update_layout(title='PD risk over time (assets that ever left the normal band)', height=440,
                          yaxis=dict(range=[0, 110], title='PD Risk (%)'), legend=dict(orientation='h', y=-0.2),
                          margin=dict(l=50, r=20, t=50, b=40))
        st.plotly_chart(fig, width=STRETCH)
        if len(evs):
            st.markdown("#### Events vs. warnings")
            st.dataframe(evs, hide_index=True, width=STRETCH)
        st.caption("'Alert episodes with no event' counts alert periods (alerts less than 120 days apart form one episode) "
                   "not followed by a recorded event within the look-back window. Interference that the noise reference "
                   "antenna also saw is suppressed before this count.")
        st.download_button("📥 Download daily asset timeline (CSV)", tl.to_csv(index=False).encode('utf-8-sig'),
                           'pdguard_historical_timeline.csv', 'text/csv')
    with st.expander("How the PD risk is calculated"):
        st.markdown(f"""
1. **Level** above the sensor background (dB). Below {pc.DETECT_DB:.0f} dB there is no PD; {pc.FULL_DB:.0f} dB gives a full score.
2. **Trend** ({pc.TREND_FULL:.0f} dB per month gives a full score) and **pulse rate**.
3. **Defect type** from the PRPD pattern (machine-learning classifier) and its **severity** at that equipment.
4. **Confirmation**: acoustic detection, SF6 by-products (≥ {pc.SF6_INDICATIVE:.0f} ppmv indicative, ≥ {pc.SF6_HIGH:.0f} ppmv high)
   and online DGA (H2 / C2H2 above IEEE C57.104 condition 1) each raise the risk.
5. **Safety rules**: a confirmed critical defect is at least 75%; a strong free particle in GIS is at least 72%;
   a persistent critical defect above 6 dB stays at least on watch; signals the noise reference antenna also sees are suppressed;
   a pattern seen on fewer than {pc.PERSIST_MIN_DAYS} of the last {pc.PERSIST_WINDOW_D} days is limited to 30%.
6. Bands: ≤ 35% normal, ≤ 70% watch, > 70% action. FMEA: RPN = S × O × D.

The defect classifier is trained on synthetic PRPD patterns built from the published signature of each defect type.
It must be re-trained on labelled field data before any accuracy figure is quoted.
""")

st.markdown(FOOTER, unsafe_allow_html=True)
