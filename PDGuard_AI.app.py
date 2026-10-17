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
STATUS_LABEL = {'EXCELLENT': 'NORMAL (Low PD Risk)', 'WATCH': 'WATCH (Elevated PD Risk)', 'ACTION': 'ACTION REQUIRED (High PD Risk)'}

st.markdown("""
<style>
.pg-header {background:#0b2e59;padding:18px 22px;border-radius:10px;margin-bottom:18px;}
.pg-header h1 {color:#fff;margin:0;font-size:1.9rem;}
.pg-header p {color:#9fd0ff;margin:4px 0 0 0;font-size:1rem;}
.pg-banner {border:2px solid;border-radius:8px;padding:10px 14px;text-align:center;margin:6px 0 12px 0;}
.pg-banner h2 {margin:0;font-size:1.5rem;}
.pg-banner p {margin:2px 0 0 0;font-size:.95rem;}
.pg-note {background:rgba(77,166,255,.10);border-left:4px solid #4da6ff;padding:8px 12px;border-radius:4px;font-size:.92rem;margin-bottom:10px;}
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
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

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
# --- 2. Header, model and data ---
# ==========================================
_, col_r = st.columns([10, 1])
with col_r:
    if st.button("Logout 🚪"):
        st.session_state['logged_in'] = False
        st.rerun()

st.markdown("""
<div class='pg-header'>
  <h1>⚡ PDGuard AI: Partial Discharge Risk Monitor</h1>
  <p>GIS, transformers, reactors and cables: online UHF / HFCT patterns, confirmed by acoustic, SF6 and DGA, ranked by risk</p>
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
def demo_bundle():
    return pc.demo_data()


@st.cache_data(show_spinner="Assessing every reading…")
def run_assessment(online, sensors, dga, acoustic, sf6):
    return pc.assess(online, sensors, MODEL, dga, acoustic, sf6)


if 'data' not in st.session_state:
    st.session_state['data'] = demo_bundle()
    st.session_state['is_demo'] = True

DATA = st.session_state['data']
try:
    A = run_assessment(DATA['online'], DATA['sensors'], DATA.get('dga'), DATA.get('acoustic'), DATA.get('sf6'))
except Exception as e:
    st.error(f"The loaded data could not be assessed: {e}")
    st.stop()
if A.empty:
    st.error("No readings matched the sensor register. Check the Sensor ID values in both files.")
    st.stop()

DAYS = sorted(A['Timestamp'].dt.floor('D').unique())
DEFAULT_DAY = DAYS[-1]
if st.session_state.get('is_demo') and pd.Timestamp('2026-03-01') in DAYS:
    DEFAULT_DAY = DAYS[DAYS.index(pd.Timestamp('2026-03-01'))]
EVENTS = DATA.get('events')

if st.session_state.get('is_demo'):
    st.markdown("<div class='pg-note'>Showing the <b>synthetic demo substation</b> (400/220 kV GIS, 4 diameters, "
                "3 transformers, 4 reactors, 3 cable sealing ends). These are not real measurements. The date sliders start "
                "on 1 March 2026, while three defects are active; move them to replay the history. Load client files "
                "in the <b>Data &amp; Formats</b> tab.</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏭 Substation Overview", "🔎 Sensor Analysis", "📈 Historical Study",
                                        "📄 Diagnostic Report", "📥 Data & Formats"])


def as_of_rows(day):
    sub = A[A['Timestamp'] <= pd.Timestamp(day) + pd.Timedelta(days=1)]
    return pc.latest(sub)


def phase_plot(row, height=340):
    h = row[pc.H_COLS].to_numpy(float)
    m = row[pc.M_COLS].to_numpy(float)
    fig = go.Figure()
    fig.add_bar(x=pc.PHASE_DEG, y=h, width=9, marker=dict(color=m, colorscale='Turbo', cmin=0, cmax=1,
                colorbar=dict(title='Relative<br>magnitude', thickness=12)), name='Pulses',
                hovertemplate='%{x}°: %{y:.0f} pulses<extra></extra>')
    x = np.linspace(0, 360, 181)
    top = max(h.max(), 1)
    fig.add_scatter(x=x, y=top / 2 + top / 2 * np.sin(np.radians(x)), mode='lines', name='Supply voltage',
                    line=dict(color='#555', dash='dash'), hoverinfo='skip')
    fig.update_layout(title='PRPD phase profile (pulses per 10° of the voltage cycle)', height=height,
                      xaxis=dict(title='Phase angle (°)', tickvals=[0, 90, 180, 270, 360], range=[0, 360]),
                      yaxis_title='Pulses in the window', showlegend=False, margin=dict(l=40, r=10, t=50, b=40))
    return fig


def gauge(value, color, title='PD Risk (%)', height=260):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=float(value), title={'text': title},
                                 gauge={'axis': {'range': [0, 100], 'tickvals': [0, 35, 70, 100]},
                                        'bar': {'color': color},
                                        'steps': [{'range': [0, 35], 'color': 'rgba(26,158,92,.25)'},
                                                  {'range': [35, 70], 'color': 'rgba(217,154,0,.25)'},
                                                  {'range': [70, 100], 'color': 'rgba(214,58,58,.25)'}]}))
    fig.update_layout(height=height, margin=dict(l=25, r=25, t=50, b=10))
    return fig


# ==========================================
# --- Tab 1: Overview ---
# ==========================================
with tab1:
    day = st.select_slider("Assessment date (the view uses only data up to this day)", options=DAYS, value=DEFAULT_DAY,
                           format_func=lambda d: pd.Timestamp(d).strftime('%Y-%m-%d'))
    L = as_of_rows(day)
    S_ = pc.asset_summary(L)
    recent = A[(A['Timestamp'] <= pd.Timestamp(day) + pd.Timedelta(days=1)) &
               (A['Timestamp'] > pd.Timestamp(day) - pd.Timedelta(days=30))]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sensors", int((~L['Noise Reference']).sum()))
    c2.metric("Assets", S_['Asset ID'].nunique())
    c3.metric("🔴 Action required", int((S_['Status'] == 'ACTION').sum()))
    c4.metric("🟠 Watch", int((S_['Status'] == 'WATCH').sum()))
    c5.metric("Interference alarms suppressed (30 days)", int(recent['Noise Suppressed'].sum()))

    left, right = st.columns([1.25, 1])
    with left:
        top = S_.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=top['PD Risk (%)'], y=top['Asset ID'], orientation='h',
                               marker_color=[STATUS_COLOR[s] for s in top['Status']],
                               text=[f"{r:.0f}%  ·  {d}" for r, d in zip(top['PD Risk (%)'], top['Defect'])],
                               textposition='outside', cliponaxis=False,
                               hovertemplate='%{y}: %{x:.1f}%<extra></extra>'))
        for x, t in [(35, 'Watch'), (70, 'Action')]:
            fig.add_vline(x=x, line_dash='dash', line_color='#888', annotation_text=t, annotation_position='top')
        fig.update_layout(title='Asset PD risk ranking (worst sensor per asset, top 15)', height=520,
                          xaxis=dict(range=[0, 118], title='PD Risk (%)'), yaxis=dict(type='category'),
                          margin=dict(l=10, r=30, t=60, b=40))
        st.plotly_chart(fig, width=STRETCH)
    with right:
        st.markdown("#### GIS diameters (breaker-and-a-half)")
        gis = L[L['Asset Class'] == 'GIS'].set_index('Asset ID')
        cells = []
        for q in range(1, 5):
            row = []
            for cb in range(1, 4):
                aid = f"Q{q}-CB{cb}"
                row.append(gis.loc[aid] if aid in gis.index else None)
            cells.append(row)
        if any(r is not None for row in cells for r in row):
            html = "<table style='width:100%;border-collapse:separate;border-spacing:6px'><tr><th></th><th>CB1</th><th>CB2 (tie)</th><th>CB3</th></tr>"
            for q, row in enumerate(cells, 1):
                html += f"<tr><td><b>Q{q}</b></td>"
                for r in row:
                    if r is None:
                        html += "<td style='background:#eee;text-align:center'>—</td>"
                    else:
                        col = STATUS_COLOR[r['Status']]
                        html += (f"<td style='background:{col}22;border:2px solid {col};border-radius:6px;text-align:center;padding:6px'>"
                                 f"<b style='color:{col}'>{r['PD Risk (%)']:.0f}%</b><br><span style='font-size:12px'>{r['Defect']}</span></td>")
                html += "</tr>"
            st.markdown(html + "</table>", unsafe_allow_html=True)
        else:
            st.caption("No GIS breaker sensors named Qn-CBm in this data set.")
        st.markdown("#### Other equipment")
        oth = S_[~S_['Asset ID'].str.match(r'^Q\d-CB\d$')][['Asset ID', 'Asset Type', 'Defect', 'PD Risk (%)', 'Status']]
        st.dataframe(oth, hide_index=True, width=STRETCH, height=300)

    st.markdown("#### Asset register: latest PD assessment")
    st.dataframe(S_.round({'Level (dB, 7d median)': 1, 'Trend (dB/30d)': 2}), hide_index=True, width=STRETCH)
    st.download_button("📥 Download asset ranking (CSV)", S_.to_csv(index=False).encode('utf-8-sig'),
                       f"PDGuard_ranking_{pd.Timestamp(day):%Y%m%d}.csv", 'text/csv')

# ==========================================
# --- Tab 2: Sensor analysis ---
# ==========================================
with tab2:
    day2 = st.select_slider("Assessment date", options=DAYS, value=DEFAULT_DAY, key='day2',
                            format_func=lambda d: pd.Timestamp(d).strftime('%Y-%m-%d'))
    L2 = as_of_rows(day2).sort_values('PD Risk (%)', ascending=False)
    labels = {r['Sensor ID']: f"{r['Sensor ID']}  ·  {r['Asset ID']}  ·  {r['PD Risk (%)']:.0f}% {r['Status']}" for _, r in L2.iterrows()}
    sid = st.selectbox("Sensor (sorted by risk)", list(labels), format_func=lambda s: labels[s])
    row = L2[L2['Sensor ID'] == sid].iloc[0]
    color = STATUS_COLOR[row['Status']]
    banner(STATUS_LABEL[row['Status']], f"{row['Location']}  ·  {pc.DEFECT_TEXT[row['Defect']]}", color)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Level above background", f"{row['Level (dB, 7d median)']:.1f} dB")
    m2.metric("Trend", f"{row['Trend (dB/30d)']:+.1f} dB / month")
    m3.metric("Pulse rate", f"{row['Pulse Rate (pps)']:.0f} /s")
    m4.metric("Pattern confidence", f"{row['Confidence (%)']:.0f} %" if row['Defect'] != pc.NO_PD else "—")
    m5.metric("Confirmations", f"{int(row['Confirmations'])} of 3")
    ca, cb = st.columns([1.5, 1])
    with ca:
        st.plotly_chart(phase_plot(row), width=STRETCH)
    with cb:
        st.plotly_chart(gauge(row['PD Risk (%)'], color), width=STRETCH)
        X = pc.features_from_profile(row[pc.H_COLS].to_numpy(float)[None], row[pc.M_COLS].to_numpy(float)[None])
        P = MODEL.predict_proba(X)[0]
        order = np.argsort(P)
        fp = go.Figure(go.Bar(x=P[order] * 100, y=np.array(MODEL.classes_)[order], orientation='h', marker_color=BLUE,
                              hovertemplate='%{y}: %{x:.0f}%<extra></extra>'))
        fp.update_layout(title='Pattern match by defect type', height=250, xaxis=dict(range=[0, 100], title='%'),
                         margin=dict(l=10, r=10, t=40, b=30))
        st.plotly_chart(fp, width=STRETCH)
    st.markdown("#### Why this result")
    for t in pc.reasons(row):
        st.markdown(f"- {t}")
    st.warning(f"**Action:** {row['Action']}")

    hist = A[(A['Sensor ID'] == sid) & (A['Timestamp'] <= pd.Timestamp(day2) + pd.Timedelta(days=1))]
    ft = go.Figure()
    ft.add_scatter(x=hist['Timestamp'], y=hist['dB Above Baseline'], mode='markers', name='Daily peak', marker=dict(color='#9aa7b8', size=3))
    ft.add_scatter(x=hist['Timestamp'], y=hist['Level (dB, 7d median)'], mode='lines', name='Level (7-day median)', line=dict(color=NAVY))
    ft.add_scatter(x=hist['Timestamp'], y=hist['PD Risk (%)'] / 100 * 30, mode='lines', name='PD risk (scaled: 30 = 100%)',
                   line=dict(color=RED, width=1), opacity=.6)
    ft.add_hline(y=pc.DETECT_DB, line_dash='dot', line_color='#888', annotation_text='Detection')
    ft.add_hline(y=pc.CRITICAL_DB, line_dash='dash', line_color=RED, annotation_text='Critical level')
    tests = []
    for name, df_ in [('Acoustic', DATA.get('acoustic')), ('SF6 test', DATA.get('sf6'))]:
        if df_ is not None and len(df_):
            for _, t in df_[df_['Asset ID'] == row['Asset ID']].iterrows():
                tests.append((pd.to_datetime(t['Date']), name))
    for d_, n_ in tests:
        if d_ <= pd.Timestamp(day2):
            ft.add_vline(x=d_, line_color=BLUE, line_dash='dot')
            ft.add_annotation(x=d_, y=1, yref='paper', text=n_, showarrow=False, font=dict(size=10, color=BLUE))
    if EVENTS is not None and len(EVENTS):
        for _, e in EVENTS[EVENTS['Asset ID'] == row['Asset ID']].iterrows():
            d_ = pd.to_datetime(e['Date'], errors='coerce')
            if pd.notna(d_) and d_ <= pd.Timestamp(day2):
                ft.add_vline(x=d_, line_color=RED, line_width=2)
                ft.add_annotation(x=d_, y=0.92, yref='paper', text='⚠ Event', showarrow=False, font=dict(color=RED))
    ft.update_layout(title='PD magnitude trend', height=380, yaxis_title='dB above sensor background',
                     legend=dict(orientation='h', y=-0.2), margin=dict(l=50, r=20, t=50, b=40))
    st.plotly_chart(ft, width=STRETCH)

    st.markdown("#### 🧮 FMEA: RPN = Severity × Occurrence × Detection")
    st.dataframe(pd.DataFrame([{'Failure Mode': f"{row['Defect']} at {row['Asset Type']}", 'S': row['S'], 'O': row['O'],
                                'D': row['D'], 'RPN': row['RPN'], 'RPN Level': row['RPN Level']}]), hide_index=True, width=STRETCH)
    st.caption("S = severity of this defect type at this equipment (a free particle in GIS is the highest). "
               "O = occurrence from the PD risk (risk ÷ 10). D = 2 for continuous online monitoring. "
               "RPN levels: Low < 40, Medium 40-99, High ≥ 100.")
    conf = pd.DataFrame([
        {'Method': 'Acoustic (periodic)', 'Latest result': row['Acoustic'] or 'No test in the last 90 days'},
        {'Method': 'SF6 decomposition, SO2 + SOF2 (periodic, GIS)',
         'Latest result': 'No test in the last 90 days' if pd.isna(row['SF6 By-products (ppmv)']) else f"{row['SF6 By-products (ppmv)']:.1f} ppmv"},
        {'Method': 'Online DGA, H2 / C2H2 (transformers and reactors)',
         'Latest result': 'Not available' if pd.isna(row['H2 (ppm)']) else f"H2 {row['H2 (ppm)']:.0f} ppm, C2H2 {row['C2H2 (ppm)']:.1f} ppm"},
    ])
    st.markdown("#### Confirmation by other methods")
    st.dataframe(conf, hide_index=True, width=STRETCH)

# ==========================================
# --- Tab 3: Historical study ---
# ==========================================
with tab3:
    st.markdown("<div class='pg-note'>Replays every reading in date order. Each assessment uses only the data that "
                "existed on that date, then the alerts are compared with the recorded failures and major maintenance "
                "findings. The result shows how early the system would have warned.</div>", unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    thr = h1.slider("Alert threshold (%)", 50, 95, 70, 5)
    lb = h2.slider("Look-back window (days)", 30, 730, 365, 30)
    tl, evs, fa, n_ep = pc.historical_study(A, EVENTS, thr, lb)
    caught = int((evs['Caught'] == 'Yes').sum()) if len(evs) else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assets", tl['Asset ID'].nunique())
    k2.metric("Events caught", f"{caught} / {len(evs)}" if len(evs) else "no events")
    lt = evs['Lead Time (days)'].dropna() if len(evs) else pd.Series(dtype=float)
    k3.metric("Median warning before event", f"{int(lt.median())} days" if len(lt) else "—")
    k4.metric("Alert episodes with no event", f"{fa} of {n_ep}")
    show = tl.groupby('Asset ID')['PD Risk (%)'].max()
    show = show[show > pc.EXCELLENT_MAX].index.tolist()
    fig = go.Figure()
    pal = ['#2a78d6', '#eb6834', '#1baf7a', '#8e5cd6', '#e87ba4', '#555555', '#eda100', '#008300']
    for i, aid in enumerate(show):
        g = tl[tl['Asset ID'] == aid]
        fig.add_scatter(x=g['Date'], y=g['PD Risk (%)'], mode='lines', name=aid, line=dict(color=pal[i % len(pal)]))
    fig.add_hline(y=thr, line_dash='dash', line_color=RED, annotation_text=f"Alert threshold {thr}%", annotation_position='top left')
    if EVENTS is not None:
        for _, e in EVENTS.iterrows():
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
               "sensor also saw is suppressed before this count.")
    st.download_button("📥 Download daily asset timeline (CSV)", tl.to_csv(index=False).encode('utf-8-sig'),
                       'pdguard_historical_timeline.csv', 'text/csv')

# ==========================================
# --- Tab 4: Report ---
# ==========================================
with tab4:
    day4 = st.select_slider("Report date", options=DAYS, value=DEFAULT_DAY, key='day4',
                            format_func=lambda d: pd.Timestamp(d).strftime('%Y-%m-%d'))
    L4 = as_of_rows(day4)
    S4 = pc.asset_summary(L4)
    aid = st.selectbox("Asset", S4['Asset ID'].tolist(),
                       format_func=lambda a: f"{a}  ·  {S4.set_index('Asset ID').loc[a, 'PD Risk (%)']:.0f}%")
    rows = L4[L4['Asset ID'] == aid].sort_values('PD Risk (%)', ascending=False)
    worst = rows.iloc[0]
    color = STATUS_COLOR[worst['Status']]
    st.markdown(f"<div class='pg-note'><b>📄 PD Diagnostic Report</b><br>Asset: <b>{aid}</b> ({worst['Asset Type']}) · "
                f"Data up to {pd.Timestamp(day4):%Y-%m-%d}</div>", unsafe_allow_html=True)
    if worst['Status'] == 'EXCELLENT':
        summary = f"{aid} shows no significant partial discharge activity (PD risk {worst['PD Risk (%)']:.0f}%). {worst['Action']}"
    else:
        summary = (f"{aid}: {pc.DEFECT_TEXT[worst['Defect']].lower()}, PD risk {worst['PD Risk (%)']:.0f}% "
                   f"({worst['Status']}), RPN {worst['RPN']} ({worst['RPN Level']}). {worst['Action']}")
    st.markdown(f"**Executive summary:** {summary}")
    rc1, rc2 = st.columns([1, 1.6])
    with rc1:
        st.plotly_chart(gauge(worst['PD Risk (%)'], color, 'Asset PD Risk (%)'), width=STRETCH)
    with rc2:
        banner(STATUS_LABEL[worst['Status']], f"Worst sensor {worst['Sensor ID']} · RPN {worst['RPN']} ({worst['RPN Level']})", color)
        for t in pc.reasons(worst):
            st.markdown(f"- {t}")
    st.dataframe(rows[['Sensor ID', 'Location', 'Defect', 'Level (dB, 7d median)', 'Trend (dB/30d)', 'PD Risk (%)', 'Status']]
                 .round(1), hide_index=True, width=STRETCH)
    rep = ["=========================================", "PDGUARD AI - PARTIAL DISCHARGE DIAGNOSTIC REPORT",
           "=========================================",
           f"Asset: {aid} ({worst['Asset Type']})", f"Data up to: {pd.Timestamp(day4):%Y-%m-%d}",
           f"Generated: {datetime.now():%Y-%m-%d %H:%M}", "", "EXECUTIVE SUMMARY:", summary, "",
           "SENSORS", "-----------------------------------------"]
    for _, r in rows.iterrows():
        rep.append(f"{r['Sensor ID']} ({r['Location']}): {r['Defect']}, {r['Level (dB, 7d median)']:.1f} dB above background, "
                   f"trend {r['Trend (dB/30d)']:+.1f} dB/month, risk {r['PD Risk (%)']:.0f}% {r['Status']}")
    rep += ["", "FINDINGS (worst sensor)", "-----------------------------------------"] + [f"* {t}" for t in pc.reasons(worst)]
    rep += ["", "FMEA", "-----------------------------------------",
            f"Failure mode: {worst['Defect']} at {worst['Asset Type']}",
            f"RPN = S {worst['S']} x O {worst['O']} x D {worst['D']} = {worst['RPN']} ({worst['RPN Level']})",
            f"Action: {worst['Action']}", "=========================================", "Generated by PDGuard AI Engine"]
    st.download_button("📥 Download Text Report", "\n".join(rep).encode('utf-8'), f"PD_Report_{aid}_{pd.Timestamp(day4):%Y%m%d}.txt", 'text/plain')

# ==========================================
# --- Tab 5: Data & formats ---
# ==========================================
with tab5:
    st.markdown("#### Load client data")
    st.markdown("<div class='pg-note'>The online files are the CSV exports of the PD monitoring system and the online "
                "DGA monitors. Acoustic and SF6 results come from periodic tests. Only the sensor register and the "
                "online PD file are required.</div>", unsafe_allow_html=True)
    u1, u2 = st.columns(2)
    f_sens = u1.file_uploader("Sensor register (required)", type='csv', key='f_sens')
    f_on = u1.file_uploader("Online PD readings (required)", type='csv', key='f_on')
    f_dga = u1.file_uploader("Online DGA (optional)", type='csv', key='f_dga')
    f_ac = u2.file_uploader("Acoustic tests (optional)", type='csv', key='f_ac')
    f_sf = u2.file_uploader("SF6 decomposition tests (optional)", type='csv', key='f_sf')
    f_ev = u2.file_uploader("Events: failures and major findings (optional)", type='csv', key='f_ev')
    b1, b2 = st.columns(2)
    if b1.button("✅ Use the uploaded files", width=STRETCH):
        need = {'sensors': ['Sensor ID', 'Asset ID', 'Asset Type', 'Location', 'Method', 'Unit', 'Baseline'],
                'online': ['Timestamp', 'Sensor ID', 'Peak', 'Pulse Rate (pps)'] + pc.H_COLS + pc.M_COLS,
                'dga': ['Asset ID', 'Timestamp', 'Hydrogen', 'Methane', 'Acethylene'],
                'acoustic': ['Date', 'Asset ID', 'Detected'], 'sf6': ['Date', 'Asset ID', 'SO2 (ppmv)', 'SOF2 (ppmv)'],
                'events': ['Asset ID', 'Date', 'Event']}
        files = {'sensors': f_sens, 'online': f_on, 'dga': f_dga, 'acoustic': f_ac, 'sf6': f_sf, 'events': f_ev}
        if f_sens is None or f_on is None:
            st.error("Load at least the sensor register and the online PD readings.")
        else:
            new, errors = {}, []
            for k, f in files.items():
                if f is None:
                    new[k] = None
                    continue
                try:
                    df_ = pd.read_csv(f)
                except Exception as e:
                    errors.append(f"{k}: cannot read the file ({e})")
                    continue
                miss = [c for c in need[k] if c not in df_.columns]
                if miss:
                    errors.append(f"{k}: missing columns {', '.join(miss[:8])}{' …' if len(miss) > 8 else ''}")
                new[k] = df_
            if errors:
                for e in errors:
                    st.error(e)
            else:
                if 'Noise Reference' not in new['sensors'].columns:
                    new['sensors']['Noise Reference'] = 'No'
                st.session_state['data'] = new
                st.session_state['is_demo'] = False
                st.rerun()
    if b2.button("↩️ Back to demo data", width=STRETCH):
        st.session_state['data'] = demo_bundle()
        st.session_state['is_demo'] = True
        st.rerun()

    st.markdown("#### File formats")
    st.markdown(f"""
- **Sensor register:** `Sensor ID, Asset ID, Asset Type, Location, Method, Unit, Baseline, Noise Reference`.
  `Asset Type` starting with *GIS*, or containing *Transformer*, *Reactor* or *Cable*. `Unit` is `dBm` (UHF) or `pC` (HFCT / IEC 60270).
  `Baseline` is the sensor background in the same unit. `Noise Reference` = Yes for an antenna that only sees outside interference.
- **Online PD readings:** `Timestamp, Sensor ID, Peak, Pulse Rate (pps)`, then `H00…H35` = pulses in each 10° window of the
  voltage cycle (the PRPD phase profile) and `M00…M35` = mean pulse magnitude in that window (0-1 of the window peak).
- **Online DGA:** `Asset ID, Timestamp, Hydrogen, Methane, Acethylene` (same names as TransformerGuard).
- **Acoustic tests:** `Date, Asset ID, Detected (Yes/No), Amplitude (mV)`.
- **SF6 decomposition tests:** `Date, Asset ID, SO2 (ppmv), SOF2 (ppmv), HF (ppmv)`.
- **Events:** `Asset ID, Date, Event`.
""")
    st.markdown("#### Demo files (open in Excel)")
    demo = demo_bundle()
    dcols = st.columns(6)
    for col, (k, name) in zip(dcols, [('sensors', 'Sensor register'), ('online', 'Online PD'), ('dga', 'Online DGA'),
                                      ('acoustic', 'Acoustic tests'), ('sf6', 'SF6 tests'), ('events', 'Events')]):
        col.download_button(name, demo[k].to_csv(index=False).encode('utf-8-sig'), f"demo_{k}.csv", 'text/csv', key=f"dl_{k}")
    with st.expander("How the PD risk is calculated"):
        st.markdown(f"""
1. **Magnitude** above the sensor background (dB). Below {pc.DETECT_DB:.0f} dB there is no PD; {pc.FULL_DB:.0f} dB gives a full score.
2. **Trend** over the last {pc.TREND_WINDOW_D} days ({pc.TREND_FULL:.0f} dB per month gives a full score) and **pulse rate**.
3. **Defect type** from the PRPD pattern (machine-learning classifier) and its **severity** at that equipment.
4. **Confirmation**: acoustic detection, SF6 by-products (≥ {pc.SF6_INDICATIVE:.0f} ppmv indicative, ≥ {pc.SF6_HIGH:.0f} ppmv high)
   and online DGA (H2 / C2H2 above IEEE C57.104 condition 1) each raise the risk.
5. **Safety rules**: a critical defect type that is confirmed is at least 75%; a strong free particle in GIS is at least 72%;
   a persistent critical defect above 6 dB stays at least on watch; readings that the noise reference sensor also sees are suppressed;
   a pattern seen on fewer than {pc.PERSIST_MIN_DAYS} of the last {pc.PERSIST_WINDOW_D} days is limited to 30%.
6. Bands: ≤ 35% normal, ≤ 70% watch, > 70% action. FMEA: RPN = S × O × D.

The defect classifier is trained on synthetic PRPD patterns built from the published signature of each defect type.
It must be re-trained on labelled field data before any accuracy figure is quoted.
""")

st.markdown(FOOTER, unsafe_allow_html=True)
