"""
PDGuard AI - partial discharge risk engine.

One engine used by every tab of the app, the batch analysis and the historical study.

Inputs (all CSV):
  * Sensor register   : Sensor ID, Asset ID, Asset Type, Location, Method, Unit, Baseline, Noise Reference
  * Online PD (UHF / HFCT export): Timestamp, Sensor ID, Peak, Pulse Rate (pps),
                        H00..H35 (pulses per 10-degree phase window = PRPD phase profile),
                        M00..M35 (mean pulse magnitude in that window, 0-1 of the window peak)
  * Online DGA (optional, transformers / reactors): Asset ID, Timestamp, Hydrogen, Methane, Acethylene
  * Acoustic tests (optional): Date, Asset ID, Detected (Yes/No), Amplitude (mV)
  * SF6 decomposition tests (optional, GIS): Date, Asset ID, SO2 (ppmv), SOF2 (ppmv), HF (ppmv)
  * Events (for the historical study): Asset ID, Date, Event
"""
import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants
N_BINS = 36                                     # 10-degree phase windows
H_COLS = [f"H{i:02d}" for i in range(N_BINS)]
M_COLS = [f"M{i:02d}" for i in range(N_BINS)]
PHASE_DEG = np.arange(N_BINS) * 10 + 5          # bin centres

DEFECTS = ['External noise', 'Corona', 'Floating', 'Void', 'Surface', 'Free particle']
NO_PD = 'No PD'
DEFECT_TEXT = {
    'No PD': 'No partial discharge above the sensor background',
    'External noise': 'External interference (not inside the equipment)',
    'Corona': 'Corona: sharp point or protrusion on a conductor',
    'Floating': 'Floating electrode: metal part not bonded to potential',
    'Void': 'Internal discharge: void or cavity inside solid insulation',
    'Surface': 'Surface discharge along an insulating surface',
    'Free particle': 'Free conducting particle moving inside the enclosure',
}

ASSET_CLASSES = ['GIS', 'Transformer', 'Reactor', 'Cable', 'Motor']
# FMEA severity of each defect at each asset class (1-10).
SEVERITY = {
    'GIS':         {'No PD': 1, 'External noise': 1, 'Corona': 6, 'Floating': 9, 'Void': 9, 'Surface': 8, 'Free particle': 10},
    'Transformer': {'No PD': 1, 'External noise': 1, 'Corona': 5, 'Floating': 8, 'Void': 9, 'Surface': 8, 'Free particle': 7},
    'Reactor':     {'No PD': 1, 'External noise': 1, 'Corona': 5, 'Floating': 8, 'Void': 9, 'Surface': 8, 'Free particle': 7},
    'Cable':       {'No PD': 1, 'External noise': 1, 'Corona': 5, 'Floating': 7, 'Void': 9, 'Surface': 8, 'Free particle': 6},
    'Motor':       {'No PD': 1, 'External noise': 1, 'Corona': 5, 'Floating': 6, 'Void': 7, 'Surface': 8, 'Free particle': 4},
}

# magnitude above the sensor baseline (dB)
DETECT_DB = 3.0          # below this: no PD
CRITICAL_DB = 18.0       # "critical level" used for the time-to-critical estimate
FULL_DB = 24.0           # magnitude score reaches 100 %
TREND_WINDOW_D = 60      # trailing window for the trend (days)
TREND_FULL = 4.0         # dB per 30 days that gives a 100 % trend score
CONFIRM_WINDOW_D = 90    # periodic tests count as confirmation for 90 days
NOISE_REF_DB = 6.0       # noise reference sensor active at or above this
LOW_CONFIDENCE = 0.50
LEVEL_WINDOW_D = 7       # level = trailing 7-day median of the daily peak
TREND_MIN = 0.5          # dB per 30 days below which the level is called stable
PERSIST_WINDOW_D = 14    # a PD pattern must appear on at least 3 of the last 14 days
PERSIST_MIN_DAYS = 3     # before it can raise more than a low-level watch

# confirmation limits
SF6_INDICATIVE = 2.0     # SO2 + SOF2 (ppmv): PD by-products present
SF6_HIGH = 12.0          # SO2 + SOF2 (ppmv): IEC 60480 limit for re-use of SF6
H2_LIMITS = (100, 700)   # IEEE C57.104-2008 condition 2 / 3 upper bounds of condition 1/2
C2H2_LIMITS = (1, 9)

EXCELLENT_MAX, WATCH_MAX = 35.0, 70.0
DETECTION_ONLINE = 2


def clamp(x, lo=0.0, hi=100.0):
    return np.minimum(hi, np.maximum(lo, x))


def asset_class(asset_type):
    t = str(asset_type).strip().lower()
    if t.startswith('gis'):
        return 'GIS'
    if 'reactor' in t:
        return 'Reactor'
    if 'cable' in t:
        return 'Cable'
    if 'motor' in t:
        return 'Motor'
    return 'Transformer'


def status_of(risk):
    if risk <= EXCELLENT_MAX:
        return 'EXCELLENT'
    if risk <= WATCH_MAX:
        return 'WATCH'
    return 'ACTION'


def rpn_level(rpn):
    return 'Low' if rpn < 40 else ('Medium' if rpn < 100 else 'High')


# ---------------------------------------------------------------- PRPD patterns
def _wrap_gauss(centre, width):
    d = (PHASE_DEG - centre + 180) % 360 - 180
    return np.exp(-0.5 * (d / width) ** 2)


def generate_pattern(kind, rng, total=None, background=None):
    """Synthetic PRPD phase profile for one acquisition window.
    Returns (counts[36], magnitude[36] in 0-1). Shapes follow the textbook PRPD signatures."""
    total = total if total is not None else int(rng.uniform(300, 20000))
    if kind == 'Void':
        c = rng.uniform(15, 75); w = rng.uniform(14, 32)
        r = rng.uniform(0.75, 1.3)
        p = _wrap_gauss(c, w) + r * _wrap_gauss(c + 180, w * rng.uniform(0.85, 1.15))
        mag = 0.45 + 0.5 * (_wrap_gauss(c + 15, w) + _wrap_gauss(c + 195, w)) + rng.normal(0, .04, N_BINS)
    elif kind == 'Surface':
        c = rng.uniform(10, 70); w = rng.uniform(18, 38)
        r = rng.uniform(0.15, 0.5)
        if rng.random() < .5:
            p = _wrap_gauss(c, w) + r * _wrap_gauss(c + 180, w * 0.8)
        else:
            p = r * _wrap_gauss(c, w * 0.8) + _wrap_gauss(c + 180, w)
        mag = 0.35 + 0.6 * p / p.max() + rng.normal(0, .06, N_BINS)
    elif kind == 'Corona':
        c = rng.normal(270, 10); w = rng.uniform(6, 16)
        p = _wrap_gauss(c, w) + rng.uniform(0, 0.25) * _wrap_gauss(c - 180, w)
        mag = 0.25 + 0.35 * _wrap_gauss(c, w * 1.5) + rng.normal(0, .03, N_BINS)
    elif kind == 'Floating':
        c = rng.uniform(25, 85); w = rng.uniform(6, 13)
        p = _wrap_gauss(c, w) + rng.uniform(0.85, 1.15) * _wrap_gauss(c + 180, w)
        mag = np.full(N_BINS, rng.uniform(0.85, 0.97)) + rng.normal(0, .015, N_BINS)
    elif kind == 'Free particle':
        a = rng.uniform(0.3, 1.2)
        p = 1 + a * np.abs(np.sin(np.radians(PHASE_DEG))) + rng.uniform(0, .6, N_BINS)
        mag = rng.uniform(0.2, 1.0, N_BINS)
    elif kind == 'External noise':
        if rng.random() < .6:                      # broadband / radio interference: flat, low spread
            p = np.ones(N_BINS) + rng.normal(0, .08, N_BINS)
            mag = np.full(N_BINS, rng.uniform(.4, .6)) + rng.normal(0, .03, N_BINS)
        else:                                      # switching / thyristor pulses: a few fixed narrow lines
            p = np.full(N_BINS, .05)
            for k in rng.choice(N_BINS, size=int(rng.integers(2, 7)), replace=False):
                p[k] += rng.uniform(.6, 1.0)
            mag = np.where(p > .3, rng.uniform(.7, .95), rng.uniform(.1, .3)) + rng.normal(0, .02, N_BINS)
    else:                                          # background only
        p = np.ones(N_BINS)
        mag = np.full(N_BINS, .3) + rng.normal(0, .05, N_BINS)
    p = np.clip(p, 1e-6, None)
    bg = background if background is not None else rng.uniform(0, 0.12)
    p = (1 - bg) * p / p.sum() + bg / N_BINS
    counts = rng.poisson(p * total).astype(float)
    mag = np.clip(mag, 0.02, 1.0)
    mag = np.where(counts > 0, mag, 0.0)
    if mag.max() > 0:
        mag = mag / mag.max()
    return counts, np.round(mag, 3)


# ---------------------------------------------------------------- features
def _circ_std(w):
    s = w.sum()
    if s <= 0:
        return 180.0
    ang = np.radians(PHASE_DEG)
    R = np.hypot((w * np.cos(ang)).sum(), (w * np.sin(ang)).sum()) / s
    return float(np.degrees(np.sqrt(max(-2 * np.log(max(R, 1e-9)), 0))))


def features_from_profile(H, M):
    """H, M: arrays (n, 36). Returns a feature matrix (n, k) independent of the sensor unit."""
    H = np.asarray(H, float); M = np.asarray(M, float)
    tot = H.sum(1, keepdims=True); tot[tot == 0] = 1
    h = H / tot
    pos, neg = h[:, :18], h[:, 18:]
    feats = [h, M]
    sp, sn = pos.sum(1), neg.sum(1)
    asym = np.abs(sp - sn) / np.maximum(sp + sn, 1e-9)
    rising = h[:, 0:9].sum(1) + h[:, 18:27].sum(1)
    shift_corr = np.array([np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0 for a, b in zip(pos, neg)])
    active = (h > 0.25 / N_BINS).mean(1)
    peak = h.max(1)
    ent = -(np.where(h > 0, h * np.log(np.where(h > 0, h, 1)), 0)).sum(1) / np.log(N_BINS)
    act_m = [m[x > 0.25 / N_BINS] for m, x in zip(M, h)]
    mcv = np.array([a.std() / a.mean() if len(a) > 1 and a.mean() > 0 else 0 for a in act_m])
    mmean = np.array([a.mean() if len(a) else 0 for a in act_m])
    cs_pos = np.array([_circ_std(r) for r in np.c_[pos, np.zeros_like(pos)]])
    cs_neg = np.array([_circ_std(r) for r in np.c_[np.zeros_like(neg), neg]])
    near_neg_peak = h[:, 24:30].sum(1)
    feats.append(np.c_[asym, rising, shift_corr, active, peak, ent, mcv, mmean, cs_pos, cs_neg, near_neg_peak])
    return np.hstack(feats)


FEATURE_NAMES = ([f"h{i:02d}" for i in range(N_BINS)] + [f"m{i:02d}" for i in range(N_BINS)] +
                 ['half_asymmetry', 'rising_quadrants', 'half_correlation', 'active_windows', 'peak_window',
                  'phase_entropy', 'magnitude_cv', 'magnitude_mean', 'spread_pos', 'spread_neg', 'neg_peak_share'])


def field_pattern(kind, rng):
    """generate_pattern plus the imperfections of field data: few pulses, strong background,
    a phase-reference offset of up to +/-20 degrees and, sometimes, a second weaker source."""
    total = int(np.exp(rng.uniform(np.log(60), np.log(20000))))
    c, m = generate_pattern(kind, rng, total=total, background=rng.uniform(0, 0.35))
    if rng.random() < 0.3:
        other = rng.choice([k for k in DEFECTS if k != kind])
        c2, m2 = generate_pattern(other, rng, total=int(total * rng.uniform(0.1, 0.35)))
        m = np.where(c + c2 > 0, (m * c + m2 * c2) / np.maximum(c + c2, 1), 0)
        c = c + c2
        if m.max() > 0:
            m = m / m.max()
    k = int(rng.integers(-2, 3))
    return np.roll(c, k), np.roll(m, k)


def make_training_set(n_per_class=1500, seed=11):
    rng = np.random.default_rng(seed)
    H, M, y = [], [], []
    for k in DEFECTS:
        for _ in range(n_per_class):
            c, m = field_pattern(k, rng)
            H.append(c); M.append(m); y.append(k)
    return features_from_profile(np.array(H), np.array(M)), np.array(y)


def classify(df, model):
    """Classify every row's PRPD profile. Returns (labels, confidence, probability matrix)."""
    X = features_from_profile(df[H_COLS].to_numpy(float), df[M_COLS].to_numpy(float))
    P = model.predict_proba(X)
    idx = P.argmax(1)
    return np.array(model.classes_)[idx], P[np.arange(len(P)), idx], P


# ---------------------------------------------------------------- assessment
def db_above(peak, baseline, unit):
    peak = np.asarray(peak, float); baseline = np.asarray(baseline, float)
    unit = np.asarray(unit).astype(str)
    is_pc = np.char.lower(unit) == 'pc'
    with np.errstate(divide='ignore', invalid='ignore'):
        lin = 20 * np.log10(np.maximum(peak, 1e-9) / np.maximum(baseline, 1e-9))
    return np.where(is_pc, lin, peak - baseline)


def _rolling_slope(dates, values, window_days):
    """Least-squares slope (dB per 30 days) over the trailing window, using only past data."""
    s = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
    t = pd.Series((s.index - s.index[0]).days.astype(float), index=s.index)
    w = f"{window_days}D"
    n = s.rolling(w).count()
    mt, my = t.rolling(w).mean(), s.rolling(w).mean()
    cov = (t * s).rolling(w).mean() - mt * my
    var = (t * t).rolling(w).mean() - mt * mt
    slope = (cov / var.replace(0, np.nan)) * 30
    slope[n < 10] = 0.0
    return slope.fillna(0).to_numpy()


def _latest_test(rows, tests, value_cols, date_col='Date'):
    """For each row (Asset ID, Timestamp) attach the latest periodic test of that asset
    done on or before the row date and no older than CONFIRM_WINDOW_D."""
    out = pd.DataFrame(index=rows.index, columns=value_cols + ['_test_date'], dtype=object)
    if tests is None or len(tests) == 0:
        return out
    t = tests.copy()
    t['_test_date'] = pd.to_datetime(t[date_col], errors='coerce')
    t = t.dropna(subset=['_test_date']).sort_values('_test_date')
    left = rows[['Asset ID', 'Timestamp']].copy()
    left['_i'] = rows.index
    left = left.sort_values('Timestamp')
    m = pd.merge_asof(left, t[['Asset ID', '_test_date'] + value_cols], left_on='Timestamp', right_on='_test_date',
                      by='Asset ID', direction='backward', tolerance=pd.Timedelta(days=CONFIRM_WINDOW_D))
    m.index = m['_i']
    return m[value_cols + ['_test_date']].reindex(rows.index)


ACTIONS = {
    'ACTION': {
        'Free particle': 'URGENT: plan an outage to open and clean the compartment; confirm location with acoustic first. Avoid switching operations.',
        'Floating': 'URGENT: locate the floating part (acoustic / UHF time-of-flight) and plan an outage to bond or replace it.',
        'Void': 'URGENT: confirm with a second method, plan an outage for diagnostic tests and repair of the affected insulation.',
        'Surface': 'URGENT: confirm with a second method; plan an outage to clean or replace the affected insulator / termination.',
        'Corona': 'Plan an inspection of the conductor and shielding at the next outage; increase monitoring.',
        'default': 'URGENT: expert review of the PD pattern and an outage plan.',
    },
    'WATCH': {
        'default': 'Increase monitoring, request an acoustic measurement to confirm and locate the source, and review the trend weekly.',
    },
    'EXCELLENT': {'default': 'Continue online monitoring.'},
}


def assess(online, sensors, model, dga=None, acoustic=None, sf6=None):
    """Assess every online reading using only the data available on that date.
    Returns one row per reading with risk, status, defect, confirmations and FMEA."""
    df = online.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    reg = sensors.copy()
    reg['Noise Reference'] = reg.get('Noise Reference', 'No').astype(str).str.lower().isin(['yes', 'true', '1', 'y'])
    df = df.merge(reg[['Sensor ID', 'Asset ID', 'Asset Type', 'Location', 'Method', 'Unit', 'Baseline', 'Noise Reference']],
                  on='Sensor ID', how='left')
    unknown = df['Asset ID'].isna()
    if unknown.any():
        df = df[~unknown]
    df = df.sort_values(['Sensor ID', 'Timestamp']).reset_index(drop=True)
    df['Asset Class'] = df['Asset Type'].map(asset_class)
    df['dB Above Baseline'] = db_above(df['Peak'], df['Baseline'], df['Unit'])
    df['Pulse Rate (pps)'] = pd.to_numeric(df['Pulse Rate (pps)'], errors='coerce').fillna(0).clip(lower=0)

    # level (7-day trailing median, removes day-to-day scatter) and trend per sensor (past data only)
    df['Trend (dB/30d)'] = 0.0
    df['Level (dB, 7d median)'] = df['dB Above Baseline']
    for sid, g in df.groupby('Sensor ID'):
        df.loc[g.index, 'Trend (dB/30d)'] = _rolling_slope(g['Timestamp'], g['dB Above Baseline'].to_numpy(), TREND_WINDOW_D)
        s_ = pd.Series(g['dB Above Baseline'].to_numpy(), index=pd.DatetimeIndex(g['Timestamp']))
        df.loc[g.index, 'Level (dB, 7d median)'] = s_.rolling(f"{LEVEL_WINDOW_D}D").median().to_numpy()
    LV = df['Level (dB, 7d median)']

    # classification
    lab, conf, _ = classify(df, model)
    df['Defect'] = lab
    df['Confidence (%)'] = conf * 100
    no_pd = df['dB Above Baseline'] < DETECT_DB
    df.loc[no_pd, 'Defect'] = NO_PD

    # noise reference: same substation day, any reference sensor active
    df['_day'] = df['Timestamp'].dt.floor('D')
    ref = df[df['Noise Reference']].groupby('_day')['dB Above Baseline'].max()
    df['Noise Ref (dB)'] = df['_day'].map(ref).fillna(-99.0)
    ext = (~df['Noise Reference']) & (df['Noise Ref (dB)'] >= NOISE_REF_DB) & df['Defect'].isin(['Corona', 'External noise', 'Free particle'])
    df['Noise Suppressed'] = ext & (df['dB Above Baseline'] >= DETECT_DB)
    df.loc[ext & ~no_pd, 'Defect'] = 'External noise'
    df.loc[df['Noise Reference'], 'Defect'] = np.where(no_pd[df['Noise Reference']], NO_PD, 'External noise')

    # confirmations
    ac = _latest_test(df, acoustic, ['Detected', 'Amplitude (mV)'])
    df['Acoustic'] = ac['Detected'].map(lambda v: None if pd.isna(v) else ('Yes' if str(v).strip().lower() in ('yes', 'y', 'true', '1') else 'No'))
    sf = _latest_test(df, sf6, ['SO2 (ppmv)', 'SOF2 (ppmv)', 'HF (ppmv)'])
    so = pd.to_numeric(sf['SO2 (ppmv)'], errors='coerce').fillna(0) + pd.to_numeric(sf['SOF2 (ppmv)'], errors='coerce').fillna(0)
    df['SF6 By-products (ppmv)'] = np.where(sf['_test_date'].isna(), np.nan, so)
    if dga is not None and len(dga):
        g = dga.copy()
        g['Timestamp'] = pd.to_datetime(g['Timestamp'], errors='coerce')
        g = g.dropna(subset=['Timestamp']).sort_values('Timestamp')
        for c in ['Hydrogen', 'Methane', 'Acethylene']:
            g[c] = pd.to_numeric(g.get(c), errors='coerce')
        left = df[['Asset ID', 'Timestamp']].copy(); left['_i'] = df.index
        mm = pd.merge_asof(left.sort_values('Timestamp'), g[['Asset ID', 'Timestamp', 'Hydrogen', 'Methane', 'Acethylene']]
                           .rename(columns={'Timestamp': '_gt'}), left_on='Timestamp', right_on='_gt', by='Asset ID',
                           direction='backward', tolerance=pd.Timedelta(days=7))
        mm.index = mm['_i']; mm = mm.reindex(df.index)
        df['H2 (ppm)'], df['CH4 (ppm)'], df['C2H2 (ppm)'] = mm['Hydrogen'], mm['Methane'], mm['Acethylene']
    else:
        df['H2 (ppm)'] = df['CH4 (ppm)'] = df['C2H2 (ppm)'] = np.nan

    is_pd = ~df['Defect'].isin([NO_PD, 'External noise'])
    # persistence: number of the last 14 days with a PD pattern on this sensor (past data only)
    df['_pd'] = is_pd.astype(float)
    df['PD Days (14d)'] = 0.0
    for sid, g in df.groupby('Sensor ID'):
        s = pd.Series(g['_pd'].to_numpy(), index=pd.DatetimeIndex(g['Timestamp']))
        df.loc[g.index, 'PD Days (14d)'] = s.rolling(f"{PERSIST_WINDOW_D}D").sum().to_numpy()
    df = score(df)
    return df.drop(columns=['_day', '_pd'])


def score(df):
    """The PD risk rules. Needs: Asset Class, dB Above Baseline, Level (dB, 7d median), Trend (dB/30d),
    Pulse Rate (pps), Defect, Confidence (%), Acoustic, SF6 By-products (ppmv), H2 (ppm), C2H2 (ppm), PD Days (14d).
    Used by the substation analysis, the historical study and the single-reading pages alike."""
    df = df.copy()
    LV = df['Level (dB, 7d median)']
    is_pd = ~df['Defect'].isin([NO_PD, 'External noise'])
    M = clamp((LV - DETECT_DB) / (FULL_DB - DETECT_DB) * 100)
    T = clamp(df['Trend (dB/30d)'] / TREND_FULL * 100)
    R = clamp(np.log10(np.maximum(df['Pulse Rate (pps)'], 1)) / 3 * 100)
    S = np.array([SEVERITY[c][d] for c, d in zip(df['Asset Class'], df['Defect'])])
    base = 0.55 * M + 0.25 * T + 0.20 * R
    risk = base * (0.35 + 0.65 * S / 10)
    df['Magnitude Score'], df['Trend Score'], df['Rate Score'] = M, T, R

    sf6v = pd.to_numeric(df['SF6 By-products (ppmv)'], errors='coerce')
    acoustic_yes = (df['Acoustic'] == 'Yes') & is_pd
    sf6_lvl = np.where(sf6v >= SF6_HIGH, 2, np.where(sf6v >= SF6_INDICATIVE, 1, 0)) * is_pd
    h2 = pd.to_numeric(df['H2 (ppm)'], errors='coerce').fillna(0)
    c2 = pd.to_numeric(df['C2H2 (ppm)'], errors='coerce').fillna(0)
    dga_lvl = np.where((h2 > H2_LIMITS[1]) | (c2 > C2H2_LIMITS[1]), 2, np.where((h2 > H2_LIMITS[0]) | (c2 > C2H2_LIMITS[0]), 1, 0))
    dga_lvl = dga_lvl * (is_pd & df['Asset Class'].isin(['Transformer', 'Reactor']))
    risk = risk + 15 * acoustic_yes + 10 * sf6_lvl + 10 * dga_lvl
    n_conf = acoustic_yes.astype(int) + (sf6_lvl > 0) + (dga_lvl > 0)
    df['Confirmations'] = n_conf

    persistent = df['PD Days (14d)'] >= PERSIST_MIN_DAYS
    df['Isolated'] = is_pd & ~persistent
    risk = np.where(df['Isolated'], np.minimum(risk, 30), risk)
    risk = np.where(is_pd & persistent & (S >= 8) & (LV >= 6), np.maximum(risk, 40), risk)
    crit = (S >= 8) & (M >= 40)
    risk = np.where(crit & (n_conf >= 1), np.maximum(risk, 75), risk)
    gis_particle = (df['Asset Class'] == 'GIS') & (df['Defect'] == 'Free particle') & (M >= 50)
    risk = np.where(gis_particle, np.maximum(risk, 72), risk)
    risk = np.where(df['Defect'] == 'External noise', np.minimum(risk, 15), risk)
    risk = np.where(df['Defect'] == NO_PD, np.minimum(risk, 10), risk)
    df['PD Risk (%)'] = np.round(clamp(risk), 1)
    df['Status'] = df['PD Risk (%)'].map(status_of)

    slope = df['Trend (dB/30d)']
    ttc = np.where((slope > TREND_MIN) & (LV < CRITICAL_DB) & is_pd,
                   (CRITICAL_DB - LV) / slope.where(slope > 0, 1) * 30, np.nan)
    ttc = np.where(is_pd & (LV >= CRITICAL_DB), 0, ttc)
    df['Days to Critical'] = np.round(ttc)

    df['S'] = S
    df['O'] = np.maximum(1, np.ceil(df['PD Risk (%)'] / 10)).astype(int)
    df['D'] = DETECTION_ONLINE
    df['RPN'] = df['S'] * df['O'] * df['D']
    df['RPN Level'] = df['RPN'].map(rpn_level)
    df['Low Confidence'] = is_pd & (df['Confidence (%)'] < LOW_CONFIDENCE * 100)
    df['Action'] = [ACTIONS[st].get(d, ACTIONS[st]['default']) for st, d in zip(df['Status'], df['Defect'])]
    return df


PATTERN_SHAPES = {
    'Two clusters on the rising part of both half-cycles, similar size': 'Void',
    'Two clusters, one much larger than the other': 'Surface',
    'One narrow cluster at the negative voltage peak': 'Corona',
    'Two narrow clusters with almost constant amplitude': 'Floating',
    'Pulses spread over the whole cycle, random amplitudes': 'Free particle',
    'Flat or fixed lines unrelated to the voltage': 'External noise',
}


def example_profile(kind, pulse_rate, seed=5):
    """Representative PRPD profile for a pattern shape picked on screen (60-second window)."""
    rng = np.random.default_rng(seed)
    return generate_pattern(kind, rng, total=int(max(pulse_rate, 1) * 60), background=0.05)


def assess_single(model, asset_type, peak, baseline, unit, pulse_rate, counts, mags,
                  prev_peak=None, days_since_prev=30, pd_days_14=14, noise_ref_db=None,
                  acoustic=None, sf6_byproducts=None, h2=None, c2h2=None, asset_id='', location=''):
    """One reading entered by hand (or one PRPD export row) assessed with the same rules as the station analysis."""
    cls = asset_class(asset_type)
    db = float(db_above([peak], [baseline], [unit])[0])
    trend = 0.0
    if prev_peak is not None and days_since_prev and days_since_prev > 0:
        prev_db = float(db_above([prev_peak], [baseline], [unit])[0])
        trend = (db - prev_db) / days_since_prev * 30
    X = features_from_profile(np.asarray(counts, float)[None], np.asarray(mags, float)[None])
    P = model.predict_proba(X)[0]
    k = int(P.argmax())
    defect, conf = model.classes_[k], P[k] * 100
    if db < DETECT_DB:
        defect = NO_PD
    noise_ref_db = -99.0 if noise_ref_db is None else noise_ref_db
    suppressed = False
    if noise_ref_db >= NOISE_REF_DB and defect in ('Corona', 'External noise', 'Free particle'):
        suppressed = defect != NO_PD
        defect = 'External noise' if defect != NO_PD else defect
    row = {'Asset ID': asset_id, 'Location': location, 'Asset Type': asset_type, 'Asset Class': cls,
           'Peak': peak, 'Baseline': baseline, 'Unit': unit,
           'dB Above Baseline': db, 'Level (dB, 7d median)': db, 'Trend (dB/30d)': trend,
           'Pulse Rate (pps)': float(pulse_rate), 'Defect': defect, 'Confidence (%)': conf,
           'Noise Ref (dB)': noise_ref_db, 'Noise Reference': False, 'Noise Suppressed': suppressed,
           'Acoustic': acoustic, 'SF6 By-products (ppmv)': np.nan if sf6_byproducts is None else sf6_byproducts,
           'H2 (ppm)': np.nan if h2 is None else h2, 'C2H2 (ppm)': np.nan if c2h2 is None else c2h2,
           'PD Days (14d)': float(pd_days_14), 'Single': True, 'Sensor ID': asset_id}
    for i in range(N_BINS):
        row[H_COLS[i]] = counts[i]
        row[M_COLS[i]] = mags[i]
    out = score(pd.DataFrame([row])).iloc[0]
    return out, dict(zip(model.classes_, P))


def reasons(row):
    """Plain-language reasons for one assessed row."""
    r = []
    d = row['Defect']
    if d == NO_PD:
        return [f"PD magnitude is {row['dB Above Baseline']:.1f} dB above the sensor background (detection starts at {DETECT_DB:.0f} dB)."]
    if d == 'External noise':
        if row.get('Noise Suppressed') or row.get('Noise Reference'):
            r.append(f"The noise reference sensor shows {row['Noise Ref (dB)']:.1f} dB at the same time: the signal comes from outside the equipment. Alarm suppressed.")
        else:
            r.append('The pattern has no fixed relation to the voltage phase that matches a PD defect: classified as external interference.')
        return r
    r.append(f"{DEFECT_TEXT[d]} (pattern confidence {row['Confidence (%)']:.0f}%).")
    single = bool(row.get('Single', False))
    if single:
        r.append(f"Level {row['dB Above Baseline']:.1f} dB above the sensor background; {row['Pulse Rate (pps)']:.0f} pulses per second.")
    else:
        r.append(f"Level {row['Level (dB, 7d median)']:.1f} dB above the sensor background (7-day median; today {row['dB Above Baseline']:.1f} dB); {row['Pulse Rate (pps)']:.0f} pulses per second.")
    span = 'since the previous reading' if single else f"over the last {TREND_WINDOW_D} days"
    if row['Trend (dB/30d)'] > TREND_MIN:
        r.append(f"Rising by {row['Trend (dB/30d)']:.1f} dB per month {span}.")
    elif row['Trend (dB/30d)'] < -TREND_MIN:
        r.append(f"Falling by {-row['Trend (dB/30d)']:.1f} dB per month.")
    else:
        r.append('Stable since the previous reading.' if single else f'Stable over the last {TREND_WINDOW_D} days.')
    if not pd.isna(row.get('Days to Critical')):
        if row['Days to Critical'] == 0:
            r.append(f"Already at or above the critical level ({CRITICAL_DB:.0f} dB).")
        else:
            r.append(f"At this rate it reaches the critical level ({CRITICAL_DB:.0f} dB) in about {row['Days to Critical']:.0f} days.")
    if row.get('Acoustic') == 'Yes':
        r.append('Confirmed by the acoustic measurement.')
    elif row.get('Acoustic') == 'No':
        r.append('The last acoustic measurement did not detect the source.')
    v = row.get('SF6 By-products (ppmv)')
    if v is not None and not pd.isna(v):
        if v >= SF6_HIGH:
            r.append(f"SF6 decomposition products {v:.1f} ppmv: above the IEC 60480 re-use limit ({SF6_HIGH:.0f} ppmv). Strong confirmation.")
        elif v >= SF6_INDICATIVE:
            r.append(f"SF6 decomposition products {v:.1f} ppmv: discharge by-products present.")
        else:
            r.append(f"SF6 decomposition products {v:.1f} ppmv: not significant.")
    h2 = row.get('H2 (ppm)')
    if h2 is not None and not pd.isna(h2) and row['Asset Class'] in ('Transformer', 'Reactor'):
        c2 = row.get('C2H2 (ppm)', 0) or 0
        lvl = 'above IEEE condition 1' if (h2 > H2_LIMITS[0] or c2 > C2H2_LIMITS[0]) else 'within normal limits'
        r.append(f"Online DGA: H2 {h2:.0f} ppm, C2H2 {c2:.1f} ppm ({lvl}).")
    if row.get('Isolated'):
        r.append(f"Isolated reading: a PD pattern was seen on only {row['PD Days (14d)']:.0f} of the last {PERSIST_WINDOW_D} days. Risk is limited until it repeats.")
    elif row['S'] >= 8 and row['Level (dB, 7d median)'] >= 6:
        r.append(f"Persistent critical defect type (seen on {row['PD Days (14d)']:.0f} of the last {PERSIST_WINDOW_D} days): kept at least on watch.")
    if row['Confirmations'] == 0 and row['Status'] != 'EXCELLENT':
        r.append('Not yet confirmed by a second method: an acoustic measurement is recommended.')
    if row.get('Low Confidence'):
        r.append('The pattern is not clear: expert review of the PRPD is recommended.')
    return r


def latest(assessed):
    return assessed.sort_values('Timestamp').groupby('Sensor ID').tail(1).reset_index(drop=True)


def asset_summary(latest_rows):
    """Worst sensor per asset."""
    rows = latest_rows[~latest_rows['Noise Reference']]
    idx = rows.groupby('Asset ID')['PD Risk (%)'].idxmax()
    out = rows.loc[idx, ['Asset ID', 'Asset Type', 'Location', 'Sensor ID', 'Defect', 'Level (dB, 7d median)',
                         'Trend (dB/30d)', 'Confirmations', 'PD Risk (%)', 'Status', 'S', 'O', 'D', 'RPN', 'RPN Level',
                         'Days to Critical', 'Action']]
    return out.sort_values('PD Risk (%)', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- historical study
def historical_study(assessed, events=None, alert_threshold=70.0, lookback_days=365, gap_days=120):
    """Daily asset risk (worst sensor), alert episodes and lead time before each recorded event."""
    a = assessed[~assessed['Noise Reference']].copy()
    a['Date'] = a['Timestamp'].dt.floor('D')
    idx = a.groupby(['Asset ID', 'Date'])['PD Risk (%)'].idxmax()
    tl = a.loc[idx, ['Asset ID', 'Date', 'Sensor ID', 'Defect', 'dB Above Baseline', 'PD Risk (%)', 'Status']].sort_values(['Asset ID', 'Date'])
    alerts = tl[tl['PD Risk (%)'] >= alert_threshold]
    episodes = []
    for asset, g in alerts.groupby('Asset ID'):
        cur = None
        for d in g['Date']:
            if cur is not None and (d - cur[2]).days <= gap_days:
                cur[2] = d
            else:
                if cur is not None:
                    episodes.append(cur)
                cur = [asset, d, d]
        if cur is not None:
            episodes.append(cur)
    ev = pd.DataFrame(columns=['Asset ID', 'Date', 'Event']) if events is None else events.copy()
    ev['_d'] = pd.to_datetime(ev['Date'], errors='coerce')
    ev = ev.dropna(subset=['_d'])
    rows = []
    for _, e in ev.iterrows():
        prior = [p for p in episodes if p[0] == e['Asset ID'] and p[1] < e['_d'] and (e['_d'] - p[2]).days <= lookback_days]
        first = min(p[1] for p in prior) if prior else None
        rows.append({'Asset ID': e['Asset ID'], 'Event Date': e['_d'].date(), 'Event': e.get('Event', ''),
                     'First Alert Before Event': first.date() if first is not None else None,
                     'Lead Time (days)': (e['_d'] - first).days if first is not None else None,
                     'Caught': 'Yes' if first is not None else 'No'})
    false_alarms = 0
    for asset, st, en in episodes:
        if not ((ev['Asset ID'] == asset) & (ev['_d'] >= st) & (ev['_d'] <= en + pd.Timedelta(days=lookback_days))).any():
            false_alarms += 1
    return tl.reset_index(drop=True), pd.DataFrame(rows), false_alarms, len(episodes)


# ---------------------------------------------------------------- demo data
def demo_register():
    """Sample 400/220 kV GIS substation: breaker-and-a-half, 4 diameters, 3 transformers,
    2 bus reactors, 2 line reactors, 3 cable sealing ends and 2 HV motors."""
    rows = []
    for q in range(1, 5):
        for cb in range(1, 4):
            rows.append([f"UHF-Q{q}-CB{cb}", f"Q{q}-CB{cb}", 'GIS circuit breaker', f"Diameter Q{q}, CB{cb} compartment", 'UHF', 'dBm', -72.0, 'No'])
    for b in (1, 2):
        for s in (1, 2):
            rows.append([f"UHF-BUS{b}-S{s}", f"BUS{b}", 'GIS busbar', f"Busbar {b}, section {s}", 'UHF', 'dBm', -72.0, 'No'])
    rows.append(['UHF-NOISE', 'NOISE-REF', 'GIS noise reference', 'Antenna outside the GIS (interference reference)', 'UHF', 'dBm', -70.0, 'Yes'])
    for t in (1, 2, 3):
        rows.append([f"UHF-T{t}", f"T{t}", 'Power transformer 400/220 kV', f"Transformer T{t}, drain-valve probe", 'UHF', 'dBm', -66.0, 'No'])
    for r in ('RB1', 'RB2', 'RL1', 'RL2'):
        rows.append([f"UHF-{r}", r, 'Shunt reactor 400 kV', f"Reactor {r}, drain-valve probe", 'UHF', 'dBm', -66.0, 'No'])
    for c in (1, 2, 3):
        rows.append([f"HFCT-C{c}", f"CABLE-{c}", 'Cable 220 kV', f"220 kV cable sealing end, transformer T{c}", 'HFCT', 'pC', 20.0, 'No'])
    for mtr in (1, 2):
        rows.append([f"CC-M{mtr}", f"MOTOR-{mtr}", 'HV motor 13.8 kV', f"Motor M{mtr} stator, coupling capacitors", 'Coupling capacitor', 'pC', 50.0, 'No'])
    return pd.DataFrame(rows, columns=['Sensor ID', 'Asset ID', 'Asset Type', 'Location', 'Method', 'Unit', 'Baseline', 'Noise Reference'])


def demo_data(seed=7, start='2025-01-01', end='2026-08-31'):
    """Synthetic but realistic history. Scenarios:
       Q2-CB2  free particle in the CB compartment -> found and removed at a planned outage
       T2      internal discharge in the HV winding -> transformer trip
       CABLE-3 surface discharge on the sealing end -> termination failure
       RB1     stable floating electrode (watch, no event)
       MOTOR-2 stable slot / surface discharge in the stator (watch)
       Q1-CB3  small corona from mid-2026
       Summer interference picked up by BUS1-S1, Q4-CB1 and the noise reference -> suppressed"""
    rng = np.random.default_rng(seed)
    reg = demo_register()
    days = pd.date_range(start, end, freq='D')
    D = lambda s: pd.Timestamp(s)
    ev_particle, ev_t2, ev_cable = D('2026-03-10'), D('2026-05-20'), D('2026-06-15')

    def ramp(day, t0, t1, v0, v1):
        if day < t0:
            return None
        f = min(1.0, (day - t0).days / max((t1 - t0).days, 1))
        return v0 + (v1 - v0) * f ** 1.4

    noise_days = set()
    for s0, n in [(D('2025-07-05'), 9), (D('2025-08-14'), 6), (D('2026-07-20'), 8)]:
        noise_days |= {s0 + pd.Timedelta(days=i) for i in range(n)}

    recs = []
    for _, s in reg.iterrows():
        sid, base = s['Sensor ID'], s['Baseline']
        for day in days:
            kind, lvl, pps = None, rng.normal(0.5, 0.8), rng.uniform(0, 3)
            if sid == 'UHF-Q2-CB2' and day < ev_particle:
                v = ramp(day, D('2025-09-01'), D('2026-02-20'), 4, 20)
                if v is not None:
                    kind, lvl, pps = 'Free particle', v + rng.normal(0, 1.5), 5 + v * 4 + rng.uniform(0, 10)
            elif sid == 'UHF-T2' and day < ev_t2:
                v = ramp(day, D('2025-06-01'), D('2026-05-15'), 3.5, 17)
                if v is not None:
                    kind, lvl, pps = 'Void', v + rng.normal(0, 1.0), 20 + v * 15 + rng.uniform(0, 20)
            elif sid == 'HFCT-C3' and day < ev_cable:
                v = ramp(day, D('2025-10-01'), D('2026-06-10'), 4, 19)
                if v is not None:
                    kind, lvl, pps = 'Surface', v + rng.normal(0, 1.0), 10 + v * 10 + rng.uniform(0, 15)
            elif sid == 'UHF-RB1':
                kind, lvl, pps = 'Floating', 10 + rng.normal(0, 0.8), 100 + rng.uniform(0, 20)
            elif sid == 'CC-M2' and day >= D('2025-04-01'):
                kind, lvl, pps = 'Surface', 8 + rng.normal(0, 0.8), 60 + rng.uniform(0, 20)
            elif sid == 'UHF-Q1-CB3' and day >= D('2026-05-01'):
                kind, lvl, pps = 'Corona', 5 + rng.normal(0, 0.7), 40 + rng.uniform(0, 20)
            if day in noise_days and sid in ('UHF-BUS1-S1', 'UHF-Q4-CB1', 'UHF-NOISE'):
                kind, lvl, pps = 'External noise', 9 + rng.normal(0, 1.5), 200 + rng.uniform(0, 100)
            counts, mag = generate_pattern(kind or 'background', rng, total=int(max(pps, 0.5) * 60))
            peak = base + lvl if s['Unit'] == 'dBm' else base * 10 ** (lvl / 20)
            recs.append([day + pd.Timedelta(hours=6), sid, round(float(peak), 2), round(float(pps), 1), *counts.astype(int), *mag])
    online = pd.DataFrame(recs, columns=['Timestamp', 'Sensor ID', 'Peak', 'Pulse Rate (pps)'] + H_COLS + M_COLS)

    # online DGA monitors (transformers and reactors), daily
    g = []
    for asset in ['T1', 'T2', 'T3', 'RB1', 'RB2', 'RL1', 'RL2']:
        for day in days:
            h2, ch4, c2h2 = 25 + rng.normal(0, 3), 8 + rng.normal(0, 1), max(0, rng.normal(0.1, 0.1))
            if asset == 'T2' and day < ev_t2:
                v = ramp(day, D('2025-07-01'), D('2026-05-15'), 0, 1)
                if v is not None:
                    h2, ch4, c2h2 = h2 + 520 * v, ch4 + 45 * v, c2h2 + 3.5 * v
            g.append([asset, day + pd.Timedelta(hours=7), round(max(h2, 0), 1), round(max(ch4, 0), 1), round(c2h2, 2)])
    dga = pd.DataFrame(g, columns=['Asset ID', 'Timestamp', 'Hydrogen', 'Methane', 'Acethylene'])

    acoustic = pd.DataFrame([
        ['2025-11-18', 'Q2-CB2', 'Yes', 38.0, 'Bouncing-particle signal, bottom of CB tank'],
        ['2026-01-25', 'Q2-CB2', 'Yes', 55.0, 'Signal stronger than November'],
        ['2025-12-10', 'T2', 'No', 0.0, 'Source not located from tank wall'],
        ['2026-03-20', 'T2', 'Yes', 12.0, 'Weak signal near HV winding, phase B'],
        ['2026-02-15', 'CABLE-3', 'Yes', 20.0, 'Sealing end, phase C'],
        ['2025-07-08', 'BUS1', 'No', 0.0, 'No internal source found'],
        ['2025-06-01', 'RB1', 'No', 0.0, 'Routine survey'],
    ], columns=['Date', 'Asset ID', 'Detected', 'Amplitude (mV)', 'Note'])
    sf6 = pd.DataFrame([
        ['2025-06-15', 'Q2-CB2', 0.1, 0.0, 0.1],
        ['2025-12-05', 'Q2-CB2', 2.6, 0.8, 0.4],
        ['2026-02-28', 'Q2-CB2', 6.5, 2.1, 1.0],
        ['2025-12-05', 'Q4-CB1', 0.1, 0.0, 0.0],
        ['2025-12-05', 'BUS1', 0.2, 0.0, 0.1],
    ], columns=['Date', 'Asset ID', 'SO2 (ppmv)', 'SOF2 (ppmv)', 'HF (ppmv)'])
    events = pd.DataFrame([
        ['Q2-CB2', ev_particle.date().isoformat(), 'Planned outage: metallic particle found and removed from CB compartment'],
        ['T2', ev_t2.date().isoformat(), 'Buchholz trip: PD damage found on HV winding lead insulation'],
        ['CABLE-3', ev_cable.date().isoformat(), 'Cable sealing end failure (phase C)'],
    ], columns=['Asset ID', 'Date', 'Event'])
    return {'sensors': reg, 'online': online, 'dga': dga, 'acoustic': acoustic, 'sf6': sf6, 'events': events}


def contributions(row):
    """How many risk points each factor adds for one assessed row (sums to the final PD risk)."""
    d = row['Defect']
    is_pd = d not in (NO_PD, 'External noise')
    f = 0.35 + 0.65 * row['S'] / 10
    out = {'Level above background': 0.55 * row['Magnitude Score'] * f,
           'Trend': 0.25 * row['Trend Score'] * f,
           'Pulse rate': 0.20 * row['Rate Score'] * f}
    sf6v = row.get('SF6 By-products (ppmv)')
    sf6v = None if sf6v is None or pd.isna(sf6v) else float(sf6v)
    h2 = row.get('H2 (ppm)'); c2 = row.get('C2H2 (ppm)')
    h2 = 0 if h2 is None or pd.isna(h2) else float(h2)
    c2 = 0 if c2 is None or pd.isna(c2) else float(c2)
    out['Acoustic confirmation'] = 15.0 if (is_pd and row.get('Acoustic') == 'Yes') else 0.0
    out['SF6 confirmation'] = (20.0 if sf6v >= SF6_HIGH else 10.0 if sf6v >= SF6_INDICATIVE else 0.0) if (is_pd and sf6v is not None) else 0.0
    dl = 2 if (h2 > H2_LIMITS[1] or c2 > C2H2_LIMITS[1]) else 1 if (h2 > H2_LIMITS[0] or c2 > C2H2_LIMITS[0]) else 0
    out['DGA confirmation'] = 10.0 * dl if (is_pd and row['Asset Class'] in ('Transformer', 'Reactor')) else 0.0
    out['Safety rules'] = float(row['PD Risk (%)']) - sum(out.values())
    return {k: round(v, 1) for k, v in out.items()}
