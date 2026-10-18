"""
CyberGuard - layered AI threat detection (MVP)
Run:  streamlit run app.py
Modules: SMS scam NLP | URL threat check (VirusTotal + heuristics) | Image OCR | Anomaly detection (Isolation Forest)
Only lightweight libraries are used (scikit-learn, pandas, numpy) to stay friendly to Linux on IBM Z (s390x).
"""
import base64
import os
import re
import sqlite3
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import pytesseract
    from PIL import Image
    OCR_OK = True
except Exception:  # OCR is optional
    OCR_OK = False

st.set_page_config(page_title="CyberGuard", page_icon="🛡️", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "cyberguard.db")
URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+|\b[a-z0-9\-]+\.(?:com|net|org|xyz|top|click|info|ru|cn|tk|ml|ga|cf|gq|link|online|site)\b[^\s]*)", re.I)

# ----------------------------------------------------------------------------
# Storage (SQLite locally; swap the connection for a cloud DB in production)
# ----------------------------------------------------------------------------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS scans (ts TEXT, module TEXT, input TEXT, verdict TEXT, score REAL)")
    return conn


def log_scan(module, text, verdict, score):
    try:
        with db() as conn:
            conn.execute("INSERT INTO scans VALUES (?,?,?,?,?)",
                         (datetime.now().isoformat(timespec="seconds"), module, text[:300], verdict, float(score)))
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Module 1: SMS scam classifier
# ----------------------------------------------------------------------------
FALLBACK = [
    ("spam", "URGENT! You have won a 1000 prize. Click http://claim-prize.xyz now to claim"),
    ("spam", "Your bank account is suspended. Verify immediately at http://secure-login-bank.top"),
    ("spam", "Congratulations! Free entry to win an iPhone. Reply WIN to 87121"),
    ("spam", "Dear customer your KYC is expired. Update now or account will be blocked http://kyc-update.click"),
    ("spam", "You have been selected for a loan of 50000. Apply now, no documents needed"),
    ("ham", "Hey are we meeting for lunch today?"),
    ("ham", "Please send me the notes from yesterday's class"),
    ("ham", "I'll call you when I reach home"),
    ("ham", "Happy birthday! Have a great day"),
    ("ham", "The meeting is moved to 4 pm, see you in the conference room"),
]


@st.cache_resource(show_spinner="Training SMS model...")
def train_sms():
    path = os.path.join(HERE, "sms.tsv")
    if os.path.exists(path):
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
        source = "SMS Spam Collection (5.5k messages)"
    else:
        df = pd.DataFrame(FALLBACK * 6, columns=["label", "text"])
        source = "tiny built-in demo set (add sms.tsv for real metrics)"
    y = (df["label"] == "spam").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(df["text"], y, test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(vec.fit_transform(Xtr), ytr)
    pred = clf.predict(vec.transform(Xte))
    metrics = {
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred, zero_division=0),
        "Recall": recall_score(yte, pred, zero_division=0),
        "F1": f1_score(yte, pred, zero_division=0),
    }
    return vec, clf, metrics, confusion_matrix(yte, pred), source


def sms_predict(text):
    vec, clf, *_ = train_sms()
    X = vec.transform([text])
    prob = float(clf.predict_proba(X)[0, 1])
    names = np.array(vec.get_feature_names_out())
    contrib = X.toarray()[0] * clf.coef_[0]
    idx = np.argsort(-contrib)[:5]
    top = [(names[i], round(float(contrib[i]), 3)) for i in idx if contrib[i] > 0]
    return prob, top


# ----------------------------------------------------------------------------
# Module 2: URL threat check (heuristics always, VirusTotal when a key is given)
# ----------------------------------------------------------------------------
SUSPICIOUS_TLDS = {"xyz", "top", "click", "tk", "ml", "ga", "cf", "gq", "link", "online", "site", "ru", "cn"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy"}
KEYWORDS = ["login", "verify", "secure", "update", "account", "bank", "kyc", "prize", "claim", "free", "wallet", "otp", "refund"]


def url_heuristics(url):
    u = url if url.startswith("http") else "http://" + url
    p = urlparse(u)
    host = (p.hostname or "").lower()
    reasons, score = [], 0.0
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        reasons.append("Uses a raw IP address"); score += 0.35
    if host.split(".")[-1] in SUSPICIOUS_TLDS:
        reasons.append(f"Suspicious TLD (.{host.split('.')[-1]})"); score += 0.3
    if host in SHORTENERS:
        reasons.append("URL shortener hides the real destination"); score += 0.25
    if "@" in u:
        reasons.append("Contains '@' (can disguise the real host)"); score += 0.25
    if host.count("-") >= 2:
        reasons.append("Many hyphens in domain"); score += 0.15
    if host.count(".") >= 3:
        reasons.append("Deeply nested subdomains"); score += 0.15
    hits = [k for k in KEYWORDS if k in u.lower()]
    if hits:
        reasons.append("Phishing-style keywords: " + ", ".join(hits[:4])); score += min(0.3, 0.1 * len(hits))
    if p.scheme == "http":
        reasons.append("Not using HTTPS"); score += 0.1
    if len(u) > 75:
        reasons.append("Unusually long URL"); score += 0.1
    return min(score, 1.0), reasons


def virustotal(url, api_key):
    try:
        uid = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        r = requests.get(f"https://www.virustotal.com/api/v3/urls/{uid}", headers={"x-apikey": api_key}, timeout=15)
        if r.status_code == 404:
            return None, "URL not yet known to VirusTotal"
        r.raise_for_status()
        stats = r.json()["data"]["attributes"]["last_analysis_stats"]
        total = max(sum(stats.values()), 1)
        return (stats.get("malicious", 0) + 0.5 * stats.get("suspicious", 0)) / total, stats
    except Exception as e:
        return None, f"VirusTotal unavailable ({e.__class__.__name__})"


def url_report(url, api_key=None):
    h_score, reasons = url_heuristics(url)
    vt_score, vt_info = (None, None)
    if api_key:
        vt_score, vt_info = virustotal(url, api_key)
    final = h_score if vt_score is None else max(h_score, min(1.0, vt_score * 5))
    return final, reasons, vt_score, vt_info


# ----------------------------------------------------------------------------
# Module 4: Anomaly detection on simulated device / application telemetry
# ----------------------------------------------------------------------------
@st.cache_data
def make_telemetry(n=500, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "cpu_pct": rng.normal(35, 8, n).clip(1, 100),
        "mem_pct": rng.normal(50, 10, n).clip(1, 100),
        "network_kb_s": rng.normal(300, 80, n).clip(1, None),
        "failed_logins": rng.poisson(0.4, n),
        "api_calls_min": rng.normal(120, 25, n).clip(1, None),
    })
    bad = rng.choice(n, size=int(n * 0.04), replace=False)
    df.loc[bad, "cpu_pct"] = rng.normal(92, 4, len(bad))
    df.loc[bad, "network_kb_s"] = rng.normal(2200, 400, len(bad))
    df.loc[bad, "failed_logins"] = rng.poisson(12, len(bad))
    df["injected_attack"] = False
    df.loc[bad, "injected_attack"] = True
    return df


def run_isolation_forest(df, contamination):
    feats = ["cpu_pct", "mem_pct", "network_kb_s", "failed_logins", "api_calls_min"]
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    out = df.copy()
    out["anomaly"] = model.fit_predict(out[feats]) == -1
    out["anomaly_score"] = -model.score_samples(out[feats])
    return out


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def verdict(score):
    if score >= 0.7:
        return "🔴 HIGH RISK"
    if score >= 0.4:
        return "🟠 SUSPICIOUS"
    return "🟢 LIKELY SAFE"


def full_text_scan(text, api_key):
    """Layered scan: NLP + every URL found in the text."""
    prob, top = sms_predict(text)
    urls = [m.group(0).rstrip(".,)") for m in URL_RE.finditer(text)]
    url_results = [(u, *url_report(u, api_key)) for u in urls[:5]]
    url_max = max([r[1] for r in url_results], default=0.0)
    combined = min(1.0, 0.6 * prob + 0.4 * url_max) if urls else prob
    combined = max(combined, 0.85 * url_max, prob * 0.9)
    return prob, top, url_results, combined


def show_scan(text, api_key, module):
    prob, top, url_results, combined = full_text_scan(text, api_key)
    c1, c2, c3 = st.columns(3)
    c1.metric("Scam-text probability", f"{prob:.0%}")
    c2.metric("URLs found", len(url_results))
    c3.metric("Combined risk score", f"{combined:.0%}")
    st.subheader(verdict(combined))
    st.progress(float(combined))
    if top:
        st.write("**Words pushing the model towards 'scam':** " + ", ".join(f"`{w}`" for w, _ in top))
    for u, s, reasons, vt_score, vt_info in url_results:
        with st.expander(f"URL: {u}  —  risk {s:.0%}", expanded=True):
            for r in reasons or ["No heuristic red flags"]:
                st.write("• " + r)
            if vt_info is not None:
                st.write("VirusTotal:", vt_info)
    log_scan(module, text, verdict(combined), combined)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🛡️ CyberGuard")
st.caption("Layered AI threat detection: scam messages • malicious URLs • suspicious images • anomalous behaviour  |  Built for Linux on IBM Z")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("VirusTotal API key (optional)", type="password",
                            value=os.environ.get("VT_API_KEY", ""),
                            help="Without a key the URL module uses built-in heuristics only.")
    st.markdown("---")
    st.markdown("**Architecture**\n\nSMS NLP → URL intel → OCR → Isolation Forest → one risk score. All components are lightweight, pure-Python/scikit-learn and portable to s390x.")

tab_sms, tab_url, tab_img, tab_anom, tab_hist = st.tabs(
    ["💬 SMS / Message", "🔗 URL Check", "🖼️ Image (OCR)", "📈 Anomaly Detection", "🗂️ Scan history"])

with tab_sms:
    st.subheader("Scan a message")
    examples = {
        "Pick an example…": "",
        "Scam": "URGENT! Your SBI account is suspended. Verify KYC now at http://sbi-kyc-update.top/login or it will be blocked",
        "Prize": "Congratulations! You won a free iPhone. Claim at bit.ly/free-iphone-win now",
        "Normal": "Hey, are we still meeting at 6 for the project discussion?",
    }
    pick = st.selectbox("Quick examples", list(examples))
    msg = st.text_area("Message text", value=examples[pick], height=120)
    if st.button("Analyze message", type="primary") and msg.strip():
        show_scan(msg, api_key, "sms")
    with st.expander("Model details & evaluation"):
        _, _, metrics, cm, source = train_sms()
        st.write(f"Training data: **{source}**  |  Model: TF-IDF (1-2 grams) + Logistic Regression")
        st.dataframe(pd.DataFrame(metrics, index=["score"]).style.format("{:.3f}"))
        st.write("Confusion matrix (rows = actual, columns = predicted; order: ham, spam)")
        st.dataframe(pd.DataFrame(cm, index=["actual ham", "actual spam"], columns=["pred ham", "pred spam"]))

with tab_url:
    st.subheader("Check a URL")
    url = st.text_input("URL", placeholder="http://secure-login-bank.top/verify")
    if st.button("Check URL", type="primary") and url.strip():
        s, reasons, vt_score, vt_info = url_report(url.strip(), api_key)
        st.metric("URL risk score", f"{s:.0%}")
        st.subheader(verdict(s))
        st.progress(float(s))
        for r in reasons or ["No heuristic red flags"]:
            st.write("• " + r)
        if api_key:
            st.write("VirusTotal:", vt_info)
        else:
            st.info("Add a VirusTotal API key in the sidebar for live threat intelligence.")
        log_scan("url", url, verdict(s), s)

with tab_img:
    st.subheader("Extract text from a suspicious image and analyze it")
    up = st.file_uploader("Upload a screenshot / image", type=["png", "jpg", "jpeg", "webp"])
    extracted = ""
    if up is not None:
        img = Image.open(up) if OCR_OK else None
        if img is not None:
            st.image(img, width=360)
            try:
                extracted = pytesseract.image_to_string(img).strip()
            except Exception as e:
                st.warning(f"OCR engine not found ({e.__class__.__name__}). Install tesseract-ocr or paste the text below.")
        else:
            st.warning("pytesseract/Pillow not installed. Paste the text manually below.")
    text_in = st.text_area("Extracted text (editable)", value=extracted, height=140)
    if st.button("Analyze extracted text", type="primary") and text_in.strip():
        show_scan(text_in, api_key, "image")

with tab_anom:
    st.subheader("Behavioural anomaly detection (Isolation Forest)")
    st.caption("Simulated device/application telemetry with a few injected attack patterns (CPU spikes, traffic bursts, brute-force logins).")
    cont = st.slider("Expected anomaly rate", 0.01, 0.15, 0.04, 0.01)
    res = run_isolation_forest(make_telemetry(), cont)
    flagged = res[res["anomaly"]]
    caught = int((res["anomaly"] & res["injected_attack"]).sum())
    total_attacks = int(res["injected_attack"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(res))
    c2.metric("Flagged anomalies", len(flagged))
    c3.metric("Injected attacks caught", f"{caught}/{total_attacks}")
    plot = res.assign(status=np.where(res["anomaly"], "anomaly", "normal"))
    st.scatter_chart(plot, x="network_kb_s", y="cpu_pct", color="status", height=380)
    st.write("**Top flagged records**")
    st.dataframe(flagged.sort_values("anomaly_score", ascending=False).head(15).round(2), use_container_width=True)

with tab_hist:
    st.subheader("Recent scans (stored in database)")
    try:
        with db() as conn:
            hist = pd.read_sql_query("SELECT * FROM scans ORDER BY ts DESC LIMIT 50", conn)
        st.dataframe(hist, use_container_width=True) if len(hist) else st.info("No scans yet.")
    except Exception as e:
        st.warning(f"History unavailable: {e}")
