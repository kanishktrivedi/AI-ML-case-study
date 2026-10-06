import joblib, pandas as pd, streamlit as st
from common import *

st.set_page_config(page_title="AlertRank – SOC Alert Prioritization", layout="wide")
art = joblib.load("outputs/model.joblib"); model, classes, opts = art["model"], art["classes"], art["options"]

def score(df):
    X = make_features(df); P = model.predict_proba(X)
    out = df.copy()
    for i, c in enumerate(classes): out[f"P_{c}"] = P[:, i].round(3)
    out["PriorityScore"] = (P[:, 2] + 0.3 * P[:, 1]).round(3)   # TP weighted fully, BP partially
    out["Priority"] = pd.cut(out.PriorityScore, [-1, .35, .6, 2], labels=["Low", "Medium", "High"])
    return out.sort_values("PriorityScore", ascending=False)

st.title("🛡️ AlertRank – ML-based Security Alert Prioritization")
st.caption("Predicts whether an alert is a True Positive, Benign Positive or False Positive and ranks the queue for analysts.")
t1, t2, t3 = st.tabs(["Score one alert", "Rank an alert queue (CSV)", "Model & EDA"])

with t1:
    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Category", opts["Category"]); ent = c1.selectbox("EntityType", opts["EntityType"])
    role = c2.selectbox("EvidenceRole", opts["EvidenceRole"]); osf = c2.selectbox("OSFamily", opts["OSFamily"])
    cc = c3.selectbox("CountryCode", opts["CountryCode"]); det = c3.selectbox("DetectorId", opts["DetectorId"])
    title = st.selectbox("AlertTitle", opts["AlertTitle"])
    d1, d2 = st.columns(2)
    date = d1.date_input("Date"); time = d2.time_input("Time")
    mitre = st.text_input("MITRE techniques (semicolon-separated, blank if none)", "T1059;T1078")
    if st.button("Score alert", type="primary"):
        row = pd.DataFrame([dict(Timestamp=f"{date} {time}", MitreTechniques=mitre or None, Category=cat, EntityType=ent,
                                 EvidenceRole=role, OSFamily=osf, CountryCode=cc, AlertTitle=title, DetectorId=det)])
        r = score(row).iloc[0]
        m1, m2 = st.columns(2); m1.metric("Priority", str(r.Priority)); m2.metric("Priority score", r.PriorityScore)
        st.bar_chart(pd.Series({c: r[f"P_{c}"] for c in classes}, name="Probability"))

with t2:
    st.write("Upload alerts with columns: Timestamp, MitreTechniques, Category, EntityType, EvidenceRole, OSFamily, CountryCode, AlertTitle, DetectorId.")
    f = st.file_uploader("CSV file", type="csv")
    if f is None and st.checkbox("Use sample from data/alerts.csv"):
        f = "data/alerts.csv"
    if f is not None:
        df = pd.read_csv(f).head(2000); res = score(df)
        st.dataframe(res.head(100), use_container_width=True)
        st.write(res.Priority.value_counts())
        st.download_button("Download ranked queue", res.to_csv(index=False), "ranked_alerts.csv")

with t3:
    st.subheader(f"Selected model: {art['name']}")
    st.dataframe(pd.read_csv("outputs/model_comparison.csv"), use_container_width=True)
    a, b = st.columns(2); a.image("outputs/gain_curve.png"); b.image("outputs/confusion.png")
    a.image("outputs/importance.png"); b.image("outputs/eda.png")
    st.warning("Limitations: labels come from analyst triage and may be noisy; random split ignores concept drift; the model supports, not replaces, analysts.")
