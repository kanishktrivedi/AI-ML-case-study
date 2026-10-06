import json, joblib, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, confusion_matrix, classification_report, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from common import *

raw = pd.read_csv("data/alerts.csv")
print("RAW shape:", raw.shape)
quality = {"rows_raw": len(raw), "duplicates": int(raw.duplicated().sum()),
           "missing_pct": (raw.isna().mean()*100).round(2).to_dict()}
df = clean(raw); quality["rows_clean"] = len(df)
df = df.sample(min(len(df), 200_000), random_state=0)
X, y = make_features(df), df[TARGET].map({c:i for i,c in enumerate(CLASSES)})
print(quality)

# ---------- EDA ----------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
df[TARGET].value_counts().reindex(CLASSES).plot.bar(ax=ax[0,0], color=["#999","#f2a900","#d62728"]); ax[0,0].set_title("Class distribution")
(pd.crosstab(df.Category, df[TARGET], normalize="index")[CLASSES]).plot.barh(stacked=True, ax=ax[0,1], color=["#999","#f2a900","#d62728"]); ax[0,1].set_title("Grade mix by Category")
tmp = X.assign(y=df[TARGET].values); (pd.crosstab(tmp.Hour, tmp.y, normalize="index")["TruePositive"]).plot(ax=ax[1,0], marker="o"); ax[1,0].set_title("True-positive rate by hour")
(pd.crosstab(tmp.MitreCount, tmp.y, normalize="index")["TruePositive"]).plot.bar(ax=ax[1,1]); ax[1,1].set_title("True-positive rate by #MITRE techniques")
plt.tight_layout(); plt.savefig("outputs/eda.png", dpi=130); plt.close()
raw.isna().mean().mul(100).sort_values().plot.barh(figsize=(6,4), title="Missing values (%) – raw"); plt.tight_layout(); plt.savefig("outputs/missing.png", dpi=130); plt.close()

# ---------- Models ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
ohe = ColumnTransformer([("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CAT)])
ordn = ColumnTransformer([("num","passthrough",NUM),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=20), CAT)])
models = {
 "Baseline (majority)": Pipeline([("p", ohe), ("m", DummyClassifier(strategy="most_frequent"))]),
 "Logistic Regression": Pipeline([("p", ohe), ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
 "Random Forest": Pipeline([("p", ordn), ("m", RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0))]),
 "Hist Gradient Boosting": Pipeline([("p", ordn), ("m", HistGradientBoostingClassifier(learning_rate=.08, max_iter=300, random_state=0))]),
}
def recall_at_k(y_true, score, k=.2):
    n = int(len(y_true)*k); top = np.argsort(-score)[:n]
    return ((np.asarray(y_true)[top]==2).sum())/max((np.asarray(y_true)==2).sum(),1)

rows, fitted = [], {}
for name, pipe in models.items():
    fit_kw = {"m__sample_weight": compute_sample_weight("balanced", ytr)} if "Boosting" in name else {}
    pipe.fit(Xtr, ytr, **fit_kw); fitted[name] = pipe
    pr = pipe.predict(Xte); pp = pipe.predict_proba(Xte)
    rows.append(dict(Model=name, Accuracy=accuracy_score(yte,pr), Macro_F1=f1_score(yte,pr,average="macro"),
        TP_Recall=recall_score(yte,pr,labels=[2],average=None)[0], TP_PR_AUC=average_precision_score(yte==2, pp[:,2]),
        Recall_in_top20pct=recall_at_k(yte, pp[:,2])))
res = pd.DataFrame(rows).round(4).sort_values("Macro_F1", ascending=False); print(res.to_string(index=False))
res.to_csv("outputs/model_comparison.csv", index=False)

best_name = res.iloc[0]["Model"]; best = fitted[best_name]
pr = best.predict(Xte); pp = best.predict_proba(Xte)
print("\nBest:", best_name); print(classification_report(yte, pr, target_names=CLASSES))
cm = confusion_matrix(yte, pr, normalize="true")
plt.figure(figsize=(5,4)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES); plt.title(f"Confusion matrix – {best_name}")
plt.tight_layout(); plt.savefig("outputs/confusion.png", dpi=130); plt.close()

# Prioritisation curve: how many TPs are found when analysts work top-k% of the queue
order = np.argsort(-pp[:,2]); cum = np.cumsum((yte.values[order]==2))/ (yte==2).sum()
plt.figure(figsize=(5.5,4)); plt.plot(np.arange(1,len(cum)+1)/len(cum), cum, label="Model ranking"); plt.plot([0,1],[0,1],"--",label="Random order")
plt.xlabel("Share of alerts reviewed"); plt.ylabel("Share of true positives found"); plt.legend(); plt.title("Prioritisation gain curve"); plt.tight_layout(); plt.savefig("outputs/gain_curve.png", dpi=130); plt.close()

# Feature importance (permutation, on a subsample)
from sklearn.inspection import permutation_importance
sub = Xte.sample(min(3000,len(Xte)), random_state=0)
pi = permutation_importance(best, sub, yte.loc[sub.index], scoring="f1_macro", n_repeats=5, random_state=0, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
imp.plot.barh(figsize=(6,5), title="Permutation importance (macro-F1 drop)"); plt.tight_layout(); plt.savefig("outputs/importance.png", dpi=130); plt.close()

# Error analysis: worst misses (true positives given lowest priority)
err = Xte.assign(true=yte.map(dict(enumerate(CLASSES))), P_TP=pp[:,2]).query("true=='TruePositive'").sort_values("P_TP").head(20)
err.to_csv("outputs/missed_true_positives.csv")

joblib.dump({"model": best, "name": best_name, "classes": CLASSES, "options": {c: sorted(X[c].unique().tolist())[:200] for c in CAT}}, "outputs/model.joblib")
json.dump(quality, open("outputs/quality.json","w"), indent=1)
imp.sort_values(ascending=False).to_csv("outputs/importance.csv")
