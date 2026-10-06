"""Synthetic SOC alert generator (fallback when the real GUIDE dataset is not available).
Schema mimics Microsoft GUIDE (Kaggle: 'Microsoft Security Incident Prediction').
Target: IncidentGrade in {TruePositive, BenignPositive, FalsePositive}."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
N = 30000
cats = ["InitialAccess","Exfiltration","Malware","CommandAndControl","SuspiciousActivity","Discovery","Impact","CredentialAccess"]
cat_tp = dict(zip(cats,[.55,.7,.65,.6,.2,.15,.6,.5]))
ents = ["User","Ip","Machine","File","Process","Url","MailMessage"]
roles = ["Impacted","Related"]
det = rng.integers(0, 60, N)
det_bias = rng.normal(0, .8, 60)          # some detectors are noisier than others
cat = rng.choice(cats, N, p=[.2,.05,.12,.08,.25,.12,.05,.13])
ent = rng.choice(ents, N, p=[.25,.2,.2,.1,.1,.1,.05])
role = rng.choice(roles, N, p=[.6,.4])
ts = pd.Timestamp("2024-06-01") + pd.to_timedelta(rng.integers(0, 14*24*3600, N), unit="s")
hour = ts.hour
off = ((hour < 6) | (hour > 21)).astype(int)
mitre_n = rng.poisson(1.2, N)
os_ = rng.choice(["Windows 11","Windows 10","Linux","macOS","Android"], N, p=[.4,.35,.15,.07,.03])
cc = rng.choice(["IN","US","GB","DE","CN","RU","BR","SG"], N, p=[.3,.3,.1,.08,.07,.05,.05,.05])
logit = (np.log(np.array([cat_tp[c] for c in cat])/(1-np.array([cat_tp[c] for c in cat])))
         + det_bias[det] + .7*off + .5*(role=="Impacted") + .25*mitre_n
         + .6*np.isin(cc,["CN","RU"]) - .6*(ent=="Url") + rng.normal(0,.7,N))
p_tp = 1/(1+np.exp(-(logit-.4)))
u = rng.random(N)
grade = np.where(u < p_tp*.75, "TruePositive", np.where(rng.random(N) < .45, "BenignPositive", "FalsePositive"))
df = pd.DataFrame(dict(Timestamp=ts, OrgId=rng.integers(0,40,N), DetectorId=det, AlertTitle=[f"{c}_Rule_{d%12}" for c,d in zip(cat,det)],
    Category=cat, MitreTechniques=[";".join(f"T{1000+i}" for i in rng.integers(0,200,k)) if k else np.nan for k in mitre_n],
    EntityType=ent, EvidenceRole=role, OSFamily=os_, CountryCode=cc, IncidentGrade=grade))
# inject realistic data-quality issues
df.loc[rng.random(N)<.08,"OSFamily"] = np.nan
df.loc[rng.random(N)<.05,"CountryCode"] = np.nan
df.loc[rng.random(N)<.03,"IncidentGrade"] = np.nan
df = pd.concat([df, df.sample(300, random_state=1)], ignore_index=True)   # duplicates
df.to_csv("data/alerts.csv", index=False); print(df.shape); print(df.IncidentGrade.value_counts(dropna=False))
