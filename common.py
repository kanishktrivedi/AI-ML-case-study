import numpy as np, pandas as pd
CAT = ["Category","EntityType","EvidenceRole","OSFamily","CountryCode","AlertTitle","DetectorId"]
NUM = ["Hour","DayOfWeek","OffHours","MitreCount","MitreMissing"]
TARGET = "IncidentGrade"
CLASSES = ["FalsePositive","BenignPositive","TruePositive"]

def clean(df, training=True):
    df = df.copy()
    if training:
        df = df.drop_duplicates()
        df = df.dropna(subset=[TARGET]); df = df[df[TARGET].isin(CLASSES)]
    return df

def make_features(df):
    d = pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    d["Hour"] = ts.dt.hour.fillna(12).astype(int)
    d["DayOfWeek"] = ts.dt.dayofweek.fillna(0).astype(int)
    d["OffHours"] = ((d.Hour < 6) | (d.Hour > 21)).astype(int)
    m = df["MitreTechniques"]
    d["MitreMissing"] = m.isna().astype(int)
    d["MitreCount"] = m.fillna("").apply(lambda s: len([x for x in str(s).split(";") if x]))
    for c in CAT:
        d[c] = df[c].astype("object").where(df[c].notna(), "Unknown").astype(str)
    return d[NUM + CAT]
