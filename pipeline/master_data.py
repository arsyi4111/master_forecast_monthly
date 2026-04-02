import pandas as pd
import os
import re
from datetime import datetime

# =========================
# CONFIG
# =========================
RAW_PATH = "../data/raw/"

# =========================
# 1. HELPERS
# =========================

def parse_filename_date(filename):
    """
    '0125.xlsx' → datetime(2025,1,1)
    """
    name = filename.replace(".xlsx", "")
    month = int(name[:2])
    year = int("20" + name[2:])
    return datetime(year, month, 1)


def clean_numeric(series):
    """
    Convert Indonesian number format to float
    Example: '1.234,56' → 1234.56
    """
    return (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )


def normalize_product(text):
    try:
        num, name = text.split(".", 1)
        return int(num.strip()), name.strip().upper()
    except:
        return None, text.strip().upper()


# =========================
# 2. LOAD RAW FILES
# =========================

def load_raw_data():
    all_data = []

    files = sorted([
        f for f in os.listdir(RAW_PATH)
        if f.endswith(".xlsx")
        and not f.startswith("~$")
        and len(f.replace(".xlsx", "")) == 4  # ensure MMYY format
    ])

    for file in files:
        path = os.path.join(RAW_PATH, file)
        date = parse_filename_date(file)

        df = pd.read_excel(path)

        df["date"] = date
        df["source_file"] = file

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


# =========================
# 3. PRODUCT STANDARDIZATION
# =========================

def build_product_reference(df):
    """
    Build product mapping with strict priority:
    1. Use 2026 names if available
    2. Else fallback to latest available
    """

    df["prod_num"], df["prod_name"] = zip(
        *df["Categori Produk"].map(normalize_product)
    )

    # =========================
    # 1. PRIORITY: 2026
    # =========================
    df_2026 = df[df["date"].dt.year == 2026]

    mapping_2026 = (
        df_2026[["prod_num", "prod_name"]]
        .dropna()
        .drop_duplicates("prod_num")
    )

    mapping_dict = dict(zip(mapping_2026["prod_num"], mapping_2026["prod_name"]))

    # =========================
    # 2. FALLBACK: other years
    # =========================
    df_rest = df[~df["prod_num"].isin(mapping_dict.keys())]

    mapping_rest = (
        df_rest
        .sort_values("date", ascending=False)
        [["prod_num", "prod_name"]]
        .dropna()
        .drop_duplicates("prod_num")
    )

    mapping_dict.update(dict(zip(mapping_rest["prod_num"], mapping_rest["prod_name"])))


    return mapping_dict


def apply_product_mapping(df, mapping_dict):
    df["prod_num"], df["prod_name_raw"] = zip(
        *df["Categori Produk"].map(normalize_product)
    )

    # 🔥 ensure prod_num is int
    df["prod_num"] = pd.to_numeric(df["prod_num"], errors="coerce").astype("Int64")

    # mapping
    df["product"] = df["prod_num"].map(mapping_dict)

    # 🔥 fallback (IMPORTANT)
    df["product"] = df["product"].fillna(df["prod_name_raw"])

    # 🔥 force string (avoid weird dtype)
    df["product"] = df["product"].astype(str)

    # 🔥 rename EBATARA → BTNPOS
    df["product"] = df["product"].str.replace("EBATARA", "BTNPOS", regex=False)

    # 🔥 merge QRIS → MERCHANT
    df.loc[df["prod_num"] == 23, "product"] = "POSPAY MERCHANT"
    df.loc[df["prod_num"] == 23, "prod_num"] = 21

    # 🚨 FINAL GUARD (VERY IMPORTANT)
    if df["product"].isna().sum() > 0:
        raise ValueError("❌ Product mapping still has NaN!")

    return df


# =========================
# 4. REVENUE ASSIGNMENT
# =========================

def safe_numeric(df, col):
    if col in df.columns:
        return clean_numeric(df[col])
    else:
        return pd.Series(0, index=df.index)


def assign_revenue(df):
    df["Kinerja 2024"] = safe_numeric(df, "Kinerja 2024")
    df["Kinerja 2025"] = safe_numeric(df, "Kinerja 2025")
    df["Kinerja 2026"] = safe_numeric(df, "Kinerja 2026")

    # ---- 2024 (from 2025 files) ----
    df_2024 = df[df["date"].dt.year == 2025].copy()
    df_2024["date"] = df_2024["date"] - pd.DateOffset(years=1)
    df_2024["revenue"] = df_2024["Kinerja 2024"]

    # ---- 2025 ----
    df_2025 = df[df["date"].dt.year == 2025].copy()
    df_2025["revenue"] = df_2025["Kinerja 2025"]

    # ---- 2026 ----
    df_2026 = df[df["date"].dt.year == 2026].copy()
    df_2026["revenue"] = df_2026["Kinerja 2026"]

    df_final = pd.concat([df_2024, df_2025, df_2026], ignore_index=True)

    # 🔥 FINAL SAFETY
    df_final["revenue"] = df_final["revenue"].fillna(0)

    return df_final


# =========================
# 5. COMPLETE TIME SERIES
# =========================

def ensure_full_timeseries(df):
    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")

    full_index = pd.MultiIndex.from_product(
        [df["branch"].unique(), df["product"].unique(), all_dates],
        names=["branch", "product", "date"]
    )

    df = df.set_index(["branch", "product", "date"]) \
           .reindex(full_index) \
           .reset_index()

    df["revenue"] = df["revenue"].fillna(0)

    return df


# =========================
# 6. MASTER DATA
# =========================

def get_master_data():
    df = load_raw_data()

    # rename
    df = df.rename(columns={
        "Reg": "region",
        "Nama KCU": "kcu",
        "Nama Kantor": "branch"
    })

    # product standardization
    mapping = build_product_reference(df)
    df = apply_product_mapping(df, mapping)

    # assign revenue
    df = assign_revenue(df)

    # select columns
    df = df[["date", "branch", "product", "revenue"]]

    # aggregate after product merge
    df = (
        df.groupby(["date", "branch", "product"], as_index=False)
        ["revenue"].sum()
    )

    # remove null
    df = df.dropna(subset=["revenue"])

    # ensure full timeline
    df = ensure_full_timeseries(df)

    # sort
    df = df.sort_values(["branch", "product", "date"])

    return df


# =========================
# 7. FEATURE ENGINEERING
# =========================

def get_engineered_data():
    df = get_master_data()

    df = df.sort_values(["branch", "product", "date"])

    # lags
    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = (
            df.groupby(["branch", "product"])["revenue"]
            .shift(lag)
        )

    df["rolling_mean_3"] = (
        df.groupby(["branch", "product"])["revenue"]
        .transform(lambda x: x.rolling(3).mean())
    )

    # time features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # drop NA from lagging
    df = df.dropna()

    return df


# =========================
# TEST RUN
# =========================

if __name__ == "__main__":
    df_master = get_master_data()
    print("Master data shape:", df_master.shape)
    print(df_master.head())

    df_feat = get_engineered_data()
    print("Engineered data shape:", df_feat.shape)
    print(df_feat.head())