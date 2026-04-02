import os
import pandas as pd

from master_data import (
    load_raw_data,
    build_product_reference,
    apply_product_mapping,
    assign_revenue,
    get_master_data,
    get_engineered_data
)

# =========================
# CONFIG
# =========================

DEBUG_PATH = "data/debug/"
os.makedirs(DEBUG_PATH, exist_ok=True)


def export(df, name):
    path = os.path.join(DEBUG_PATH, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"[EXPORT] {name} → {path}")


# =========================
# RUNNER
# =========================

def run_all_tests():
    print("=" * 50)
    print("TESTING MASTER DATA PIPELINE")
    print("=" * 50)

    test_load_raw()
    test_product_mapping()
    test_revenue_assignment()
    test_master_output()
    test_engineered_output()

    print("\nALL TESTS COMPLETED ✅")


# =========================
# 1. LOAD RAW
# =========================

def test_load_raw():
    print("\n[TEST] Load Raw Data")

    df = load_raw_data()

    export(df, "01_raw_data")

    print("Shape:", df.shape)
    print(df.head())

    assert len(df) > 0, "Raw data is empty"
    assert "date" in df.columns, "Missing 'date' column"

    print("✔ Raw data loaded")


# =========================
# 2. PRODUCT MAPPING
# =========================

def test_product_mapping():
    print("\n[TEST] Product Mapping")

    df = load_raw_data()

    mapping = build_product_reference(df)
    df = apply_product_mapping(df, mapping)

    export(df, "02_product_mapping")

    print("\nSample Mapping:")
    print(df[["Categori Produk", "product"]].drop_duplicates().head(10))

    # 🔥 DEBUG: check product 13
    print("\n[DEBUG] Product 13 mapping:")
    test_13 = df[df["prod_num"] == 13][
        ["date", "Categori Produk", "product"]
    ].drop_duplicates()

    print(test_13.head(20))

    assert df["product"].notna().all(), "Some products are not mapped"

    print("✔ Product mapping OK")


# =========================
# 3. REVENUE ASSIGNMENT
# =========================

def test_revenue_assignment():
    print("\n[TEST] Revenue Assignment")

    df = load_raw_data()

    mapping = build_product_reference(df)
    df = apply_product_mapping(df, mapping)

    df = assign_revenue(df)

    export(df, "03_revenue_assignment")

    print(df[["date", "source_file", "revenue"]].head())

    # DEBUG nulls
    nulls = df[df["revenue"].isna()]
    if len(nulls) > 0:
        print("\n[DEBUG] NULL revenue rows:")
        print(nulls.head())

    assert "revenue" in df.columns, "Revenue column missing"
    assert df["revenue"].isna().sum() == 0, "Revenue has null values"

    print("✔ Revenue assignment OK")


# =========================
# 4. MASTER DATA
# =========================

def test_master_output():
    print("\n[TEST] Master Data Output")

    df = get_master_data()

    export(df, "04_master_data")

    print("Shape:", df.shape)
    print(df.head())

    assert set(["date", "branch", "product", "revenue"]).issubset(df.columns)
    assert df["revenue"].isna().sum() == 0

    dup = df.duplicated(subset=["date", "branch", "product"]).sum()
    assert dup == 0, f"Duplicate rows found: {dup}"

    print("✔ Master dataset OK")


# =========================
# 5. ENGINEERED DATA
# =========================

def test_engineered_output():
    print("\n[TEST] Engineered Data")

    df = get_engineered_data()

    export(df, "05_engineered_data")

    print("Shape:", df.shape)
    print(df.head())

    expected_cols = [
        "lag_1", "lag_2", "lag_3",
        "rolling_mean_3",
        "month", "year"
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing feature: {col}"

    assert df.isna().sum().sum() == 0, "Engineered data has NA values"

    print("✔ Engineered dataset OK")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    run_all_tests()