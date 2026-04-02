import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error


# =========================
# CONFIG
# =========================
MODEL_PATH = "../output/model/xgb_model.pkl"
ENCODER_PATH = "../output/model/encoders.pkl"


# =========================
# 1. PREPARE DATA
# =========================

def prepare_data(df):
    df = df.copy()

    # encode categorical
    le_branch = LabelEncoder()
    le_product = LabelEncoder()

    df["branch_enc"] = le_branch.fit_transform(df["branch"])
    df["product_enc"] = le_product.fit_transform(df["product"])

    # save encoders
    joblib.dump(
        {"branch": le_branch, "product": le_product},
        ENCODER_PATH
    )

    FEATURES = [
        "branch_enc",
        "product_enc",
        "month",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3"
    ]

    TARGET = "revenue"

    return df, FEATURES, TARGET


# =========================
# 2. TRAIN MODEL
# =========================

def train_model(df):
    df, FEATURES, TARGET = prepare_data(df)

    # time split
    train = df[df["date"] < "2026-01-01"]
    test  = df[df["date"] >= "2026-01-01"]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    # evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"MAE: {mae:.4f}")

    # baseline
    baseline = test["lag_1"]
    baseline_mae = mean_absolute_error(y_test, baseline)
    print(f"Baseline (lag_1) MAE: {baseline_mae:.4f}")

    # save model
    joblib.dump(model, MODEL_PATH)

    return model


# =========================
# 3. LOAD MODEL
# =========================

def load_model():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders


# =========================
# 4. FORECAST (RECURSIVE)
# =========================

def forecast(df, model, encoders, horizon=10):
    df = df.copy()

    le_branch = encoders["branch"]
    le_product = encoders["product"]

    results = []

    # loop per series
    for (branch, product), group in df.groupby(["branch", "product"]):

        group = group.sort_values("date").copy()

        # encode
        branch_enc = le_branch.transform([branch])[0]
        product_enc = le_product.transform([product])[0]

        history = group.copy()

        for step in range(horizon):

            last_row = history.iloc[-1]

            # build next date
            next_date = last_row["date"] + pd.DateOffset(months=1)

            # construct features
            lag_1 = last_row["revenue"]
            lag_2 = history.iloc[-2]["revenue"]
            lag_3 = history.iloc[-3]["revenue"]

            rolling_mean_3 = np.mean([lag_1, lag_2, lag_3])

            month = next_date.month

            X = pd.DataFrame([{
                "branch_enc": branch_enc,
                "product_enc": product_enc,
                "month": month,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "rolling_mean_3": rolling_mean_3
            }])

            # predict
            pred = model.predict(X)[0]

            # append to results
            results.append({
                "date": next_date,
                "branch": branch,
                "product": product,
                "forecast": pred
            })

            # append to history (IMPORTANT)
            new_row = {
                "date": next_date,
                "branch": branch,
                "product": product,
                "revenue": pred
            }

            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(results)