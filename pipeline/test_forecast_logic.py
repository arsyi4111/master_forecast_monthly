import os
import pandas as pd

from master_data import get_engineered_data
from forecast_logic import train_model, load_model, forecast


OUTPUT_PATH = "data/output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def main():
    print("=" * 50)
    print("FORECAST PIPELINE")
    print("=" * 50)

    # =========================
    # 1. LOAD DATA
    # =========================
    print("\nLoading engineered data...")
    df = get_engineered_data()
    print("Data shape:", df.shape)

    # =========================
    # 2. USER INPUT
    # =========================
    print("\nChoose mode:")
    print("1. Train model + forecast")
    print("2. Use existing model + forecast")
    print("3. Only prepare data (no model)")

    choice = input("Enter choice (1/2/3): ").strip()

    # =========================
    # 3. TRAIN / LOAD MODEL
    # =========================
    if choice == "1":
        print("\nTraining model...")
        model = train_model(df)
        model, encoders = load_model()

    elif choice == "2":
        print("\nLoading existing model...")
        model, encoders = load_model()

    elif choice == "3":
        print("\nSkipping model & forecast.")
        df.to_csv(os.path.join(OUTPUT_PATH, "engineered_data.csv"), index=False)
        print("Saved engineered data.")
        return

    else:
        print("Invalid choice.")
        return

    # =========================
    # 4. FORECAST SETTINGS
    # =========================
    horizon = input("\nEnter forecast horizon (months, default=10): ").strip()
    horizon = int(horizon) if horizon else 10

    print(f"\nRunning forecast for {horizon} months...")

    # =========================
    # 5. FORECAST
    # =========================
    forecast_df = forecast(df, model, encoders, horizon=horizon)

    # =========================
    # 6. SAVE OUTPUT
    # =========================
    output_file = os.path.join(OUTPUT_PATH, "forecast_result.csv")
    forecast_df.to_csv(output_file, index=False)

    print(f"\nForecast saved to: {output_file}")
    print("\nSample output:")
    print(forecast_df.head())


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()