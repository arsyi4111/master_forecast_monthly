import os
import pandas as pd
import subprocess
import sys

from master_data import get_master_data, get_engineered_data
from forecast_logic import train_model, load_model, forecast


OUTPUT_PATH = "data/output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def create_master_and_engineered():
    print("\n[STEP] Creating Master Data...")
    master_df = get_master_data()
    master_path = os.path.join(OUTPUT_PATH, "master_data.csv")
    master_df.to_csv(master_path, index=False)
    print(f"Master data saved to: {master_path}")

    print("\n[STEP] Creating Engineered Data...")
    eng_df = get_engineered_data()
    eng_path = os.path.join(OUTPUT_PATH, "engineered_data.csv")
    eng_df.to_csv(eng_path, index=False)
    print(f"Engineered data saved to: {eng_path}")


def create_model_and_forecast():
    print("\n[STEP] Loading Engineered Data...")
    df = get_engineered_data()

    print("\n[STEP] Training Model...")
    model = train_model(df)

    print("\n[STEP] Loading Model...")
    model, encoders = load_model()

    horizon = input("\nEnter forecast horizon (default=10): ").strip()
    horizon = int(horizon) if horizon else 10

    print(f"\n[STEP] Forecasting {horizon} months ahead...")
    forecast_df = forecast(df, model, encoders, horizon=horizon)

    output_file = os.path.join(OUTPUT_PATH, "forecast_result.csv")
    forecast_df.to_csv(output_file, index=False)

    print(f"\nForecast saved to: {output_file}")
    print("\nSample:")
    print(forecast_df.head())


def forecast_only():
    print("\n[STEP] Loading Engineered Data...")
    df = get_engineered_data()

    print("\n[STEP] Loading Existing Model...")
    model, encoders = load_model()

    horizon = input("\nEnter forecast horizon (default=10): ").strip()
    horizon = int(horizon) if horizon else 10

    print(f"\n[STEP] Forecasting {horizon} months ahead...")
    forecast_df = forecast(df, model, encoders, horizon=horizon)

    output_file = os.path.join(OUTPUT_PATH, "forecast_result.csv")
    forecast_df.to_csv(output_file, index=False)

    print(f"\nForecast saved to: {output_file}")

def run_dashboard():
    print("\n[STEP] Launching Dashboard...")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
            check=True
        )
    except Exception as e:
        print("Failed to launch dashboard.")
        print(e)


def main():
    print("=" * 50)
    print("FINANCIAL SERVICES FORECAST SYSTEM")
    print("=" * 50)

    print("\nSelect Option:")
    print("1. Create Master Data and Engineered Data")
    print("2. Create Forecasting Model and Forecast")
    print("3. Forecast Using Existing Model")
    print("4. Full Pipeline (Data + Model + Forecast)")
    print("5. Run Dashboard")

    choice = input("\nEnter choice (1/2/3/4/5): ").strip()

    if choice == "1":
        create_master_and_engineered()

    elif choice == "2":
        create_model_and_forecast()

    elif choice == "3":
        forecast_only()

    elif choice == "4":
        create_master_and_engineered()
        create_model_and_forecast()

    elif choice == "5":
        run_dashboard()

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()