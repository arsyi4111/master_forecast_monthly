# Financial Services Global Forecast Model

This Project is intended to create a global ML Forecasting model that can be applied to predict nation-wide Financial Services Revenue.

## Main Feature

1. All-in Forecasting Model (Individual Branch (KC) and All 23 Products), the main feature of this project is to be able to do forecasting for every financial services product for every KC.
2. A single click data cleaning, this project can create both master data and feature engineered dataset automatically by using monthly revenue data from Pos Indonesia's Financial Services Dashboard.

### Limitation and Future Improvement

1. Based on literature studies, a specific model trained for a specific branch or product will yield more accurate forecasting results. But, this general model is used to keep the training process efficient yet still yield enough holistic insight.
2. Future improvements include hybrid modeling approaches to handle sparse and anomaly-heavy time series more effectively.

---

## Technical Specification

### Data Cleaning

All data cleaning processes are handled inside *master_data.py*.

The pipeline performs the following steps:

#### 1. Raw Data Ingestion

* Reads monthly Excel files with naming format `MMYY.xlsx`
* Supports multiple year structures:

  * 2025 files contain 2024 and 2025 revenue
  * 2026 files contain 2025 and 2026 revenue

#### 2. Product Standardization

* Extracts product number and name from `Categori Produk`
* Handles product renaming inconsistencies (e.g. EBATARA → BTNPOS)
* Merges specific products (e.g. POSPAY QRIS → POSPAY MERCHANT)

#### 3. Revenue Construction

* Builds a continuous monthly dataset:

  * 2024 derived from 2025 files
  * 2025 from 2025 files
  * 2026 from 2026 files

#### 4. Data Aggregation

Data is aggregated at the following level:

```text
(date, branch, product) → revenue
```

#### 5. Output

* Clean master dataset
* Ready for feature engineering and modeling

---

### Feature Engineering

Feature engineering is also handled inside *master_data.py*.

The following features are generated:

#### Lag Features

* `lag_1`: revenue at t-1
* `lag_2`: revenue at t-2
* `lag_3`: revenue at t-3

#### Rolling Features

* `rolling_mean_3`: average of last 3 months

#### Time Features

* `month`
* `year`

#### Important Design Note

Rows without complete lag history are removed to ensure model input consistency.

---

### Model Architecture

The forecasting model is implemented in *forecast_logic.py*.

#### Model Type

* XGBoost Regressor

#### Input Features

```text
branch_enc
product_enc
month
lag_1
lag_2
lag_3
rolling_mean_3
```

#### Target Variable

```text
revenue
```

#### Encoding

* Label Encoding is applied to:

  * branch
  * product
* Encoders are saved for reuse during inference

---

### Training Strategy

#### Data Split

* Training set: 2024–2025
* Testing set: 2026

#### Evaluation Metrics

* Mean Absolute Error (MAE)
* Baseline comparison using `lag_1`

#### Key Consideration

Random splitting is avoided to preserve temporal dependency.

---

### Forecasting Method

Forecasting is performed using a recursive approach.

#### Process

```text
1. Use latest known data
2. Predict next month
3. Append prediction to history
4. Recompute features
5. Repeat for desired horizon
```

#### Characteristics

* Multi-step forecasting
* Per (branch, product) series
* Sequential dependency on previous predictions

---

### Execution

The pipeline is executed via:

```bash
python run_forecast_test.py
```

#### Available Modes

```text
1. Train model + forecast
2. Use existing model + forecast
3. Only prepare data
```

---

### Output

Forecast results are saved as:

```text
data/output/forecast_result.csv
```

With structure:

| date | branch | product | forecast |
| ---- | ------ | ------- | -------- |

---

## Data Characteristics and Considerations

### 1. Sparse Series

Some products exhibit:

* long periods of zero values
* occasional small spikes

These are not well-suited for standard regression models.

---

### 2. Outliers

Certain branches/products may contain extreme spikes that do not represent normal behavior.

Example:

* normal range: 0.5 – 2.0
* anomaly: > 10.0

These can distort model learning and should be handled appropriately.

---

### 3. Cold Start Period

Due to lag-based features:

* early months without sufficient history are excluded
* this is a deliberate design choice to maintain feature validity

---

## Future Improvement

### Modeling Enhancements

* Hybrid modeling (ML + rule-based)
* Separate handling for sparse time series
* Outlier detection and capping mechanisms

### Feature Engineering

* Additional lag features (e.g. lag_6)
* Extended rolling windows

### Business Layer

* Gap analysis vs target
* Top-performing branch/product identification
* Integration with visualization/dashboard tools

---

## Conclusion

This project provides a scalable and automated forecasting framework for financial services revenue across branches and products.

The main challenge lies not in model complexity, but in handling real-world data characteristics such as:

* sparsity
* anomalies
* temporal inconsistencies

A combination of robust data engineering and appropriate modeling strategy is essential to produce reliable forecasts.

---

## Author

Muhamad Arsyi
