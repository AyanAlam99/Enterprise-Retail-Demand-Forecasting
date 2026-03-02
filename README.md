# 📈 Enterprise Retail Store Sales Forecasting (MLOps Pipeline)

An end-to-end, production-ready Machine Learning Operations (MLOps) pipeline for time-series forecasting of retail store sales. This project predicts daily sales for various product families across multiple stores using a highly optimized XGBoost Regressor, complete with automated data ingestion from MongoDB, robust custom feature engineering, and strict quality gates.

##  Project Overview

Unlike standard cross-sectional ML projects, this pipeline is designed specifically for **Time-Series Forecasting**. It handles the complexities of temporal data, including chronological splitting, dynamic lag feature generation, and stateful vs. stateless preprocessing, packaged cleanly into a custom wrapper for seamless deployment.

### 🌟 Key Architectural Highlights

* **Chronological OOT Split:** Replaces random `train_test_split` with a strict Out-of-Time split to prevent data leakage and simulate real-world forecasting accuracy.
* **Custom Time-Series Transformation:** Bypasses standard `ColumnTransformer` to perform complex multi-table Pandas merges (`oil`, `holidays`, `transactions`, `stores`) and generates dynamic features like `lag_1`, `lag_7`, and `rolling_mean_7`.
* **Stateful Preprocessing Dictionary:** Serializes only stateful rules (Label Encoders, Family Mappings, Big Event flags) into a `preprocessing.pkl` dictionary, allowing dynamic lag calculation at inference time.
* **The `MyModel` Wrapper:** A custom Python class that encapsulates both the Data Preprocessor and the Trained XGBoost Model into a single entity, ensuring inference APIs only need to make one `.predict()` call.
* **YAML-Driven Hyperparameters:** Model parameters are strictly decoupled from code and loaded dynamically from `config/model.yaml`.
* **Automated Quality Gates:** The pipeline automatically fails and prevents model saving if the Test $R^2$ Score drops below a defined threshold (85%) or if the Overfitting delta (Train $R^2$ - Test $R^2$) exceeds 5%.

## 📊 Model Performance

The current XGBoost model generalizes exceptionally well without memorizing the training data.

* **Train $R^2$ Score:** 96.56%
* **Test $R^2$ Score:** 97.45%
* **Test RMSE:** 201.30
* **Overfitting Delta:** ~0.89% (Well within the 5% threshold)

## 🗂️ Project Structure

```text
├── artifact/                  # Dynamically generated outputs (models, arrays, logs)
├── config/                    
│   ├── schema.yaml            # Strict schema definitions for data validation
│   └── model.yaml             # XGBoost hyperparameter configurations
├── src/
│   ├── components/            # Core pipeline engines
│   │   ├── data_ingestion.py      # MongoDB fetch & OOT Split
│   │   ├── data_validation.py     # Schema enforcement
│   │   ├── data_transformation.py # Feature engineering & merging
│   │   └── model_trainer.py       # XGBoost training & Quality Gates
│   ├── entity/                
│   │   ├── config_entity.py       # Dataclasses for component inputs
│   │   ├── artifacts_entity.py    # Dataclasses for component outputs
│   │   └── estimator.py           # Custom MyModel wrapper class
│   ├── pipeline/
│   │   ├── train_pipeline.py      # Automated trigger for training workflow
│   │   └── prediction_pipeline.py # Inference logic & dynamic lag generation
│   ├── exception.py           # Custom structured exception handling
│   ├── logger.py              # Centralized logging system
│   └── utils/                 # Helper functions (YAML read/write, model save/load)
├── demo.py                    # Entry point to trigger the training pipeline
├── requirements.txt           # Project dependencies
└── README.md
