#  Enterprise Retail Store Sales Forecasting (MLOps Pipeline)

An end-to-end, production-ready Machine Learning Operations (MLOps) pipeline for time-series forecasting of retail store sales. This project predicts daily sales for various product families across multiple stores using a highly optimized XGBoost Regressor. It features automated data ingestion from MongoDB Atlas, robust custom feature engineering, strict quality gates, and a fully automated CI/CD deployment pipeline to AWS.

## Project Overview

Unlike standard cross-sectional ML projects, this pipeline is designed specifically for **Time-Series Forecasting**. It handles the complexities of temporal data, including chronological splitting, dynamic lag feature generation, and stateful vs. stateless preprocessing, packaged cleanly into a custom wrapper for seamless real-time inference.

### Key Architectural Highlights

* **Chronological OOT Split:** Replaces random `train_test_split` with a strict Out-of-Time split to prevent data leakage and simulate real-world forecasting accuracy.
* **Custom Time-Series Transformation:** Bypasses standard `ColumnTransformer` to perform complex multi-table Pandas merges (`oil`, `holidays`, `transactions`, `stores`) and generates dynamic features like `lag_1`, `lag_7`, and `rolling_mean_7`.
* **Stateful Preprocessing Dictionary:** Serializes only stateful rules (Label Encoders, Family Mappings, Big Event flags) into a `preprocessing.pkl` dictionary, allowing dynamic lag calculation at inference time.
* **The `MyModel` Wrapper:** A custom Python class that encapsulates both the Data Preprocessor and the Trained XGBoost Model into a single entity, ensuring inference APIs only need to make one `.predict()` call.
* **Automated Quality Gates:** The pipeline automatically fails and prevents model saving if the Test $R^2$ Score drops below a defined threshold (85%) or if the Overfitting delta (Train $R^2$ - Test $R^2$) exceeds 5%.

### Deployment & CI/CD Architecture

* **FastAPI Web Interface:** A lightweight, high-performance web API serving the model with a user-friendly HTML front-end for generating forecasts.
* **Dynamic In-Memory Caching:** Utilizes in-memory dictionaries for static data (like Store City/State mappings) to ensure O(1) lookup times and reduce database load during inference.
* **MongoDB Atlas Integration:** Connects to a cloud NoSQL database to fetch real-time, past 30-day historical contexts (sales and transactions) for dynamic lag feature generation during live predictions.
* **Amazon S3 Model Registry:** Automatically uploads trained, quality-passed model artifacts to an S3 bucket and pulls the latest production model during application startup.
* **Dockerized Environment:** The entire application is containerized using a slim Python image, optimized with a strict `.dockerignore` to keep image sizes minimal.
* **GitHub Actions CI/CD:** A fully automated workflow that builds the Docker image, pushes it to Amazon Elastic Container Registry (ECR), and triggers a pull-and-run deployment on a self-hosted AWS EC2 instance.

## Model Performance

The current XGBoost model generalizes exceptionally well without memorizing the training data.

* **Train $R^2$ Score:** 96.56%
* **Test $R^2$ Score:** 97.45%
* **Test RMSE:** 201.30
* **Overfitting Delta:** ~0.89% (Well within the 5% threshold)

## 🗂️ Project Structure

```text
├── .github/workflows/         # CI/CD pipelines (AWS ECR/EC2 Deployment)
├── artifact/                  # Dynamically generated outputs (models, arrays, logs)
├── config/                    
│   ├── schema.yaml            # Strict schema definitions for data validation
│   └── model.yaml             # XGBoost hyperparameter configurations
├── src/
│   ├── components/            # Core pipeline engines
│   │   ├── data_ingestion.py  # MongoDB fetch & OOT Split
│   │   ├── data_validation.py # Schema enforcement
│   │   ├── data_transformation.py # Feature engineering & merging
│   │   └── model_trainer.py   # XGBoost training & Quality Gates
│   ├── entity/                
│   │   ├── config_entity.py   # Dataclasses for component inputs
│   │   ├── artifacts_entity.py# Dataclasses for component outputs
│   │   └── estimator.py       # Custom MyModel wrapper class
│   ├── pipeline/
│   │   ├── train_pipeline.py  # Automated trigger for training workflow
│   │   └── prediction_pipeline.py # Inference logic & dynamic MongoDB lag generation
│   ├── exception.py           # Custom structured exception handling
│   ├── logger.py              # Centralized logging system
│   └── utils/                 # Helper functions (S3 interactions, YAML read/write)
├── templates/                 # HTML templates for the FastAPI front-end
├── app.py                     # FastAPI application entry point
├── Dockerfile                 # Containerization instructions
├── .dockerignore              # Exclusions for optimized Docker builds
├── demo.py                    # Entry point to trigger the training pipeline locally
├── requirements.txt           # Project dependencies
└── README.md
