<div align="center">

# 🏪 Enterprise Retail Store Sales Forecasting
### A Production-Ready MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-v1.4-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB_Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/atlas)
[![AWS](https://img.shields.io/badge/AWS_EC2%2FS3%2FECR-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Online-success?style=for-the-badge)](http://54.165.246.236:8000/)

<br/>

> *An end-to-end, production-ready MLOps pipeline for time-series forecasting of retail store sales — from raw MongoDB data to a live AWS-hosted prediction API, fully automated with CI/CD.*

<br/>

---

## 📊 Model Performance at a Glance

| Metric | Value | Status |
|--------|-------|--------|
| 🎯 **Train R² Score** | 96.56% | ✅ Excellent |
| 🎯 **Test R² Score** | **97.45%** | ✅ Excellent |
| 📉 **Test RMSE** | 201.30 | ✅ Low Error |
| ⚖️ **Overfitting Delta** | ~0.89% | ✅ Well within 5% threshold |

> The model **generalises better on unseen data than on training data** — a hallmark of a truly production-ready ML system.

</div>

---

## 🗺️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END MLOPS PIPELINE                            │
│                                                                             │
│  ┌──────────────┐    ┌─────────────────────────────────────────────────┐   │
│  │  MongoDB     │───▶│              TRAINING PIPELINE                  │   │
│  │  Atlas       │    │  ┌──────────┐ ┌──────────┐ ┌─────────────────┐ │   │
│  │  (Raw Data)  │    │  │  Ingest  │▶│ Validate │▶│  Transform &    │ │   │
│  └──────────────┘    │  │  + OOT   │ │  Schema  │ │  Feature Eng.   │ │   │
│                      │  │  Split   │ └──────────┘ └────────┬────────┘ │   │
│                      │  └──────────┘                       │          │   │
│                      │                          ┌──────────▼────────┐ │   │
│                      │                          │  XGBoost Trainer  │ │   │
│                      │                          │  + Quality Gates  │ │   │
│                      │                          └──────────┬────────┘ │   │
│                      └──────────────────────────────────────┼─────────┘   │
│                                                             │              │
│                                               ┌─────────────▼──────────┐  │
│                                               │     Amazon S3          │  │
│                                               │   (Model Registry)     │  │
│                                               └─────────────┬──────────┘  │
│                                                             │              │
│  ┌──────────────────────────────────────────────────────────▼──────────┐  │
│  │                     INFERENCE & DEPLOYMENT                          │  │
│  │                                                                     │  │
│  │  GitHub Actions ──▶ Docker Build ──▶ ECR Push ──▶ EC2 Pull & Run   │  │
│  │                                                                     │  │
│  │  Client ──▶ FastAPI (EC2) ──▶ MyModel.predict() ──▶ Forecast       │  │
│  │                      │                                              │  │
│  │                      └──▶ MongoDB Atlas (Live 30-day lag context)   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Engineering Highlights

### 🔒 Anti-Leakage: Chronological OOT Split
> Replaces naive `random train_test_split` with a **strict Out-of-Time (OOT) split**, ensuring the model is always evaluated on the future — just like real-world forecasting. Zero data leakage guaranteed.

### 🛠️ Custom Time-Series Feature Engineering
> Bypasses standard `ColumnTransformer` entirely. Performs complex **multi-table Pandas merges** (`oil`, `holidays`, `transactions`, `stores`) and generates dynamic temporal features:
> - `lag_1` — Yesterday's sales
> - `lag_7` — Sales from one week ago
> - `rolling_mean_7` — 7-day rolling average trend

### 🧠 Stateful vs. Stateless Preprocessing Architecture
> Serialises only **stateful artefacts** (Label Encoders, Family Mappings, Big Event flags) into a `preprocessing.pkl` dictionary. Stateless features like lag values are computed dynamically at inference time, enabling real-time predictions without stale data.

### 📦 The `MyModel` Wrapper — One Call to Rule Them All
> A custom Python class encapsulates both the **Data Preprocessor** and the **Trained XGBoost Model** into a single entity. Inference APIs call just `.predict()` — no pipeline stitching, no state management.

```python
# All complexity hidden behind a single interface
forecast = model.predict(input_df)
```

### 🚦 Automated Quality Gates
> The pipeline **automatically fails and prevents model saving** if:
> - Test R² drops below **85%**
> - Overfitting delta (Train R² − Test R²) exceeds **5%**
>
> Bad models never reach production. Period.

### ⚡ O(1) In-Memory Caching
> Static reference data (Store City/State mappings) is cached in-memory at startup, delivering **constant-time lookups** and eliminating redundant database hits during high-throughput inference.

---

## 🚀 Deployment & CI/CD

Every `git push` to `main` triggers a **fully automated, zero-downtime deployment**:

```
git push main
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  GitHub Actions Workflow                                        │
│                                                                 │
│  1. Checkout code                                               │
│  2. Configure AWS credentials                                   │
│  3. Login to Amazon ECR                                         │
│  4. docker build ──▶ docker push ──▶ ECR Registry              │
│                                                                 │
│  (Self-hosted EC2 runner picks up next)                         │
│                                                                 │
│  5. docker pull (latest image from ECR)                         │
│  6. docker stop + rm (old container)                            │
│  7. docker run -d --name retail-sales-app -p 8000:8000          │
└─────────────────────────────────────────────────────────────────┘
```

**Infrastructure Stack:**
- 🐳 **Docker** — Slim Python base image, strict `.dockerignore` for minimal layer size
- 📦 **Amazon ECR** — Private container registry
- ☁️ **Amazon EC2** — Self-hosted runner & production server
- 🗄️ **Amazon S3** — Versioned model artefact registry
- 🍃 **MongoDB Atlas** — Cloud NoSQL store for real-time sales context

---

## 📁 Project Structure

```
├── .github/workflows/          # CI/CD: AWS ECR → EC2 automated deployment
├── artifact/                   # Dynamically generated outputs (models, arrays, logs)
├── config/
│   ├── schema.yaml             # Strict schema definitions for data validation
│   └── model.yaml              # XGBoost hyperparameter configurations
├── src/
│   ├── components/             # Core pipeline engines
│   │   ├── data_ingestion.py   # MongoDB fetch & chronological OOT split
│   │   ├── data_validation.py  # Schema enforcement & data quality checks
│   │   ├── data_transformation.py  # Multi-table feature engineering & merging
│   │   └── model_trainer.py    # XGBoost training + automated quality gates
│   ├── entity/
│   │   ├── config_entity.py    # Dataclasses for component configuration inputs
│   │   ├── artifacts_entity.py # Dataclasses for component outputs/artefacts
│   │   └── estimator.py        # Custom MyModel wrapper class
│   ├── pipeline/
│   │   ├── train_pipeline.py   # Automated orchestrator for training workflow
│   │   └── prediction_pipeline.py  # Inference logic & dynamic MongoDB lag fetch
│   ├── exception.py            # Custom structured exception handling
│   ├── logger.py               # Centralised logging system
│   └── utils/                  # Helper functions (S3 I/O, YAML read/write)
├── templates/                  # HTML templates for FastAPI front-end
├── app.py                      # FastAPI application entry point
├── Dockerfile                  # Container build instructions
├── .dockerignore               # Optimised exclusions for minimal image size
├── demo.py                     # Local entry point to trigger training pipeline
├── requirements.txt            # Project dependencies
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- Docker
- MongoDB Atlas URI
- AWS credentials (S3, ECR, EC2)

### Local Training Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd retail-sales-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export MONGO_URL="your_mongodb_atlas_uri"
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_DEFAULT_REGION="us-east-1"

# 4. Trigger the training pipeline
python demo.py
```

### Run with Docker

```bash
docker build -t retail-sales-app .

docker run -d -p 8000:8000 \
  -e MONGO_URL=$MONGO_URL \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  retail-sales-app
```

Then visit `http://localhost:8000` to access the forecasting UI.

---

## 🌐 Live Demo

**The application is deployed and running on AWS EC2:**

> 🔗 **[http://54.165.246.236:8000/](http://54.165.246.236:8000/)**

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|------------|
| **ML Model** | XGBoost v1.4 |
| **API Framework** | FastAPI |
| **Data Store** | MongoDB Atlas |
| **Model Registry** | Amazon S3 |
| **Container Runtime** | Docker |
| **Container Registry** | Amazon ECR |
| **Compute** | Amazon EC2 |
| **CI/CD** | GitHub Actions |
| **Language** | Python 3.10+ |

---






