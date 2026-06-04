# MLOps Model Deployment Platform

## Overview

This project demonstrates an end-to-end MLOps workflow for training, tracking, deploying, monitoring, and retraining machine learning models using modern cloud-native tools.

The platform includes:

* Machine Learning Training Pipeline
* MLflow Experiment Tracking
* FastAPI Inference Service
* Docker Containerization
* Kubernetes Deployment
* AWS EKS Integration
* GitHub Actions CI/CD
* Monitoring and Drift Detection
* Automated Retraining Pipeline

---

## Tech Stack

* Python
* Scikit-Learn
* MLflow
* FastAPI
* Docker
* Kubernetes
* AWS EKS
* GitHub Actions
* Prometheus
* Evidently AI

---

## Project Architecture

Dataset

↓

Training Pipeline

↓

MLflow Tracking

↓

Model Registry

↓

FastAPI Serving Layer

↓

Docker Container

↓

AWS EKS Deployment

↓

Monitoring

↓

Automated Retraining

---

## Features

### Model Training

* Data ingestion
* Data preprocessing
* Feature engineering
* Model training
* Evaluation

### Experiment Tracking

* MLflow experiment logging
* Metrics tracking
* Artifact storage

### Model Serving

* REST API endpoints
* Real-time predictions
* Health monitoring

### Deployment

* Dockerized application
* Kubernetes manifests
* AWS EKS deployment

### Monitoring

* Drift detection
* Performance monitoring
* Model health tracking

### Retraining

* Automated retraining pipeline
* Model replacement workflow

---

## Run Locally

Install dependencies:

pip install -r requirements.txt

Train model:

python src/train.py

Run API:

uvicorn api.app:app --reload

API URL:

http://localhost:8000

Swagger Documentation:

http://localhost:8000/docs

---

## Future Improvements

* Kubeflow Integration
* Feature Store
* Advanced Monitoring
* Multi-Model Deployment
* A/B Testing
