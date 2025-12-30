# E-commerce Fraud Detection System

A production-ready machine learning pipeline designed to detect fraudulent e-commerce transactions. This project implements a modular architecture, automated testing through GitHub Actions, and model interpretability using SHAP values.

## Project Overview

Fraud detection is a classic imbalanced classification problem. This project uses a **Random Forest Classifier** optimized for **Precision-Recall AUC** to identify suspicious patterns in user behavior, such as transaction velocity and geographic inconsistencies.

### Key Features

* **Modular Architecture:** Clean separation of concerns across `preprocess`, `model`, `train`, and `eval`.
* **Automated CI/CD:** Integrated GitHub Actions to run unit tests on every push.
* **SHAP Interpretability:** Global and local explanations to understand *why* a transaction was flagged.
* **Handling Imbalance:** Implementation of SMOTE (or Stratified Sampling) to handle the minority fraud class.

---

## Project Structure

```text
ecom-fraud-detection/
├── .github/workflows/ml_pipeline.yml   
├── notebooks/           
│   ├── eda.ipynb         
│   └── model.ipynb      
├── src/                 
    ├── preprocess.py      
    ├── model.py         
    ├── train.py         
    └── eval.py          
├── tests/test_preprocess.py                         
├── requirements.txt   
├── .gitignore     
```

---

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Kidus-Yoseph1/ecom-fraud-detection.git
cd ecom-fraud-detection

```


2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

##  How to Run

### 1. Data Preprocessing & Training

The pipeline is designed to be run from the root directory. To train the model with the current configuration:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/train.py

```

### 2. Running Tests

We use `pytest` to ensure data integrity and model consistency:

```bash
python -m pytest tests/

```

---

