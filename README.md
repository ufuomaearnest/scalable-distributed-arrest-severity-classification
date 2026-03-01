
# 7006SCN – Machine Learning & Big Data  
## Comparative Evaluation of Scalable Multi-Class Classification Models for NYPD Arrest Severity Prediction

---

## Project Overview

This project develops a scalable distributed machine learning framework using PySpark to predict arrest severity categories:

- Felony  
- Misdemeanor  
- Violation  
- Other  

The dataset contains approximately 5.99 million records (~1.2GB) obtained from open government data.

The project emphasizes:

- Distributed data engineering  
- Feature engineering  
- Cross-validated model training  
- Scalability analysis (strong and weak scaling)  
- Tableau-based visualization  

---

## Project Structure

```
7006SCN_Project/
│
├── notebooks/
│   ├── 1_data_ingestion.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_model_training.ipynb
│   └── 4_evaluation.ipynb
│
├── scripts/
│   ├── run_pipeline_from_my_notebook.py
│   ├── performance_profiler.py
│   └── test_pipeline.py
│
├── config/
│   ├── spark_config.yaml
│   └── tableau_config.json
│
├── data/
│   ├── schemas/
│   │   ├── nypd_raw_schema.json
│   │   └── ml_ready_schema.json
│   └── samples/
│       ├── sample_nypd_raw.csv
│       └── sample_ml_ready.csv
│
├── Dockerfile
├── environment.yml
├── .gitignore
└── README.md
```

---

## Setup Instructions

### Option 1 – Google Colab

1. Upload the dataset to `/content/`
2. Upload `run_pipeline_from_my_notebook.py`
3. Run:

```
!python run_pipeline_from_my_notebook.py
```

---

### Option 2 – Conda Environment

```
conda env create -f environment.yml
conda activate nypd_bigdata_env
python run_pipeline_from_my_notebook.py
```

---

### Option 3 – Docker

Build the image:

```
docker build -t nypd-bigdata-project .
```

Run:

```
docker run -it nypd-bigdata-project
```

---

## Models Implemented

- Decision Tree  
- Logistic Regression (3-fold Cross-Validation)  
- One-vs-Rest LinearSVC  
- Multilayer Perceptron (MLP)  

Evaluation metrics:

- Accuracy  
- Weighted F1-Score  
- Weighted Precision  
- Weighted Recall  
- ROC Curve  
- Confusion Matrix  

---

## Scalability Analysis

The project evaluates:

- Strong scaling (shuffle partitions tuning)  
- Weak scaling (runtime versus data size)  
- Training time comparison  
- Memory persistence strategies  

Results are exported as CSV files for Tableau visualization.

---

## Tableau Dashboards

1. Data Overview  
2. Model Evaluation  
3. Scalability Analysis  
4. Analytical Insights  

---

## Outputs

Generated outputs are stored in:

```
/content/outputs/
```

Including:

- Model comparison table  
- Confusion matrix  
- ROC curve points  
- Strong scaling results  
- Weak scaling results  

---

## Key Findings

- Decision Tree and Logistic Regression achieved the highest performance.  
- Classical models outperformed the neural network architecture.  
- Distributed Spark processing ensured stable and scalable performance.  

---

## Author

Ofierohor Ufuoma Earnest  
MSc Data Science  
Coventry University  
Module: 7006SCN – Machine Learning & Big Data  

---

## Notes

- The full dataset (1.2GB) is not included in the repository due to size constraints.  
- Sample data is provided for structural validation.  
- All results are reproducible using the provided scripts.

