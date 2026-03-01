# README_tableau.md
7006SCN – Machine Learning and Big Data  
Ofierohor Ufuoma Earnest – 16506275

---

## Overview

This Tableau file was built using CSV outputs generated directly from my PySpark notebook executed in Google Colab.

After completing distributed data cleaning, modelling, and evaluation in Spark, I exported Tableau-ready CSV files using `coalesce(1)` to ensure single-part outputs before downloading them locally.

All dashboards in Tableau were created using these exported CSV files.

---

## 1. Cleaned Dataset Export

After data cleaning and preparation, I exported the modelling-ready dataset using:

df.coalesce(1) \
  .write \
  .mode("overwrite") \
  .option("header", "true") \
  .csv(OUT_DIR)

The final extracted file:
/content/NYPD_Tableau_Cleaned.csv

Used in Tableau for:
- Arrest distribution visuals
- Borough analysis
- Age distribution charts
- Severity category breakdown

---

## 2. Model Comparison Results Export

After training the following models:
- Decision Tree
- Logistic Regression
- One-vs-Rest LinearSVC
- Multilayer Perceptron (MLP)

I exported performance metrics using:

results_df.coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(OUT_RESULTS_DIR)

Final exported file:
/content/model_comparison_results.csv

Contains:
- Accuracy
- Weighted F1
- Weighted Precision
- Weighted Recall

Used in Tableau for:
- Model comparison bar charts
- Performance ranking visuals
- Evaluation dashboards

---

## 3. Confusion Matrix Export

The confusion matrix was exported using:

confusion_df.coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(OUT_CONF_DIR)

Final file:
/content/confusion_matrix_counts.csv

Used in Tableau to build:
- Confusion matrix heatmap
- Class-level prediction performance visuals

---

## 4. Manual Model Results CSV

A summarized CSV was also created using pandas:

df.to_csv("model_results.csv", index=False)

Contains:
- Model
- Accuracy
- F1
- WeightedPrecision
- WeightedRecall

Used for simplified Tableau comparison dashboard.

---

## 5. Tableau Workflow

1. Downloaded exported CSV files from Colab  
2. Opened Tableau Desktop  
3. Connected to CSV files  
4. Created worksheets  
5. Built dashboards combining multiple visuals  
6. Saved final file as packaged workbook (.twbx)

---

## 6. Dashboards Created

The Tableau workbook includes:
- Data Overview Dashboard
- Model Performance Comparison Dashboard
- Confusion Matrix Dashboard
- Feature & Arrest Insights Dashboard

All visuals were built strictly from Spark-generated outputs.
