#!/usr/bin/env python3
# ============================================
# 7006SCN - performance_profiler.py
# Big Data Scalability & Performance Analysis
# ============================================

import time
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# --------------------------------------------
# Spark Session
# --------------------------------------------

spark = (
    SparkSession.builder
    .appName("NYPD_Performance_Profiler")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

print("Spark version:", spark.version)

# --------------------------------------------
# Load Prepared Dataset (must already exist)
# --------------------------------------------

TRAIN_PATH = "/content/train_ready.parquet"

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError("train_ready.parquet not found in /content/. Run pipeline first.")

train_df = spark.read.parquet(TRAIN_PATH)
train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)

print("Training rows:", train_df.count())

# --------------------------------------------
# 1. Training Time Measurement
# --------------------------------------------

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)

start = time.time()
model = lr.fit(train_df)
end = time.time()

training_time = end - start
print("Logistic Regression training time:", round(training_time, 2), "seconds")

# --------------------------------------------
# 2. Strong Scaling Test (Shuffle Partitions)
# --------------------------------------------

partition_tests = [50, 100, 200, 400]
strong_scaling_results = []

for p in partition_tests:
    spark.conf.set("spark.sql.shuffle.partitions", str(p))
    
    start = time.time()
    lr.fit(train_df)
    end = time.time()
    
    runtime = end - start
    strong_scaling_results.append((p, runtime))
    print(f"Partitions: {p} | Runtime: {round(runtime, 2)} sec")

# --------------------------------------------
# 3. Weak Scaling Test (Data Size %)
# --------------------------------------------

weak_scaling_results = []
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

for frac in fractions:
    sample_df = train_df.sample(fraction=frac, seed=42)
    
    start = time.time()
    lr.fit(sample_df)
    end = time.time()
    
    runtime = end - start
    weak_scaling_results.append((frac, runtime))
    print(f"Data Fraction: {frac} | Runtime: {round(runtime, 2)} sec")

# --------------------------------------------
# Save Results
# --------------------------------------------

output_dir = "/content/outputs"
os.makedirs(output_dir, exist_ok=True)

spark.createDataFrame(strong_scaling_results, ["ShufflePartitions", "RuntimeSeconds"]) \
    .coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(os.path.join(output_dir, "strong_scaling"))

spark.createDataFrame(weak_scaling_results, ["DataFraction", "RuntimeSeconds"]) \
    .coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(os.path.join(output_dir, "weak_scaling"))

print("Performance profiling completed successfully.")

spark.stop()
