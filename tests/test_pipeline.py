#!/usr/bin/env python3
# ============================================
# 7006SCN - test_pipeline.py
# Basic Structural & Integrity Tests
# ============================================

import os
import sys
from pyspark.sql import SparkSession

print("Starting pipeline tests...")

# --------------------------------------------
# 1. Check Required Files
# --------------------------------------------

required_files = [
    "run_pipeline_from_my_notebook.py",
    "performance_profiler.py",
    "spark_config.yaml",
    "tableau_config.json"
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    print("Missing required files:", missing)
else:
    print("All required project files exist.")

# --------------------------------------------
# 2. Spark Session Test
# --------------------------------------------

try:
    spark = SparkSession.builder.appName("TestPipeline").getOrCreate()
    print("Spark initialized successfully.")
except Exception as e:
    print("Spark failed to initialize:", e)
    sys.exit(1)

# --------------------------------------------
# 3. Check Sample Data (if present)
# --------------------------------------------

sample_files = [
    "sample_nypd_raw.csv",
    "sample_ml_ready.csv"
]

for f in sample_files:
    if os.path.exists(f):
        print(f"{f} found.")
    else:
        print(f"{f} not found (optional).")

# --------------------------------------------
# 4. Simple DataFrame Test
# --------------------------------------------

try:
    df = spark.createDataFrame([(1, "test")], ["id", "value"])
    if df.count() == 1:
        print("Spark DataFrame test passed.")
    else:
        print("Spark DataFrame test failed.")
except Exception as e:
    print("Spark DataFrame creation failed:", e)

spark.stop()

print("Pipeline test completed.")
