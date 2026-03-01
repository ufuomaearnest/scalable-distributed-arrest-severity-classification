#!/bin/bash

# ============================================
# 7006SCN - Machine Learning & Big Data
# Environment Setup Script
# ============================================

echo "Installing required Python packages..."

pip install --upgrade pip

# Core Big Data & ML Libraries
pip install pyspark==3.5.1
pip install findspark
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn

echo "Verifying Spark installation..."
python -c "import pyspark; print('PySpark version:', pyspark.__version__)"

echo "Environment setup completed successfully."
