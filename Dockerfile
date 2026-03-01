
# ============================================
# 7006SCN - Dockerfile
# NYPD Arrest Severity Big Data Project
# ============================================

# Base Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Java required for Spark)
RUN apt-get update &&     apt-get install -y openjdk-11-jdk &&     apt-get clean

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Copy project files into container
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install pyspark==3.5.1     numpy     pandas     scikit-learn     matplotlib     seaborn     findspark

# Expose port (if needed for Jupyter or Spark UI)
EXPOSE 8888

# Default command
CMD ["python", "run_pipeline_from_my_notebook.py"]
