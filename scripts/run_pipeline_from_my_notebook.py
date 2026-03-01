# ============================================
# Auto-generated from uploadedcsvfile (5).ipynb
# This file contains EXACTLY your notebook code
# ============================================

!apt-get install openjdk-8-jdk-headless -qq > /dev/null

!java -version

!pip install pyspark==3.5.1

import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/usr/local/lib/python3.10/dist-packages/pyspark"

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("NYC_Cleaning")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.network.timeout", "600s")
    .config("spark.executor.heartbeatInterval", "60s")
    .getOrCreate()
)

print("Spark version:", spark.version)

from google.colab import files
uploaded = files.upload()
print(uploaded.keys())

!ls -lh /content | sed -n '1,200p'

import os, glob

# Most likely exact path
CSV_PATH = "/content/NYPD_Arrests_Data_Historic_.csv"

# If it uploaded with a slightly different name, auto-pick a matching file
if not os.path.exists(CSV_PATH):
    matches = glob.glob("/content/*NYPD*Arrests*Historic*.csv")
    print("Matches found:", matches)
    if matches:
        CSV_PATH = matches[0]

print("Using CSV_PATH:", CSV_PATH)
print("Exists?", os.path.exists(CSV_PATH))

!pip -q install pyspark==3.5.1

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("NYPD_Arrests_Cleaning")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.network.timeout", "600s")
    .config("spark.executor.heartbeatInterval", "60s")
    .getOrCreate()
)

print("Spark version:", spark.version)

from pyspark.sql import functions as F

df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "true")
    .option("escape", "\"")
    .csv(CSV_PATH)
)

print("Rows:", df_raw.count())
print("Cols:", len(df_raw.columns))
df_raw.printSchema()
df_raw.show(5, truncate=False)

import re

def clean_colname(c: str) -> str:
    c = c.strip()
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^0-9a-zA-Z_]", "", c)
    return c

df = df_raw.toDF(*[clean_colname(c) for c in df_raw.columns])
print(df.columns)

for c in df.columns:
    df = df.withColumn(
        c,
        F.when(F.trim(F.col(c).cast("string")) == "", F.lit(None)).otherwise(F.col(c))
    )

# quick null count snapshot
df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).show(truncate=False)

# Date parsing (NYPD arrest data usually supports to_date directly)
if "ARREST_DATE" in df.columns:
    df = df.withColumn("ARREST_DATE", F.to_date(F.col("ARREST_DATE")))

# Cast numeric fields safely
for c in ["PD_CD","KY_CD","ARREST_PRECINCT","JURISDICTION_CODE"]:
    if c in df.columns:
        df = df.withColumn(c, F.col(c).cast("int"))

for c in ["X_COORD_CD","Y_COORD_CD","Latitude","Longitude"]:
    if c in df.columns:
        df = df.withColumn(c, F.col(c).cast("double"))

df.printSchema()

before = df.count()

if "ARREST_KEY" in df.columns:
    df = df.dropDuplicates(["ARREST_KEY"])
else:
    df = df.dropDuplicates()

after = df.count()
print("Before:", before, "| After:", after, "| Removed:", before - after)

# LAW_CAT_CD is typically: F (Felony), M (Misdemeanor), V (Violation)
if "LAW_CAT_CD" in df.columns:
    df = df.withColumn(
        "SEVERITY",
        F.when(F.upper(F.col("LAW_CAT_CD")).isin("F", "FELONY"), "FELONY")
         .when(F.upper(F.col("LAW_CAT_CD")).isin("M", "MISDEMEANOR"), "MISDEMEANOR")
         .when(F.upper(F.col("LAW_CAT_CD")).isin("V", "VIOLATION"), "VIOLATION")
         .otherwise("OTHER")
    )

df.select([c for c in ["LAW_CAT_CD","SEVERITY"] if c in df.columns]).show(20, truncate=False)

if "Latitude" in df.columns and "Longitude" in df.columns:
    df = df.filter(
        (F.col("Latitude").isNull() | ((F.col("Latitude") >= -90) & (F.col("Latitude") <= 90))) &
        (F.col("Longitude").isNull() | ((F.col("Longitude") >= -180) & (F.col("Longitude") <= 180)))
    )

print("Rows after coord filter:", df.count())

OUT_DIR = "/content/NYPD_Tableau_Cleaned"
(
    df.coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(OUT_DIR)
)

!ls -lh {OUT_DIR}

import glob, shutil

part_files = glob.glob(f"{OUT_DIR}/part-*.csv")
print("Part files:", part_files)

FINAL_CSV = "/content/NYPD_Tableau_Cleaned.csv"
shutil.copy(part_files[0], FINAL_CSV)

!ls -lh /content | grep NYPD_Tableau_Cleaned.csv

from google.colab import files
files.download(FINAL_CSV)

from pyspark.sql import functions as F

# I keep only rows where my target (SEVERITY) exists
df_ml = df.filter(F.col("SEVERITY").isNotNull())

# I standardize text columns I will use (safe cleanup)
for c in ["ARREST_BORO", "AGE_GROUP", "PERP_SEX", "PERP_RACE", "OFNS_DESC"]:
    if c in df_ml.columns:
        df_ml = df_ml.withColumn(c, F.upper(F.trim(F.col(c))))

# I also make sure my target is clean
df_ml = df_ml.withColumn("SEVERITY", F.upper(F.trim(F.col("SEVERITY"))))

print("Rows for ML:", df_ml.count())
df_ml.groupBy("SEVERITY").count().orderBy(F.desc("count")).show(truncate=False)

# I define my feature columns (categorical + numeric) based on the dataset columns that exist
candidate_categoricals = ["ARREST_BORO", "AGE_GROUP", "PERP_SEX", "PERP_RACE", "OFNS_DESC"]
candidate_numerics     = ["PD_CD", "KY_CD", "ARREST_PRECINCT", "JURISDICTION_CODE",
                          "X_COORD_CD", "Y_COORD_CD", "Latitude", "Longitude"]

cat_cols = [c for c in candidate_categoricals if c in df_ml.columns]
num_cols = [c for c in candidate_numerics if c in df_ml.columns]

print("Categorical features:", cat_cols)
print("Numeric features:", num_cols)

# I keep only the needed columns
keep_cols = ["SEVERITY"] + cat_cols + num_cols
df_ml = df_ml.select(*keep_cols)

df_ml.show(5, truncate=False)

from pyspark.storagelevel import StorageLevel

# I split my data into train and test for fair evaluation
train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)

# I cache for speed (big data best practice)
train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
test_df  = test_df.persist(StorageLevel.MEMORY_AND_DISK)

print("Train:", train_df.count(), "Test:", test_df.count())

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

LABEL_COL = "SEVERITY"

# I create my label indexer (multiclass)
label_indexer = StringIndexer(
    inputCol=LABEL_COL,
    outputCol="label",
    handleInvalid="keep"
)

# I index + one-hot encode categoricals
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in cat_cols
]

encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in cat_cols],
    outputCols=[f"{c}_ohe" for c in cat_cols],
    handleInvalid="keep"
)

# I assemble numeric + encoded features into one vector
assembler_inputs = [f"{c}_ohe" for c in cat_cols] + num_cols

assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="features_raw",
    handleInvalid="keep"
)

# I scale features (helps LR/SVM/MLP)
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=False
)

# I fit the shared pipeline ONCE on train (fairness + speed)
feat_pipe = Pipeline(stages=[label_indexer] + indexers + [encoder, assembler, scaler]).fit(train_df)

train_ready = feat_pipe.transform(train_df).select("label", "features")
test_ready  = feat_pipe.transform(test_df).select("label", "features")

print("Prepared train rows:", train_ready.count(), "Prepared test rows:", test_ready.count())
train_ready.show(3, truncate=False)

from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, LinearSVC, OneVsRest, MultilayerPerceptronClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# I define my evaluators
eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
eval_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
eval_wp  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
eval_wr  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

# I count number of classes for MLP
num_classes = int(train_ready.select("label").distinct().count())
print("Number of classes:", num_classes)

models = {}

# 1) Logistic Regression
models["Logistic Regression"] = LogisticRegression(
    featuresCol="features", labelCol="label",
    maxIter=50, regParam=0.0, elasticNetParam=0.0
)

# 2) Decision Tree
models["Decision Tree"] = DecisionTreeClassifier(
    featuresCol="features", labelCol="label",
    maxDepth=10, minInstancesPerNode=50
)

# 3) SVM (multiclass using OneVsRest + LinearSVC)
svm_binary = LinearSVC(featuresCol="features", labelCol="label", maxIter=50, regParam=0.1)
models["SVM (OneVsRest LinearSVC)"] = OneVsRest(classifier=svm_binary, labelCol="label", featuresCol="features")

# 4) Deep Learning (Multilayer Perceptron)
# I set a simple architecture: input -> hidden -> hidden -> output
input_size = train_ready.select("features").first()["features"].size
layers = [input_size, 64, 32, num_classes]
models["Deep Learning (MLP)"] = MultilayerPerceptronClassifier(
    featuresCol="features", labelCol="label",
    layers=layers, maxIter=50, seed=42, blockSize=256
)

print("MLP layers:", layers)

from pyspark.sql import Row

results = []

for name, model in models.items():
    print("\n==============================")
    print("Training:", name)
    print("==============================")

    fitted = model.fit(train_ready)
    preds  = fitted.transform(test_ready).cache()

    acc = float(eval_acc.evaluate(preds))
    f1  = float(eval_f1.evaluate(preds))
    wp  = float(eval_wp.evaluate(preds))
    wr  = float(eval_wr.evaluate(preds))

    print(f"{name} -> Accuracy: {acc:.4f} | F1: {f1:.4f} | W-Precision: {wp:.4f} | W-Recall: {wr:.4f}")

    results.append(Row(Model=name, Accuracy=acc, F1=f1, WeightedPrecision=wp, WeightedRecall=wr))

    preds.unpersist()

results_df = spark.createDataFrame(results).orderBy(F.desc("F1"))
results_df.show(truncate=False)

from pyspark.sql import functions as F

# I quickly check NaN / Infinity in numeric columns
checks = []
for c in num_cols:
    checks.append(
        F.sum(F.when(F.isnan(F.col(c)) | F.col(c).isin(float("inf"), float("-inf")), 1).otherwise(0)).alias(c)
    )

nan_inf_counts = df_ml.select(checks)
nan_inf_counts.show(truncate=False)

from pyspark.sql import functions as F

df_clean = df_ml

# 1) I replace NaN/Infinity in numeric columns with NULL
for c in num_cols:
    df_clean = df_clean.withColumn(
        c,
        F.when(F.col(c).isin(float("inf"), float("-inf")), None)
         .when(F.isnan(F.col(c)), None)
         .otherwise(F.col(c))
    )

# 2) I fill NULL numeric values with the median (robust) or 0 if you prefer
# Median in Spark: approxQuantile
fill_map = {}
for c in num_cols:
    med = df_clean.approxQuantile(c, [0.5], 0.01)[0]  # 1% relative error
    if med is None:
        med = 0.0
    fill_map[c] = float(med)

df_clean = df_clean.fillna(fill_map)

print("Numeric fill values (medians):")
for k,v in fill_map.items():
    print(k, "->", v)

from pyspark.storagelevel import StorageLevel

train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)

train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
test_df  = test_df.persist(StorageLevel.MEMORY_AND_DISK)

print("Train:", train_df.count(), "Test:", test_df.count())

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

LABEL_COL = "SEVERITY"

label_indexer = StringIndexer(inputCol=LABEL_COL, outputCol="label", handleInvalid="keep")

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]

encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in cat_cols],
    outputCols=[f"{c}_ohe" for c in cat_cols],
    handleInvalid="keep"
)

assembler_inputs = [f"{c}_ohe" for c in cat_cols] + num_cols

assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_raw", handleInvalid="keep")

scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)

feat_pipe = Pipeline(stages=[label_indexer] + indexers + [encoder, assembler, scaler]).fit(train_df)

train_ready = feat_pipe.transform(train_df).select("label", "features")
test_ready  = feat_pipe.transform(test_df).select("label", "features")

print("Prepared train rows:", train_ready.count(), "Prepared test rows:", test_ready.count())

from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

# I verify that my feature vectors contain no NaN/Inf
arr_df = train_ready.select(vector_to_array("features").alias("fa"))

bad = arr_df.select(
    F.sum(
        F.expr("aggregate(fa, 0, (acc, x) -> acc + IF(isnan(x) OR x = double('inf') OR x = -double('inf'), 1, 0))")
    ).alias("bad_values")
).collect()[0]["bad_values"]

print("Bad values in train features:", bad)

from pyspark.sql import Row
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, LinearSVC, OneVsRest, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F

# Evaluators
eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
eval_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
eval_wp  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
eval_wr  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

num_classes = int(train_ready.select("label").distinct().count())
input_size = train_ready.select("features").first()["features"].size
layers = [input_size, 64, 32, num_classes]
print("Classes:", num_classes, " | MLP layers:", layers)

models = {}
models["Logistic Regression"] = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
models["Decision Tree"] = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=10, minInstancesPerNode=50)

svm_binary = LinearSVC(featuresCol="features", labelCol="label", maxIter=50, regParam=0.1)
models["SVM (OneVsRest LinearSVC)"] = OneVsRest(classifier=svm_binary, labelCol="label", featuresCol="features")

models["Deep Learning (MLP)"] = MultilayerPerceptronClassifier(
    featuresCol="features", labelCol="label",
    layers=layers, maxIter=50, seed=42, blockSize=256
)

results = []

for name, model in models.items():
    print("\n==============================")
    print("Training:", name)
    print("==============================")

    fitted = model.fit(train_ready)
    preds  = fitted.transform(test_ready).cache()

    acc = float(eval_acc.evaluate(preds))
    f1  = float(eval_f1.evaluate(preds))
    wp  = float(eval_wp.evaluate(preds))
    wr  = float(eval_wr.evaluate(preds))

    print(f"{name} -> Accuracy: {acc:.4f} | F1: {f1:.4f} | W-Precision: {wp:.4f} | W-Recall: {wr:.4f}")

    results.append(Row(Model=name, Accuracy=acc, F1=f1, WeightedPrecision=wp, WeightedRecall=wr))

    preds.unpersist()

results_df = spark.createDataFrame(results).orderBy(F.desc("F1"))
results_df.show(truncate=False)

best_model_name = results_df.first()["Model"]
print("Best model by F1:", best_model_name)

best_fitted = models[best_model_name].fit(train_ready)

best_preds = (
    best_fitted.transform(test_ready)
              .select("label", "prediction")
              .cache()
)

# Confusion matrix counts: label vs prediction
confusion_df = (
    best_preds.groupBy("label", "prediction")
              .count()
              .orderBy("label", "prediction")
)

confusion_df.show(200, truncate=False)

best_preds.unpersist()

import glob, shutil, os

# --- 1) Export model comparison results ---
OUT_RESULTS_DIR = "/content/model_comparison_results"

(
    results_df.coalesce(1)
              .write.mode("overwrite")
              .option("header", "true")
              .csv(OUT_RESULTS_DIR)
)

part_results = glob.glob(f"{OUT_RESULTS_DIR}/part-*.csv")[0]
FINAL_RESULTS_CSV = "/content/model_comparison_results.csv"
shutil.copy(part_results, FINAL_RESULTS_CSV)

print("Saved model comparison CSV:", FINAL_RESULTS_CSV)


# --- 2) Export confusion matrix
OUT_CONF_DIR = "/content/confusion_matrix_counts"

try:
    (
        confusion_df.coalesce(1)
                    .write.mode("overwrite")
                    .option("header", "true")
                    .csv(OUT_CONF_DIR)
    )

    part_conf = glob.glob(f"{OUT_CONF_DIR}/part-*.csv")[0]
    FINAL_CONF_CSV = "/content/confusion_matrix_counts.csv"
    shutil.copy(part_conf, FINAL_CONF_CSV)

    print("Saved confusion matrix CSV:", FINAL_CONF_CSV)

except NameError:
    print("confusion_df not found. Please run CELL 7 first if you want the confusion matrix export.")


# --- 3) Show files in /content so I can download them ---
!ls -lh /content | sed -n '1,200p'

from google.colab import files
files.download("/content/model_comparison_results.csv")

import pandas as pd

data = {
    "Model": [
        "Decision Tree",
        "Logistic Regression",
        "SVM (OneVsRest LinearSVC)",
        "Deep Learning (MLP)"
    ],
    "Accuracy": [
        0.9931683484901992,
        0.99294210169084,
        0.9167152972056433,
        0.6462752346580042
    ],
    "F1": [
        0.9929643067073213,
        0.9925716602114856,
        0.9130844253402577,
        0.5074148109477526
    ],
    "WeightedPrecision": [
        0.9930656850715233,
        0.992965313352576,
        0.9246960137674678,
        0.4176725671786969
    ],
    "WeightedRecall": [
        0.9931683484901992,
        0.99294210169084,
        0.9167152972056434,
        0.6462752346580042
    ]
}

df = pd.DataFrame(data)
df.to_csv("model_results.csv", index=False)

print("CSV file created successfully!")

from google.colab import files
files.download("model_results.csv")

from pyspark import StorageLevel

train_df, test_df = df_feat.randomSplit([0.8, 0.2], seed=42)
train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
test_df  = test_df.persist(StorageLevel.MEMORY_AND_DISK)

print("Train:", train_df.count(), "Test:", test_df.count())

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)

lr_grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.0, 0.01, 0.1])
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
    .build()
)

lr_cv = CrossValidator(
    estimator=lr,
    estimatorParamMaps=lr_grid,
    evaluator=evaluator_f1,
    numFolds=3,
    parallelism=2,
    seed=42
)

lr_cv_model = lr_cv.fit(train_df)
best_lr = lr_cv_model.bestModel

print("LR best regParam:", best_lr.getRegParam())
print("LR best elasticNetParam:", best_lr.getElasticNetParam())
print("LR best CV F1:", float(max(lr_cv_model.avgMetrics)))

lr_test_preds = best_lr.transform(test_df)
print("LR test F1:", float(evaluator_f1.evaluate(lr_test_preds)))
print("LR test ACC:", float(evaluator_acc.evaluate(lr_test_preds)))

