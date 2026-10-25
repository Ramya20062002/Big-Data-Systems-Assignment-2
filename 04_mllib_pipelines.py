from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, avg, count
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("MLlibPipeline").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load and Flatten
df_raw = spark.read.json("data/raw_transactions.json")
df_flat = df_raw.select(
    col("transaction_id"),
    col("sender.account_id").alias("sender_account"),
    col("sender.risk_rating").alias("sender_risk"),
    col("amount"),
    col("category"),
    col("device_footprint.device_os").alias("device_os"),
    to_timestamp(col("timestamp")).alias("tx_timestamp"),
    col("is_fraud_label").alias("label")
)

# Apply Window Feature
account_window = Window.partitionBy("sender_account").orderBy("tx_timestamp").rowsBetween(-5, 0)
df_windowed = df_flat.withColumn("rolling_avg_amount", avg("amount").over(account_window))

# MLlib Pipelines Setup
indexer_cat = StringIndexer(inputCol="category", outputCol="category_idx")
indexer_risk = StringIndexer(inputCol="sender_risk", outputCol="risk_idx")
indexer_device = StringIndexer(inputCol="device_os", outputCol="device_idx")

assembler = VectorAssembler(
    inputCols=["amount", "rolling_avg_amount", "category_idx", "risk_idx", "device_idx"],
    outputCol="features"
)

rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20)

# Create & Run Pipeline
pipeline = Pipeline(stages=[indexer_cat, indexer_risk, indexer_device, assembler, rf])

train_data, test_data = df_windowed.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

# Evaluate
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)

print(f"\n==========================================")
print(f"  PySpark MLlib Model AUC Score: {auc:.4f}")
print(f"==========================================\n")

# Save outputs to CSV for Dashboard
predictions.select("transaction_id", "sender_account", "amount", "prediction") \
           .toPandas().to_csv("data/predictions_output.csv", index=False)

print("Saved model prediction results to data/predictions_output.csv")
