from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp

# 1. Initialize PySpark Engine
spark = SparkSession.builder \
    .appName("TrackB_PurePySpark_Engine") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to avoid cluttering screen
spark.sparkContext.setLogLevel("ERROR")

# 2. Read raw JSON directly into PySpark DataFrame
df_raw = spark.read.json("data/raw_transactions.json")

# 3. Distributed Schema Flattening
df_flat = df_raw.select(
    col("transaction_id"),
    col("sender.account_id").alias("sender_account"),
    col("sender.risk_rating").alias("sender_risk"),
    col("receiver.account_id").alias("receiver_account"),
    col("amount"),
    col("category"),
    col("device_footprint.device_os").alias("device_os"),
    col("location.coordinates")[0].alias("longitude"),
    col("location.coordinates")[1].alias("latitude"),
    to_timestamp(col("timestamp")).alias("tx_timestamp"),
    col("is_fraud_label").alias("label")
)

print("--- Print Flattened PySpark Schema ---")
df_flat.printSchema()

print("--- Display First 5 Flattened Records ---")
df_flat.show(5, truncate=False)
