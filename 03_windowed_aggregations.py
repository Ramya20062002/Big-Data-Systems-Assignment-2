from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, avg, sum, count
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("WindowedAggregations").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df_raw = spark.read.json("data/raw_transactions.json")

df_flat = df_raw.select(
    col("transaction_id"),
    col("sender.account_id").alias("sender_account"),
    col("amount"),
    to_timestamp(col("timestamp")).alias("tx_timestamp")
)

# Define Window: Group by sender account, sort by time, look back 5 rows
account_window = Window.partitionBy("sender_account").orderBy("tx_timestamp").rowsBetween(-5, 0)

# Apply Window Analytics
df_windowed = df_flat.withColumn("rolling_avg_amount", avg("amount").over(account_window)) \
                     .withColumn("rolling_tx_count", count("transaction_id").over(account_window))

print("--- Windowed Aggregation Results ---")
df_windowed.select("sender_account", "tx_timestamp", "amount", "rolling_avg_amount", "rolling_tx_count").show(10)
