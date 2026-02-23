from pyspark.sql import SparkSession

def create_spark():
    return SparkSession.builder \
        .appName("Stock Intelligence Elite India") \
        .getOrCreate()
