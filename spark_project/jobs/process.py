import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, date_sub, desc, lit
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import avg, to_date, month, regexp_replace, col, row_number
from pyspark.sql.window import Window

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(CURRENT_DIR, "out")

# Середня тривалість поїздки на день
def avg_trip_duration_per_day(df):
    df = df.withColumn("trip_duration", regexp_replace(col("tripduration"), ",", "").cast("double"))

    # Дістати дату з start_time
    df = df.withColumn("trip_date", to_date(col("start_time"), 'yyyy-MM-dd HH:mm:ss'))

    # Групування за датою та обчислення середньої тривалості
    result = (
        df.groupBy("trip_date")
        .agg(avg("trip_duration").alias("avg_duration"))
        .orderBy("trip_date")
    )

    # Запис у каталог
    result.write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_PATH, "avg_trip_duration_per_day")
    )
    return result

# Кількість поїздок за день
def trip_count_per_day(df):
    # Дістати дату з start_time
    df = df.withColumn("trip_date", to_date(col("start_time"), 'yyyy-MM-dd HH:mm:ss'))

    # Групування за датою та обчислення кількості поїздок за день
    result = (
        df.groupBy("trip_date")
        .agg(count("trip_date").alias("trip_count"))
        .orderBy("trip_date")
    )

    # Запис у каталог
    result.write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_PATH, "trip_count_per_day")
    )
    return result

# Найпопулярніша початкова станція для кожного місяця
def most_popular_start_station_per_month(df):
    # Додаємо стовпець "month" з дати поїздки
    df = df.withColumn("month", month(col("start_time")))

    # Групуємо за місяцем та станцією — рахуємо кількість поїздок з кожної станції
    grouped = (
        df.groupBy("month", "from_station_id", "from_station_name")
        .agg(count("*").alias("trip_count"))
    )

    # Вікно по місяцю: сортування по trip_count у спадному порядку
    window = Window.partitionBy("month").orderBy(col("trip_count").desc())

    # Додаємо row_number і відбираємо лише найпопулярніші станції по кожному місяцю
    ranked = grouped.withColumn("rank", row_number().over(window)).filter(col("rank") == 1)

    # Запис у файл
    ranked.write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_PATH, "most_popular_station_per_month")
    )

    return ranked

# Cтанції, які входять у трійку лідерів станцій для поїздок кожного дня протягом останніх двох тижнів
def top3_stations_last_2weeks(df):
    # Дістати дату з start_time
    df = df.withColumn("trip_date", to_date(col("start_time"), 'yyyy-MM-dd HH:mm:ss'))
    max_date = df.agg(spark_max("trip_date")).collect()[0][0]
    last_2weeks = df.filter(col("trip_date") >= date_sub(lit(max_date), 13))
    station_counts = (
        last_2weeks.groupBy("trip_date", "from_station_id", "from_station_name")
        .agg(count("*").alias("trip_count"))
    )
    window = Window.partitionBy("trip_date").orderBy(col("trip_count").desc())
    result = station_counts.withColumn("rank", row_number().over(window)).filter(col("rank") <= 3)

    result.write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_PATH, "top_stations_last_2weeks")
    )
    return result

# Чоловіки чи жінки їздять довше в середньому?
def avg_trip_duration_by_gender(df):
    df = df.withColumn("trip_duration", regexp_replace(col("tripduration"), ",", "").cast("double"))
    df_gender = df.filter((col("gender").isNotNull()) & (col("gender") != ""))
    result = (
        df_gender.groupBy("gender")
        .agg(avg("trip_duration").alias("avg_duration"))
    )
    result.write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_PATH, "avg_trip_duration_by_gender")
    )

    return result

def main() -> None:
    spark = SparkSession.builder.appName("TripAnalysis").getOrCreate()
    df = spark.read.option("header", True).csv("/opt/bitnami/spark/jobs/Divvy_Trips_2019_Q4.csv")
    
    avg_trip_duration_per_day(df)
    trip_count_per_day(df)
    most_popular_start_station_per_month(df)
    top3_stations_last_2weeks(df)
    avg_trip_duration_by_gender(df)

    spark.stop()


if __name__ == "__main__":
    main()

# docker exec -it spark-master spark-submit /opt/bitnami/spark/jobs/process.py