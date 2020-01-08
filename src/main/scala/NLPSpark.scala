import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.functions._


object NLPSpark {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .config("spark.master", "local[*]")
      .appName("nlp_spark")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val df_train = spark.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("multiline", "true")
      .load("/Users/gbusch/tmp/nlp-getting-started/train.csv")
    val df_test = spark.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("multiline", "true")
      .load("/Users/gbusch/tmp/nlp-getting-started/test.csv")

    val ml_df = df_train.withColumn("only_str", regexp_replace(col("text"), "(\\d+)", ""))
    val ml_df_test = df_test.withColumn("only_str", regexp_replace(col("text"), "(\\d+)", ""))

    val regex_tokenizer = new RegexTokenizer()
      .setInputCol("only_str")
      .setOutputCol("words")
      .setPattern("\\W")
    val raw_words = regex_tokenizer.transform(ml_df)
    val raw_words_test = regex_tokenizer.transform(ml_df_test)

    import java.util.Locale
    Locale.setDefault(Locale.forLanguageTag("en_US"))

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")
      .setLocale("en_US")
    val words_df = remover.transform(raw_words)
    val words_df_test = remover.transform(raw_words_test)

    val cv = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
    val model = cv.fit(words_df)
    val countVectorizer_train = model.transform(words_df).withColumn("label", col("target"))
    val countVectorizer_test = model.transform(words_df_test)

    val Array(training, validate) = countVectorizer_train.randomSplit(Array(0.8, 0.2), seed = 42)

    import org.apache.spark.ml.classification.LogisticRegression

    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("target")
      .setMaxIter(10)

    val lrModel = lr.fit(training)
    val lrPreds = lrModel.transform(validate)

    import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}

    println("Spark NLP")
    println("=========")
    val lrEval = new BinaryClassificationEvaluator()
    println("Test Area Under ROC: " + lrEval.evaluate(lrPreds))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    println("Accuracy: " + evaluator.evaluate(lrPreds))

    val testPred = lrModel.transform(countVectorizer_test)

    testPred
      .select(col("id"), col("prediction").alias("target").cast("Int"))
      .write.format("csv")
      .option("header", "true")
      .mode("overwrite")
      .save("/Users/gbusch/tmp/nlp-getting-started/submission_v1.csv")

    spark.close()
  }
}
