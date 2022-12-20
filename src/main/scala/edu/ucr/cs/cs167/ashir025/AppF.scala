package edu.ucr.cs.cs167.ashir025


import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions.{col, concat_ws, explode, lit, map}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}



object AppF {

  def main(args : Array[String]) {

    val inputFile = args(0)

    val conf = new SparkConf().setAppName("CS167 Final")
    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    val spark: SparkSession.Builder = SparkSession.builder().config(conf)
    val sparkSession: SparkSession = spark.getOrCreate()
    import sparkSession.implicits._
    val sparkContext = sparkSession.sparkContext


    //********************   task 1   *****************************************

    //input json file read as a view named "tweets" with hashtags column renamed to topic.
    //Load in json file specified in program arguments
    val tweetsDF = sparkSession.read.format("json").load(inputFile)
    //Create dataframe (cleanTweets) with selected attributes
    val cleanTweetsDF = tweetsDF.selectExpr("id", "text", "entities.hashtags.text AS hashtags", "user.description AS user_description", "retweet_count", "reply_count", "quoted_status_id")
    //Write cleanTweets to new json file "tweets_clean.json"
    cleanTweetsDF.write.format("json").save("tweets_clean.json")
    //cleanTweetsDF.printSchema()

    //TopHashtag will contain vector of keywords
    var theKeywordsArrStr = new Array[String](20);
    cleanTweetsDF.createOrReplaceTempView("tweets")

    theKeywordsArrStr = sparkSession.sql(
      s"""
                      SELECT explode(hashtags) as hashtags, count(*) AS count
                      FROM tweets
                      GROUP BY hashtags
                      ORDER BY count DESC
                      LIMIT 20
                    """).map(f=>f.getString(0)).collect()

    //println("Top 20 Keywords: " + theKeywordsArrStr.mkString(","))


    // ********************* end of task1 **************************************

    //********************   task 2   *****************************************
    //input json file read as a view named "tweets" with hashtags column renamed to topic.
    val keywords: String = "'"+ theKeywordsArrStr.mkString("','") + "'" //formatted csv styled string to array styled string to pass into sql.
    //println(keywords)
    // 'keyword1','keyword2'
    //selects the intended output
    val task2DF: DataFrame  = sparkSession.sql(
      s"""
    SELECT id, quoted_status_id, reply_count, retweet_count, text, user_description, element_at(new_hash_tags, 1) AS topic FROM (
      SELECT *, array_intersect(hashtags, array($keywords)) AS new_hash_tags FROM tweets
    ) AS t1
     WHERE size(new_hash_tags) > 0;
    """)
    //task2DF.printSchema()
    //task2DF.show(40)
    //counts the total intended result(re-run of previous line but with COUNT(*)
    val totalCount: DataFrame = sparkSession.sql(
      s"""
          SELECT COUNT(*) AS totalCountFromTask2 FROM (
    SELECT id, quoted_status_id, reply_count, retweet_count, text, user_description, element_at(new_hash_tags, 1) AS topic FROM (
      SELECT *, array_intersect(hashtags, array($keywords)) AS new_hash_tags FROM tweets
    ) AS t1
     WHERE size(new_hash_tags) > 0);
    """)
    totalCount.show()
    //38 total count from sample file.
    //253 total counts for 10k dataset

    //write to json file
    task2DF.write.json("tweets_topic.json")

    // ********************* end of task2 **************************************
    //TODO for task 3, the variable <task2DF> is the dataframe that contains the full data

    // ********************* task 3 **************************************

    val t1 = System.nanoTime

    //reading data into DataFrame

    /*
    val tweetsDF: DataFrame = spark.read.format("json")
      .load(inputfile)
    */
    val tweetsDFT3: DataFrame = task2DF

    //concatenate text and user_description into one column
    val tweetsreloaded = tweetsDFT3.withColumn("text/user_description", concat_ws(",",col("text"),col("user_description")))

    //tokenzier that finds all the tokens (words) from the tweets text and user description
    val tokenzier = new Tokenizer()
      .setInputCol("text/user_description")
      .setOutputCol("words")

    //hashingTF transformer that converts the tokens into a set of numeric features
    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("features")

    //stringIndexer that converts each topic to an index
    val stringIndexer = new StringIndexer()
      .setInputCol("topic")
      .setOutputCol("label")
      .setHandleInvalid("skip")

    //logistic Regression that predicts the topic from the set of features
    val logisticRegression = new LogisticRegression()
      .setMaxIter(100)

    //setting stages of pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenzier, hashingTF, stringIndexer, logisticRegression))

    //setting params, numFeatures and regParam
    val paramGrid: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures,Array(5,10, 40))
      .addGrid(logisticRegression.regParam,Array(0.1, 0.01, 0.001))
      .build()

    //training-test split
    val cv = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("label"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
      .setParallelism(2)

    //splitting the data
    val Array(trainingData: Dataset[Row], testData: Dataset[Row]) = tweetsreloaded.randomSplit(Array(0.8, 0.2))

    // Running cross-validation, and choosing the best set of parameters
    val logisticModel: TrainValidationSplitModel = cv.fit(trainingData)

    //best parameters
    val numFeatures: Int = logisticModel.bestModel.asInstanceOf[PipelineModel].stages(1).asInstanceOf[HashingTF].getNumFeatures
    val regParam: Double = logisticModel.bestModel.asInstanceOf[PipelineModel].stages(3).asInstanceOf[LogisticRegressionModel].getRegParam
    //println(s"Number of features in the best model = $numFeatures")
    //println(s"RegParam the best model = $regParam")

    //getting predictions from our model
    val predictions: DataFrame = logisticModel.transform(testData)
    predictions.select("id","text","topic", "user_description", "label", "prediction").show()

    //using MultiClassificationEvaluator() to set label and predictions
    val multiClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    //computing accuracy
    val accuracy: Double = multiClassificationEvaluator.evaluate(predictions)
    //println(s"Accuracy of the test set is $accuracy")

    //code to compute precision/recall
    val metrics = multiClassificationEvaluator.getMetrics(predictions)
    //println(s"Weighted Precision: ${metrics.weightedPrecision}")
    //println(s"Weighted Recall: ${metrics.weightedRecall}")

    val t2 = System.nanoTime
    //println(s"Applied tweets analysis classification algorithm on file $inputFile in ${(t2 - t1) * 1E-9} seconds")




    println("Top 20 Keywords: " + theKeywordsArrStr.mkString(","))
    totalCount.show()
    predictions.select("id","text","topic", "user_description", "label", "prediction").show()
    println(s"Accuracy of the test set is $accuracy")
    println(s"Weighted Precision: ${metrics.weightedPrecision}")
    println(s"Weighted Recall: ${metrics.weightedRecall}")
    println(s"Applied tweets analysis classification algorithm on file $inputFile in ${(t2 - t1) * 1E-9} seconds")

  }
}