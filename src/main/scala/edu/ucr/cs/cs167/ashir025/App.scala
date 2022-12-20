package edu.ucr.cs.cs167.ashir025


import org.apache.spark.sql.functions.{col, concat_ws, lit, map}
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object App {

  def main(args : Array[String]) {

    if (args.length != 1) {
      println("Usage <input file>")
      println("  - <input file> path to a CSV file input")
      sys.exit(0)
    }
    val inputfile = args(0)
    val conf = new SparkConf
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    println(s"Using Spark master '${conf.get("spark.master")}'")

    val spark = SparkSession
      .builder()
      .appName("CS167 FinalProject")
      .config(conf)
      .getOrCreate()

    val t1 = System.nanoTime
    try {
      //reading data into DataFrame
      val tweetsDF: DataFrame = spark.read.format("json")
        .load(inputfile)

      //concatenate text and user_description into one column
      val tweetsreloaded = tweetsDF.withColumn("text/user_description", concat_ws(",",col("text"),col("user_description")))

      //tokenizer that finds all the tokens (words) from the tweets text and user description
      val tokenzier = new Tokenizer()
        .setInputCol("text/user_description")
        .setOutputCol("words")

      //hashingTF transformer that converts the tokens into a set of numeric features
      val hashingTF = new HashingTF()
        .setInputCol("words")
        .setOutputCol("features")

      //stringIndexer that converts each topic to an index
      val stringIndexer = new StringIndexer()
        .setInputCol("hashtag")
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
      predictions.select("id","text","hashtag", "user_description", "label", "prediction").show()

      //using MultiClassificationEvaluator() to set label and predictions
      val multiClassificationEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")

      //computing accuracy
      val accuracy: Double = multiClassificationEvaluator.evaluate(predictions)
      println(s"Accuracy of the test set is $accuracy")

      //val mapD = predictionLabel.columns.flatMap(c => Seq(lit(c), col(c)))
      //predictionLabel.withColumn("Map", map(mapD: _*)).show(false)

      //code to compute precision/recall
      val metrics = multiClassificationEvaluator.getMetrics(predictions)
      println(s"Weighted Precision: ${metrics.weightedPrecision}")
      println(s"Weighted Recall: ${metrics.weightedRecall}")

      val t2 = System.nanoTime
      println(s"Applied tweets analysis classification algorithm on input $inputfile in ${(t2 - t1) * 1E-9} seconds")
    } finally {
      spark.stop
    }
  }
}