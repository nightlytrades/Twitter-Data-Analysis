		
<h2 align="center"> Big Data Management Final Project - Project D: Twitter Data Analysis
</h2>

Jeana Tijerina - Task 1

Carl Che - Task 2

Amir Shirazi - Task 3

Using twitter data we applied a real big-data application to build a machine-learning classifier
that extracts a topic from tweets and assigns the topic to each tweet using the most frequent
hashtags. The Machine learning Pipeline consists of three transformations and one estimator.

Task #1:
Prepare the given data for later tasks by loading the given file in json format using Spark load
function. A new dataframe is produced only selecting the specified attributes (using selectExpr)
and outputted to a json file. It should be noted that the entities.hashtags.txt and user.description
attribute had to be renamed to avoid duplication errors and match the expected schema. A top-k
SQL query was used to select the top 20 most frequent hashtags from the data and outputted to
an array that will be used for Task 2.
Top 20 Keywords found for the 10k dataset:
ALDUBxEBLovei, FurkanPalalı, LalOn, no309, sbhawks, Top3Apps, chien, trouvé, chien,
perdu, ShowtimeLetsCelebr8, trndnl, omnibusla7, TwitterIn4Words, BhaiDooj,
NewSuperstitions, Zumbicando, Mersal, SayaDukungDenayuAmeliaBisa, AndaBisa

Task #2:
Use the top 20 keywords found from Task #1 along with the array_intersect() function to find
tweets that have at least 1 keyword within its hashtags. Array_intersect() is useful in this case
because it takes in two array variables as its parameter and finds the common elements.
The total number of records in the tweets_topic dataset for the 10k dataset is 253.

Task #3:
Built a Machine Learning model using spark.ml that assigns a topic for each tweet based on the
classified tweets. The Machine learning Pipeline consists of three transformations, tokenizer,
hashingTF, stringIndexer, and one estimator, logistic regression, that predicts the topic from the
features. Our model uses TrainValidationSplit validator for regular training-test split, 80% to
train on one set and 20% to test on the other.
The Accuracy of the test set is 0.8459, the Weighted Precision is 0.8509 , and the Weighted
Recall is 0.8333 on the 10k dataset.