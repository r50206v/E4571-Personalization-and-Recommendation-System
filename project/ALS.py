#!/usr/bin/env python
# coding: utf-8
import os
import pyspark
from pyspark import SparkFiles
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import rand, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

seed = 100
sc = SparkContext()
filePath = "/home/jovyan/work/Personalization/ml-20m/ratings.csv"


# read dataset into spark RDD
sc.addFile(filePath)
sqlContext = SQLContext(sc)
df = sqlContext.read.csv(
    SparkFiles.get("ratings.csv"), 
    header=True, 
    inferSchema=True
)

sqlContext.registerDataFrameAsTable(df, "df")
df = sqlContext.sql('''
    SELECT 
        userId AS user, 
        movieId AS item,
        rating
    FROM df
''')


# adding uniform random numbers for train/validation/test set
df = df.withColumn('TrainTest', rand(seed=seed))
# print("(row, col): ", (df.count(), len(df.columns)))
dftrain = df.where(col('TrainTest') < 0.75).drop(*["TrainTest"])
dftest = df.where(col('TrainTest') >= 0.75).drop(*["TrainTest"])
# print(dftrain.printSchema())
# print("(row, col): ", (dftrain.count(), len(dftrain.columns)))
# dftrain.show(n=5)
# print(dftest.printSchema())
# print("(row, col): ", (dftest.count(), len(dftest.columns)))
# dftest.show(n=5)


# building model
als = ALS(nonnegative=True, checkpointInterval=3, coldStartStrategy="drop")
paramGrid = ParamGridBuilder()\
    .addGrid(als.rank, [5, 30, 70])\
    .addGrid(als.regParam, [0.1, 1, 10])\
    .build()


rmse = RegressionEvaluator(metricName="rmse", labelCol="rating")
# trainRatio makes train:0.5 valid:0.25 and test:0.25
tvs = TrainValidationSplit(
    estimator=als,
    estimatorParamMaps=paramGrid,
    evaluator=rmse,
    seed=seed,
    trainRatio=0.66,
    parallelism=3
)


model = tvs.fit(dftrain)
model.transform(dftrain).show()


testPred = model.transform(dftest)
testPred.show(5)
rmse.evaluate(testPred)


model_path = os.getcwd() + '/ALS_model2'
model.save(model_path)