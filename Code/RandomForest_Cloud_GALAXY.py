from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

sc = SparkContext("local", "test_script")
trainingFile = "gs://vishu/TrainingData/Training_Data.csv"
data = sc.textFile(trainingFile)

def parsePoint(line):
    attlist = line.split(",")
    label = 0
    if attlist[1] == "QSO":
        label = 1
    elif attlist[1] == "GALAXY":
        label = 2
    else:
        label = 0
    features = [float(attlist[2]) - float(attlist[4]),
                float(attlist[5]) - float(attlist[4]),
                float(attlist[4]) - float(attlist[3]),
                float(attlist[3]) - float(attlist[6])]
    return LabeledPoint(label, features)

#Remove Header
header = data.first()
data = data.filter(lambda line: line != header)

parsedData = data.map(parsePoint)

model = RandomForest.trainClassifier(parsedData, numClasses=3, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='entropy', maxDepth=20, maxBins=32)

testingFile = "gs://vishu/TestingData/"
testData = sc.textFile(testingFile)

def parseTestData(line):
    attlist = line.split(",")
    label = int(attlist[0])
    features = [float(attlist[4]) - float(attlist[6]),
                float(attlist[7]) - float(attlist[6]),
                float(attlist[6]) - float(attlist[5]),
                float(attlist[5]) - float(attlist[8])]
    return (label, features)


parsedTestData = testData.map(parseTestData)

predictions = model.predict(parsedTestData.map(lambda x: x[1]))
labelsAndPredictions = parsedTestData.map(lambda lp: lp[0]).zip(predictions)

def getResults(t):
    result = ""
    if t[1] == 0:
        return (int(t[0]),"STAR")
    elif t[1] == 2:
        return (int(t[0]),"GALAXY")
    else:
        return (int(t[0]),"QSO")


resultList = labelsAndPredictions.map(getResults)
resultList.saveAsTextFile("gs://vishu/RandomForest_3_Result")
print resultList.take(5)

