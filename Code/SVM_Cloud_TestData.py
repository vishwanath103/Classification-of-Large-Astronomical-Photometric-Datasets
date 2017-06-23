from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

sc = SparkContext("local", "test_script")
trainingFile = "gs://vishu/TrainingData/Training_Data.csv"
data = sc.textFile(trainingFile)

def parsePoint(line):
    attlist = line.split(",")
    label = 0
    if attlist[1] == "QSO":
        label = 1
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


model = SVMWithSGD.train(parsedData, iterations=100)

testingFile = "gs://vishu/TestingData/"
testData = sc.textFile(testingFile)

def parseTestData(line):
    attlist = line.split(",")
    label = int(attlist[0])
    features = [float(attlist[4]) - float(attlist[6]),
                float(attlist[7]) - float(attlist[6]),
                float(attlist[6]) - float(attlist[5]),
                float(attlist[5]) - float(attlist[8])]
    return LabeledPoint(label, features)

parsedTestData = testData.map(parseTestData)

labelsAndPreds = parsedTestData.map(lambda p: (p.label, model.predict(p.features)))

def getResults(t):
    result = ""
    if t[1] == 0:
        return (int(t[0]),"NQSO")
    else:
        return (int(t[0]),"QSO")


resultList = labelsAndPreds.map(getResults)
resultList.saveAsTextFile("gs://vishu/SVMTestResult")
print resultList.take(5)
