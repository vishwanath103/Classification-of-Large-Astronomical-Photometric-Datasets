'''
This file is part of Classification of Large Atsronomical Photometric Datasets.

Classification of Large Atsronomical Photometric Datasets is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Classification of Large Atsronomical Photometric Datasets is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
'''

'''
The goal of this script is to classify the Astronomical Datasets into Stars, Quasars and Galaxies
using distributed Random Forest implementation in pySpark over Google Cloud framework
'''

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

# Create spark context and read files from the cloud storage
sc = SparkContext("local", "test_script")
trainingFile = "gs://vishu/TrainingData/Training_Data.csv"
data = sc.textFile(trainingFile)

# Function to compute object features as g-r, u-r, r-i, i-z
# and map class label to numerical vector
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

# Extract the features from the data using distributed framework
parsedData = data.map(parsePoint)

# Create a RF model and train the model using training dataset
model = RandomForest.trainClassifier(parsedData, numClasses=3, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='entropy', maxDepth=20, maxBins=32)


# Read the testing file and create a RDD object
testingFile = "gs://vishu/TestingData/"
testData = sc.textFile(testingFile)

# Function to obtain features from the testing
# file as g-r, u-r, r-i, i-z
def parseTestData(line):
    attlist = line.split(",")
    label = int(attlist[0])
    features = [float(attlist[4]) - float(attlist[6]),
                float(attlist[7]) - float(attlist[6]),
                float(attlist[6]) - float(attlist[5]),
                float(attlist[5]) - float(attlist[8])]
    return (label, features)

# Parse the testing file using a distributed framework
parsedTestData = testData.map(parseTestData)

# Using the parsed data to predict the classes in a distributed framework
predictions = model.predict(parsedTestData.map(lambda x: x[1]))
labelsAndPredictions = parsedTestData.map(lambda lp: lp[0]).zip(predictions)

# Function to map numerical vector to class labels
def getResults(t):
    result = ""
    if t[1] == 0:
        return (int(t[0]),"STAR")
    elif t[1] == 2:
        return (int(t[0]),"GALAXY")
    else:
        return (int(t[0]),"QSO")

# Parse the results and store it in a cloud storage
resultList = labelsAndPredictions.map(getResults)
resultList.saveAsTextFile("gs://vishu/RandomForest_3_Result")

