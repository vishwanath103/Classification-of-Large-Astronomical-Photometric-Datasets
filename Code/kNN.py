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
using distributed kNN implementation in pySpark over Google Cloud framework
'''

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.sql import Row
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
import pandas as pd
from pyspark.rdd import RDD
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Create spark context and read files from the cloud storage
sc = SparkContext("local", "test_script")
trainingFile = "gs://vishu/TrainingData/Training_Data.csv"
data = sc.textFile(trainingFile).repartition(60)

# Remove Header
header = data.first()
data = data.filter(lambda line: line != header)


# Function to compute object features as g-r, u-r, r-i, i-z
def transformToLabelledPoint(inputStr):
    attlist = inputStr.split(",")
    data = (float(attlist[2]) - float(attlist[4]),
         float(attlist[5]) - float(attlist[4]),
         float(attlist[4]) - float(attlist[3]),
         float(attlist[3]) - float(attlist[6]))
    return data

# Function to transform class label to a numerical vector
def tranformToClass(inputStr):
    attlist = inputStr.split(",")
    classes = (float(attlist[0]), attlist[1])
    return classes

# Extract features from all the objects in a distributed framework
autolp = data.map(transformToLabelledPoint)
autolpCollect = autolp.collect()
vectorsCollected = np.vstack(tuple(autolpCollect))

# Transform class label to numerical vector in distributed framework
classes = data.map(tranformToClass)
classDF = sqlContext.createDataFrame(classes, ["id","class"])
dfclass = classDF.toPandas()
classPD = pd.Series(dfclass['class'])

# Create kNN model and fit the model to the training dataset
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(vectorsCollected, classPD)

# Create a broadcasted object for a knn model
bc_knnobj = sc.broadcast(knn)

# Predict the classes for all the testing objects in a distributed framework and accumulate the results
# into a RDD object
results = autolp.map(lambda x: bc_knnobj.value.predict(x))

# Convert RDD object to a tuple and store the results in a text file in google cloud storage
resultlist = results.map(lambda x: (x.item(0)))
resultlist.saveAsTextFile("gs://vishu/Result")
