# Classification-of-Large-Astronomical-Photometric-Datasets
Abstract:<br />
We have used kNN, SVM and Random Forest algorithms in a distributed environment over cloud to classify 1,183,850,913 unclassified, photometric objects present in the SDSS catalog. The SDSS III catalog contains photometric data for all objects viewed through a telescope and spectroscopic data for a small part of these. Although it is possible to classify objects using spectroscopic data, it is impractical to obtain this data for all objects. We used the photometric data of the spectroscopically defined objects to train and test all our algorithms. We have used spark framework to implement distributed computing environment over the cloud. We found writing of results to the cloud storage is very slow and it is linear in time. Though using SVM writing the results is being done in a parallel manner, its accuracy is around 87% due to lack of kernel implementation in spark. We then used Random Forest algorithm to classify the entire set of 1,183,850,913 objects with an accuracy of 94% in about 17 hours of time. This is significant as even to collect spectroscopic data for these many objects would take decades.

Requirements:<br />
Data: SDSS Catalog<br />
Language: Python<br />
Framework: PySpark<br />
Platform: Google Cloud
