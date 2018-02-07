# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.utils import shuffle
from sklearn import tree
#from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

#import matplotlib.pyplot as plt

data2 = pd.read_csv('C:/Users/surya/Desktop/SpringSemester/IDA/HW1/Biomechanical_Data_column_2C_weka.csv')

print("Data2 Shape: \n", data2.shape)
print("Data2 Head: \n", data2.head())

shuffledata2 = shuffle(data2)

print("Data2Shuffle Shape: \n", shuffledata2.head())

#Gives first 210 records
train_data2 = shuffledata2[:210]
#Gives last 100 records
test_data2 = shuffledata2[-100:]

print("Train Data2 Shape: \n", train_data2.shape)
print("Train Data2 head: \n", train_data2.head())
print("Test Data2 Shape: \n", test_data2.shape)
print("Test Data2 head: \n", test_data2.head())

trainFeatures = list(train_data2.columns[:6])
print("Features: \n", trainFeatures)

X = train_data2[trainFeatures]
print("X: \n", X.head())

y = train_data2["class"]
print("Y: \n", y.head())

#Criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
classifier = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf = 10)
classifier = classifier.fit(X,y)
print("Classifier: \n", classifier)

tFeatures = list(test_data2.columns[:6])
print("Features: \n", tFeatures)

testFeatures = test_data2[tFeatures]
print("Test Features: \n", testFeatures.head())

testPrediction = classifier.predict(testFeatures)
print("Test Prediction: \n", testPrediction)

testActual = list(test_data2["class"])
print("Test Actual: \n", testActual)

print("Accuracy: %.3f" %accuracy_score(testActual, testPrediction))
print("Precision: %.3f" %precision_score(testActual, testPrediction, average="macro"))
print("Recall: %.3f" %recall_score(testActual, testPrediction, average="macro"))

confusionMatrix = confusion_matrix(testActual, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

#plt.figure()
#plt.matshow(confusionMatrix)
#plt.title('Confusion Matrix')
#plt.colorbar()
#plt.ylabel('True Label')
#plt.xlabel('Predicted Label')  
#plt.show()

#tree.export_graphviz(classifier, out_file='tree.dot')   

import graphviz 
dot_data = tree.export_graphviz(classifier, out_file=None, feature_names=tFeatures, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data) 
graph.render("DTrees")

import seaborn as sns

ax= plt.subplot()
sns.heatmap(confusionMatrix, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Abnormal', 'Normal']); ax.yaxis.set_ticklabels(['Normal', 'Abnormal']);
#ax.xaxis.set_ticklabels(['', '']); ax.yaxis.set_ticklabels(['', '']);
ax.plot()
plt.show()
