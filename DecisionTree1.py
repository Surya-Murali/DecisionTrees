# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

#Read Data
data2 = pd.read_csv('C:/Users/surya/Desktop/SpringSemester/IDA/HW1/Dataset/Biomechanical_Data_column_2C_weka.csv')

#The dataset contains 310 records
#See the Shape of data and check the first 5 records of your data
print("Data2 Shape: \n", data2.shape)
print("Data2 Head: \n", data2.head())

#Shuffling data before training the dataset
shuffledata2 = shuffle(data2)

#Print the first 5 records of the shuffled data
print("Data2Shuffle head: \n", shuffledata2.head())

#Assign the first 210 records to the Training data
train_data2 = shuffledata2[:210]
#Assign the last 100 records to the Testing data
test_data2 = shuffledata2[-100:]

#Check the shape and head of the training and testing data
print("Train Data2 Shape: \n", train_data2.shape)
print("Train Data2 head: \n", train_data2.head())
print("Test Data2 Shape: \n", test_data2.shape)
print("Test Data2 head: \n", test_data2.head())

#Assign the first 6 column names as the Train Features
trainFeatures = list(train_data2.columns[:6])
print("Features: \n", trainFeatures)

#Get the data under these 6 columns
X = train_data2[trainFeatures]
print("X: \n", X.head())

#Get the data under the 'class' column
y = train_data2["class"]
print("Y: \n", y.head())

print("-------------------------------------#Minimum Records per Leaf Node = 5----------------------------------------------------------")
#Create a Decision Tree Classifier
#Criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
#Here we are using criterion = 'entropy' for the information gain
#Minimum Records per Leaf Node = 5
classifier = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf = 5)

#Using the Training data to fit the classifier
classifier = classifier.fit(X,y)
print("Classifier: \n", classifier)

#Now take the Testing data:
#Get the first 6 column names of the Test data and assign it to the Test Features (tFeatures)
tFeatures = list(test_data2.columns[:6])
print("Features: \n", tFeatures)

#Get the data of the Test Features
testFeatures = test_data2[tFeatures]
print("Test Features: \n", testFeatures.head())

#Use the classifier to predict the class of the Testing data
testPrediction = classifier.predict(testFeatures)
print("Test Prediction: \n", testPrediction)

#For measuring the parameters like Accuracy, Recall and Precision, compare this Predicted class with the Actul class of the Testing data
#Get the Actual class of the Testing data
testActual = list(test_data2["class"])
print("Test Actual: \n", testActual)

#Measuring the Accuracy, Precision and Recall scores of the Classifier
#Accuracy is measured for the entire classifier, while Precision and Recall are computed for each class
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(testActual, testPrediction)),"%")

print("Precision: %.3f" %precision_score(testActual, testPrediction, average="macro"))
print("Recall: %.3f" %recall_score(testActual, testPrediction, average="macro"))

#Creating a Confusion Matrix for the Actual Vs Prediction
confusionMatrix = confusion_matrix(testActual, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

print("-------------------------------------#Minimum Records per Leaf Node = 10----------------------------------------------------------")
#Create a Decision Tree Classifier
#Criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
#Here we are using criterion = 'entropy' for the information gain
#Minimum Records per Leaf Node = 5
classifier = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf = 10)

#Using the Training data to fit the classifier
classifier = classifier.fit(X,y)
print("Classifier: \n", classifier)

#Now take the Testing data:
#Get the first 6 column names of the Test data and assign it to the Test Features (tFeatures)
tFeatures = list(test_data2.columns[:6])
print("Features: \n", tFeatures)

#Get the data of the Test Features
testFeatures = test_data2[tFeatures]
print("Test Features: \n", testFeatures.head())

#Use the classifier to predict the class of the Testing data
testPrediction = classifier.predict(testFeatures)
print("Test Prediction: \n", testPrediction)

#For measuring the parameters like Accuracy, Recall and Precision, compare this Predicted class with the Actul class of the Testing data
#Get the Actual class of the Testing data
testActual = list(test_data2["class"])
print("Test Actual: \n", testActual)

#Measuring the Accuracy, Precision and Recall scores of the Classifier
#Accuracy is measured for the entire classifier, while Precision and Recall are computed for each class
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(testActual, testPrediction)),"%")

print("Precision: %.3f" %precision_score(testActual, testPrediction, average="macro"))
print("Recall: %.3f" %recall_score(testActual, testPrediction, average="macro"))

#Creating a Confusion Matrix for the Actual Vs Prediction
confusionMatrix = confusion_matrix(testActual, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

print("-------------------------------------#Minimum Records per Leaf Node = 15----------------------------------------------------------")
#Create a Decision Tree Classifier
#Criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
#Here we are using criterion = 'entropy' for the information gain
#Minimum Records per Leaf Node = 5
classifier = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf = 15)

#Using the Training data to fit the classifier
classifier = classifier.fit(X,y)
print("Classifier: \n", classifier)

#Now take the Testing data:
#Get the first 6 column names of the Test data and assign it to the Test Features (tFeatures)
tFeatures = list(test_data2.columns[:6])
print("Features: \n", tFeatures)

#Get the data of the Test Features
testFeatures = test_data2[tFeatures]
print("Test Features: \n", testFeatures.head())

#Use the classifier to predict the class of the Testing data
testPrediction = classifier.predict(testFeatures)
print("Test Prediction: \n", testPrediction)

#For measuring the parameters like Accuracy, Recall and Precision, compare this Predicted class with the Actul class of the Testing data
#Get the Actual class of the Testing data
testActual = list(test_data2["class"])
print("Test Actual: \n", testActual)

#Measuring the Accuracy, Precision and Recall scores of the Classifier
#Accuracy is measured for the entire classifier, while Precision and Recall are computed for each class
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(testActual, testPrediction)),"%")

print("Precision: %.3f" %precision_score(testActual, testPrediction, average="macro"))
print("Recall: %.3f" %recall_score(testActual, testPrediction, average="macro"))

#Creating a Confusion Matrix for the Actual Vs Prediction
confusionMatrix = confusion_matrix(testActual, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

print("-------------------------------------#Minimum Records per Leaf Node = 20----------------------------------------------------------")
#Create a Decision Tree Classifier
#Criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
#Here we are using criterion = 'entropy' for the information gain
#Minimum Records per Leaf Node = 5
classifier = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf = 20)

#Using the Training data to fit the classifier
classifier = classifier.fit(X,y)
print("Classifier: \n", classifier)

#Now take the Testing data:
#Get the first 6 column names of the Test data and assign it to the Test Features (tFeatures)
tFeatures = list(test_data2.columns[:6])
print("Features: \n", tFeatures)

#Get the data of the Test Features
testFeatures = test_data2[tFeatures]
print("Test Features: \n", testFeatures.head())

#Use the classifier to predict the class of the Testing data
testPrediction = classifier.predict(testFeatures)
print("Test Prediction: \n", testPrediction)

#For measuring the parameters like Accuracy, Recall and Precision, compare this Predicted class with the Actul class of the Testing data
#Get the Actual class of the Testing data
testActual = list(test_data2["class"])
print("Test Actual: \n", testActual)

#Measuring the Accuracy, Precision and Recall scores of the Classifier
#Accuracy is measured for the entire classifier, while Precision and Recall are computed for each class
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(testActual, testPrediction)), "%")

print("Precision: %.3f" %precision_score(testActual, testPrediction, average="macro"))
print("Recall: %.3f" %recall_score(testActual, testPrediction, average="macro"))

#Creating a Confusion Matrix for the Actual Vs Prediction
confusionMatrix = confusion_matrix(testActual, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

print("-------------------------------------#Minimum Records per Leaf Node = 25----------------------------------------------------------")
#Create a Decision Tree Classifier
#Criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
#Here we are using criterion = 'entropy' for the information gain
#Minimum Records per Leaf Node = 5
classifier = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf = 25)

#Using the Training data to fit the classifier
classifier = classifier.fit(X,y)
print("Classifier: \n", classifier)

#Now take the Testing data:
#Get the first 6 column names of the Test data and assign it to the Test Features (tFeatures)
tFeatures = list(test_data2.columns[:6])
print("Features: \n", tFeatures)

#Get the data of the Test Features
testFeatures = test_data2[tFeatures]
print("Test Features: \n", testFeatures.head())

#Use the classifier to predict the class of the Testing data
testPrediction = classifier.predict(testFeatures)
print("Test Prediction: \n", testPrediction)

#For measuring the parameters like Accuracy, Recall and Precision, compare this Predicted class with the Actul class of the Testing data
#Get the Actual class of the Testing data
testActual = list(test_data2["class"])
print("Test Actual: \n", testActual)

#Measuring the Accuracy, Precision and Recall scores of the Classifier
#Accuracy is measured for the entire classifier, while Precision and Recall are computed for each class
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(testActual, testPrediction)), "%")

print("Precision: %.3f" %precision_score(testActual, testPrediction, average="macro"))
print("Recall: %.3f" %recall_score(testActual, testPrediction, average="macro"))

#Creating a Confusion Matrix for the Actual Vs Prediction
confusionMatrix = confusion_matrix(testActual, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

#Now we will visualize the Decision Tree using the graphviz package. Here we are visualizing a Decision Tree with Minimum-Records-Per-Leaf-Node as 25
import graphviz 
decisionTreeViz = tree.export_graphviz(classifier, out_file=None, feature_names=tFeatures, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(decisionTreeViz) 
#Naming it as 'DecisionTreeViz'
graph.render("DecisionTreeViz")

#Visualizing the Confusion Matrix using the seaborn package
import seaborn as sns
ax= plt.subplot()
#annot=True to annotate cells - This provides labels(numbers) in the Confusion Matrix
sns.heatmap(confusionMatrix, annot=True, ax = ax);
#Setting the labels, title and tick marks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Abnormal', 'Normal']); ax.yaxis.set_ticklabels(['Normal', 'Abnormal']);
ax.plot()
plt.show()
