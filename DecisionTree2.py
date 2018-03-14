# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

#Read Data
data2 = pd.read_csv('C:/Users/surya/Desktop/SpringSemester/IDA/HW1/Dataset/BiomechanicalData_column_3C_weka.csv')

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

#For measuring parameters like Accuracy, Recall and Precision, compare this Predicted class with the Actul class of the Testing data
#Get the Actual class of the Testing data
testActual = list(test_data2["class"])
print("Test Actual: \n", testActual)

#Creating a Confusion Matrix for the Actual Vs Prediction
confusionMatrix = confusion_matrix(testActual, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

#Measuring the Accuracy, Precision and Recall scores of the Classifier
#Accuracy is measured for the entire classifier, while Precision and Recall are computed for each class
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(testActual, testPrediction)), "%")

#Calculating Precision values for the multilabel / multiclass (3 classes) [Considering each class separately as the Positive Class]
#precision_score(testActual, testPrediction, average = None) - returns the Precision values for all 3 classes in the Ascending order of their class names/labels
#However, if you specify the 'labels' parameter in the function, the specified label appears in the first position of the array
#Precision for the 3 classes [Considering each class separately as the Positive Class]:
print("Precision of the 3 classes in the Ascending Order of their names: ", precision_score(testActual, testPrediction, average = None))

#Precision of Normal Class - Specify Normal class in the 'labels' parameter - [Considering Normal class as the Positive Class]
PrecisionNormal = str(round((100*precision_score(testActual, testPrediction, labels= "Normal", average = None))[0], 2))
print("Precision of Normal Class: ", PrecisionNormal, "%")

#Precision of Spondylolisthesis Class - Specify Spondylolisthesis class in the 'labels' parameter - [Considering Spondylolisthesis class as the Positive Class]
PrecisionSpondylolisthesis = str(round((100*precision_score(testActual, testPrediction, labels= "Spondylolisthesis", average = None))[0], 2))
print("Precision of Spondylolisthesis Class: ", PrecisionSpondylolisthesis, "%")

#Precision of Hernia Class - Specify Hernia class in the 'labels' parameter - [Considering Hernia class as the Positive Class]
PrecisionHernia = str(round((100*precision_score(testActual, testPrediction, labels= "Hernia", average = None))[0], 2))
print("Precision of Hernia Class: ", PrecisionHernia, "%")

#Calculating Recall values for the multilabel / multiclass (3 classes) [Considering each class separately as the Positive Class]
#recall_score(testActual, testPrediction, average = None) - returns the Recall values for all 3 classes in the Ascending order of their class names/labels
#However, if you specify the 'labels' parameter in the function, the specified label appears in the first position of the array
#Recall for the 3 classes [Considering each class separately as the Positive Class]:
print("Recall of the 3 classes in the Ascending Order of their names: ", recall_score(testActual, testPrediction, average = None))

#Recall of Normal Class - Specify Normal class in the 'labels' parameter - [Considering Normal class as the Positive Class]
RecallNormal = str(round((100*recall_score(testActual, testPrediction, labels= "Normal", average = None))[0], 2))
print("Recall of Normal Class: ", RecallNormal, "%")

#Recall of Spondylolisthesis Class - Specify Spondylolisthesis class in the 'labels' parameter - [Considering Spondylolisthesis class as the Positive Class]
RecallSpondylolisthesis = str(round((100*recall_score(testActual, testPrediction, labels= "Spondylolisthesis", average = None))[0], 2))
print("Recall of Spondylolisthesis Class: ", RecallSpondylolisthesis, "%")

#Recall of Hernia Class - Specify Hernia class in the 'labels' parameter - [Considering Hernia class as the Positive Class]
RecallHernia = str(round((100*recall_score(testActual, testPrediction, labels= "Hernia", average = None))[0], 2))
print("Recall of Hernia Class: ", RecallHernia, "%")

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

#For measuring parameters like Accuracy, Recall and Precision, compare this Predicted class with the Actul class of the Testing data
#Get the Actual class of the Testing data
testActual = list(test_data2["class"])
print("Test Actual: \n", testActual)

#Creating a Confusion Matrix for the Actual Vs Prediction
confusionMatrix = confusion_matrix(testActual, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

#Measuring the Accuracy, Precision and Recall scores of the Classifier
#Accuracy is measured for the entire classifier, while Precision and Recall are computed for each class
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(testActual, testPrediction)), "%")

#Calculating Precision values for the multilabel / multiclass (3 classes) [Considering each class separately as the Positive Class]
#precision_score(testActual, testPrediction, average = None) - returns the Precision values for all 3 classes in the Ascending order of their class names/labels
#However, if you specify the 'labels' parameter in the function, the specified label appears in the first position of the array
#Precision for the 3 classes [Considering each class separately as the Positive Class]:
print("Precision of the 3 classes in the Ascending Order of their names: ", precision_score(testActual, testPrediction, average = None))

#Precision of Normal Class - Specify Normal class in the 'labels' parameter - [Considering Normal class as the Positive Class]
PrecisionNormal = str(round((100*precision_score(testActual, testPrediction, labels= "Normal", average = None))[0], 2))
print("Precision of Normal Class: ", PrecisionNormal, "%")

#Precision of Spondylolisthesis Class - Specify Spondylolisthesis class in the 'labels' parameter - [Considering Spondylolisthesis class as the Positive Class]
PrecisionSpondylolisthesis = str(round((100*precision_score(testActual, testPrediction, labels= "Spondylolisthesis", average = None))[0], 2))
print("Precision of Spondylolisthesis Class: ", PrecisionSpondylolisthesis, "%")

#Precision of Hernia Class - Specify Hernia class in the 'labels' parameter - [Considering Hernia class as the Positive Class]
PrecisionHernia = str(round((100*precision_score(testActual, testPrediction, labels= "Hernia", average = None))[0], 2))
print("Precision of Hernia Class: ", PrecisionHernia, "%")

#Calculating Recall values for the multilabel / multiclass (3 classes) [Considering each class separately as the Positive Class]
#recall_score(testActual, testPrediction, average = None) - returns the Recall values for all 3 classes in the Ascending order of their class names/labels
#However, if you specify the 'labels' parameter in the function, the specified label appears in the first position of the array
#Recall for the 3 classes [Considering each class separately as the Positive Class]:
print("Recall of the 3 classes in the Ascending Order of their names: ", recall_score(testActual, testPrediction, average = None))

#Recall of Normal Class - Specify Normal class in the 'labels' parameter - [Considering Normal class as the Positive Class]
RecallNormal = str(round((100*recall_score(testActual, testPrediction, labels= "Normal", average = None))[0], 2))
print("Recall of Normal Class: ", RecallNormal, "%")

#Recall of Spondylolisthesis Class - Specify Spondylolisthesis class in the 'labels' parameter - [Considering Spondylolisthesis class as the Positive Class]
RecallSpondylolisthesis = str(round((100*recall_score(testActual, testPrediction, labels= "Spondylolisthesis", average = None))[0], 2))
print("Recall of Spondylolisthesis Class: ", RecallSpondylolisthesis, "%")

#Recall of Hernia Class - Specify Hernia class in the 'labels' parameter - [Considering Hernia class as the Positive Class]
RecallHernia = str(round((100*recall_score(testActual, testPrediction, labels= "Hernia", average = None))[0], 2))
print("Recall of Hernia Class: ", RecallHernia, "%")

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

#For measuring parameters like Accuracy, Recall and Precision, compare this Predicted class with the Actul class of the Testing data
#Get the Actual class of the Testing data
testActual = list(test_data2["class"])
print("Test Actual: \n", testActual)

#Creating a Confusion Matrix for the Actual Vs Prediction
confusionMatrix = confusion_matrix(testActual, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

#Measuring the Accuracy, Precision and Recall scores of the Classifier
#Accuracy is measured for the entire classifier, while Precision and Recall are computed for each class
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(testActual, testPrediction)), "%")

#Calculating Precision values for the multilabel / multiclass (3 classes) [Considering each class separately as the Positive Class]
#precision_score(testActual, testPrediction, average = None) - returns the Precision values for all 3 classes in the Ascending order of their class names/labels
#However, if you specify the 'labels' parameter in the function, the specified label appears in the first position of the array
#Precision for the 3 classes [Considering each class separately as the Positive Class]:
print("Precision of the 3 classes in the Ascending Order of their names: ", precision_score(testActual, testPrediction, average = None))

#Precision of Normal Class - Specify Normal class in the 'labels' parameter - [Considering Normal class as the Positive Class]
PrecisionNormal = str(round((100*precision_score(testActual, testPrediction, labels= "Normal", average = None))[0], 2))
print("Precision of Normal Class: ", PrecisionNormal, "%")

#Precision of Spondylolisthesis Class - Specify Spondylolisthesis class in the 'labels' parameter - [Considering Spondylolisthesis class as the Positive Class]
PrecisionSpondylolisthesis = str(round((100*precision_score(testActual, testPrediction, labels= "Spondylolisthesis", average = None))[0], 2))
print("Precision of Spondylolisthesis Class: ", PrecisionSpondylolisthesis, "%")

#Precision of Hernia Class - Specify Hernia class in the 'labels' parameter - [Considering Hernia class as the Positive Class]
PrecisionHernia = str(round((100*precision_score(testActual, testPrediction, labels= "Hernia", average = None))[0], 2))
print("Precision of Hernia Class: ", PrecisionHernia, "%")

#Calculating Recall values for the multilabel / multiclass (3 classes) [Considering each class separately as the Positive Class]
#recall_score(testActual, testPrediction, average = None) - returns the Recall values for all 3 classes in the Ascending order of their class names/labels
#However, if you specify the 'labels' parameter in the function, the specified label appears in the first position of the array
#Recall for the 3 classes [Considering each class separately as the Positive Class]:
print("Recall of the 3 classes in the Ascending Order of their names: ", recall_score(testActual, testPrediction, average = None))

#Recall of Normal Class - Specify Normal class in the 'labels' parameter - [Considering Normal class as the Positive Class]
RecallNormal = str(round((100*recall_score(testActual, testPrediction, labels= "Normal", average = None))[0], 2))
print("Recall of Normal Class: ", RecallNormal, "%")

#Recall of Spondylolisthesis Class - Specify Spondylolisthesis class in the 'labels' parameter - [Considering Spondylolisthesis class as the Positive Class]
RecallSpondylolisthesis = str(round((100*recall_score(testActual, testPrediction, labels= "Spondylolisthesis", average = None))[0], 2))
print("Recall of Spondylolisthesis Class: ", RecallSpondylolisthesis, "%")

#Recall of Hernia Class - Specify Hernia class in the 'labels' parameter - [Considering Hernia class as the Positive Class]
RecallHernia = str(round((100*recall_score(testActual, testPrediction, labels= "Hernia", average = None))[0], 2))
print("Recall of Hernia Class: ", RecallHernia, "%")

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

#For measuring parameters like Accuracy, Recall and Precision, compare this Predicted class with the Actul class of the Testing data
#Get the Actual class of the Testing data
testActual = list(test_data2["class"])
print("Test Actual: \n", testActual)

#Creating a Confusion Matrix for the Actual Vs Prediction
confusionMatrix = confusion_matrix(testActual, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

#Measuring the Accuracy, Precision and Recall scores of the Classifier
#Accuracy is measured for the entire classifier, while Precision and Recall are computed for each class
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(testActual, testPrediction)), "%")

#Calculating Precision values for the multilabel / multiclass (3 classes) [Considering each class separately as the Positive Class]
#precision_score(testActual, testPrediction, average = None) - returns the Precision values for all 3 classes in the Ascending order of their class names/labels
#However, if you specify the 'labels' parameter in the function, the specified label appears in the first position of the array
#Precision for the 3 classes [Considering each class separately as the Positive Class]:
print("Precision of the 3 classes in the Ascending Order of their names: ", precision_score(testActual, testPrediction, average = None))

#Precision of Normal Class - Specify Normal class in the 'labels' parameter - [Considering Normal class as the Positive Class]
PrecisionNormal = str(round((100*precision_score(testActual, testPrediction, labels= "Normal", average = None))[0], 2))
print("Precision of Normal Class: ", PrecisionNormal, "%")

#Precision of Spondylolisthesis Class - Specify Spondylolisthesis class in the 'labels' parameter - [Considering Spondylolisthesis class as the Positive Class]
PrecisionSpondylolisthesis = str(round((100*precision_score(testActual, testPrediction, labels= "Spondylolisthesis", average = None))[0], 2))
print("Precision of Spondylolisthesis Class: ", PrecisionSpondylolisthesis, "%")

#Precision of Hernia Class - Specify Hernia class in the 'labels' parameter - [Considering Hernia class as the Positive Class]
PrecisionHernia = str(round((100*precision_score(testActual, testPrediction, labels= "Hernia", average = None))[0], 2))
print("Precision of Hernia Class: ", PrecisionHernia, "%")

#Calculating Recall values for the multilabel / multiclass (3 classes) [Considering each class separately as the Positive Class]
#recall_score(testActual, testPrediction, average = None) - returns the Recall values for all 3 classes in the Ascending order of their class names/labels
#However, if you specify the 'labels' parameter in the function, the specified label appears in the first position of the array
#Recall for the 3 classes [Considering each class separately as the Positive Class]:
print("Recall of the 3 classes in the Ascending Order of their names: ", recall_score(testActual, testPrediction, average = None))

#Recall of Normal Class - Specify Normal class in the 'labels' parameter - [Considering Normal class as the Positive Class]
RecallNormal = str(round((100*recall_score(testActual, testPrediction, labels= "Normal", average = None))[0], 2))
print("Recall of Normal Class: ", RecallNormal, "%")

#Recall of Spondylolisthesis Class - Specify Spondylolisthesis class in the 'labels' parameter - [Considering Spondylolisthesis class as the Positive Class]
RecallSpondylolisthesis = str(round((100*recall_score(testActual, testPrediction, labels= "Spondylolisthesis", average = None))[0], 2))
print("Recall of Spondylolisthesis Class: ", RecallSpondylolisthesis, "%")

#Recall of Hernia Class - Specify Hernia class in the 'labels' parameter - [Considering Hernia class as the Positive Class]
RecallHernia = str(round((100*recall_score(testActual, testPrediction, labels= "Hernia", average = None))[0], 2))
print("Recall of Hernia Class: ", RecallHernia, "%")

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

#For measuring parameters like Accuracy, Recall and Precision, compare this Predicted class with the Actul class of the Testing data
#Get the Actual class of the Testing data
testActual = list(test_data2["class"])
print("Test Actual: \n", testActual)

#Creating a Confusion Matrix for the Actual Vs Prediction
confusionMatrix = confusion_matrix(testActual, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

#Measuring the Accuracy, Precision and Recall scores of the Classifier
#Accuracy is measured for the entire classifier, while Precision and Recall are computed for each class
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(testActual, testPrediction)), "%")

#Calculating Precision values for the multilabel / multiclass (3 classes) [Considering each class separately as the Positive Class]
#precision_score(testActual, testPrediction, average = None) - returns the Precision values for all 3 classes in the Ascending order of their class names/labels
#However, if you specify the 'labels' parameter in the function, the specified label appears in the first position of the array
#Precision for the 3 classes [Considering each class separately as the Positive Class]:
print("Precision of the 3 classes in the Ascending Order of their names: ", precision_score(testActual, testPrediction, average = None))

#Precision of Normal Class - Specify Normal class in the 'labels' parameter - [Considering Normal class as the Positive Class]
PrecisionNormal = str(round((100*precision_score(testActual, testPrediction, labels= "Normal", average = None))[0], 2))
print("Precision of Normal Class: ", PrecisionNormal, "%")

#Precision of Spondylolisthesis Class - Specify Spondylolisthesis class in the 'labels' parameter - [Considering Spondylolisthesis class as the Positive Class]
PrecisionSpondylolisthesis = str(round((100*precision_score(testActual, testPrediction, labels= "Spondylolisthesis", average = None))[0], 2))
print("Precision of Spondylolisthesis Class: ", PrecisionSpondylolisthesis, "%")

#Precision of Hernia Class - Specify Hernia class in the 'labels' parameter - [Considering Hernia class as the Positive Class]
PrecisionHernia = str(round((100*precision_score(testActual, testPrediction, labels= "Hernia", average = None))[0], 2))
print("Precision of Hernia Class: ", PrecisionHernia, "%")

#Calculating Recall values for the multilabel / multiclass (3 classes) [Considering each class separately as the Positive Class]
#recall_score(testActual, testPrediction, average = None) - returns the Recall values for all 3 classes in the Ascending order of their class names/labels
#However, if you specify the 'labels' parameter in the function, the specified label appears in the first position of the array
#Recall for the 3 classes [Considering each class separately as the Positive Class]:
print("Recall of the 3 classes in the Ascending Order of their names: ", recall_score(testActual, testPrediction, average = None))

#Recall of Normal Class - Specify Normal class in the 'labels' parameter - [Considering Normal class as the Positive Class]
RecallNormal = str(round((100*recall_score(testActual, testPrediction, labels= "Normal", average = None))[0], 2))
print("Recall of Normal Class: ", RecallNormal, "%")

#Recall of Spondylolisthesis Class - Specify Spondylolisthesis class in the 'labels' parameter - [Considering Spondylolisthesis class as the Positive Class]
RecallSpondylolisthesis = str(round((100*recall_score(testActual, testPrediction, labels= "Spondylolisthesis", average = None))[0], 2))
print("Recall of Spondylolisthesis Class: ", RecallSpondylolisthesis, "%")

#Recall of Hernia Class - Specify Hernia class in the 'labels' parameter - [Considering Hernia class as the Positive Class]
RecallHernia = str(round((100*recall_score(testActual, testPrediction, labels= "Hernia", average = None))[0], 2))
print("Recall of Hernia Class: ", RecallHernia, "%")

#Now we will visualize the Decision Tree using the graphviz package. Here we are visualizing a Decision Tree with Minimum-Records-Per-Leaf-Node as 25
import graphviz 
decisionTreeViz = tree.export_graphviz(classifier, out_file=None, feature_names=tFeatures, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(decisionTreeViz) 
#Naming it as 'DecisionTreeViz'
graph.render("DecisionTreeViz")

#Visualizing the Confusion Matrix using the seaborn package
import seaborn as sns
fig, ax= plt.subplots(figsize=(8,6))

#When you set xticklabels=True, yticklabels=True in the Heatmap function, it automatically provides the labels for Confusion matrix with 0s, 1s and 2s in the Ascending order of the Class labels  
#Use this to find out the labels of the Confusion Matrix:
#sns.heatmap(confusionMatrix, annot=True, linewidths=.5, xticklabels=True, yticklabels=True);

#After you find out the labels in the Confusion Matrix, remove the xticklabels & yticklabels parameters. Set the labels manually now
#annot=True to annotate cells - This provides labels(numbers) in the Confusion Matrix

sns.heatmap(confusionMatrix, annot=True, linewidths=.5);
ax.xaxis.set_ticklabels(['Hernia', 'Normal', 'Spondylolisthesis']); 
ax.yaxis.set_ticklabels(['Spondylolisthesis', 'Normal', 'Hernia']);

#Setting the labels, title and tick marks
#Usually in the Confusion Matrix, the Y-Axis represents the True Values and X-Axis represents the Predicted Values
ax.set_title('Confusion Matrix', fontsize=20);
ax.set_xlabel('Predicted Labels', fontsize=12);
ax.set_ylabel('True Labels', fontsize=12); 

ax.plot()
plt.show()
