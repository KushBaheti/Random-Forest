# Random Forest Algorithm on Sonar Dataset
from random import sample
from csv import reader
from math import sqrt, floor, ceil
import sys

# Load CSV file
def loadCSV(filename):
    dataset = list()
    with open(filename, 'r') as file:
        fileContents = reader(file)
        for row in fileContents:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Create a random subsample from the dataset with replacement
def createSubset(dataset, subsetRatio):
    subset = list()
    number_samples = round(len(dataset)*subsetRatio)
    subset = sample(dataset, number_samples)
    return subset

# Select features without replacement
def selectFeatures():
    numOfFeatures = ceil(sqrt(len(attributes)))
    features = list()
    index_list = []
    for i in range(len(attributes)):
        index_list.append(i)
    features = sample(index_list, numOfFeatures)
    return features

def giniIndex(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Find best split, given the attribute values
def findBestSplit(subset):
    classValues = list(set(row[-1] for row in subset))
    features = selectFeatures()
    bestIndex, bestValue, bestScore, bestGroups = float("inf"), float("inf"), float("inf"), None # CHANGED
    for index in features:
        for row in subset:
            falseValues, trueValues = splitOnAttribute(subset, index, row[index])
            # Calculating the info gain 
            gini = giniIndex((falseValues, trueValues), classValues)
            if gini < bestScore:
                bestIndex, bestValue, bestScore, bestGroups = index, row[index], gini, (falseValues, trueValues)
    returnValue = {"groups": bestGroups, "value": bestValue, "index": bestIndex}
    return returnValue

# Split a dataset based on an attribute and an attribute value
def splitOnAttribute(subset, index, value):
    left, right = list(), list()
    for row in subset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Make the node passed as arguent a leaf of the decision tree currently being built
def makeTerminal(group):
    #get outcome column
    let_cols = []
    for row in group:
        let_cols.append(row[-1])
    #get max freq outcome
    return max(set(let_cols), key=let_cols.count)

# Function that initially takes a root node and then recursively builds rest of the tree  
def splitNodes(node, currDepth, maxDepth, minSize):

    falseValues, trueValues = node["groups"]
    del(node["groups"])

    # Check for split with a children containing zero values
    if not falseValues or not trueValues:
        node["trueValues"] = makeTerminal(falseValues+trueValues)
        node["falseValues"] = node["trueValues"]
        return 
    
    # Check if reached max depth
    if currDepth>=maxDepth:
        node["trueValues"] = makeTerminal(trueValues)
        node["falseValues"] = makeTerminal(falseValues)        
        return    

    # Value length smaller than minSize then terminal node
    if len(trueValues)<=minSize:
        node["trueValues"] = makeTerminal(trueValues)
    else: # create subtree
        node["trueValues"]=findBestSplit(trueValues)
        splitNodes(node["trueValues"], currDepth+1, maxDepth, minSize)

    # Value length smaller than minSize then terminal node
    if len(falseValues)<=minSize:
        node["falseValues"] = makeTerminal(falseValues)
    else: # create subtree
        node["falseValues"]=findBestSplit(falseValues)
        splitNodes(node["falseValues"], currDepth+1, maxDepth, minSize)  

# Build the decision tree
def makeRoot(subset, maxDepth, minSize):
    root = findBestSplit(subset)
    splitNodes(root, 1, maxDepth, minSize)
    return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value'] :
		if isinstance(node['falseValues'], dict):
			return predict(node['falseValues'], row)
		else:
			return node['falseValues']
	else:
		if isinstance(node['trueValues'], dict):
			return predict(node['trueValues'], row)
		else:
			return node['trueValues']

# Make a prediction with a list of bagged trees
def baggingPredict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

# The Random Forest Algorithm
# First it generates random trees using bootstrap aggregation and feature selection
# Then it makes predictions (classifies) unseen data 
def randomForest(trainingData, testData, maxDepth, minSize, subsetRatio, nTrees):
    trees = list()
    for i in range(nTrees):
        subset = createSubset(trainingData, subsetRatio)
        tree = makeRoot(subset, maxDepth, minSize)
        trees.append(tree)
    predictions = []
    for row in testData:
        pred = baggingPredict(trees, row)
        predictions.append(pred)
    return(predictions)

# Convert string column to float
def strColumnToFloat(data, column):
    for row in data:
        row[column] = float(row[column].strip())

# Convert string column to integer
def strColumnToInt(trainData, testData, column):
    class_values = [row[column] for row in trainData]
    unique = set(class_values)
    lookup = dict()

    # creating a mapping of attributes -> ints
    for i, value in enumerate(unique):
        lookup[value] = i
	
    # mapping trainData
    for row in trainData:
        row[column] = lookup[row[column]]

    # mapping testDate
    for row in testData:
        row[column] = lookup[row[column]]
	
# Measure accuracy of predictions made on unseen data
def accuracyMetric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Identify which dataset the user has selected to work with
datasetName = sys.argv
if len(datasetName) < 2:
    print("Usage: python random-forest.py [cancer/car]")
    exit()

# Load the required datasets
def loadTrainingAndTestData(datasetName, labelsName):
    data = loadCSV(datasetName)
    attributes = data.pop(0)
    labels = loadCSV(labelsName)
    labels.pop(0)
    if (datasetName.find('train') == -1):
        return (data, labels)
    else:
        return (data, labels, attributes)

# Load the appropriate datasets and clean
trainingData, trainingLabels, testData, testLabels, attributes = [], [], [], [], []
if datasetName[1] == 'cancer':
    # Load training data set and its corresponding labels
    trainingData, trainingLabels, attributes = loadTrainingAndTestData('cancer_X_train.csv', 'cancer_y_train.csv')
    # Load test data set and its corresponding labels
    testData, testLabels = loadTrainingAndTestData('cancer_X_test.csv', 'cancer_y_test.csv')

    # Merge labels to rest of dataset for convenience
    for i in range(len(trainingData)):
        trainingData[i].append(trainingLabels[i][0])
    # Merge labels to rest of dataset for convenience
    for i in range(len(testData)):
        testData[i].append(testLabels[i][0])
    
    # Convert numbers in str to float
    for i in range(0, len(trainingData[0])):
        strColumnToFloat(trainingData, i)
    # Convert numbers in str to float
    for i in range(0, len(testData[0])):
        strColumnToFloat(testData, i)

elif datasetName[1] == 'car':
    # Load training data set and its corresponding labels
    trainingData, trainingLabels, attributes = loadTrainingAndTestData('car_X_train.csv', 'car_y_train.csv')
    # Load test data set and its corresponding labels
    testData, testLabels = loadTrainingAndTestData('car_X_test.csv', 'car_y_test.csv')
    # Merge labels to rest of dataset for convenience
    for i in range(len(trainingData)):
        trainingData[i].append(trainingLabels[i][0])
    # Merge labels to rest of dataset for convenience
    for i in range(len(testData)):
        testData[i].append(testLabels[i][0])

    # converting categorical values to Int
    for i in range(len(trainingData[0])):
        strColumnToInt(trainingData, testData, i)

# Initialize values of certain parameters
minSize = 1

# nTrees = 10
maxDepth = 10
subsetRatio = 0.75

print("Dataset:", datasetName[1].upper())

# Use the Random Forest with di fferent tree sizes and check accuracy
for nTrees in [5, 10, 15]:
    predicted = randomForest(trainingData, testData, maxDepth, minSize, subsetRatio, nTrees)
    actual = []
    for row in testData:
        actual.append(row[-1])
    accuracy = accuracyMetric(actual, predicted)
    print("Max depth:", maxDepth, "; subset ratio:", subsetRatio, "; number of trees:", nTrees)
    print("ACCURACY =", accuracy, "%")

# BY:
# JANMAJAYA MALL (3035492159)
# KUSH BAHETI (3035436583)