#CS545 programming assignment 2
#Naive bayes classifier
#Guy Cutting

import csv
import random
import math
import numpy as np
    
def load_from_csv(filename):
    #load the data from csv file
    lines = csv.reader(open(filename))
    dataset = list(lines)
    #read in in rows
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def split_dataset(dataset, ratio):
    #split data according to specified ratio
    #make sure training set size is an integer
    train_size = int(len(dataset) * ratio)
    train_set = []
    data_copy = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(data_copy))
        train_set.append(data_copy.pop(index))
    return [train_set, data_copy]

def separate_classes(dataset):
    #make dictionary of feature sets for each class
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    #return mean of a set of numbers
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    #return std. dev of a set of numbers
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    #return list of summary values (mean and std. dev)
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarize_by_class(dataset):
    separated = separate_classes(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries

def calculate_prob_dist(x, mean, stdev):
    #calculate gaussian distribution
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev+.000001,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev + .000001)) * exponent

def class_probs(summaries, in_vector):
    #calculate independent feature probabilities P(x | class) used for prediction
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        #calculate product of all probability values
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = in_vector[i]
            #wasn't necessary to take the log
            #in fact i had a divide by zero error when using log() of the probabilities so I did it just as the model specified
            probabilities[class_value] *= calculate_prob_dist(x, mean, stdev)
    return probabilities

def predict(summaries, in_vector):
    probabilities = class_probs(summaries, in_vector)
    best_label, best_prob = None, -1
    #find the probability-maximizing class
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def get_predictions(summaries, test_set):
    #initialize list
    predictions = []
    #make a prediction for each element in the set
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions

def get_accuracy(test_set, predictions):
    correct, total_pos, true_pos, pred_pos = 0,0,0,0
    cm = np.zeros((2,2))
    
    #counts needed to calculate recall and precision
    for i in range(len(test_set)):
        #count positive predictions
        if predictions[i] == 1:
            pred_pos += 1
        #count where predictions is correct
        if test_set[i][-1] == predictions[i]:
            correct += 1
            #count where positive prediction is correct
            if test_set[i][-1] == 1:
                true_pos += 1
        #count where predicted value is positive
        if test_set[i][-1] == 1:
            total_pos += 1
    
    #confusion matrix
    cm = np.zeros((2,2))
    actual_vals = np.array([element[-1] for element in test_set])
    predictions = np.array(predictions)
    for i in range(2):
        for j in range(2):
            blah = 1
            cm[i,j] = np.sum(np.where(predictions==i,1,0)*np.where(actual_vals==j,1,0))
    
    print("Confusion matrix: \n")
    print(cm)
    
    #calculate and display recall and precision
    print("\nrecall: ", true_pos/total_pos)
    print("precision:", true_pos/pred_pos)
    return (correct/float(len(test_set))) * 100.0

def main():
    filename = 'downloads/spambase.data.csv'
    split_ratio = 0.5
    dataset = load_from_csv(filename)
    training_set, test_set = split_dataset(dataset, split_ratio)
    # prepare model
    summaries = summarize_by_class(training_set)
    # test model
    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}%'.format(accuracy))

main()
