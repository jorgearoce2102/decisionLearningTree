from nose.tools import *
import DecisionTree as DT
import numpy as np


def entropy_function_test():

    """
    Function that gives the entropy H for a given value(s)
    """

    e = (5/8) * DT.entropy.get_boolean_entropy(4/5)

    eq_(0.451, round(e,3))

def read_dataset_test():
    """construct numpy arrays from dataset"""
    filename = "/home/jorge/Documents/2-Programming/AI/DecisionTree/Dataset/players.txt"
    data = np.loadtxt(filename, dtype = "S4", delimiter="\t")
    dataset = DT.Dataset(filename, _delimiter = "\t")
    #TODO make sure that the arguments are valid so that the dataset can be correcly loaded
    eq_(3,dataset.NbAttr)
    eq_(np.array(['No','Yes']),dataset.classes)

def create_decision_tree():
    pass

def gain_test():
    """"Function to test given dataset and their gain. The dataset to test from
    is the one shown in the book in page 700. Only Pat, Type  and WillWait columns"""

    #create dataset
    filename = "/home/jorge/Documents/2-Programming/AI/DecisionTree/Dataset/restaurant.txt"
    dataset = DT.Dataset(filename, _delimiter = '\t')

    Tree = DT.DecisionTree(dataset)

    #test entropy function. Because the probs are equal for each class, then 1.0
    eq_(1.0,Tree.entropy(Tree.dataset[:,-1]))

    #test two attribues, calculations got from book page 704
    eq_(0.541,Tree.gain(4))
    eq_(0.0,Tree.gain(1))

    print(Tree.getImportance())

def decisionTreeLearning_test():
    """"Function to test the general decision tree learning function."""

    #create dataset
    filename = "/home/jorge/Documents/2-Programming/AI/DecisionTree/Dataset/restaurant.txt"
    dataset = DT.Dataset(filename, _delimiter = '\t')

    Tree = DT.DecisionTree(dataset)

    #show first branch of decisition tree shown in page 702
    # print(Tree.root)
    # print(Tree.leaf)

    #return tree for classify function next
    return Tree

def classify_test():
    """Test classify function that classifies a given exemple"""

    tree = decisionTreeLearning_test()
    exemple = np.array([b"Yes",b"Yes",b"Yes",b"Yes",b"Some",b"$",b"No",b"No",b"Burg",b"c"])
    class_ = tree.classify(exemple)

    eq_(class_, b' Yes')

def classify_dataset_test():
    #create dataset
    filename = "Dataset/iris.data"
    dataset = DT.Dataset(filename,_delimiter = ',')
    Tree = DT.DecisionTree(dataset)

    #load exemples
    exemple1 = np.array([5.4,3.9,1.3,0.4]).astype('S15')
    exemple2 = np.array([6.3,2.5,4.9,1.5]).astype('S15')
    exemple3 = np.array([6.5,3.0,5.5,1.8,]).astype('S15')

    #classify exemples
    class1 = Tree.classify(exemple1)
    class2 = Tree.classify(exemple2)
    class3 = Tree.classify(exemple3)

    #verify classification
    eq_(class1, b'Iris-setosa')
    eq_(class2, b'Iris-versicolor')
    eq_(class3, b'Iris-virginica')
