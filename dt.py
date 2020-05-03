#!/usr/bin/python

import sys, getopt
import DecisionTree as DT
import numpy as np

def main(argv):

    exemple = ""

    #load dataset
    try:
        filename = argv[0]
        dataset = DT.Dataset(filename,_delimiter = ',')
    except getopt.GetoptError:
        print('dt.py -i <dataset dir>')
        sys.exit(2)

    #create and train decision tree
    Tree = DT.DecisionTree(dataset)

    #load exemple
    try:
        exemple = np.array(argv[1].split(",")).astype('S15')
    except:
        print("error loading exemple")

    # classify
    eclass = Tree.classify(exemple)
    print(eclass)

if __name__ == "__main__":
   main(sys.argv[1:])
