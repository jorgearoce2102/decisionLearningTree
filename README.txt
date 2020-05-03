

Implementation of the Decision Tree (DT) algorithm for supervised learning.
DT theory is taken from AI a modern approach (Russel, Norving) chapter 18.3

Prerequisites:

Numpy

Usage:

python dt.py <dataset dir> <exemple>

<dataset dir> : the directory in which the dataset is saved (read Dataset requirements listed below)

<exemple> : attributes of a classification exemple, all together separated by a coma


Dataset requirements:

Dataset must be saved in a text file readible by Numpy and composed of
attribute columns from left to right and a classification column at the
extreme right coulumn. No column name title. For numerical attributes, values
must be grouped otherwise for numerical values, the tree grows unnecessarily bigger.

#TODO:

-Include a python script with cross validation to measure optimal training-validation dataset 
