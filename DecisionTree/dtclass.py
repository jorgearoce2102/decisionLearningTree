import numpy as np
import math

log = math.log
from . import entropy

class Dataset(object):

    def __init__(self,filename, classPosition = -1, _dtype = "S15", _delimiter = ','):

        self.classPosition = classPosition
        #Try to read filename onto a numpy array otherwise show error
        try:
            self.data = np.loadtxt(filename, dtype = _dtype, delimiter=_delimiter)
        except ValueError:
            print("Error loading dataset.")

        #Get dataset information (number attributes, size, etc)
        self.dataset_info()

    def dataset_info(self):
        """Function that defines all the info of the dataset so it is ready when needed"""

        #identify classes
        self.classes = np.unique(self.data[:,self.classPosition])
        self.NbClasses = len(self.classes)

class DecisionTree(object):

    def __init__(self, dataset):

        #Initiate dataset class
        self.dataset = dataset.data

        #number of attributes in every exemple, attr will be identified by index num
        self.NbAttributes = np.shape(self.dataset)[1] - 1

        #create tree
        self.Tree = self.decisionTreeLearning(self.dataset)

    def decisionTreeLearning(self, dataset):

        #if all exemples all have the same classification
        classes = np.unique(dataset[:,-1])
        if len(classes) == 1:
            return classes

        #if attributres is empty
        elif np.shape(dataset)[1]==2:
            (_,counts) = np.unique(dataset[:,0].flatten(),return_counts=True)
            ind=np.argmax(counts)
            return dataset[:,0][ind]

        else:
            #get most important attr. A is formed of vk. a is the attr name
            A,a= self.getImportance(dataset)

            #create tree with root A
            tree = Tree((A,a))
            for vk in A:
                #get examples corresponding to value vk
                exs = dataset[dataset[:,a] == vk]
                subtree = self.decisionTreeLearning(exs)
                tree.addBranch(subtree)
            return tree

    def getImportance(self, dataset):
        """get the most important attribute add branch to tree
        returns
        mostImportantAttr: The attribute with the most gain of inf  """

        #init values
        highestGain = 0
        mostImportantAttr = 0

        #iterate over each attribute to get their gain and select the highest one
        for attrID in range(np.shape(dataset)[1]-1):
            gain = self.gain(attrID,dataset)
            if gain > highestGain:
                highestGain = gain
                mostImportantAttr = attrID

        # self.dataset = np.delete(self.dataset, mostImportantAttr, 1)

        return np.unique(dataset[:,mostImportantAttr]), mostImportantAttr

    def gain(self, attrID, dataset):
        """Function that returns the gain from a given coulumn attrID
        args
        attrID: the column from which to get gain. Must not be the rightmost
            column since thats the one for the classes
        """
        #values that the attribute may take
        d = np.unique(dataset[:,attrID])
        NbExemples = len(dataset[:,attrID])

        #init remainder and gain
        R = 0

        #attribute and its corresponding classes table
        a_c = dataset[:,[attrID,-1]]

        for k in d:
            #probability that the attribute d will take value k
            kexemples = dataset[:,attrID][dataset[:,attrID] == k]
            pk = len(kexemples)/NbExemples

            #probability distribution given k
            #a_c table of the attributeID column and its corresponding class
            a_c = dataset[:,[attrID,-1]]

            Hk = self.entropy(a_c[:,1][a_c[:,0]==k])

            #summ probability of the given value times its entropy
            R += pk*Hk

        G = self.entropy(dataset[:,-1]) - R

        return round(G, 3)

    def entropy(self, V):

        """get entropy of a given set V of values vk"""

        #number k of values v in V and their probabilities
        (k, kcount)  =  np.unique(V, return_counts = True)
        Pvk = kcount/len(V)

        #Entropy
        H = 0

        for vk, pvk in zip(k,Pvk):
            H += pvk*log(pvk,2)

        return round(-H, 3)

    def classify(self, exemple):
        """Classify an given exemple
        args

        exemple: cointains the value of the total d attributes of the exemple"""

        assert len(exemple) == self.NbAttributes, \
        "exemple not valid, must contain {} arguments".format(self.NbAttributes)

        Tree = self.Tree
        while True:
            try:
                attrID = Tree.root_name
            except:
                return Tree
            v = exemple[attrID]
            i = np.where(Tree.root == v )[0][0]
            Tree = Tree.leaf[i]


class Tree(object):

    def __init__(self, root):
        #root: set of values, root name: the attrID from parents
        self.root, self.root_name = root
        self.leaf = np.empty((0), dtype = "S1")

    def addBranch(self, branch):
        self.leaf = np.append(self.leaf, branch)
