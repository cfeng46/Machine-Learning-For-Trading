"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
Note, this is NOT a correct DTLearner; Replace with your own implementation.
"""

import numpy as np

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose = False):
	self.leaf_size = leaf_size

    def author(self):
	return "cfeng46"

    def build_tree(self, dataX, dataY):
	if dataX.shape[0] <= self.leaf_size:
	    return np.array([[-1.0, dataY.mean(), np.nan, np.nan]])
	if (np.std(dataY) == 0):
	    return np.array([[-1.0, dataY.mean(), np.nan, np.nan]])
	#if (np.std(dataX) == 0):
	 #   return np.array([[-1.0, dataY.mean(), np.nan, np.nan]])
	else:
	    features = abs(np.corrcoef(dataX.T, dataY)[-1,0:-1])
	    bestVar = np.argmax(np.nan_to_num(features))
	    SplitVal = np.median(dataX[:,bestVar])
	    if len(dataX[dataX[:,bestVar] <= SplitVal]) == 0:
		return np.array([[-1.0, dataY[dataX[:,bestVar] > SplitVal].mean(), np.nan, np.nan]])
	    if len(dataX[dataX[:,bestVar] > SplitVal]) == 0:
		return np.array([[-1.0, dataY[dataX[:,bestVar] <= SplitVal].mean(), np.nan, np.nan]])
	    if (np.std(dataX[:,bestVar]) == 0):
		return np.array([[-1.0, dataY.mean(), np.nan, np.nan]])
	    lefttree = self.build_tree(dataX[dataX[:,bestVar] <= SplitVal], dataY[dataX[:,bestVar] <= SplitVal])
	    righttree = self.build_tree(dataX[dataX[:,bestVar] > SplitVal], dataY[dataX[:,bestVar] > SplitVal])
	    root = np.array([[bestVar, SplitVal, 1, lefttree.shape[0] + 1]])
	    return (np.vstack([root, lefttree, righttree]))
    def addEvidence(self, dataX, dataY):
	self.tree = self.build_tree(dataX, dataY)
			
    def pred(self, point, node):
	if self.tree[node,0] == -1.0:
	    return self.tree[node,1]
	else:
	    index = int(self.tree[node,0])
	    if point[index] <= self.tree[node,1]:
		newNode = node + int(self.tree[node, 2])
	    else:
		newNode = node + int(self.tree[node, 3])
	    return self.pred(point, newNode)
    def query(self, points):
	result = []
	for i in points:
	    result.append(self.pred(i, 0))
	return result

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
