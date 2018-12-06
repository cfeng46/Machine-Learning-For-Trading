import numpy as np

class BagLearner(object):
    
    def __init__(self, learner, bags=20, kwargs=None, boost=False, verbose=False):
	self.learner = learner
	self.bags = bags
	self.kwargs = kwargs
	self.learners = []
	for k in range(0,self.bags):
	    self.learners.append(self.learner(**self.kwargs))

    def author(self):
	return "cfeng46"

    def addEvidence(self, dataX, dataY):
	self.Xtrain = []
        self.Ytrain = []
        for i in range(0,self.bags):
            data_size = len(dataY)
            index = np.random.randint(len(dataX),size=data_size)
            self.Xtrain.append(dataX[index,:])
            self.Ytrain.append(dataY[index])


    def query(self, points):
	result = np.zeros(len(points))
	final = []
	for i in range(len(self.learners)):
            self.learners[i].addEvidence(self.Xtrain[i], self.Ytrain[i])
            predY = self.learners[i].query(points)
            final.append(predY)
        result = np.mean(final, axis = 0)
        return result
