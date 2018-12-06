import BagLearner as bl
import LinRegLearner as lrl
import numpy as np
class InsaneLearner(object):
    def __init__(self, verbose = False):
	pass
    def author(self):
	return "cfeng46"
    def addEvidence(self, dataX, dataY):
	self.dataX = dataX
	self.dataY = dataY
    def query(self, points):
	learner = bl.BagLearner(learner = bl.BagLearner, kwargs = {'learner':lrl.LinRegLearner,'bags':20, 'kwargs':{}}, bags = 20, boost = False, verbose = False)
	learner.addEvidence(self.dataX, self.dataY)
	Y = learner.query(points)
	return Y
