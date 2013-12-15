from __future__ import division
import numpy as np
import operator
from scipy import stats
import timeit as t

def main():

	nb = NaiveBayes()
	
	#TWITTER:
	data,targets = load_features_targets('../data/features.csv', '../data/targets.csv')



	#HOUSING DATA:
	#data,targets = loadHousingData('RealEstate.csv')

	#data, targets, testdata, testtargets = splitdata(data, targets)

	#nb.train(data, targets)
	#predictions = nb.predict(testdata)

	#errors = evaluate(predictions, testtargets)
	#print 'Housing Data: %d/%d predictions wrong.' %(errors, len(predictions))

	return 


def evaluate(predictions, correct):
	errors = 0
	for i in range(0,len(predictions)):
		if(predictions[i] != correct[i]):
			errors += 1
	return errors

#split in training set & test set:
def splitdata(data, targets):
	cut = .7 * len(data)
	testdata = data[cut:]
	testtargets = targets[cut:]
	data = data[:cut]
	targets = targets[:cut]
	
	return data, targets, testdata, testtargets

#this loads the housing data, extracts data and targets etc.
def loadHousingData(filename):
	dataset = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(2,3,4,5))
	np.random.shuffle(dataset)

	#extract target:
	num_targets = dataset[:,0]
	num_targets = num_targets.reshape((dataset.shape[0],1))

	#binarize the target values:
	targets = list()
	for y in num_targets:
		if y <= 295000: targets.append(1)
		else: targets.append(0)
	targets = np.array(targets)
	targets = targets.reshape((targets.shape[0], 1))


	# extract data:
	data = np.array(dataset[:,1:])

	return data, targets

def load_features_targets(feature_file, target_file):
	#determine number of columns in order to skip the first column
	with open(feature_file) as f:
		columns = int(f.readline())
		f.close()
	use_columns = range(1,columns)
	print use_columns
	dataset = np.loadtxt(feature_file, delimiter=',', skiprows=2, usecols=(use_columns))

	print dataset[:5]
	
	targets = []

	return dataset, targets


	
class NaiveBayes(object):

	def __init__(self):
		self.targetprob = {}
		self.targets = []
		self.means = {}
		self.sds = {}

	def train(self, data, targets):
		'''data and targets are assumed to be np.arrays
		'''

		# make a list of unique target-values: (they need to be in a certain order in order because argmax just retrieves an index, that's why no set)
		targets = [t[0] for t in targets]	#extract values from lists in numpy array
		self.targets = list(set(targets))

		#calculate all frequencies of target-values:
		self.targetprob = {}
		tfreq = {}
		for v in targets:
			tfreq.setdefault(v, 0)
			tfreq[v] += 1
		total = len(targets)
		for v,freq in tfreq.items():
			self.targetprob[v] = freq/total

		#calculate means and standard deviations of attribute-values:
		all_attr_values = {}
		for v in set(targets):
			col_attr_values = []
			for col in range(0, data.shape[1]):
				attr_values = []
				for row in range(0, data.shape[0]):
					if(targets[row] == v): attr_values.append(data[row][col])
				col_attr_values.append(attr_values)
			all_attr_values[v] = col_attr_values

		self.means = {}
		for v,attr in all_attr_values.items():
			self.means[v] = [np.mean(a) for a in attr]
		
		#calculate standard deviations:
		self.sds = {}
		for v,attr in all_attr_values.items():
			self.sds[v] = [np.std(a) for a in attr]
		#change standard deviations that are zero to something near to zero (otherwise divide by 0 error):
		for v,sds in self.sds.items():
			new_sds = []
			for sd in sds:
				if(sd==0):
					new_sds.append(0.0001)
				else:
					new_sds.append(sd)
			self.sds[v] = new_sds
		

	def predict(self, data):
		predictions = []
		i = 0
		for datapoint in data:
			#print 'datapoint: %s' % str(datapoint)
			probabilities = []
			for v in self.targets:	
				#print 'target-value: %s' % str(v)
				p_v = self.targetprob[v]
				attributes = [attr for attr in datapoint]
				gaussians = stats.norm.pdf(attributes, self.means[v], self.sds[v])
				
				#/stats.norm.pdf(self.means[v], self.means[v], self.sds[v])
				#print self.means[v]
				#print self.sds[v]
				#print [attr for attr in datapoint]
				#print 'attr=%s, means=%s, sds=%s --> %s' %(str(attributes), str(self.means[v]), str(self.sds[v]), str(gaussians))
				
				product_p_a = reduce(operator.mul, gaussians, 1)
				#print 'p_v * product_p_a = p;    %s * %s = %s' %(str(p_v), str(product_p_a), str(p_v * product_p_a))
				probabilities.append(p_v * product_p_a)
			
			#if(i==0): 
			#	print datapoint
			#	print probabilities
			#i += 1
			
			pred = self.targets[np.argmax(probabilities)]
			predictions.append(pred)
			#print 'prediction: %s \n' %str(pred)

		return predictions


if __name__ == '__main__':
	main()
