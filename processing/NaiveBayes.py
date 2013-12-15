from __future__ import division
import numpy as np
import operator
from scipy import stats
import timeit as t

def main():

	nb = NaiveBayes()
	
	#TWITTER:
	features,all_targets,IDs = load_features_targets('../data/features.csv',
			'../data/targets.csv')
	#test,test_indeces = load_testdata('../test.csv')
	
	features, targets, IDs, testfeatures, testtargets, testIDs = splitdata(features, 
			all_targets, IDs)


	prediction_file = open('../predictions.csv', 'w')

	write_csv_row(prediction_file, ['id','s1','s2','s3','s4','s5','w1','w2','w3','w4',\
			'k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12',\
			'k13','k14','k15'])


	#run Naive Bayes for each target separately
	#(not 1 vs all because different targets are independent)
	num_targets = all_targets.shape[1]
	for targ_index in xrange(num_targets):
		targets = all_targets[targ_index]

		#Naive Bayes:
		nb.train(features, targets)
		predictions = nb.predict(testfeatures)
		#errors = evaluate(predictions, testtargets)
		#print 'target %d: %d/%d predictions wrong.' %(targ_index,errors, len(predictions))
    print predictions


	#nb.train(data, targets)
	#predictions = nb.predict(testdata)

	#errors = evaluate(predictions, testtargets)
	#print 'Housing Data: %d/%d predictions wrong.' %(errors, len(predictions))

	return 

def write_csv_row(f,row):
	for a in row[:-1]:
		f.write(str(a) + ',')
	f.write(str(row[-1]) + '\n')

def load_features_targets(feature_file, target_file):
	#determine number of columns in order to skip the first column
	features = np.loadtxt(feature_file, delimiter=',', skiprows=1)
	features = features[:,1:]	#first column is ID
	
	targets = np.loadtxt(target_file, delimiter=',', skiprows=1)
	targets = targets[:,1:]		#first column is ID

	IDs = features[:0]

	return features, targets, IDs


def evaluate(predictions, correct):
	errors = 0
	for i in range(0,len(predictions)):
		if(predictions[i] != correct[i]):
			errors += 1
	return errors

#split in training set & test set:
def splitdata(data, targets, IDs):
	cut = .7 * len(data)
	testdata = data[cut:]
	testtargets = targets[cut:]
	testIDs = IDs[cut:]
	data = data[:cut]
	targets = targets[:cut]
	IDs = IDs[:cut]
	
	return data, targets, IDs, testdata, testtargets, testIDs

	
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
