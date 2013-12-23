from __future__ import division
import numpy as np
import operator
from scipy import stats

class NaiveBayes(object):

    def __init__(self):
        self.targetprob = {}
        self.targets = []
        self.means = {}
        self.sds = {}

    def train(self, data, targets):
        '''data and targets are assumed to be np.arrays
        '''

        # make a list of unique target-values: (they need to be in a
        # certain order in order because argmax just retrieves an index,
        # that's why no set)

        self.targets = list(set(targets))
#         print 'all_targets: ' + str(self.targets)

        # calculate all frequencies of target-values:
        self.targetprob = {}
        tfreq = {}
        for v in targets:
            tfreq.setdefault(v, 0)
            tfreq[v] += 1
        total = len(targets)
        for v, freq in tfreq.items():
            self.targetprob[v] = freq / total

        # calculate means and standard deviations of attribute-values:
        all_attr_values = {}
        for v in set(targets):
            col_attr_values = []
            for col in range(0, data.shape[1]):
                attr_values = []
                for row in range(0, data.shape[0]):
                    if(targets[row] == v): attr_values.append(data[row][col])
                col_attr_values.append(attr_values)
            all_attr_values[v] = col_attr_values

#         print 'all_attr_values: ' + str(all_attr_values)
#         self.means = {}
        for v, attr in all_attr_values.items():
            self.means[v] = [np.mean(a) for a in attr]

        # calculate standard deviations:
        self.sds = {}
        for v, attr in all_attr_values.items():
            self.sds[v] = [np.std(a) for a in attr]
        # change standard deviations that are zero to something near to zero (otherwise divide by 0 error):
        for v, sds in self.sds.items():
            new_sds = []
            for sd in sds:
                new_sds.append(max(sd, 0.1))
            self.sds[v] = new_sds


    def predict(self, data):
        predictions = []
#         i = 0
        for datapoint in data:
#             print 'datapoint: %s' % str(datapoint)
            probabilities = []
            for v in self.targets:
#                 print 'target-value: %s' % str(v)
                p_v = self.targetprob[v]
                attributes = [attr for attr in datapoint]
                gaussians = stats.norm.pdf(attributes, self.means[v], self.sds[v])

#                 /stats.norm.pdf(self.means[v], self.means[v], self.sds[v])
#                 print self.means[v]
#                 print self.sds[v]
#                 print [attr for attr in datapoint]
#                 print 'attr=%s, means=%s, sds=%s --> %s' %(str(attributes), str(self.means[v]), str(self.sds[v]), str(gaussians))

                product_p_a = reduce(operator.mul, gaussians, 1)
#                 print 'p_v * product_p_a = p;    %s * %s = %s' %(str(p_v), str(product_p_a), str(p_v * product_p_a))
                probabilities.append(p_v * product_p_a)

#             if(i==0):
#                print datapoint
#                print probabilities
#             i += 1

            pred = self.targets[np.argmax(probabilities)]
            predictions.append(pred)
#             print 'prediction: %s \n' %str(pred)

        return predictions
