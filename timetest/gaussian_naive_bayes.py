'''
Created on 22.11.2013

@author: Nick
'''

import math
import numpy as np
from gaussian import gaussian

class GaussianNaiveBayes(object):

    def __init__(self):
        self.class_count = {}
        self.classes = {}
        self.classes_meanvar = {}
        
    def __increase_class_count(self, current_class):
        if self.classes.has_key(current_class):
            model = self.classes[current_class]
            self.class_count[current_class] += 1
        else:
            model = {}
            self.class_count[current_class]  = 1
        return model
    
    def train(self, data, targets):
        index = 0
        max_index = len(data)
        
        while index < max_index:
            current_class = targets[index][0]
            vals = data[index]
            att = 0
            att_vals = []
            
            for val in vals:
                att_vals.append((att, val))
                att += 1
                
            model = self.__increase_class_count(current_class)
                
            for att_val in att_vals:
                attribute = att_val[0]
                value = att_val[1]
                if model.has_key(attribute):
                    attributes  = model[attribute]
                    attributes.append(value)
                    model[attribute] = attributes
                else:
                    attributes = []
                    attributes.append(value)
                    model[attribute] = attributes
                    
            index += 1
                              
            self.classes[current_class] = model
        
        for clss in self.classes.keys():
            class_model = self.classes[clss]
            class_meanvar = {}
            
            for attribute in class_model.keys():
                mean = np.array(class_model[attribute]).mean()
                std  = np.array(class_model[attribute]).std()
                std = max(std, 0.1)
                class_meanvar[attribute] =  (mean, std)
                
            self.classes_meanvar[clss] = class_meanvar
        
    def test(self, test_data):
        
        predictions = []
        
        for vals in test_data:
            att = 0
            att_vals = []
            
            for val in vals:
                att_vals.append((att, val))
                att += 1
                
            Pmodel = {}
        
            for clss in self.classes_meanvar.keys():
                Pclss = 0.0
                class_meanvar = self.classes_meanvar[clss]
            
                for attribute_value in att_vals:
                    attribute = attribute_value[0]
                    value = attribute_value[1]
                
                    if class_meanvar.has_key(attribute):
                        att_mean = class_meanvar[attribute][0]
                        att_std = class_meanvar[attribute][1]
                        g = gaussian(value, att_mean, att_std)
                        if g != 0:
                            Pclss = Pclss +  math.log10(g)
                        else:
                            Pclss = 0
               
                Pmodel[Pclss] = clss
                
            probs = Pmodel.keys()
            probs.sort()
            Pmax = probs[-1]
            best = Pmodel[Pmax]
            predictions.append(best)
            
        predictions = np.array(predictions)
        predictions = predictions.reshape((predictions.size, 1))
            
        return predictions
