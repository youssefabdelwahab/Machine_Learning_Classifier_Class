import math 
import numpy as np 

class artificial_neural_network(self): 
    
def __init__(self): 
    self.w1 = 1
    self.w2 = 1 
    self.bias = 0

def binary_crossentropy(self , y_test , y_predicted): 
        epsilon = 1e-18
        y_predicted_adj = [max(i,epsilon) for i in y_predicted]
        y_predicted_adj = [min(i,1-epsilon) for i in y_predicted_new]
        y_predicted_adj = np.array(y_predicted_adj)
        return -np.mean(y_test * np.log(y_predicted_adj) + (1-y_test) * np.log(1-y_predicted_adj) )
    
def sigmoid(self , X): 
    return 1/(1 + np.exp(-X))

        
def gradient_descent(self , feature_1 , feature_2, epochs , learning_rate ):
    w1 , w2 = 1
    bias = 0 
    rate = learning_rate
    n = len(feature_1)

    for i in range(epochs): 
        sum_weights  = w1*feature_1 + w2*feature_2 + bias 
        y_predicted = sigmoid(sum_weights)

        loss = binary_crossentropy(y_test, y_predicted)

        w1d = (1/n)(np.dot(np.transpose(feature_1), (y_predicted - y_test)))
        w2d = (1/n)(np.dot(np.transpose(feature_2), (y_predicted - y_test)))
        bias_d = np.mean(y_predicted - y_test)

        w1 = w1 - (rate(w1d))
        w2 = w2 - (rate(w2d))
        bias = bias - (rate(bias_d))

        print(f"Epochs:{i} , loss:{loss}")

        return w1 , w2 , bias 
    

def fit(self , X, y , epochs , loss_threshold): 
    self.w1 , self.w2 , self.bias = self.gradient_descent(feature_1 , feature_2 , y , epochs , learning_rate)


def predict(self ,X): 
    weighted_sum = self.w1 * feature_1 + self.w2 * feature_2 + self.bias
    return sigmoid(weighted_sum)
        
    








   
    

