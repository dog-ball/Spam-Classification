import pandas as pd
import numpy as np
import pickle

"""To train the model from scratch see comment in the create_classifier() function."""


def get_kernel_matrix(x1, x2, sigma):
    """Function for the gram matrix used in the Kernel trick"""
    num_samples_x1 = x1.shape[0]
    num_samples_x2 = x2.shape[0]
    kernel_matrix = np.zeros([num_samples_x1, num_samples_x2])
    for nrow in range(num_samples_x1):
        for ncol in range(num_samples_x2):
            kernel_matrix[nrow][ncol] = np.exp(- sigma*(np.dot(x1[nrow]-x2[ncol], x1[nrow]-x2[ncol])))
    return kernel_matrix


class SVMSpamClassifier:
    """Support Vector Machine Classifier for spam email"""
    def __init__(self, sigma, C, train=False):
        self.sigma = sigma
        self.C = C
        self.save_weights = train
        # Load presaved weights if train set to False
        if train is False:
            self.support_x, self.support_y, self.support_lamb, self.param_b = self.load_data()
        else:
            self.support_ind = None
            self.support_x = None
            self.support_y = None
            self.support_lamb = None
            self.lamb = None
            self.param_b = None
            self.count = 0
            self.objective_func = -np.Inf
        
    
    def predict(self, x):
        """Function for classifying unseen data"""
        K = get_kernel_matrix(self.support_x, x, self.sigma)
        pred_y = []
        for i in range(x.shape[0]):
            z = np.dot(self.support_lamb * self.support_y, K[:,i]) +  self.param_b  
            if z < 0:
                z = -1
            else:
                z = 1
            pred_y.append(z)
        if self.train is True:
            return np.array(pred_y)
        else:
            return np.array([x if x == 1 else 0 for x in pred_y])
        
        
    def train(self, x, y, max_iter=1E6, epsilon=1E-4):
        """Function for training the SVM"""
        num_samples = x.shape[0]
        """Initialization and solve dual problem with SMO"""
        K = get_kernel_matrix(x, x, self.sigma)
        C = self.C
        self.lamb = np.zeros(num_samples)
        self.param_b = np.random.normal()
        #SMO algorithm
        while True:
            # Randomly select a pair (a, b) to optimize
            [a, b] = np.random.choice(num_samples, 2, replace=False)
            if K[a, a] + K[b, b] - 2 * K[a,b] == 0:
                continue
                
            lamb_a_old, lamb_b_old = self.lamb[a], self.lamb[b]
            Ea =  np.dot(self.lamb * y, K[:,a]) + self.param_b - y[a]
            Eb =  np.dot(self.lamb * y, K[:,b]) + self.param_b - y[b]
            lamb_a_new_unclip = lamb_a_old + y[a] *(Eb - Ea)/(K[a,a] + K[b,b] - 2 * K[a, b])
            xi = - lamb_a_old * y[a] - lamb_b_old * y[b]
            
            if y[a] != y[b]:
                L = max(xi * y[b], 0)
                H = min(C + xi * y[b], C)
            else:
                L = max(0, -C - xi * y[b])
                H = min(C, -xi * y[b])
                
            if lamb_a_new_unclip < L:
                lamb_a_new = L
            elif lamb_a_new_unclip > H:
                lamb_a_new = H
            else:
                lamb_a_new = lamb_a_new_unclip
                
            lamb_b_new = lamb_b_old + (lamb_a_old - lamb_a_new) * y[a] * y[b]
            if lamb_a_new > 0 and lamb_a_new < C:
                self.param_b =  self.param_b - Ea + (lamb_a_old - lamb_a_new) * y[a] * K[a,a] + (lamb_b_old - lamb_b_new) * y[b] * K[b,a]
            elif lamb_b_new > 0 and lamb_b_new < C:
                self.param_b = self.param_b - Eb + (lamb_a_old - lamb_a_new) * y[a] * K[a,b] + (lamb_b_old - lamb_b_new) * y[b] * K[b,b]
            self.lamb[a], self.lamb[b] = lamb_a_new, lamb_b_new
            
            self.count += 1
            local_count += 1

            # Determine whether to stop training
            if local_count >= max_iter or self.count % 10000 ==0:
                self.support_ind =  self.lamb > 0
                self.support_x = x[self.support_ind]
                self.support_y = y[self.support_ind]
                self.support_lamb = self.lamb[self.support_ind]	
                support_K = K[self.support_ind, :][:, self.support_ind]
                new_objective_func = np.sum(self.support_lamb) - 0.5 * np.dot(np.matmul((self.support_lamb * self.support_y).T, support_K).T, self.support_lamb * self.support_y) 
                
                # If the change of dual objective function is less than epsilon stop training
                if abs(new_objective_func - self.objective_func) <= epsilon:
                    break
                else:
                    self.objective_func = new_objective_func
                if local_count >= max_iter:
                    break
                    
        # If True will save weights to be used again            
        if self.save_weights:
            self.save_data()
    

    def save_data(self):
        """Function for saving data to be used later"""
        with open('support_x.pkl', 'wb') as f:
            pickle.dump(self.support_x, f)  
        with open('support_y.pkl', 'wb') as f:
            pickle.dump(self.support_y, f)
        with open('support_lamb.pkl', 'wb') as f:
            pickle.dump(self.support_lamb, f)
        with open('param_b.pkl', 'wb') as f:
            pickle.dump(self.param_b, f)
            
            
    def load_data(self):
        """Function for loading data when initialising classifier"""
        with open('support_x.pkl', 'rb') as f:
            support_x = pickle.load(f)
        with open('support_y.pkl', 'rb') as f:
            support_y = pickle.load(f)
        with open('support_lamb.pkl', 'rb') as f:
            support_lamb = pickle.load(f)
        with open('param_b.pkl', 'rb') as f:
            param_b = pickle.load(f)
        return support_x, support_y, support_lamb, param_b 
    
    
def create_classifier():
    """To train the model from scratch set train to True"""
    classifier = SVMSpamClassifier(0.01, 100, train=False)
    return classifier

classifier = create_classifier()
