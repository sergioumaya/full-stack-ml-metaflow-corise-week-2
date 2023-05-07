
# TODO: build the majority class baseline model. 
# TODO: find the majority class in the labels. 🤔
# TODO: score the model on valdf with a 2D metric space: sklearn.metrics.accuracy_score, sklearn.metrics.roc_auc_score
    # Documentation on suggested model-scoring approach: https://scikit-learn.org/stable/modules/model_evaluation.html

from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

class MajorityClassifier():
    def __init__(self):
        self.majority_class = None
    
    def fit(self, y_train):
        self.majority_class = self.find_majority_class(y_train)
        return self
    
    def predict(self, X):
        return np.array([self.majority_class] * len(X))

    def eval_acc(self,X,y):
         return accuracy_score(y, self.predict(X))
    
    def eval_auc(self,X,y):
        return roc_auc_score(y, self.predict(X))
    
    def find_majority_class(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]
