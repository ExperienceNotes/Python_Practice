import numpy as np
from scipy.sparse import data
from sklearn.metrics import precision_score,recall_score,fbeta_score
y_pred = np.random.randint(2,size = 100)
y_true = np.random.randint(2,size = 100)

def custom_fbeta_socre(y_pred,y_true,beta = 1):
    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    fbeta = (1+(beta)**2) * (precision*recall)/(beta**2*precision + recall)
    return fbeta
print(custom_fbeta_socre(y_pred,y_true,beta=2))
print(fbeta_score(y_true, y_pred, beta=2))