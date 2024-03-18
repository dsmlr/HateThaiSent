from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.callbacks import Callback
import numpy as np


def batch_roc_auc_score(y_true, y_pred):
    num_classes = y_true.shape[1]
    auc_scores = []
    
    for class_idx in range(num_classes):
        y_true_class = y_true[:, class_idx]
        y_pred_class = y_pred[:, class_idx]
        auc = roc_auc_score(y_true_class, y_pred_class)
        auc_scores.append(auc)
    
    return np.mean(auc_scores)


def calculate_custom_metrics(val_data, model, roc_auc_scores, classification_reports, acc_scores, f1_scores):
    
    data, labels = val_data
    
    hate_label = labels[0]
    sent_label = labels[1]
    
    hate_pred , sent_pred = model.predict(data)
    
    roc_auc_scores.append([batch_roc_auc_score(hate_label, hate_pred), batch_roc_auc_score(sent_label, sent_pred)])
    
    hate_pred = np.argmax(hate_pred, axis=1)
    sent_pred = np.argmax(sent_pred, axis=1)
    hate_label = np.argmax(hate_label, axis=1)
    sent_label = np.argmax(sent_label, axis=1)
    
    classification_reports.append([classification_report(hate_label, hate_pred), classification_report(sent_label, sent_pred)])
    acc_scores.append([accuracy_score(hate_label, hate_pred),accuracy_score(sent_label, sent_pred)])
    f1_scores.append([f1_score(hate_label, hate_pred, average="weighted"),f1_score(sent_label, sent_pred, average="weighted")])
    
    return roc_auc_scores, classification_reports, acc_scores, f1_scores
    
    
class CustomMetricsCallback(Callback):
    def __init__(self, validation_data):
        super(CustomMetricsCallback, self).__init__()
        #access these class attributes to get the test metrics
        self.validation_data = validation_data 
        self.roc_auc_scores = []
        self.classification_reports = []
        self.acc_scores = []
        self.f1_scores = []
        
    def on_epoch_end(self, epoch, logs=None):
        self.roc_auc_scores, self.classification_reports, self.acc_scores, self.f1_scores = calculate_custom_metrics(self.validation_data, self.model, self.roc_auc_scores, self.classification_reports, self.acc_scores, self.f1_scores)
        