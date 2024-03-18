from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.nn.functional as F
import warnings
import os
import re
from tqdm import tqdm
import torch
import math
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from helper_functions import get_sentencewise_tokenized, get_ft_embedding_dicts, get_numerical_labels
from data_utils import pad_and_tokens_to_ids, get_bert_input_ids_attn_masks, Multitask_Dataset
from nn import SharedEncoderBERT
import torch.nn as nn
from metrics import evaluate_model_sklearn_with_accuracy_and_reports


dataset_path = "dataset.csv"
max_len_ft = 509
actual_max_seq_len = 256
batch_size = 64
num_epochs = 40
device = torch.device("cuda:0")


dataset = pd.read_csv(dataset_path)
dataset.drop(dataset.columns[dataset.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
dataset.iloc[:,0:1].fillna('other', inplace=True)


tokenized_sent = get_sentencewise_tokenized(dataset, max_len_ft)

bert_input_ids, bert_attn_masks = get_bert_input_ids_attn_masks(tokenized_sent, actual_max_seq_len)

hate_labels , sent_labels = get_numerical_labels(dataset)

input_ids_train, input_ids_test, attention_masks_train, attention_masks_test, hate_label_train, hate_label_test, sent_label_train, sent_label_test= train_test_split(bert_input_ids, bert_attn_masks , hate_labels, sent_labels, test_size = 0.20, random_state = 42)

train_ds = Multitask_Dataset(input_ids=input_ids_train, attention_masks=attention_masks_train, labels_task1=hate_label_train, labels_task2=sent_label_train)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_ds = Multitask_Dataset(input_ids=input_ids_test, attention_masks=attention_masks_test, labels_task1=hate_label_test, labels_task2=sent_label_test)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

model = SharedEncoderBERT(num_labels_task1=2, num_labels_task2=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()


task1_accuracies = []
task2_accuracies = []
task1_f1s = []
task2_f1s = []
task1_reports = []
task2_reports = []
task1_auc_rocs = []
task2_auc_rocs = []


for epoch in range(num_epochs):
    
    model.train()
    
    for batch in tqdm(train_loader):
        batch_input_ids, batch_attention_masks, batch_labels_task1, batch_labels_task2 = batch
        
       
        optimizer.zero_grad()
        logits_task1, logits_task2 = model(input_ids_in = batch_input_ids.to(device), attention_mask_in = batch_attention_masks.to(device))

        loss_task1 = criterion(logits_task1, batch_labels_task1.to(device))
        loss_task2 = criterion(logits_task2, batch_labels_task2.to(device))
        total_loss = loss_task1 + loss_task2
        
        total_loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss_task2.item()}")
    
    # Evaluate the model and store accuracy values and classification reports
    task1_auc_roc, task2_auc_roc, task1_accuracy, task2_accuracy, task1_f1, task2_f1, report_task1, report_task2 = evaluate_model_sklearn_with_accuracy_and_reports(model, test_loader, device)

    task1_auc_rocs.append(task1_auc_roc)
    task2_auc_rocs.append(task2_auc_roc)
    task1_accuracies.append(task1_accuracy)
    task2_accuracies.append(task2_accuracy)
    task1_f1s.append(task1_f1)
    task2_f1s.append(task2_f1)
    task1_reports.append(report_task1)
    task2_reports.append(report_task2)

    # Print accuracies for the current epoch
    print(f"Epoch {epoch+1} - Task 1 F1: {task1_f1:.4f}")
    print(f"Epoch {epoch+1} - Task 2 F1: {task2_f1:.4f}")