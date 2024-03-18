from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.nn as nn

class SharedEncoderBERT(nn.Module):
    def __init__(self, num_labels_task1, num_labels_task2):
        super(SharedEncoderBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        
        self.dropout = nn.Dropout(0.1)
        
        self.classifier_task1 = nn.Linear(768, num_labels_task1)
        self.classifier_task2 = nn.Linear(768, num_labels_task2)

    def forward(self, input_ids_in, attention_mask_in):
        outputs = self.bert(input_ids=input_ids_in, attention_mask=attention_mask_in)
        
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        pooled_output = self.dropout(pooled_output)
        
        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)
        
        return logits_task1, logits_task2
