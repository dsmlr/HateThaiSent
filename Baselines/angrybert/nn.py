from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch.nn as nn
from nn_components import Word_Embedding, myLSTM, Gate_Attention, SimpleClassifier
import torch.nn.functional as F
import numpy as np

class AngryBERT(nn.Module):
    def __init__(self, embedding_dict_from_id):
        super(AngryBERT, self).__init__()
        
        self.hidden_dim_lstm = 128
        
        self.embedding_task1 = Word_Embedding(ntoken=len(embedding_dict_from_id), emb_dim=300)
        self.embedding_task2 = Word_Embedding(ntoken=len(embedding_dict_from_id), emb_dim=300)
        
        self.bert_model = BertForSequenceClassification.from_pretrained(
                'bert-base-multilingual-uncased',
                num_labels=2,
                output_attentions=False,
                output_hidden_states=True
            )
        
        self.proj = nn.Linear(768,self.hidden_dim_lstm*2)

        self.lstm_task1 = myLSTM(input_size=300, hidden_size=self.hidden_dim_lstm, num_layers=1)
        self.lstm_task2 = myLSTM(input_size=300, hidden_size=self.hidden_dim_lstm, num_layers=1)
        
        self.gate_task1 = Gate_Attention(self.hidden_dim_lstm*2, self.hidden_dim_lstm*2, self.hidden_dim_lstm*2)
        self.gate_task2 = Gate_Attention(self.hidden_dim_lstm*2, self.hidden_dim_lstm*2, self.hidden_dim_lstm*2)
        
        self.classifier_task1 = SimpleClassifier(in_dim=2*self.hidden_dim_lstm, hid_dim=256, out_dim=2, dropout=0.1)
        self.classifier_task2 = SimpleClassifier(in_dim=2*self.hidden_dim_lstm, hid_dim=256, out_dim=3, dropout=0.1)
        
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, nn_embed_token_ids, bert_input_ids, bert_attn_masks):
        
        task1 = self.embedding_task1(nn_embed_token_ids)
        task2 = self.embedding_task2(nn_embed_token_ids)
        
        bert_output = self.bert_model(input_ids=bert_input_ids, attention_mask=bert_attn_masks)
        bert_output = bert_output[1][-1][:,0,:]
        bert_output = self.proj(bert_output)
        bert_output = F.relu(bert_output)
        
        task1 = self.lstm_task1(task1)
        task2 = self.lstm_task2(task2)
        
        task1 = self.gate_task1(bert_output, task1)
        task2 = self.gate_task2(bert_output, task2)
        
        task1_logits = self.classifier_task1(task1)
        task2_logits = self.classifier_task2(task2)
        
        task1_logits = self.softmax(task1_logits)
        task2_logits = self.softmax(task2_logits)
        
        return task1_logits, task2_logits
    