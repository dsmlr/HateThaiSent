from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import one_hot
from tqdm import tqdm

def get_bert_input_ids_attn_masks(tokenized_sent, max_len):
    
    tokenized_= []
    for it in tokenized_sent:
        tokenized_.append(' '.join(itr for itr in it))

    texts = tokenized_
    input_ids = []
    attention_masks = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

    for text in tqdm(texts):
        encoded_dict = tokenizer.encode_plus(
                            text,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_len,           # Pad & truncate all sentences.
                            truncation = True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        
    return input_ids, attention_masks


def pad_and_tokens_to_ids(tokens, embedding_dict_from_id, token_to_id, max_seq_len):
    
    max_sequence_length = max_seq_len
    sentence_ids = []
    
    pad_index = len(embedding_dict_from_id)
    
    for token in tokens:
        if token in token_to_id.keys():
            sentence_ids.append(token_to_id[token])
                
    if len(sentence_ids) > max_sequence_length:
        sentence_ids = sentence_ids[:max_sequence_length]
    else:
        sentence_ids = sentence_ids + [pad_index] * (max_sequence_length - len(sentence_ids))
           
    # print(padded_token_ids)
    return sentence_ids
    
class Multitask_Dataset(Dataset):
    def __init__(self, nn_tokens, input_ids, attention_masks, labels_task1, labels_task2):
        self.nn_tokens = nn_tokens
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels_task1 = labels_task1
        self.labels_task2 = labels_task2

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        
        nn_tokens = self.nn_tokens[idx]
        
        input_ids = self.input_ids[idx]
        attention_masks = self.attention_masks[idx]
        
        label_task1 = torch.tensor(self.labels_task1[idx])
        label_task2 = torch.tensor(self.labels_task2[idx])
        
        label_task1 = one_hot(label_task1, num_classes=2)
        label_task2 = one_hot(label_task2, num_classes=3)
        
        return torch.tensor(nn_tokens), input_ids.squeeze(dim=0), attention_masks.squeeze(dim=0), label_task1.to(dtype = torch.float64), label_task2.to(dtype = torch.float64)