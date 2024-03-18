from pythainlp.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import fasttext
import numpy as np
import re
from tqdm import tqdm


def pre_processing(text):
    text = re.sub('http://\S+|https://\S+', '', text)
    text = re.sub('[!/,*_.@:#($)?-]', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[A-Za-z]+', '', text)
    text= re.sub("\s\s+" , " ", text)
    text = text.replace(" ", "")
    return text

def get_numerical_labels(dataset):
    le = LabelEncoder()
    hate_labels, sent_labels = [], []
    for idx in range(len(dataset)):
        hate_labels.append(dataset.hatespeech[idx])
        sent_labels.append(dataset.sentiment[idx])
    return np.asarray(le.fit_transform(hate_labels)), np.asarray(le.fit_transform(sent_labels))

def get_sentencewise_tokenized(dataset, max_text_len):
    tokenized_sent=[]
    
    for it in tqdm(range(len(dataset))):
        text = dataset.iloc[it,0]
        text = pre_processing(text)
        tokenized_text = word_tokenize(text, engine="newmm")
        if len(tokenized_text)>=max_text_len:
            tokenized_text=tokenized_text[:max_text_len]
        tokenized_sent.append(tokenized_text)
        
    return tokenized_sent

def get_ft_embedding_dicts(tokenized_sent,  fasttext_model_path):
    
    ft = fasttext.load_model(fasttext_model_path)
    tokens_flattened = [token for sentence in tokenized_sent for token in sentence]
    token_freq = dict(Counter(tokens_flattened))
    sorted_token_freq = {k: v for k, v in sorted(token_freq.items(), key=lambda item: item[1], reverse=True)}
    sorted_token_freq_trunc = {}
    for ind, i in enumerate(sorted_token_freq.items()):
        if(i[1] >= 2):
            sorted_token_freq_trunc[i[0]] = i[1]
            
    embedding_dict_from_token =  {}
    embedding_dict_from_id = {}
    for ind, token in tqdm(enumerate(sorted_token_freq_trunc.keys())):
            embedding = ft.get_word_vector(token)
            embedding_dict_from_token[token] = embedding
            embedding_dict_from_id[ind] = embedding
    return embedding_dict_from_id, embedding_dict_from_token



