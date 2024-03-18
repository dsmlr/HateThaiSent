from pythainlp.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
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

def get_sentencewise_tokenized(dataset, max_text_len):
    tokenized_sent=[]
    
    for it in tqdm(range(len(dataset))):
        text = dataset.iloc[it,0]
        text = pre_processing(text)
        tokenized_text = word_tokenize(text, engine="newmm")
        if len(tokenized_text)>=max_text_len:
            tokenized_text=tokenized_text[:max_text_len]
        tokenized_text = ' '.join(tokenized_text)
        tokenized_sent.append(tokenized_text)
        
    return tokenized_sent


def get_one_hot_labels(dataset):
    le = LabelEncoder()
    hate_labels, sent_labels = [], []
    for idx in range(len(dataset)):
        hate_labels.append(dataset.hatespeech[idx])
        sent_labels.append(dataset.sentiment[idx])
    return to_categorical(np.asarray(le.fit_transform(hate_labels))), to_categorical(np.asarray(le.fit_transform(sent_labels)))

def get_word_embedding(model, word):
    try:
        embedding=model[word]
    except:
        embedding=np.zeros((300,))
    return embedding


def pad_embeddings(Word_Embeddings, max_length=128):
    dummy_embedding = np.zeros(768)  # Assuming the embedding dimension is 768
    padded_embeddings = []

    for embedding in Word_Embeddings:
        original_embedding = embedding.copy()
        
        if len(original_embedding) < max_length:
            padding_needed = max_length - len(original_embedding)
            original_embedding.extend([dummy_embedding] * padding_needed)

        if len(original_embedding) > max_length:
            original_embedding = original_embedding[:max_length]

        padded_embeddings.append(original_embedding)

    return np.array(padded_embeddings)

