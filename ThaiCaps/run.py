import pandas as pd
import numpy as np
from argparse import Namespace 
from helper_functions import get_sentencewise_tokenized, get_one_hot_labels, pad_embeddings, get_word_embedding
from components import AttLayer, CapsuleLayer, PrimaryCap, Length
from metrics import CustomMetricsCallback
from sklearn.preprocessing import LabelEncoder
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPool1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, concatenate, add
from keras.models import Model
from sklearn.model_selection import train_test_split


config = Namespace(max_token_len = 509,
                   max_seq_length_fasttext = 128,
                   dataset_path = "dataset.csv",
                   bert_embeddings_path ="path_to_bert_embeddings",
                   fasttext_full_embeddings_path = "path_to_fasttext_model",
                   fasttext_embeddings_path = 'path_to_fasttext_embeddings',
                   fasttext_embedding_dimension = 300,
                   num_words_tokenizer = 50000,
                   batch_size = 32,
                   epochs = 20,
                   )

dataset=pd.read_csv(config.dataset_path)
preprocessed_data = get_sentencewise_tokenized(dataset, config.max_token_len)

hate_labels, sent_labels = get_one_hot_labels(dataset)

bert_embeddings = np.load(config.bert_embeddings_path, allow_pickle=True)

tokenizer = Tokenizer(num_words=config.num_words_tokenizer)
tokenizer.fit_on_texts(preprocessed_data)
input_ids_fasttext = tokenizer.texts_to_sequences(preprocessed_data)

vocab_size = len(tokenizer.word_index) + 1
input_ids_fasttext_padded = pad_sequences(input_ids_fasttext, padding='post', maxlen=config.max_seq_length_fasttext)

# fasttext_model = FastText.load_fasttext_format(config.fasttext_full_embeddings_path)
# embedding_matrix = np.zeros((vocab_size, 300))
# for word, index in tokenizer.word_index.items():
#     embedding_vector = get_word_embedding(word)
#     if embedding_vector is not None:
#         embedding_matrix[index] = embedding_vector

# fasttext_embedding_matrix = np.array(embedding_matrix)

fasttext_embedding_matrix = np.load(config.fasttext_embeddings_path, allow_pickle=True)

fasttext_embedding_layer = Embedding(vocab_size, config.fasttext_embedding_dimension, weights=[fasttext_embedding_matrix],
                        input_length=config.max_seq_length_fasttext,
                        trainable=False,
                        mask_zero=True)

vecmap_input = Input(shape=(config.max_seq_length_fasttext,), dtype='int32')
fasttext_embedded_sequences = fasttext_embedding_layer(vecmap_input)
fasttext_embed_model = Model(inputs = [vecmap_input] , outputs = [fasttext_embedded_sequences])
fasttext_embeddings = fasttext_embed_model(input_ids_fasttext_padded)
fasttext_embeddings = np.array(fasttext_embeddings)

fasttext_X_train, fasttext_X_test, bert_embeddings_X_train, bert_embeddings_X_test, hate_y_train, hate_y_test, sent_y_train, sent_y_test = train_test_split(fasttext_embeddings, bert_embeddings, hate_labels, sent_labels, test_size = 0.20, random_state = 0)

#making train and test as multiples of batch_size
fasttext_X_train = np.vstack([fasttext_X_train, fasttext_X_train[-3:, :, :]])
bert_embeddings_X_train = np.vstack([bert_embeddings_X_train, bert_embeddings_X_train[-3:, :, :]])
hate_y_train = np.vstack([hate_y_train, hate_y_train[-3:, :]])
sent_y_train = np.vstack([sent_y_train, sent_y_train[-3:, :]])

fasttext_X_test = np.vstack([fasttext_X_test, fasttext_X_test[-16:, :, :]])
bert_embeddings_X_test = np.vstack([bert_embeddings_X_test, bert_embeddings_X_test[-16:, :, :]])
hate_y_test = np.vstack([hate_y_test, hate_y_test[-16:, :]])
sent_y_test = np.vstack([sent_y_test, sent_y_test[-16:, :]])

custom_metrics_callback = CustomMetricsCallback(validation_data=([fasttext_X_test, bert_embeddings_X_test], [hate_y_test, sent_y_test]))


#fasttext
l_gru = Bidirectional(GRU(256, return_sequences=True))(fasttext_embedded_sequences)
l_att = AttLayer(100)(l_gru)
l_att_sent1 = Dropout(0.25)(l_att)

#bert
bert_input = Input(shape=(config.max_seq_length_fasttext,768), batch_size=config.batch_size)
conv1 = Conv1D(filters=64, kernel_size=2, strides=1, padding='valid', activation='relu', name='conv1')(bert_input)
primarycaps = PrimaryCap(conv1, dim_vector=16, n_channels=8, kernel_size=2, strides=2, padding='valid')
digitcaps = CapsuleLayer(num_capsule=2, dim_vector=16, num_routing=3, name='digitcaps')(primarycaps)
# out_caps = Length(name='out_caps')(digitcaps)
out_caps = Flatten()(digitcaps)
l_att_sent2 = Dropout(0.25)(out_caps)

#sentiment head
out_sent = Dense(100, activation='relu')(concatenate([l_att_sent1, l_att_sent2], axis=-1))
out_sent = Dropout(0.25)(out_sent)
out_sent1 = Dense(60, activation='relu')(out_sent)
out_sent = Dropout(0.25)(out_sent)
out_sent = Dense(3, activation='softmax', name='out_sent')(out_sent)

#offensive head
out_bully = Dense(100, activation='relu')(concatenate([l_att_sent1, l_att_sent2], axis=-1))
out_bully = Dropout(0.25)(out_bully)
out_bully = Dense(60, activation='relu')(out_bully)
out_bully = Dropout(0.25)(out_bully)
out_bully = Dense(2, activation='softmax', name='out_bully')(concatenate([out_bully, out_sent1]))



thaicaps_mt = Model(inputs=[fasttext_embedded_sequences, bert_input], outputs=[out_bully, out_sent])



thaicaps_mt.compile(optimizer='adam',
                       loss={'out_bully': 'categorical_crossentropy', 'out_sent': 'categorical_crossentropy'},
                       loss_weights=[1, 1],
                       metrics={'out_bully': 'categorical_accuracy', 'out_sent': 'categorical_accuracy'})



thaicaps_mt.fit([fasttext_X_train, bert_embeddings_X_train], [hate_y_train, sent_y_train],
                   validation_data=([fasttext_X_test, bert_embeddings_X_test], [hate_y_test, sent_y_test]), 
                   callbacks=[custom_metrics_callback], 
                   batch_size=config.batch_size, epochs=config.epochs, shuffle=True, verbose=1)

