from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras. preprocessing.sequence import pad_sequences

max_length_address = 25

# Tokenizing train_addresses and train_POI
def tokenize_and_pad(sentences_to_tokenize, data, oov_tok="<OOV>", trunc_type = 'post', padding = 'post'):
    tokenizer = Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences_to_tokenize)

    word_index = tokenizer.word_index
    print(len(word_index))
    total_words = len(word_index)+1

    data_sequences = tokenizer.texts_to_sequences(data)
    data_padded = pad_sequences(data_sequences, maxlen=max_length_address, truncating=trunc_type, padding=padding)
    return data_padded, word_index