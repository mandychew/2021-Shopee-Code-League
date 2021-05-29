#!pip install -q tensorflow_text_nightly
!pip install -q tf-nightly
!pip install tokenizers -q

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from tokenizers import BertWordPieceTokenizer

# Initialize an empty BERT tokenizer
tokenizer = BertWordPieceTokenizer(
  clean_text=False,
  handle_chinese_chars=False,
  strip_accents=False,
  lowercase=True,
)

# prepare text files to train vocab on them
test_string = ['/content/test.txt']
#[sentences_to_tokenize]
#

# train BERT tokenizer
tokenizer.train(
  test_string,
  vocab_size=20,
  min_frequency=2,
  show_progress=True,
  special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
  limit_alphabet=1000,
  wordpieces_prefix="##"
)

# save the vocab
tokenizer.save('/content/bert-vocab.txt')

# create a BERT tokenizer with trained vocab
vocab = '/content/bert-vocab.txt'
tokenizer = BertWordPieceTokenizer(vocab)

# test the tokenizer with some text
encoded = tokenizer.encode('...')
print(encoded.tokens)

#bert_tokenizer_params=dict(lower_case=True)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size = 97192,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=dict(lower_case=True),
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)

vocab = bert_vocab.bert_vocab_from_dataset(
    sentences_to_tokenize, 
    **bert_vocab_args
)

# Tokenize corpus (all words AI should learn) without vocab_size limit 

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences_to_tokenize)

word_index = tokenizer.word_index
print(len(word_index))
