from nltk.tokenize import word_tokenize

# !pip install inflect

import pandas as pd
import numpy as np
import spacy
import inflect
from collections import defaultdict

nlp = spacy.load('en_core_web_sm')
p = inflect.engine()

train_csv = '/kaggle/input/dl-project/isp-match-dl-2024_v2/train.csv'
val_csv = '/kaggle/input/dl-project/isp-match-dl-2024_v2/val.csv'
test_csv = '/kaggle/input/dl-project/isp-match-dl-2024_v2/test.csv'


train_data = pd.read_csv(train_csv)
val_data = pd.read_csv(val_csv)
test_data = pd.read_csv(test_csv)


# function to convert numbers to words
def convert_numbers_to_text(text):
    words = text.split()
    for i, word in enumerate(words):
        if word.isdigit():  # check if the word is a number
            words[i] = p.number_to_words(word)  # convert number to words
    return ' '.join(words)

def preprocess_text(text):
    text = convert_numbers_to_text(text)
    doc = nlp(text.lower())
    # remove stop words and nonalphabetic tokens, lemmatize
    cleaned_tokens = [
        token.lemma_ for token in doc if token.is_alpha and not token.is_stop
    ]
    return ' '.join(cleaned_tokens)

train_data['caption'] = train_data['caption'].apply(preprocess_text)
val_data['caption'] = val_data['caption'].apply(preprocess_text)
test_data['caption'] = test_data['caption'].apply(preprocess_text)


corpus = list(train_data['caption']) + list(val_data['caption']) + list(test_data['caption'])

print(len(corpus))

tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]


class SimpleWord2Vec:
    def __init__(self, sentences, vector_size=100, window=5, min_count=1, learning_rate=0.01, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.wordvectors = {}
        self.context_vectors = {}
        self.vocab = set()
        
        # build vocabulary and initialize random vectors
        counts = defaultdict(int)
        for sent in sentences:
            for w in sent:
                counts[w] += 1
        for word, cnt in counts.items():
            if cnt >= min_count:
                self.vocab.add(word)
                self.wordvectors[word] = np.random.rand(vector_size) * 0.1 - 0.05
                self.context_vectors[word] = np.random.rand(vector_size) * 0.1 - 0.05

        self.train(sentences)

    def train(self, sentences):
        for epoch in range(self.epochs):
            for sent in sentences:
                for i, word in enumerate(sent):
                    if word not in self.vocab:
                        continue
                    context_start = max(0, i - self.window)
                    context_end = min(len(sent), i + self.window + 1)
                    context_words = [sent[j] for j in range(context_start, context_end) if j != i and sent[j] in self.vocab]
                    for context_word in context_words:
                        self.update_vectors(word, context_word)

    def update_vectors(self, word, context_word):
        word_vector = self.wordvectors[word]
        context_vector = self.context_vectors[context_word]
        
        # calculate loss
        score = np.dot(word_vector, context_vector)
        error = 1 - score
        
        # update vectors
        self.wordvectors[word] += self.learning_rate * error * context_vector
        self.context_vectors[context_word] += self.learning_rate * error * word_vector

    def __contains__(self, word):
        return word in self.wordvectors

    def __getitem__(self, word):
        return self.wordvectors[word]


tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]
word2vec_model = SimpleWord2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=7,
    min_count=1
)

def encode_caption(caption, model, vector_size=100):
    if not caption:
        return np.zeros(vector_size)
    valid_vectors = [model[word] for word in caption if word in model]
    return np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(vector_size) # average of word vectors


train_embeddings = np.array([encode_caption(caption, word2vec_model) for caption in train_data['caption']])
val_embeddings   = np.array([encode_caption(caption, word2vec_model) for caption in val_data['caption']])
test_embeddings  = np.array([encode_caption(caption, word2vec_model) for caption in test_data['caption']])
print(f"Train Embeddings Shape: {train_embeddings.shape}")
print(f"Validation Embeddings Shape: {val_embeddings.shape}")
print(f"Test Embeddings Shape: {test_embeddings.shape}")

np.save('train_embeddings_w2v.npy', train_embeddings)
np.save('val_embeddings_w2v.npy', val_embeddings)
np.save('test_embeddings_w2v.npy', test_embeddings)

