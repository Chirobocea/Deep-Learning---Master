import pandas as pd

df_train = pd.read_csv('/kaggle/input/dl-project/isp-match-dl-2024_v2/train.csv')
df_val = pd.read_csv('/kaggle/input/dl-project/isp-match-dl-2024_v2/val.csv')
df_test = pd.read_csv('/kaggle/input/dl-project/isp-match-dl-2024_v2/test.csv')

corpus = df_train.iloc[:, 1].tolist() + df_val.iloc[:, 1].tolist() + df_test.iloc[:, 1].tolist()

print(len(corpus))


import matplotlib.pyplot as plt

label_column = df_train.columns[-1] 
label_counts = df_train[label_column].value_counts()

plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar')
plt.title('Label Distribution in Training Data')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()



# !pip install spacy


# !pip install inflect


import spacy
import inflect

nlp = spacy.load('en_core_web_sm')
p = inflect.engine()

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

cleaned_corpus = [preprocess_text(doc) for doc in corpus]
print(cleaned_corpus[:5])


from collections import Counter

all_words = ' '.join(corpus).split()


word_freq = Counter(all_words)
unique_words_count = len(word_freq)
print(f"Number of unique words: {unique_words_count}")

frequencies = list(word_freq.values())

plt.figure(figsize=(10, 6))
plt.hist(frequencies, bins=50, color='blue', edgecolor='black', log=True)
plt.title("Word Frequency Histogram")
plt.xlabel("Word Frequency")
plt.ylabel("Count (Log Scale)")
plt.grid(axis='y')
plt.show()


from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



from tokenizers import Tokenizer, pre_tokenizers, trainers
from tokenizers.models import BPE
from tokenizers import normalizers


tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.normalizer = normalizers.NFKC()

trainer = trainers.BpeTrainer(
    vocab_size=1500,
    min_frequency=2,
    show_progress=True,
    special_tokens=[
        "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"
    ]
)

tokenizer.train_from_iterator(cleaned_corpus, trainer=trainer)
tokenizer.save("roberta_tokenizer.json")

import pickle
with open("roberta_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)



sentence_lengths = [len(tokenizer.encode(sentence).tokens) for sentence in cleaned_corpus]

plt.figure(figsize=(12, 6))
plt.hist(sentence_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Sentence Lengths (in tokens)')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.show()



num_samples_greater_than = sum(1 for length in sentence_lengths if length > 32)
print(f"Number of sentences with more than  tokens: {num_samples_greater_than}")



import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = 32
padded_samples = pad_sequences(
    [tokenizer.encode(sentence).ids for sentence in cleaned_corpus],
    maxlen=max_length,
    padding='post', 
    truncating='post',  
    value=0 
)

train_samples = padded_samples[:10000]
val_samples = padded_samples[10000:13000]  
test_samples = padded_samples[13000:15000] 

np.save('train_tokens.npy', train_samples)
np.save('val_tokens.npy', val_samples)
np.save('test_tokens.npy', test_samples)