
# !pip install inflect

import pandas as pd
import os
import cv2
import numpy as np

import spacy
import inflect
from sklearn.feature_extraction.text import TfidfVectorizer


nlp = spacy.load('en_core_web_sm')
p = inflect.engine()

train_csv = '/kaggle/input/dl-project/isp-match-dl-2024_v2/train.csv'
val_csv = '/kaggle/input/dl-project/isp-match-dl-2024_v2/val.csv'
test_csv = '/kaggle/input/dl-project/isp-match-dl-2024_v2/test.csv'

train_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/train_images'
val_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/val_images'
test_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/test_images'


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

all_captions = pd.concat([
    train_data['caption'], 
    val_data['caption'], 
    test_data['caption']
])


# tf-idf features
vectorizer = TfidfVectorizer(max_features=2000)
vectorizer.fit(all_captions)

train_tfidf = vectorizer.transform(train_data['caption']).toarray()
val_tfidf = vectorizer.transform(val_data['caption']).toarray()
test_tfidf = vectorizer.transform(test_data['caption']).toarray()

print(f"TF-IDF Train Feature Shape: {train_tfidf.shape}")
print(f"TF-IDF Validation Feature Shape: {val_tfidf.shape}")
print(f"TF-IDF Test Feature Shape: {test_tfidf.shape}")

np.save('train_tfidf.npy', train_tfidf)
np.save('val_tfidf.npy', val_tfidf)
np.save('test_tfidf.npy', test_tfidf)


# hog features
def extract_hog_features(image_dir, image_names):
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64), 
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=6
    )
    features = []
    for img_name in image_names:
        img_path = os.path.join(image_dir, str(img_name) + '.jpg')
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Image {img_path} not found.")
            continue
        image = cv2.resize(image, (100, 100))
        hog_features = hog.compute(image).flatten()
        features.append(hog_features)
    return np.array(features)



train_hog = extract_hog_features(train_image_dir, train_data['image_id'])
val_hog = extract_hog_features(val_image_dir, val_data['image_id'])
test_hog = extract_hog_features(test_image_dir, test_data['image_id'])

print(f"HOG Train Feature Shape: {train_hog.shape}")
print(f"HOG Validation Feature Shape: {val_hog.shape}")
print(f"HOG Test Feature Shape: {test_hog.shape}")

np.save('train_hog.npy', train_hog)
np.save('val_hog.npy', val_hog)
np.save('test_hog.npy', test_hog)
