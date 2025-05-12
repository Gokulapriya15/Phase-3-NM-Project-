EXPOSING THE TRUTH WITH ADVANCED FAKE DETECTION POWERED BY NATURAL LANGUAGE PROCESSING 

1. Problem Statement
   
The spread of fake news on digital platforms has become a significant threat to society, influencing public opinion, elections, and even endangering lives during crises. The inability to distinguish between factual and misleading content on social media and news websites can lead to misinformation and social unrest. This project aims to address this issue by developing an effective and scalable fake news detection system using natural language processing (NLP) techniques.

3. Abstract
   
This project proposes an advanced fake news detection system powered by Natural Language Processing (NLP). The system will employ a combination of machine learning techniques and deep learning models to analyze text content, identify linguistic patterns indicative of fake news, and classify news articles as either true or false. The system will be trained on a diverse dataset of news articles, incorporating features derived from text analysis, source credibility, and external fact-checking resources. The goal is to create a reliable and efficient tool for identifying fake news and supporting informed decision-making in an increasingly digital world

4. Exploratory Data Analysis (EDA)
   
Exploratory Data Analysis (EDA) was conducted to gain insights into the structure, patterns, and distribution of fake vs real news articles. This step is critical to understand the dataset before applying machine learning models.
1. Dataset Overview
Total Records: X articles (e.g., 44,000 in ISOT)
Columns:
title

text

label (1 = Fake, 0 = Real)
Sample Output:
df.info()
df.head()
2. Class Distribution
Checked the balance between fake and real news classes.
Code:
sns.countplot(x='label', data=df)
plt.title("Fake vs Real News Distribution")
Observation:
If imbalanced, consider using SMOTE or class weights during training.
3. Text Length Analysis
Article Length: Number of words/tokens in fake vs real articles.
Code:
df['text_length'] = df['text'].apply(lambda x: len(x.split()))
sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True)
Observation:
Fake news articles may be shorter or more repetitive.
4. Most Common Words
Word frequency analysis after removing stop words.
Code:
from collections import Counter
from wordcloud import WordCloud

fake_words = ' '.join(df[df['label']==1]['text'])
real_words = ' '.join(df[df['label']==0]['text'])

WordCloud().generate(fake_words).to_image()
WordCloud().generate(real_words).to_image()
Observation:
Fake news may use emotionally charged or clickbait terms like "shocking", "breaking", etc.
5. N-gram Analysis (Bigrams/Trigrams)
To identify common phrases in fake and real news.
Code:
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english')
bigrams = vectorizer.fit_transform(df[df['label']==1]['text'])
Observation:
Real news may focus more on informative phrases, while fake news may contain provocative or vague statements.
6. Sentiment Analysis
Basic sentiment polarity score (optional) using TextBlob or VADER.
Code:
from textblob import TextBlob

df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
sns.boxplot(x='label', y='sentiment', data=df)
Observation:
Fake news may skew toward extreme sentiment values (highly negative or overly positive).
7. Correlation & Word Embeddings Visualization (Optional)
Use t-SNE or PCA to visualize text embeddings (e.g., TF-IDF or BERT) in 2D space.
Code:
from sklearn.decomposition import PCA

# Reduce TF-IDF features for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(tfidf_matrix.toarray())
Observation:
Helps in identifying natural clustering between fake and real news.

4. Model Building
   
The goal of model building is to classify news articles as fake or real using features extracted from the text data. Both traditional machine learning models and deep learning models were considered.
1. Model Selection Strategy
Two modeling pipelines were explored:
Traditional Machine Learning Models using BoW/TF-IDF

Deep Learning Models using word embeddings (Word2Vec, GloVe, BERT)
A. Traditional Machine Learning Models
1. Logistic Regression
Simple and interpretable baseline classifier.

Works well with TF-IDF features.
Code:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
2. Naive Bayes
Particularly effective for text classification.

Assumes word independence.
Code:
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
3. Random Forest / XGBoost
Ensemble models that capture nonlinear relationships and feature interactions.
Code:
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
B. Deep Learning Models
1. LSTM / GRU (Sequential Models)
Suitable for learning long-term dependencies in text.

Input: Word embeddings or tokenized sequences.
Code:
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
4. BERT-Based Classifier
Fine-tuned transformer model for binary classification.

Offers state-of-the-art accuracy.
Code:
from transformers import BertTokenizer, TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

5. Source Code
   
import pandas as pd
import numpy as np
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Add labels
true_df['label'] = 1  # Real
fake_df['label'] = 0  # Fake

# Combine datasets
data = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle

# Preprocessing
stop_words = stopwords.words('english')
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    return ' '.join([word for word in tokens if word not in stop_words])

data['text'] = data['title'] + " " + data['text']
data['text'] = data['text'].apply(clean_text)

# Train/Test split
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# Evaluation
y_pred = model.predict(tfidf_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



 
