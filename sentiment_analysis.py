"""
============================================================
AI-Based Customer Sentiment Analysis for Predicting Product Success
MBA Capstone Project | Gautam Buddha University
Author: Abhijeet Srivastava | Roll No: 246 PBA 021
Supervisor: Mr. Alok Sharma
============================================================

Datasets Used:
  - Amazon Product Reviews (1,465 records) - amazon.csv
  - Twitter / Social Media Tweets (9,093 records) - final_data.csv
  - Product Reviews Multi-Category (1,000 records) - product_reviews.csv
  
Total Records: 11,558 across 3 complementary datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────
# SECTION 1: DATA LOADING
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

amazon   = pd.read_csv('amazon.csv')
tweets   = pd.read_csv('final_data.csv')
prod_rev = pd.read_csv('product_reviews.csv')

print(f"[1] Amazon Product Reviews    : {len(amazon):,} records, {amazon.shape[1]} features")
print(f"[2] Twitter Sentiment Dataset : {len(tweets):,} records, {tweets.shape[1]} features")
print(f"[3] Product Reviews Dataset   : {len(prod_rev):,} records, {prod_rev.shape[1]} features")
print(f"    TOTAL RECORDS             : {len(amazon)+len(tweets)+len(prod_rev):,}")

# ─────────────────────────────────────────────────────────
# SECTION 2: DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Amazon - numeric conversion
amazon['rating_num']     = pd.to_numeric(amazon['rating'], errors='coerce')
amazon['price_disc']     = amazon['discounted_price'].str.replace('[₹,]', '', regex=True)
amazon['price_disc']     = pd.to_numeric(amazon['price_disc'], errors='coerce')
amazon['price_actual']   = amazon['actual_price'].str.replace('[₹,]', '', regex=True)
amazon['price_actual']   = pd.to_numeric(amazon['price_actual'], errors='coerce')
amazon['discount_pct']   = amazon['discount_percentage'].str.replace('%', '', regex=True)
amazon['discount_pct']   = pd.to_numeric(amazon['discount_pct'], errors='coerce')
amazon['main_category']  = amazon['category'].str.split('|').str[0]

print(f"Amazon missing ratings after cleaning: {amazon['rating_num'].isna().sum()}")
print(f"Amazon avg rating: {amazon['rating_num'].mean():.2f}")

# Tweets - sentiment mapping (3-class)
def map_sentiment(label):
    label = str(label)
    if 'Positive' in label: return 'positive'
    if 'Negative' in label: return 'negative'
    return 'neutral'

tweets['sentiment'] = tweets[
    'is_there_an_emotion_directed_at_a_brand_or_product'
].apply(map_sentiment)

print(f"\nTweet sentiment distribution:")
print(tweets['sentiment'].value_counts().to_string())

# Amazon - derive sentiment from rating
def rating_to_sentiment(r):
    if   r >= 4.0: return 'positive'
    elif r <= 2.5: return 'negative'
    return 'neutral'

amazon['sentiment'] = amazon['rating_num'].apply(rating_to_sentiment)
print(f"\nAmazon sentiment from ratings:")
print(amazon['sentiment'].value_counts().to_string())

# ─────────────────────────────────────────────────────────
# SECTION 3: TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────
STOPWORDS = set("""
i me my myself we our ours ourselves you your yours he him his she her hers
it its they them their what which who is are was were be been being have has
had do does did will would shall should may might must can could a an the and
but or nor not so yet both either neither for at by in on to up as of off rt
via amp just got like really now get getting one two new also want need make
""".split())

def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove URLs, mentions, HTML tags
    3. Remove hashtag symbol (retain word)
    4. Remove punctuation and numbers
    5. Tokenize and remove stopwords
    6. Filter short tokens
    """
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # URLs
    text = re.sub(r'<[^>]+>', '', text)                   # HTML tags
    text = re.sub(r'@\w+', '', text)                      # Mentions
    text = re.sub(r'#(\w+)', r'\1', text)                 # Hashtags → word
    text = re.sub(r'[^a-z\s]', ' ', text)                 # Non-alpha
    tokens = [
        t for t in text.split()
        if t not in STOPWORDS and len(t) > 2
    ]
    return ' '.join(tokens)

# Apply preprocessing
tweets['clean_text'] = tweets['tweet_text'].apply(preprocess_text)

# For Amazon reviews
amazon['clean_review'] = amazon['review_content'].apply(preprocess_text)

print(f"\nSample preprocessed tweet:")
sample = tweets[tweets['clean_text'].str.len() > 20].iloc[0]
print(f"  Original : {sample['tweet_text'][:80]}...")
print(f"  Cleaned  : {sample['clean_text'][:80]}...")

# ─────────────────────────────────────────────────────────
# SECTION 4: FEATURE ENGINEERING (TF-IDF)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FEATURE ENGINEERING — TF-IDF")
print("=" * 60)

# Filter labeled tweets (binary: positive vs negative)
labeled = tweets[tweets['sentiment'].isin(['positive', 'negative'])].copy()
labeled = labeled[labeled['clean_text'].str.len() > 5].reset_index(drop=True)

print(f"Labeled tweets for supervised learning: {len(labeled):,}")
print(f"  Positive: {(labeled['sentiment']=='positive').sum():,}")
print(f"  Negative: {(labeled['sentiment']=='negative').sum():,}")

X = labeled['clean_text']
y = labeled['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTrain/Test split (80/20):")
print(f"  Training samples : {len(X_train):,}")
print(f"  Test samples     : {len(X_test):,}")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),   # Unigrams + Bigrams
    min_df=2,
    sublinear_tf=True
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)
print(f"  TF-IDF matrix shape (train): {X_train_vec.shape}")

# ─────────────────────────────────────────────────────────
# SECTION 5: MACHINE LEARNING MODELS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, C=1.0, solver='lbfgs'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, max_depth=10
    ),
    "Naive Bayes": MultinomialNB(alpha=1.0)
}

results = {}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc  = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_m = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    results[name] = {
        'model': model, 'y_pred': y_pred,
        'accuracy': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m,
        'precision': prec, 'recall': rec
    }
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  F1 (wtd)  : {f1_w:.4f}")
    print(f"  F1 (macro): {f1_m:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, 
          target_names=['Negative','Positive'], indent=4))

# Best model
best_name = max(results, key=lambda k: results[k]['f1_weighted'])
print(f"\n★ Best Model: {best_name}")
print(f"  Accuracy : {results[best_name]['accuracy']*100:.2f}%")
print(f"  F1 Score : {results[best_name]['f1_weighted']:.4f}")

# ─────────────────────────────────────────────────────────
# SECTION 6: AMAZON REVIEW ANALYSIS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("AMAZON PRODUCT ANALYSIS")
print("=" * 60)
cat_stats = amazon.groupby('main_category').agg(
    products=('product_id', 'count'),
    avg_rating=('rating_num', 'mean'),
    avg_discount=('discount_pct', 'mean'),
    avg_price=('price_disc', 'mean')
).round(2).sort_values('products', ascending=False).head(5)
print(cat_stats.to_string())

# ─────────────────────────────────────────────────────────
# SECTION 7: RULE-BASED SENTIMENT (Lexicon Approach)
# ─────────────────────────────────────────────────────────
POS_LEX = set(['good','great','excellent','amazing','love','best','awesome','perfect',
               'wonderful','fantastic','happy','satisfied','recommend','superb','nice',
               'impressive','outstanding','brilliant','quality','worth','helpful','easy',
               'fast','reliable','comfortable','beautiful','clear','smooth','powerful',
               'innovative','value','sturdy','durable','efficient','effective','pleased'])
NEG_LEX = set(['bad','poor','terrible','worst','hate','awful','disappointing','horrible',
               'broken','useless','waste','cheap','slow','difficult','problem','issue',
               'complaint','defect','return','refund','damaged','fake','faulty','annoying',
               'frustrating','disappointed','unhappy','defective','weak','overpriced',
               'misleading','inconsistent','unreliable','flimsy','regret'])

def lexicon_sentiment(text):
    if not isinstance(text, str): return 'neutral'
    tokens = set(text.lower().split())
    pos = len(tokens & POS_LEX)
    neg = len(tokens & NEG_LEX)
    if pos > neg: return 'positive'
    if neg > pos: return 'negative'
    return 'neutral'

tweets['lex_sentiment']  = tweets['tweet_text'].apply(lexicon_sentiment)
amazon['lex_sentiment']  = amazon['review_content'].apply(lexicon_sentiment)

# Rule-based accuracy on labeled set
labeled['lex_pred'] = labeled['tweet_text'].apply(lexicon_sentiment)
valid = labeled[labeled['lex_pred'] != 'neutral']
lex_acc = accuracy_score(
    labeled['sentiment'],
    labeled['lex_pred'].replace('neutral', 'positive')
)
print(f"\nRule-based Lexicon Accuracy: {lex_acc*100:.2f}%")

# ─────────────────────────────────────────────────────────
# SECTION 8: BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY BUSINESS INSIGHTS")
print("=" * 60)

total_amazon = len(amazon)
pos_amazon   = len(amazon[amazon['sentiment']=='positive'])
neg_amazon   = len(amazon[amazon['sentiment']=='negative'])
print(f"Amazon: {pos_amazon/total_amazon*100:.1f}% positive, {neg_amazon/total_amazon*100:.1f}% negative reviews")

brand_pos = tweets[tweets['sentiment']=='positive']['emotion_in_tweet_is_directed_at'].value_counts()
brand_neg = tweets[tweets['sentiment']=='negative']['emotion_in_tweet_is_directed_at'].value_counts()
print(f"\nTop brands by positive tweet volume:")
for brand, cnt in brand_pos.head(5).items():
    print(f"  {brand}: {cnt}")

cat_pr = prod_rev.groupby('category')['sentiment'].value_counts().unstack(fill_value=0)
cat_pr['positive_rate'] = (cat_pr.get('positive',0) / cat_pr.sum(axis=1) * 100).round(1)
print(f"\nCategory positive sentiment rates:")
print(cat_pr['positive_rate'].sort_values(ascending=False).to_string())

feat_neg = prod_rev.groupby('feature_mentioned')['sentiment'].value_counts().unstack(fill_value=0)
feat_neg['neg_rate'] = (feat_neg.get('negative',0) / feat_neg.sum(axis=1) * 100).round(1)
print(f"\nFeatures with highest negative sentiment:")
print(feat_neg['neg_rate'].sort_values(ascending=False).head(8).to_string())

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Total data points analysed: {len(amazon)+len(tweets)+len(prod_rev):,}")
print(f"Best ML model: {best_name} ({results[best_name]['accuracy']*100:.2f}% accuracy)")
