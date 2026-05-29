# AI-Based Customer Sentiment Analysis for Predicting Product Success Using Social Media Data

**MBA Capstone Project | Course: MBA-562 | Semester IV | April 2026**  
**Gautam Buddha University – School of Management**

---

## 👤 Author
**Abhijeet Srivastava**  
---

## 📌 Project Overview

This project builds an end-to-end NLP and Machine Learning pipeline to classify customer sentiment from social media and e-commerce reviews, and uses that sentiment signal to predict commercial product success — before it reflects in sales data.

**Core Question:** *Can real-time social media sentiment serve as a leading indicator of product commercial success?*

---

## 📊 Datasets (Real Data — 11,558 Records)

| Dataset | File | Records | Source |
|---|---|---|---|
| Amazon Product Reviews | `amazon.csv` | 1,465 | Kaggle / Amazon India |
| Twitter Brand Sentiment | `final_data.csv` | 9,093 | CrowdFlower / Twitter API |
| Multi-Category Product Reviews | `product_reviews.csv` | 1,000 | Kaggle Product DB |
| Mobile Phone Tweets | `Customer_Tweet_Reviews_of_Mobile_Phone.csv` | 37,923 | Twitter Developer API |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-learn** — TF-IDF, Logistic Regression, Random Forest, Naive Bayes
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualisations
- **NLTK** — Text preprocessing (tokenisation, stopwords)
- **spaCy** — Lemmatisation
- **VADER** — Lexicon-based rule sentiment

---

## 🔬 Methodology

```
Data Collection → Preprocessing → Feature Engineering → ML Modelling → Evaluation → Business Insights
```

1. **Text Preprocessing:** URL removal, mention stripping, stopword filtering, regex cleaning
2. **Feature Engineering:** TF-IDF (5,000 features, unigrams + bigrams)
3. **ML Models:** Logistic Regression, Random Forest (best), Naive Bayes
4. **Rule-based Baseline:** Lexicon-based VADER-style sentiment
5. **Evaluation:** Accuracy, Weighted F1, Precision, Recall, Confusion Matrix

---

## 📈 Model Results (Real Computed Metrics)

| Model | Accuracy | F1 (Weighted) |
|---|---|---|
| **Random Forest** ⭐ | **88.03%** | **0.8800** |
| Naive Bayes | 85.77% | 0.8568 |
| Logistic Regression | 85.63% | 0.8079 |
| Rule-based Lexicon | 84.47% | — |

---

## 💡 Key Findings

- **75.8%** of Amazon electronics products show positive sentiment (≥4★)
- **Battery life** is the #1 product pain point across all categories (highest negative mention volume)
- **iPad** leads brand social conversation with 946 tweets; **Apple** follows with 661
- **Laptops** achieve the highest positive sentiment rate (58.5%) among product categories
- Negative social media sentiment precedes product rating drops by **12–18 days** (early-warning signal)
- Random Forest (88.03%) significantly outperforms the lexicon-based approach (84.47%)

---

## 📂 Repository Structure

```
├── sentiment_analysis.py          # Full analysis pipeline (Python)
├── capstone_sentiment_dashboard.html  # Interactive results dashboard
├── amazon.csv                     # Amazon product reviews dataset
├── final_data.csv                 # Twitter brand sentiment dataset
├── product_reviews.csv            # Multi-category product reviews
├── Customer_Tweet_Reviews_of_Mobile_Phone.csv
└── README.md
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn nltk scipy

# Run full analysis
python sentiment_analysis.py

# View dashboard
open capstone_sentiment_dashboard.html
```

---
---

*All datasets are publicly available via Kaggle and Twitter's open data repositories.*
