# 🤖 AI-Based Customer Sentiment Analysis for Predicting Product Success

> **MBA Capstone Project** — Abhijeet Srivastava | Roll No. 246 PBA 021
> MBA – Business Analytics & Data Science | Batch 2024–2026
> Supervisor: Mr. Alok Sharma | School of Management, Gautam Buddha University | May 2026

---

## 📌 Overview

This project builds an end-to-end NLP + Machine Learning pipeline that analyses customer
sentiment from real-world social media and e-commerce data to predict consumer electronics
product success — *before* traditional sales metrics catch up.

**Core Research Question:**
> *Can AI-based sentiment analysis of real-world Kaggle data reliably classify customer
> sentiment towards consumer electronics, and translate those insights into actionable
> business intelligence?*

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Best Model | Logistic Regression |
| Accuracy | **73.5%** (baseline: 56.8%) |
| ROC-AUC | **0.829** |
| Negative Precision | **94.6%** |
| Total Dataset Size | **36,401 records** (4 Kaggle datasets) |
| PSR → BSR Lead Time | **~2 weeks** (r = −0.67, p < 0.001) |

---

## 🗂️ Datasets Used

| Dataset | Records | Description |
|---|---|---|
| Amazon India Product Sales | 1,464 | Product ratings & reviews from Amazon.in. Avg rating: 4.10/5 |
| CrowdFlower Apple-Google Tweets | 9,093 | Crowd-annotated product launch tweets |
| Consumer Electronics Reviews | 1,000 | 5 categories: Smartphones, Laptops, Audio, Wearables, Smart Home |
| Customer Mobile Phone Tweets | 25,000 | Mobile product tweets, keyword-scored |

All datasets are publicly available, organically collected — no synthetic data used.

---

## 🔬 Methodology (CRISP-DM Pipeline)

1. **Data Acquisition** — Kaggle CSVs, integrity validation
2. **Preprocessing** — URL/HTML removal, lemmatisation, negation handling (`NOT_good`, `NOT_work`)
3. **EDA** — Sentiment distributions, category breakdowns, temporal trends
4. **Feature Engineering** — TF-IDF (5,000 features, bigrams), SMOTE for class balancing
5. **ML Modelling** — Logistic Regression ✓, Random Forest, Naive Bayes (stratified 85/15 split)
6. **Business Intelligence** — PSR alerts, BI dashboard concept, FastAPI deployment plan

---

## 🛠️ Tech Stack

`Python 3.10` · `Pandas` · `NumPy` · `Scikit-learn` · `NLTK` · `VADER` · `Matplotlib` · `Seaborn` · `SMOTE` · `WordCloud` · `Google Colab` · `joblib`

---

## 📈 Key Findings

- 📦 **Mobile Accessories** lead satisfaction at **86% Positive**; Headphones lag at **52%**
- ⚠️ **Smart Home** has the highest negative rate — **19% Negative** — needs quality audit
- 📷 **Camera & Display** is the top positive driver (VADER compound: **+0.52**)
- 😤 **Customer Service** is the only negative-scoring aspect (**−0.12**) — critical gap
- 📉 PSR consistently **precedes Amazon Best Seller Rank improvement by ~2 weeks**

---

## 💼 Business Recommendations

| Recommendation | Action |
|---|---|
| R1 | Deploy LR pipeline as FastAPI microservice → Power BI/Tableau dashboard |
| R2 | 30-day post-launch monitoring: PSR < 60% triggers Rapid Response Protocol |
| R3 | Integrate weekly PSR as exogenous variable in supply-chain demand forecasting |
| R4 | Invest in battery life & value-for-money R&D (weakest-scoring aspects) |
| R5 | Fine-tune mBERT/XLM-RoBERTa for Hindi-English code-mixed content |

---

## 🚀 Future Scope

- Fine-tune **RoBERTa / BERT** for +8–12 pp accuracy improvement
- **Apache Kafka** real-time streaming (< 5 min latency)
- **Multilingual NLP** for Hindi-English social media
- **SHAP Explainability Dashboard** for non-technical stakeholders
- **Competitor benchmarking** module

---

## 📁 Repository Structure
├── capstone_report.pdf          # Full project report (51 pages)
├── capstone_presentation.pptx   # 15-slide summary presentation
├── notebook.ipynb               # Complete Python pipeline (Google Colab)
├── sentiment_pipeline.joblib    # Serialised trained model
└── README.md

Abhijeet Srivastava
*Feel free to connect on www.linkedin.com/in/abhijeet-srivastava-r07 or raise an issue for any queries.*
