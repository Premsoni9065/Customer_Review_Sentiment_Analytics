# 📱 Customer Review & Sentiment Analytics

An interactive **Customer Voice / E-Commerce Review Analytics Dashboard** built with Python, Pandas, NLTK (VADER), SQL, and Streamlit. Designed as a beginner-friendly, portfolio-ready Data Analyst project.

👨‍💻 **Developed & Deployed by Prem Kumar**
*Data Analyst | Python | SQL | Power BI | Excel*

---

## 📋 Project Overview

Customer reviews are one of the richest sources of feedback an e-commerce business has — but reading thousands of them manually isn't practical. This project turns a raw customer reviews dataset into an interactive analytics dashboard that surfaces sentiment, complaints, product performance, and business insights automatically.

## 🎯 Business Problem

An e-commerce company receives thousands of product reviews every month but has no easy way to:
- Understand overall customer satisfaction
- Identify which products/categories are underperforming
- Spot recurring complaints before they affect sales
- Track how sentiment changes over time

## 🎯 Project Objective

Build an interactive dashboard that allows a business user to explore customer reviews and answer questions like:
- What percentage of our customers are happy?
- Which products get the most complaints?
- What words come up most often in negative reviews?
- Is customer sentiment improving or declining over time?

## ✨ Features

- 🏠 **Overview** — KPI cards, sentiment & rating distribution, trends, top products
- ⭐ **Rating Analysis** — Rating distribution, rating by category/product, rating vs sentiment
- 😊 **Sentiment Analysis** — Sentiment distribution, sentiment by rating & category
- 📦 **Product Analysis** — Top products by review count/rating/sentiment + category breakdown
- 📝 **Review Explorer** — Searchable, filterable review table with CSV export
- 🔤 **Word Analysis** — Most frequent positive/negative words + optional word cloud
- 💡 **Business Insights** — Auto-generated insights + Customer Satisfaction Score
- 📊 **Data Explorer** — Data quality report (missing values, duplicates, etc.)
- 🔍 Sidebar filters: Product, Category, Rating, Sentiment, Date Range, Verified Purchase

## 🗂️ Dataset Description

`customer_reviews.csv` — ~7,000 synthetic but realistic e-commerce reviews across 29 products and 8 categories (Electronics, Home & Kitchen, Clothing, Beauty & Personal Care, Sportswear, etc.)

| Column | Description |
|---|---|
| Review_ID | Unique review identifier |
| Product_ID | Unique product identifier |
| Product_Name | Name of the product |
| Category | Product category |
| Customer_ID | Unique customer identifier |
| Review_Date | Date the review was submitted |
| Rating | Star rating (1–5) |
| Review_Title | Short review headline |
| Review_Text | Full review text |
| Verified_Purchase | Yes / No |
| Helpful_Votes | Number of users who found the review helpful |

## 🧠 NLP Methodology

1. **Text preprocessing**: lowercase conversion → punctuation & special character removal → extra whitespace removal → stopword removal
2. **Sentiment scoring**: Each cleaned review is scored with **NLTK VADER** (`SentimentIntensityAnalyzer`), a lexicon-based sentiment tool well suited to short, informal text like product reviews — no deep learning required.

### Sentiment Analysis Method

```
compound_score = VADER.polarity_scores(review_text)["compound"]

if compound_score >= 0.05:
    sentiment = "Positive"
elif compound_score <= -0.05:
    sentiment = "Negative"
else:
    sentiment = "Neutral"
```

VADER is chosen because it is lightweight, requires no training data or GPU, and performs well on short customer-review-style text.

## 🗃️ SQL Analysis

`review_analysis.sql` contains 15 beginner-level SQL queries covering `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, `COUNT`, `AVG`, and `CASE WHEN` — including total reviews, average rating, rating distribution, category performance, top/bottom rated products, verified vs non-verified reviews, and average helpful votes by rating.

## 📊 Dashboard Features

- Modern dark navy SaaS-style theme with light KPI/section cards
- Rounded cards, soft shadows, purple/blue accents
- Fully dynamic — no hard-coded numbers or insights
- Responsive, wide layout with a clean sidebar

## 🛠️ Technology Stack

- **Python** — core programming language
- **Pandas / NumPy** — data cleaning & analysis
- **NLTK (VADER)** — sentiment analysis
- **Matplotlib / Seaborn** — data visualization
- **WordCloud** *(optional)* — word cloud visualization
- **SQL** — analytical queries
- **Streamlit** — interactive web dashboard

## 📁 Project Structure

```
customer-review-sentiment/
│
├── app.py                  # Main Streamlit application
├── customer_reviews.csv    # Sample dataset (~7,000 reviews)
├── review_analysis.sql     # SQL analysis queries
├── requirements.txt        # Python dependencies
├── README.md                # Project documentation
└── .gitignore
```

## 💻 Installation & Local Setup

**1. Create a virtual environment**
```
py -3.11 -m venv venv
```

**2. Activate it (Windows PowerShell)**
```
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**
```
pip install -r requirements.txt
```

**4. Run the app**
```
streamlit run app.py
```

The app will automatically download the VADER lexicon the first time it runs — no manual NLTK setup needed.

## 🐙 GitHub Setup

1. Create a new GitHub repository named `customer-review-sentiment`
2. Upload: `app.py`, `customer_reviews.csv`, `review_analysis.sql`, `requirements.txt`, `README.md`, `.gitignore`
3. Commit and push to the `main` branch

## ☁️ Streamlit Deployment

1. Create the GitHub repository and upload all project files
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Click **"New app"** and connect your GitHub account
4. Select the `customer-review-sentiment` repository
5. Set the main file path to `app.py`
6. Click **Deploy**
7. Test the live public app URL once the build finishes

> ✅ The app uses only relative file paths (`customer_reviews.csv`) and downloads NLTK data programmatically, so it works out-of-the-box on Streamlit Community Cloud.

## 🚀 Future Improvements

- Support direct CSV upload from the UI for user-supplied datasets
- Add topic modeling to auto-cluster complaint themes
- Add multi-language sentiment support
- Add authentication for multi-user business dashboards
- Integrate a live database connection instead of a static CSV

## 👨‍💻 Author

**Prem Kumar**
Data Analyst | Python | SQL | Power BI | Excel

website link = https://customer-review-sentiment-analytics-by-pks.streamlit.app/

🚀 Developed & Deployed by Prem Kumar
