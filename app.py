"""
📱 Customer Review & Sentiment Analytics
------------------------------------------------
An interactive Streamlit dashboard that analyzes customer product
reviews using beginner-friendly NLP (NLTK VADER), Pandas, and
data visualization.

Developed & Deployed by Prem Kumar
Data Analyst | Python | SQL | Power BI | Excel
"""

import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ------------------------------------------------------------------
# 0. PAGE CONFIG (must be the first Streamlit command)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Review & Sentiment Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# 1. NLTK VADER SETUP (works on Streamlit Community Cloud)
# ------------------------------------------------------------------
@st.cache_resource
def load_sentiment_analyzer():
    """
    Download the VADER lexicon if it isn't already available, then
    return a ready-to-use SentimentIntensityAnalyzer.
    This runs once per app session thanks to st.cache_resource.
    """
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()


# ------------------------------------------------------------------
# 2. CUSTOM CSS - modern dark navy / SaaS dashboard styling
# ------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0f1b2d;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a2942 0%, #16213e 100%);
        padding: 1.6rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        border: 1px solid #26375a;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #9fb3d1;
        font-size: 0.95rem;
        margin-bottom: 0.6rem;
    }
    .badge {
        display: inline-block;
        background: linear-gradient(90deg, #7c3aed, #4f46e5);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
    }
    .dev-credit {
        color: #7c9cd4;
        font-size: 0.85rem;
        margin-top: 0.4rem;
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
        border: 1px solid #eef0f5;
        text-align: left;
    }
    .kpi-label {
        color: #6b7280;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .kpi-value {
        color: #1a2942;
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    /* Section cards */
    .section-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
        border: 1px solid #eef0f5;
    }
    .section-title {
        color: #1a2942;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    /* Insight box */
    .insight-box {
        background: #f5f3ff;
        border-left: 4px solid #7c3aed;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        color: #312e81;
        font-size: 0.92rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #12203a;
    }
    section[data-testid="stSidebar"] * {
        color: #e5ecf7 !important;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    .app-footer {
        text-align: center;
        color: #7c9cd4;
        padding: 1.4rem 0 0.6rem 0;
        font-size: 0.85rem;
        border-top: 1px solid #26375a;
        margin-top: 1.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------
# 3. CORE DATA FUNCTIONS
# ------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "Review_ID", "Product_ID", "Product_Name", "Category", "Customer_ID",
    "Review_Date", "Rating", "Review_Title", "Review_Text",
    "Verified_Purchase", "Helpful_Votes",
]


@st.cache_data
def load_data(file_path="customer_reviews.csv"):
    """Load the customer reviews dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df, None
    except FileNotFoundError:
        return None, f"Could not find '{file_path}'. Please make sure the file is in the app folder."
    except pd.errors.EmptyDataError:
        return None, "The dataset file is empty."
    except Exception as e:
        return None, f"Unexpected error while loading data: {e}"


def clean_data(df):
    """
    Clean the raw reviews dataframe:
    - Validate required columns
    - Remove duplicate rows
    - Handle missing values
    - Validate rating range (1-5)
    - Convert Review_Date to datetime
    """
    df = df.copy()

    # Drop exact duplicate rows
    df.drop_duplicates(inplace=True)

    # Fill missing text fields so downstream NLP doesn't break
    df["Review_Title"] = df["Review_Title"].fillna("No Title")
    df["Review_Text"] = df["Review_Text"].fillna("")
    df["Helpful_Votes"] = pd.to_numeric(df["Helpful_Votes"], errors="coerce").fillna(0).astype(int)

    # Validate rating: keep only 1-5, drop anything invalid
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df[df["Rating"].between(1, 5)]
    df["Rating"] = df["Rating"].astype(int)

    # Convert date column
    df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")
    df = df.dropna(subset=["Review_Date"])

    df.reset_index(drop=True, inplace=True)
    return df


def preprocess_text(text):
    """
    Beginner-friendly text cleaning pipeline:
    lowercase -> remove punctuation/special characters -> remove extra spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # remove punctuation/numbers/special chars
    text = re.sub(r"\s+", " ", text).strip()        # remove extra spaces
    return text


STOPWORDS = set("""
a an the and or but if is are was were be been being to of in on for with
as at by from this that these those it its it's i you he she we they them
his her our your their not no so very just also can will would could should
my me mine yours ours theirs than then there here up down out over under again
""".split())


def remove_stopwords(text):
    """Remove common stopwords from a cleaned text string."""
    return " ".join([w for w in text.split() if w not in STOPWORDS])


@st.cache_data
def perform_sentiment_analysis(df):
    """
    Apply text preprocessing and VADER sentiment analysis to every review.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) scores each
    piece of text and returns a 'compound' score between -1 (very negative)
    and +1 (very positive). We classify sentiment using standard VADER
    thresholds:

        compound >= 0.05   -> Positive
        compound <= -0.05  -> Negative
        otherwise           -> Neutral
    """
    analyzer = load_sentiment_analyzer()
    df = df.copy()

    df["Clean_Review_Text"] = df["Review_Text"].apply(preprocess_text)

    compounds = []
    sentiments = []
    for text in df["Clean_Review_Text"]:
        score = analyzer.polarity_scores(text)["compound"]
        compounds.append(score)
        if score >= 0.05:
            sentiments.append("Positive")
        elif score <= -0.05:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")

    df["Sentiment_Score"] = compounds
    df["Sentiment"] = sentiments
    return df


# ------------------------------------------------------------------
# 4. ANALYTICS FUNCTIONS
# ------------------------------------------------------------------
def calculate_kpis(df):
    """Calculate the core KPIs shown at the top of the dashboard."""
    total_reviews = len(df)
    avg_rating = round(df["Rating"].mean(), 2) if total_reviews else 0
    positive = int((df["Sentiment"] == "Positive").sum())
    neutral = int((df["Sentiment"] == "Neutral").sum())
    negative = int((df["Sentiment"] == "Negative").sum())
    positive_pct = round(100 * positive / total_reviews, 1) if total_reviews else 0

    return {
        "total_reviews": total_reviews,
        "avg_rating": avg_rating,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "positive_pct": positive_pct,
    }


def apply_filters(df, products, categories, ratings, sentiments, date_range, verified):
    """Apply sidebar filter selections to the dataframe."""
    filtered = df.copy()

    if products:
        filtered = filtered[filtered["Product_Name"].isin(products)]
    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    if ratings:
        filtered = filtered[filtered["Rating"].isin(ratings)]
    if sentiments:
        filtered = filtered[filtered["Sentiment"].isin(sentiments)]
    if verified and verified != "All":
        filtered = filtered[filtered["Verified_Purchase"] == verified]
    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered["Review_Date"] >= start) & (filtered["Review_Date"] <= end)]

    return filtered


def get_word_frequencies(text_series, top_n=20):
    """Count the most frequently used words in a series of cleaned text."""
    all_words = []
    for text in text_series:
        cleaned = remove_stopwords(preprocess_text(text))
        all_words.extend([w for w in cleaned.split() if len(w) > 2])
    return Counter(all_words).most_common(top_n)


def generate_business_insights(df):
    """Dynamically generate business insight bullet points from the data."""
    insights = []
    if df.empty:
        return ["Not enough data to generate insights. Try adjusting your filters."]

    cat_rating = df.groupby("Category")["Rating"].mean().sort_values(ascending=False)
    if not cat_rating.empty:
        insights.append(f"📌 **Highest-rated category:** {cat_rating.index[0]} "
                         f"(avg rating {cat_rating.iloc[0]:.2f}⭐)")
        insights.append(f"📌 **Lowest-rated category:** {cat_rating.index[-1]} "
                         f"(avg rating {cat_rating.iloc[-1]:.2f}⭐)")

    cat_sentiment = df[df["Sentiment"] == "Negative"].groupby("Category").size()
    cat_total = df.groupby("Category").size()
    if not cat_sentiment.empty:
        neg_pct = (cat_sentiment / cat_total * 100).dropna().sort_values(ascending=False)
        if not neg_pct.empty:
            insights.append(f"📌 **Category with highest negative sentiment:** {neg_pct.index[0]} "
                             f"({neg_pct.iloc[0]:.1f}% negative reviews)")

    prod_pos = df[df["Sentiment"] == "Positive"].groupby("Product_Name").size()
    if not prod_pos.empty:
        top_prod = prod_pos.sort_values(ascending=False).index[0]
        insights.append(f"📌 **Product with highest positive sentiment:** {top_prod}")

    negative_reviews = df[df["Sentiment"] == "Negative"]["Clean_Review_Text"]
    if not negative_reviews.empty:
        common_words = get_word_frequencies(negative_reviews, top_n=1)
        if common_words:
            insights.append(f"📌 **Most common customer complaint word:** \"{common_words[0][0]}\"")

    insights.append(f"📌 **Average rating across all reviews:** {df['Rating'].mean():.2f}⭐")

    pos_pct = 100 * (df["Sentiment"] == "Positive").sum() / len(df)
    insights.append(f"📌 **Overall positive sentiment:** {pos_pct:.1f}% of reviews are positive")

    return insights


def satisfaction_interpretation(score):
    if score >= 80:
        return "Excellent", "#16a34a"
    elif score >= 60:
        return "Good", "#65a30d"
    elif score >= 40:
        return "Needs Improvement", "#d97706"
    else:
        return "Poor", "#dc2626"


# ------------------------------------------------------------------
# 5. LOAD & PREPARE DATA
# ------------------------------------------------------------------
raw_df, load_error = load_data("customer_reviews.csv")

if load_error:
    st.error(f"⚠️ {load_error}")
    st.info("Please add a valid 'customer_reviews.csv' file to the app folder and reload the page.")
    st.stop()

if raw_df.empty:
    st.warning("The dataset is empty. Please upload a dataset with review data.")
    st.stop()

missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
if missing_cols:
    st.error(f"⚠️ The dataset is missing required columns: {', '.join(missing_cols)}")
    st.stop()

data_quality = {
    "total_records": len(raw_df),
    "missing_values": int(raw_df.isnull().sum().sum()),
    "duplicate_records": int(raw_df.duplicated().sum()),
}

cleaned_df = clean_data(raw_df)
data_quality["unique_products"] = cleaned_df["Product_Name"].nunique()
data_quality["unique_categories"] = cleaned_df["Category"].nunique()

df = perform_sentiment_analysis(cleaned_df)

if df.empty:
    st.warning("No valid reviews remain after data cleaning. Please check your dataset.")
    st.stop()


# ------------------------------------------------------------------
# 6. SIDEBAR - NAVIGATION + FILTERS
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💬 CUSTOMER VOICE")
    st.caption("Customer Review & Sentiment Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "⭐ Rating Analysis",
            "😊 Sentiment Analysis",
            "📦 Product Analysis",
            "📝 Review Explorer",
            "🔤 Word Analysis",
            "💡 Business Insights",
            "📊 Data Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🔍 Filters")

    all_products = sorted(df["Product_Name"].unique().tolist())
    all_categories = sorted(df["Category"].unique().tolist())
    all_ratings = sorted(df["Rating"].unique().tolist())
    all_sentiments = ["Positive", "Neutral", "Negative"]
    min_date, max_date = df["Review_Date"].min(), df["Review_Date"].max()

    f_category = st.multiselect("Category", all_categories)
    f_product = st.multiselect("Product", all_products)
    f_rating = st.multiselect("Rating", all_ratings)
    f_sentiment = st.multiselect("Sentiment", all_sentiments)
    f_verified = st.selectbox("Verified Purchase", ["All", "Yes", "No"])
    f_date = st.date_input("Date Range", value=(min_date, max_date))

    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()

filtered_df = apply_filters(df, f_product, f_category, f_rating, f_sentiment, f_date, f_verified)

if filtered_df.empty:
    st.warning("⚠️ No reviews match the selected filters. Showing full dataset instead.")
    filtered_df = df


# ------------------------------------------------------------------
# 7. HEADER
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="main-header">
        <span class="badge">CUSTOMER VOICE ANALYTICS</span>
        <h1>📱 Customer Review & Sentiment Analytics</h1>
        <p>Understand what customers think about your products.</p>
        <div class="dev-credit">👨‍💻 Developed &amp; Deployed by Prem Kumar</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def kpi_card(col, label, value):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def section_header(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 8. PAGE: OVERVIEW
# ------------------------------------------------------------------
if page == "🏠 Overview":
    st.subheader("🏠 Review Analytics Overview")

    kpis = calculate_kpis(filtered_df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_card(c1, "📝 Total Reviews", f"{kpis['total_reviews']:,}")
    kpi_card(c2, "⭐ Average Rating", f"{kpis['avg_rating']}")
    kpi_card(c3, "😊 Positive", f"{kpis['positive']:,}")
    kpi_card(c4, "😐 Neutral", f"{kpis['neutral']:,}")
    kpi_card(c5, "😞 Negative", f"{kpis['negative']:,}")
    kpi_card(c6, "📈 Positive %", f"{kpis['positive_pct']}%")

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Sentiment Distribution")
        sentiment_counts = filtered_df["Sentiment"].value_counts()
        fig, ax = plt.subplots(figsize=(4.5, 4))
        colors = {"Positive": "#16a34a", "Neutral": "#f59e0b", "Negative": "#dc2626"}
        ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            colors=[colors.get(s, "#999") for s in sentiment_counts.index],
            wedgeprops=dict(width=0.45),
            startangle=90,
        )
        ax.set_title("")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Rating Distribution")
        rating_counts = filtered_df["Rating"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=rating_counts.index, y=rating_counts.values, color="#4f46e5", ax=ax)
        ax.set_xlabel("Rating (Stars)")
        ax.set_ylabel("Number of Reviews")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Sentiment Trend Over Time")
        trend = (
            filtered_df.assign(Month=filtered_df["Review_Date"].dt.to_period("M").astype(str))
            .groupby(["Month", "Sentiment"]).size().unstack(fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        for sentiment, color in colors.items():
            if sentiment in trend.columns:
                ax.plot(trend.index, trend[sentiment], marker="o", label=sentiment, color=color)
        ax.set_xticks(range(len(trend.index)))
        ax.set_xticklabels(trend.index, rotation=45, ha="right", fontsize=7)
        ax.legend()
        ax.set_ylabel("Reviews")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Reviews by Category")
        cat_counts = filtered_df["Category"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=cat_counts.values, y=cat_counts.index, color="#7c3aed", ax=ax)
        ax.set_xlabel("Number of Reviews")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("🏆 Top Products by Average Rating")
    top_products = (
        filtered_df.groupby("Product_Name")
        .agg(Review_Count=("Review_ID", "count"), Average_Rating=("Rating", "mean"))
        .query("Review_Count >= 3")
        .sort_values("Average_Rating", ascending=False)
        .head(10)
        .round(2)
    )
    st.dataframe(top_products, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 9. PAGE: RATING ANALYSIS
# ------------------------------------------------------------------
elif page == "⭐ Rating Analysis":
    st.subheader("⭐ Rating Analysis")

    cat_avg = filtered_df.groupby("Category")["Rating"].mean().sort_values(ascending=False)
    if not cat_avg.empty:
        c1, c2 = st.columns(2)
        kpi_card(c1, "🏆 Highest Rated Category", f"{cat_avg.index[0]} ({cat_avg.iloc[0]:.2f}⭐)")
        kpi_card(c2, "⚠️ Lowest Rated Category", f"{cat_avg.index[-1]} ({cat_avg.iloc[-1]:.2f}⭐)")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Rating Distribution")
        rating_counts = filtered_df["Rating"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=rating_counts.index, y=rating_counts.values, color="#4f46e5", ax=ax)
        ax.set_xlabel("Rating (Stars)")
        ax.set_ylabel("Number of Reviews")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Average Rating by Category")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=cat_avg.values, y=cat_avg.index, color="#7c3aed", ax=ax)
        ax.set_xlabel("Average Rating")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Average Rating by Product (Top 15)")
    prod_avg = (
        filtered_df.groupby("Product_Name")
        .agg(Review_Count=("Review_ID", "count"), Average_Rating=("Rating", "mean"))
        .query("Review_Count >= 3")
        .sort_values("Average_Rating", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=prod_avg["Average_Rating"], y=prod_avg.index, color="#4f46e5", ax=ax)
    ax.set_xlabel("Average Rating")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Rating vs Sentiment")
    cross = pd.crosstab(filtered_df["Rating"], filtered_df["Sentiment"])
    fig, ax = plt.subplots(figsize=(7, 4))
    cross.plot(kind="bar", stacked=True, color=["#dc2626", "#f59e0b", "#16a34a"], ax=ax)
    ax.set_xlabel("Rating (Stars)")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 10. PAGE: SENTIMENT ANALYSIS
# ------------------------------------------------------------------
elif page == "😊 Sentiment Analysis":
    st.subheader("😊 Sentiment Analysis")

    kpis = calculate_kpis(filtered_df)
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "😊 Positive", f"{kpis['positive']:,}")
    kpi_card(c2, "😐 Neutral", f"{kpis['neutral']:,}")
    kpi_card(c3, "😞 Negative", f"{kpis['negative']:,}")
    kpi_card(c4, "📈 Positive %", f"{kpis['positive_pct']}%")

    st.write("")
    st.info("ℹ️ Sentiment is calculated using **NLTK VADER**: each review's text gets a "
            "compound score from -1 (very negative) to +1 (very positive). "
            "Score ≥ 0.05 → Positive, ≤ -0.05 → Negative, otherwise Neutral.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Sentiment Distribution")
        sentiment_counts = filtered_df["Sentiment"].value_counts()
        colors = {"Positive": "#16a34a", "Neutral": "#f59e0b", "Negative": "#dc2626"}
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.pie(
            sentiment_counts.values, labels=sentiment_counts.index, autopct="%1.1f%%",
            colors=[colors.get(s, "#999") for s in sentiment_counts.index],
            wedgeprops=dict(width=0.45), startangle=90,
        )
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Sentiment by Rating")
        cross = pd.crosstab(filtered_df["Rating"], filtered_df["Sentiment"])
        fig, ax = plt.subplots(figsize=(5, 4))
        cross.plot(kind="bar", stacked=True, color=["#dc2626", "#f59e0b", "#16a34a"], ax=ax)
        ax.set_xlabel("Rating")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Sentiment by Category")
    cat_sent = (
        filtered_df.groupby(["Category", "Sentiment"]).size().unstack(fill_value=0)
    )
    cat_sent_pct = (cat_sent.div(cat_sent.sum(axis=1), axis=0) * 100).round(1)
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in cat_sent_pct.columns:
            cat_sent_pct[col] = 0.0
    cat_sent_pct = cat_sent_pct[["Positive", "Neutral", "Negative"]]
    cat_sent_pct.columns = ["Positive %", "Neutral %", "Negative %"]
    st.dataframe(cat_sent_pct.sort_values("Positive %", ascending=False), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 11. PAGE: PRODUCT ANALYSIS (includes Category Analysis section)
# ------------------------------------------------------------------
elif page == "📦 Product Analysis":
    st.subheader("📦 Product Analysis")

    prod_stats = (
        filtered_df.groupby(["Product_Name", "Category"])
        .agg(
            Review_Count=("Review_ID", "count"),
            Average_Rating=("Rating", "mean"),
            Positive_Pct=("Sentiment", lambda s: 100 * (s == "Positive").sum() / len(s)),
            Negative_Pct=("Sentiment", lambda s: 100 * (s == "Negative").sum() / len(s)),
        )
        .reset_index()
        .round(2)
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("📊 Top 10 Products by Number of Reviews")
    st.dataframe(
        prod_stats.sort_values("Review_Count", ascending=False).head(10),
        use_container_width=True, hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("⭐ Top 10 by Average Rating")
        top_rated = prod_stats[prod_stats["Review_Count"] >= 3].sort_values("Average_Rating", ascending=False).head(10)
        st.dataframe(top_rated, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("😊 Top 10 by Positive Sentiment %")
        top_positive = prod_stats[prod_stats["Review_Count"] >= 3].sort_values("Positive_Pct", ascending=False).head(10)
        st.dataframe(top_positive, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("😞 Top 10 by Negative Sentiment %")
    top_negative = prod_stats[prod_stats["Review_Count"] >= 3].sort_values("Negative_Pct", ascending=False).head(10)
    st.dataframe(top_negative, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🗂️ Category Analysis")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Review Count by Category")
        cat_counts = filtered_df["Category"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=cat_counts.values, y=cat_counts.index, color="#4f46e5", ax=ax)
        ax.set_xlabel("Reviews")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Average Rating by Category")
        cat_avg = filtered_df.groupby("Category")["Rating"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=cat_avg.values, y=cat_avg.index, color="#7c3aed", ax=ax)
        ax.set_xlabel("Average Rating")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Positive vs Negative Sentiment by Category")
    cat_sent = filtered_df.groupby(["Category", "Sentiment"]).size().unstack(fill_value=0)
    cat_sent_pct = (cat_sent.div(cat_sent.sum(axis=1), axis=0) * 100).round(1)
    fig, ax = plt.subplots(figsize=(8, 4))
    cat_sent_pct.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0).plot(
        kind="bar", stacked=True, color=["#16a34a", "#f59e0b", "#dc2626"], ax=ax
    )
    ax.set_ylabel("Percent of Reviews")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 12. PAGE: REVIEW EXPLORER
# ------------------------------------------------------------------
elif page == "📝 Review Explorer":
    st.subheader("📝 Review Explorer")

    search_term = st.text_input("🔎 Search Reviews", placeholder="e.g. battery, delivery, quality, price, service")

    explorer_df = filtered_df.copy()
    if search_term:
        mask = (
            explorer_df["Review_Text"].str.contains(search_term, case=False, na=False)
            | explorer_df["Review_Title"].str.contains(search_term, case=False, na=False)
        )
        explorer_df = explorer_df[mask]
        st.caption(f"Found {len(explorer_df):,} reviews containing \"{search_term}\"")

    display_cols = [
        "Review_ID", "Product_Name", "Category", "Rating", "Review_Title",
        "Review_Text", "Sentiment", "Verified_Purchase", "Helpful_Votes",
    ]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.dataframe(explorer_df[display_cols], use_container_width=True, hide_index=True, height=450)

    csv_data = explorer_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Filtered Reviews",
        data=csv_data,
        file_name="filtered_reviews.csv",
        mime="text/csv",
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 13. PAGE: WORD ANALYSIS
# ------------------------------------------------------------------
elif page == "🔤 Word Analysis":
    st.subheader("🔤 Customer Language Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("😊 Top 20 Positive Words")
        pos_words = get_word_frequencies(filtered_df[filtered_df["Sentiment"] == "Positive"]["Review_Text"], 20)
        if pos_words:
            words, counts = zip(*pos_words)
            fig, ax = plt.subplots(figsize=(5, 6))
            sns.barplot(x=list(counts), y=list(words), color="#16a34a", ax=ax)
            ax.set_xlabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("Not enough positive reviews to analyze.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("😞 Top 20 Negative Words")
        neg_words = get_word_frequencies(filtered_df[filtered_df["Sentiment"] == "Negative"]["Review_Text"], 20)
        if neg_words:
            words, counts = zip(*neg_words)
            fig, ax = plt.subplots(figsize=(5, 6))
            sns.barplot(x=list(counts), y=list(words), color="#dc2626", ax=ax)
            ax.set_xlabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("Not enough negative reviews to analyze.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("🔤 Most Frequently Used Words (All Reviews)")
    all_words = get_word_frequencies(filtered_df["Review_Text"], 25)
    if all_words:
        word_df = pd.DataFrame(all_words, columns=["Word", "Frequency"])
        st.dataframe(word_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Optional word cloud (kept lightweight / non-mandatory)
    try:
        from wordcloud import WordCloud
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("☁️ Word Cloud (All Reviews)")
        text_blob = " ".join(remove_stopwords(preprocess_text(t)) for t in filtered_df["Review_Text"])
        if text_blob.strip():
            wc = WordCloud(width=1000, height=400, background_color="white", colormap="viridis").generate(text_blob)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    except ImportError:
        st.caption("💡 Install the optional 'wordcloud' package to see a visual word cloud here.")


# ------------------------------------------------------------------
# 14. PAGE: BUSINESS INSIGHTS
# ------------------------------------------------------------------
elif page == "💡 Business Insights":
    st.subheader("💡 Business Insights")

    # Customer Satisfaction Score
    kpis = calculate_kpis(filtered_df)
    satisfaction_score = kpis["positive_pct"]
    label, color = satisfaction_interpretation(satisfaction_score)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("😊 Customer Satisfaction Score")
    st.markdown(
        f"""
        <div style="display:flex; align-items:baseline; gap:14px;">
            <div style="font-size:2.4rem; font-weight:800; color:{color};">{satisfaction_score}%</div>
            <div style="font-size:1.1rem; font-weight:600; color:{color};">{label}</div>
        </div>
        <p style="color:#6b7280; margin-top:0.4rem;">
            Satisfaction Score = Positive Reviews / Total Reviews × 100
            &nbsp;|&nbsp; 80%+ Excellent · 60-79% Good · 40-59% Needs Improvement · Below 40% Poor
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("📌 Key Business Insights")
    for insight in generate_business_insights(filtered_df):
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("⚠️ Customer Complaints — Top Negative Feedback Areas")
    neg_df = filtered_df[filtered_df["Sentiment"] == "Negative"]
    if not neg_df.empty:
        complaint_table = (
            filtered_df.groupby("Product_Name")
            .agg(
                Negative_Reviews=("Sentiment", lambda s: (s == "Negative").sum()),
                Total_Reviews=("Sentiment", "count"),
                Average_Rating=("Rating", "mean"),
            )
            .assign(Negative_Pct=lambda d: round(100 * d["Negative_Reviews"] / d["Total_Reviews"], 1))
            .query("Total_Reviews >= 3 and Negative_Reviews > 0")
            .sort_values("Negative_Pct", ascending=False)
            .round(2)
            .head(10)
        )
        st.dataframe(complaint_table, use_container_width=True)

        st.markdown("**Most common words in negative reviews:**")
        neg_words = get_word_frequencies(neg_df["Review_Text"], 10)
        st.write(", ".join([w for w, _ in neg_words]))
    else:
        st.info("No negative reviews found in the current filter selection. 🎉")
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 15. PAGE: DATA EXPLORER (Data Quality)
# ------------------------------------------------------------------
elif page == "📊 Data Explorer":
    st.subheader("📊 Data Explorer & Data Quality")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("📊 Data Quality Report (Before Cleaning)")
    dq1, dq2, dq3, dq4, dq5 = st.columns(5)
    kpi_card(dq1, "Total Records", f"{data_quality['total_records']:,}")
    kpi_card(dq2, "Missing Values", f"{data_quality['missing_values']:,}")
    kpi_card(dq3, "Duplicate Records", f"{data_quality['duplicate_records']:,}")
    kpi_card(dq4, "Unique Products", f"{data_quality['unique_products']:,}")
    kpi_card(dq5, "Unique Categories", f"{data_quality['unique_categories']:,}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("✅ Cleaned Dataset Preview")
    st.caption(f"After cleaning: {len(df):,} valid reviews remain (from {data_quality['total_records']:,} raw records).")
    st.dataframe(filtered_df.head(200), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("🧾 Column Summary")
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(t) for t in df.dtypes],
            "Non-Null Count": [df[c].notnull().sum() for c in df.columns],
        }),
        use_container_width=True, hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 16. FOOTER
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>
        📱 <b>Customer Review & Sentiment Analytics</b><br>
        Built with Python • NLP • SQL • Streamlit<br>
        👨‍💻 Developed &amp; Deployed by Prem Kumar<br>
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    </div>
    """,
    unsafe_allow_html=True,
)
