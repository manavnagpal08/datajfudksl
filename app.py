import streamlit as st
import pandas as pd
import pickle
import os
import json 
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import Counter
import re
import time
import random
import numpy as np
from io import StringIO

# --- Configuration and Constants ---
PRODUCTS_FILE = "data/products.csv"
REVIEWS_FILE = "data/reviews.csv"
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
PRODUCTS_COLUMNS = ['id', 'name', 'price', 'region', 'image_url', 'description', 'category'] 

# Sentiment Emojis
POSITIVE_EMOJI = "✨"
NEGATIVE_EMOJI = "⚠️"
NEUTRAL_EMOJI = "⏸️" 

# Category Emojis and map for visual flair
CATEGORY_ICONS = {
    "Electronics": "💻",
    "Clothing & Footwear": "👚",
    "Furniture": "🛋️",
    "Cosmetics": "💄",
    "Groceries": "🍎",
    "Books": "📚",
    "Uncategorized": "📦"
}

# Custom Credentials provided by user
USERS = {
    "admin": {"password": "admin123", "role": "Admin"},
    "manager": {"password": "manager123", "role": "Manager"},
    "user": {"password": "user123", "role": "User"}
}

# --- Internal Review Synthesis Logic & Sentiment Functions ---

def calculate_ema_trend(df_daily_counts, days=30, smoothing_factor=0.2):
    """
    Calculates Exponential Moving Average (EMA) for Positive Rate.
    
    FIX: Ensures 'Positive', 'Neutral', and 'Negative' columns exist 
    to prevent KeyError if a sentiment is missing on all dates.
    """
    if df_daily_counts.empty: return pd.Series()
    
    sentiment_cols = ['Positive', 'Neutral', 'Negative']

    # 1. Fill in missing dates for a smoother time series
    start_date = df_daily_counts['date'].min()
    end_date = df_daily_counts['date'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    df_ts = df_daily_counts.set_index('date')
    df_ts = df_ts.reindex(date_range, fill_value=0).reset_index().rename(columns={'index': 'date'})
    
    # **FIX APPLIED HERE:** Ensure all required sentiment columns exist and are integers
    for col in sentiment_cols:
        if col not in df_ts.columns:
            df_ts[col] = 0
        else:
            # Ensure they are integers
            df_ts[col] = df_ts[col].astype(int) 
            
    # 2. Calculate Daily Positive Rate (handle division by zero)
    # This line now safely accesses 'Positive', 'Neutral', and 'Negative'
    df_ts['Total'] = df_ts['Positive'] + df_ts['Neutral'] + df_ts['Negative']
    df_ts['Daily_Pos_Rate'] = np.where(df_ts['Total'] > 0, (df_ts['Positive'] / df_ts['Total']) * 100, 0)
    
    # 3. Calculate EMA (Weighted average, higher factor = more responsive to new data)
    df_ts['EMA_Pos_Rate'] = df_ts['Daily_Pos_Rate'].ewm(alpha=smoothing_factor, adjust=False).mean()
    
    # 4. Simple lookahead forecast (just extends the last EMA value)
    last_ema = df_ts['EMA_Pos_Rate'].iloc[-1]
    forecast_dates = [(end_date + timedelta(days=i)) for i in range(1, 8)] # Forecast next 7 days
    
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'Daily_Pos_Rate': [np.nan] * 7,
        'EMA_Pos_Rate': [last_ema] * 7,
        'is_forecast': [True] * 7
    })
    
    df_ts['is_forecast'] = False
    return pd.concat([df_ts, forecast_df], ignore_index=True)


def get_top_sentiment_words(df_subset, sentiment, n=3):
    """Calculates top N words for a specific sentiment."""
    if df_subset.empty: return []
    
    stop_words = set([
        'the', 'a', 'an', 'is', 'it', 'and', 'but', 'or', 'to', 'of', 'in', 'for', 
        'with', 'on', 'this', 'that', 'i', 'was', 'my', 'had', 'have', 'very', 'not',
        'would', 'me', 'be', 'so', 'get', 'product', 'item', 'just', 'too', 'great', 
        'good', 'bad', 'best', 'worst', 'really', 'much', 'like', 'for', 'about', 'is', 'i',
        'can', 'will', 'use', 'one', 'get', 'it', 'if', 'this'
    ])
    
    subset = df_subset[df_subset['sentiment'] == sentiment]
    if subset.empty: return []

    text = ' '.join(subset['review'].astype(str).str.lower().tolist())
    words = re.findall(r'\b\w{3,}\b', text) 
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    word_counts = Counter(filtered_words)
    return [word.capitalize() for word, count in word_counts.most_common(n)]

def generate_product_summary_internal(product_name, reviews_df):
    """Generates a synthesized product review summary based *only* on internal analytics."""
    if reviews_df.empty or len(reviews_df) < 5:
        return "Insufficient reviews (requires 5+ reviews) to generate a comprehensive, objective summary."

    total = len(reviews_df)
    pos_count = len(reviews_df[reviews_df['sentiment'] == 'Positive'])
    pos_rate = pos_count / total
    
    # 1. Determine Overall Sentiment Score
    if pos_rate >= 0.8:
        overall_sentiment = "Outstanding"
        sentiment_description = "The customer reception is overwhelmingly positive, marking this as a top-tier product."
    elif pos_rate >= 0.6:
        overall_sentiment = "Strong"
        sentiment_description = "The feedback is mostly positive, suggesting high customer satisfaction with minor caveats."
    elif pos_rate >= 0.4:
        overall_sentiment = "Mixed/Neutral"
        sentiment_description = "The product receives balanced feedback, with strengths and weaknesses equally highlighted by customers."
    else:
        overall_sentiment = "Critical"
        sentiment_description = "A significant portion of reviews express negative experiences, indicating serious issues need immediate attention."

    # 2. Identify Strengths (Top positive keywords)
    top_pos_words = get_top_sentiment_words(reviews_df, 'Positive', n=3)
    if top_pos_words:
        strengths = f"Key strengths frequently mentioned by positive reviewers include **{', '.join(top_pos_words[:-1])}**, and **{top_pos_words[-1]}**."
    else:
        strengths = "No distinct positive features were clearly highlighted by reviewers."

    # 3. Identify Weaknesses (Top negative keywords)
    top_neg_words = get_top_sentiment_words(reviews_df, 'Negative', n=3)
    if top_neg_words:
        weaknesses = f"The primary areas for improvement, according to negative feedback, concern **{', '.join(top_neg_words[:-1])}**, and **{top_neg_words[-1]}**."
    else:
        weaknesses = "No specific critical issues were clearly highlighted in negative feedback."


    # 4. Synthesize Final Paragraph
    final_summary = (
        f"<p style='font-size: 1.1em; line-height: 1.6;'><b>Overall Assessment: <span style='color:#4338ca;'>{overall_sentiment}</span></b> ({pos_count} Positive reviews out of {total}). "
        f"{sentiment_description} "
        f"{strengths} "
        f"{weaknesses}</p>"
    )
    
    return final_summary


# --- Aesthetics (Custom CSS for Light Mode ONLY) ---
def apply_theme_css():
    """Applies custom CSS hardcoded for Light Mode."""
    
    # Define LIGHT MODE color variables
    bg_color = "#f9fafb"
    text_color = "#1f2937"
    card_bg = "#ffffff"
    card_shadow = "0 25px 50px -12px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(0, 0, 0, 0.05)"
    login_bg = "radial-gradient(circle at 10% 20%, rgba(203, 213, 225, 0.4) 0%, #f9fafb 100%)"
    accent_color = "#4338ca"
    hr_color = "#e5e7eb"
    score_bar_bg = "#f3f4f6"
    pos_color = "#059669" # Emerald 600
    neg_color = "#dc2626" # Red 600
    neu_color = "#f59e0b" # Amber 500


    st.markdown(f"""
    <style>
        /* Global Styling */
        .stApp, .stApp > header {{
            background-color: {bg_color};
            color: {text_color};
            font-family: 'Inter', sans-serif;
        }}
        h1, h2, h3, h4, .st-emotion-cache-10trblm {{
            color: {text_color} !important;
            font-weight: 700;
        }}
        h1 {{
            color: {accent_color} !important; 
            border-bottom: 4px solid {accent_color}; 
            padding-bottom: 15px;
            margin-top: 0;
        }}
        
        /* Login Page Styling */
        .login-container {{ background: {login_bg}; }}
        .login-box {{
            background-color: {card_bg};
            box-shadow: {card_shadow}; 
            border-top: 5px solid {accent_color}; 
        }}
        
        /* Product Card Styling with Enhanced Hover */
        .product-card {{
            border: 1px solid {hr_color}; 
            background-color: {card_bg};
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02); 
            transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.3s ease-out; 
        }}
        .product-card:hover {{
            transform: translateY(-10px); 
            box-shadow: 0 25px 50px -12px rgba(67,56,202,0.25); 
            border: 1px solid {accent_color};
        }}

        /* Custom button styling (Primary action) */
        .stButton>button {{
            color: white !important;
            background-color: {accent_color}; 
            box-shadow: 0 4px #2e288e; /* Darker shade for 3D effect */
        }}
        .stButton>button:hover {{
            background-color: {accent_color}; 
            transform: translateY(-2px);
            box-shadow: 0 6px #2e288e; 
        }}
        
        /* Sentiment Colors */
        .pos-text {{ color: {pos_color}; font-weight: bold; }}
        .neg-text {{ color: {neg_color}; font-weight: bold; }}
        .neu-text {{ color: {neu_color}; font-weight: bold; }}

        /* Custom Metrics Boxes */
        .metric-box {{
            background-color: {card_bg};
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }}
        .total-metric {{ border-left: 6px solid {accent_color}; }}
        
        /* Card Review Score Bar */
        .score-bar {{
            background-color: {score_bar_bg}; 
            border: 1px solid {hr_color};
        }}
        
        /* General Separator */
        hr {{ border-top: 1px solid {hr_color}; }}

        /* Keyword Highlighting */
        .highlight-pos {{ background-color: rgba(5, 150, 105, 0.2); padding: 2px 4px; border-radius: 4px; }}
        .highlight-neg {{ background-color: rgba(220, 38, 38, 0.2); padding: 2px 4px; border-radius: 4px; }}

    </style>
    """, unsafe_allow_html=True)


# ----------------------------
# Data Loading and Utility Functions
# ----------------------------

def generate_sample_products(current_max_id):
    """Generates sample product data with defined categories if the file is empty."""
    next_id = current_max_id + 1
    sample_data = [
        {'id': next_id, 'name': 'Ultra-HD Monitor', 'price': 34999.00, 'region': 'North', 'image_url': 'https://placehold.co/150x150/4338ca/ffffff?text=Monitor', 'description': '4K monitor with high refresh rate.', 'category': 'Electronics'},
        {'id': next_id + 1, 'name': 'Organic Cotton T-Shirt', 'price': 1299.00, 'region': 'South', 'image_url': 'https://placehold.co/150x150/059669/ffffff?text=T-Shirt', 'description': 'Sustainable and comfortable wear.', 'category': 'Clothing & Footwear'},
        {'id': next_id + 2, 'name': 'Luxury Leather Sofa', 'price': 129000.00, 'region': 'West', 'image_url': 'https://placehold.co/150x150/f97316/ffffff?text=Sofa', 'description': 'Three-seater genuine leather.', 'category': 'Furniture'},
        {'id': next_id + 3, 'name': 'Moisturizing Cream Set', 'price': 4500.00, 'region': 'East', 'image_url': 'https://placehold.co/150x150/ec4899/ffffff?text=Cream', 'description': 'Day and night moisturizing routine.', 'category': 'Cosmetics'},
        {'id': next_id + 4, 'name': 'Assorted Fresh Fruit Box', 'price': 999.00, 'region': 'North', 'image_url': 'https://placehold.co/150x150/f59e0b/ffffff?text=Fruit', 'description': 'Weekly box of seasonal fruits.', 'category': 'Groceries'},
        {'id': next_id + 5, 'name': 'The Great Classic Novel', 'price': 750.00, 'region': 'South', 'image_url': 'https://placehold.co/150x150/a855f7/ffffff?text=Book', 'description': 'A must-read for all students.', 'category': 'Books'},
    ]
    return pd.DataFrame(sample_data)

def generate_initial_reviews(df_products):
    """Generates initial reviews for sample data with new columns."""
    reviews = []
    
    # Ensure data is ready for iteration
    if df_products.empty:
        return pd.DataFrame(columns=['product_id', 'review', 'sentiment', 'timestamp', 'upvotes', 'downvotes', 'reviewer_type'])

    product_ids = df_products['id'].tolist()
    
    base_reviews = {
        'Ultra-HD Monitor': [
            ("Picture quality is stunning!", "Positive", 5),
            ("Overpriced for the features.", "Negative", 1),
            ("Solid performance, no complaints.", "Positive", 3),
        ],
        'Organic Cotton T-Shirt': [
            ("Very comfortable and fits perfectly.", "Positive", 8),
            ("Shrank slightly after first wash.", "Negative", 2),
        ],
        'Luxury Leather Sofa': [
            ("Amazing centerpiece for my living room.", "Positive", 12),
            ("Delivery was slow, but the sofa is great.", "Neutral", 4),
        ],
        'Moisturizing Cream Set': [
            ("My skin feels fantastic, highly recommend!", "Positive", 15),
        ],
    }

    for idx, row in df_products.iterrows():
        product_name = row['name']
        product_id = row['id']
        
        # Use specific reviews if available, otherwise general ones
        review_list = base_reviews.get(product_name, [
            ("Good value for money.", "Positive", 3),
            ("It was okay, nothing special.", "Neutral", 1),
            ("Disappointed with the quality.", "Negative", 2),
        ])
        
        for review_text, sentiment, num_reviews in review_list:
            for _ in range(num_reviews):
                reviews.append({
                    'product_id': product_id,
                    'review': review_text,
                    'sentiment': sentiment,
                    'timestamp': datetime.now() - timedelta(days=random.randint(1, 30), hours=random.randint(1, 24)),
                    'upvotes': random.randint(0, 10),
                    'downvotes': random.randint(0, 5),
                    'reviewer_type': random.choice(['Verified Buyer', 'Guest']),
                    'manager_reply': None
                })
    
    df = pd.DataFrame(reviews)
    
    # Ensure all new columns are present
    if 'upvotes' not in df.columns: df['upvotes'] = 0
    if 'downvotes' not in df.columns: df['downvotes'] = 0
    if 'reviewer_type' not in df.columns: df['reviewer_type'] = random.choice(['Verified Buyer', 'Guest'])
    if 'manager_reply' not in df.columns: df['manager_reply'] = None
    
    return df


@st.cache_data(show_spinner="Loading Data...")
def load_initial_data():
    """Loads and initializes products and reviews DataFrames."""
    os.makedirs("data", exist_ok=True)
    
    # Load products, ensuring 'category' column exists
    df_products = pd.read_csv(PRODUCTS_FILE) if os.path.exists(PRODUCTS_FILE) and os.path.getsize(PRODUCTS_FILE) > 0 else pd.DataFrame(columns=PRODUCTS_COLUMNS)
    
    # --- FIX: Ensure Product IDs are unique and non-zero to prevent gauge key collision ---
    # 1. Convert to numeric, errors -> NaN (which becomes None in Int64)
    df_products['id'] = df_products['id'].astype(str).str.strip()
    
    # 2. Determine the maximum existing valid ID
    valid_ids = df_products['id'].dropna()
    # Start max_id at 0, or 1 if a product ID max is 0 (to avoid starting new IDs at 1, 2, 3...)
    current_max_id = valid_ids.max() if not valid_ids.empty else 0
    if current_max_id == 0 and not valid_ids.empty: current_max_id = 1
    
    # 3. Assign new unique IDs to rows where ID is missing, invalid (NaN/None), or 0.
    for index, row in df_products.iterrows():
        if pd.isna(row['id']) or row['id'] == 0:
            current_max_id += 1
            df_products.loc[index, 'id'] = current_max_id
    # --- END FIX ---

    # Ensure all required product columns are present
    for col in PRODUCTS_COLUMNS:
        if col not in df_products.columns:
            default_value = 'Uncategorized' if col == 'category' else None
            df_products.loc[:, col] = default_value
    
    # Generate sample data if the product file is truly empty
    max_id = df_products['id'].max() if not df_products.empty else 0
    if df_products.empty or len(df_products.dropna(how='all')) == 0:
        df_products = generate_sample_products(max_id)
        max_id = df_products['id'].max()

    df_products['category'] = df_products['category'].fillna('Uncategorized').astype(str)
    
    # Load reviews with new columns
    REVIEW_COLUMNS = ['product_id', 'review', 'sentiment', 'timestamp', 'upvotes', 'downvotes', 'reviewer_type', 'manager_reply'] 
    df_reviews = pd.DataFrame(columns=REVIEW_COLUMNS)
    
    if os.path.exists(REVIEWS_FILE) and os.path.getsize(REVIEWS_FILE) > 0:
        try:
            loaded_df = pd.read_csv(REVIEWS_FILE)
            for col in REVIEW_COLUMNS:
                if col not in loaded_df.columns:
                    if col in ['upvotes', 'downvotes']: loaded_df[col] = 0
                    elif col == 'reviewer_type': loaded_df[col] = random.choice(['Verified Buyer', 'Guest'])
                    elif col == 'manager_reply': loaded_df[col] = None
                    else: loaded_df[col] = None
            df_reviews = loaded_df
        except Exception as e:
            st.warning(f"Error loading reviews: {e}. Starting with sample reviews.")
            
    df_reviews['product_id'] = df_reviews['product_id'].astype(str).str.strip()
    df_reviews['timestamp'] = pd.to_datetime(df_reviews['timestamp'], errors='coerce').fillna(pd.to_datetime('2024-01-01 00:00:00'))
    df_reviews['upvotes'] = df_reviews['upvotes'].astype(int)
    df_reviews['downvotes'] = df_reviews['downvotes'].astype(int)
    df_reviews['reviewer_type'] = df_reviews['reviewer_type'].fillna(random.choice(['Verified Buyer', 'Guest']))

    # Generate sample reviews if necessary
    if df_reviews.empty or len(df_reviews.dropna(how='all')) < 10:
        df_reviews = generate_initial_reviews(df_products)
        
    # Final save to ensure integrity
    df_reviews.to_csv(REVIEWS_FILE, index=False)
    df_products.to_csv(PRODUCTS_FILE, index=False) 

    return df_products, df_reviews

def save_reviews():
    """Saves the reviews DataFrame from session state to CSV."""
    st.session_state['df_reviews'].to_csv(REVIEWS_FILE, index=False)
    
def save_products():
    """Saves the products DataFrame from session state to CSV."""
    st.session_state['df_products'].to_csv(PRODUCTS_FILE, index=False)

@st.cache_data(show_spinner="Loading Model...")
def load_model_and_vectorizer():
    """Loads the sentiment model and vectorizer."""
    try:
        vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
        clf = pickle.load(open(MODEL_PATH, "rb"))
        return vectorizer, clf
    except FileNotFoundError:
        return None, None

def predict_sentiment(text, vectorizer, clf):
    """Uses the loaded model to predict sentiment."""
    try:
        X = vectorizer.transform([text])
        return clf.predict(X)[0]
    except Exception:
        return "Model Error"

def highlight_review_keywords(review_text, product_id):
    """Highlights top positive/negative words in a review."""
    
    # Get top 5 words for both sentiments across all reviews for this product
    product_reviews = st.session_state['df_reviews'][st.session_state['df_reviews']['product_id'] == product_id]
    pos_words = [w.lower() for w in get_top_sentiment_words(product_reviews, 'Positive', n=5)]
    neg_words = [w.lower() for w in get_top_sentiment_words(product_reviews, 'Negative', n=5)]

    words = review_text.split()
    highlighted_review = []

    for word in words:
        # Clean word for comparison (remove punctuation, lower case)
        clean_word = re.sub(r'\W+', '', word).lower()
        
        if clean_word in pos_words:
            highlighted_review.append(f"<span class='highlight-pos'>{word}</span>")
        elif clean_word in neg_words:
            highlighted_review.append(f"<span class='highlight-neg'>{word}</span>")
        else:
            highlighted_review.append(word)

    return " ".join(highlighted_review)


# ----------------------------
# Session State Initialization & Setup
# ----------------------------

# Initialize all required keys explicitly to prevent KeyErrors on first run/rerun
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_role' not in st.session_state:
    st.session_state['current_role'] = 'Guest'
if 'show_detail_id' not in st.session_state:
    st.session_state['show_detail_id'] = None 
if 'product_summary_cache' not in st.session_state:
    st.session_state['product_summary_cache'] = {} 
if 'auto_refresh' not in st.session_state:
    st.session_state['auto_refresh'] = False
if 'last_refresh_time' not in st.session_state:
    st.session_state['last_refresh_time'] = datetime.now()


# Initialize DataFrames into Session State
if 'df_products' not in st.session_state or 'df_reviews' not in st.session_state:
    st.session_state['df_products'], st.session_state['df_reviews'] = load_initial_data()

# Load model once and store in state
if 'vectorizer' not in st.session_state or 'clf' not in st.session_state:
    st.session_state['vectorizer'], st.session_state['clf'] = load_model_and_vectorizer()

# Data access shortcuts
df_products = st.session_state['df_products']
df_reviews = st.session_state['df_reviews']

model_ready = st.session_state['vectorizer'] is not None and st.session_state['clf'] is not None
if not model_ready:
    st.error("🚨 Sentiment Model Not Found! Prediction functionality is disabled.")

# Apply theme CSS (now hardcoded for Light Mode)
apply_theme_css()

# --- Auto Refresh Logic (Runs before content display) ---
if st.session_state.get('auto_refresh', False):
    refresh_interval = 10 # Seconds
    time_since_last_refresh = (datetime.now() - st.session_state['last_refresh_time']).total_seconds()
    
    if time_since_last_refresh >= refresh_interval:
        st.session_state['last_refresh_time'] = datetime.now()
        # Clear specific caches that rely on state and rerun
        st.cache_data.clear()
        # This will trigger a full script execution including data loading/processing
        st.rerun() 
    else:
        # Schedule the next rerun
        time.sleep(refresh_interval - time_since_last_refresh)
        st.session_state['last_refresh_time'] = datetime.now() # Reset time to prevent immediate re-rerun
        st.rerun()


# ----------------------------
# Authentication
# ----------------------------

def main_login_screen():
    """Renders the central, polished login interface."""
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown("<h2 style='text-align: center; color: #4338ca;'>📈 E-Commerce Analytics Hub</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6b7280; margin-bottom: 25px;'>Securely access your data insights.</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        username = st.text_input("Username", key="login_username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
        
        submitted = st.form_submit_button("Secure Login")
        
        if submitted:
            if username in USERS and USERS[username]["password"] == password:
                role = USERS[username]["role"]
                st.session_state['logged_in'] = True
                st.session_state['current_role'] = role
                st.success(f"Logged in successfully as **{role}**! Redirecting...")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
def logout():
    """Logs out the current user."""
    st.session_state['logged_in'] = False
    st.session_state['current_role'] = 'Guest'
    st.session_state['show_detail_id'] = None
    st.session_state['product_summary_cache'] = {}
    st.info("You have been logged out.")
    time.sleep(0.5)
    st.rerun()

# ----------------------------
# Review Voting and Reply Logic
# ----------------------------
def handle_vote(index, vote_type):
    """Increments upvote or downvote count for a review."""
    df = st.session_state['df_reviews']
    if vote_type == 'up':
        df.loc[index, 'upvotes'] += 1
        st.toast("Upvoted! Thanks for your feedback.")
    elif vote_type == 'down':
        df.loc[index, 'downvotes'] += 1
        st.toast("Downvoted! We appreciate your input.")
    
    save_reviews()
    # Rerunning is optional, but often nice to see the number update
    st.session_state['product_summary_cache'] = {}
    st.rerun()

def handle_manager_reply(index, reply_text):
    """Adds a manager reply to a specific review."""
    df = st.session_state['df_reviews']
    df.loc[index, 'manager_reply'] = reply_text.strip()
    save_reviews()
    st.success("Manager reply posted.")
    st.session_state['product_summary_cache'] = {}
    st.rerun()


# ----------------------------
# Product Detail View Function
# ----------------------------

def show_product_detail(product_id):
    """Shows detailed analytics for a single product."""
    product = df_products[df_products['id'] == product_id].iloc[0]
    product_reviews = df_reviews[df_reviews['product_id'] == product_id]
    
    icon = CATEGORY_ICONS.get(product['category'], '📦')
    st.header(f"Product Detail Analysis: {icon} {product['name']} (ID: {product_id})")
    
    col_back, col_toggle = st.columns([1, 4])
    col_back.button("← Back to Catalog", on_click=lambda: st.session_state.update({'show_detail_id': None}))
    
    if product_reviews.empty:
        st.warning("No reviews available for detailed analysis yet.")
        return

    # Split layout for product info and summary
    col_img, col_summary = st.columns([1, 2])
    
    with col_img:
        st.image(product['image_url'], caption=f"{product['name']}", width=250, use_container_width='auto', output_format='PNG',)
        st.markdown(f"**Price:** ₹{product['price']:.2f}")
        st.markdown(f"**Region:** {product['region']}")
        st.markdown(f"**Category:** **{product['category']}**")

    with col_summary:
        st.subheader("🤖 Internal Synthesis & Projected Sentiment Analysis")
        
        summary_placeholder = st.empty()
        if product_id not in st.session_state['product_summary_cache']:
            with summary_placeholder:
                with st.spinner("Analyzing all reviews and synthesizing summary..."):
                    summary = generate_product_summary_internal(product['name'], product_reviews)
                    st.session_state['product_summary_cache'][product_id] = summary
        
        summary_placeholder.markdown(st.session_state['product_summary_cache'][product_id], unsafe_allow_html=True)

    # 3. Product Summary and Metrics
    st.markdown("---")
    st.subheader("Review Metrics Breakdown")

    total_reviews = len(product_reviews)
    sentiment_counts = product_reviews['sentiment'].value_counts().to_dict()
    pos_count = sentiment_counts.get('Positive', 0)
    neu_count = sentiment_counts.get('Neutral', 0)
    neg_count = sentiment_counts.get('Negative', 0)
    
    pos_p = (pos_count / total_reviews) * 100 if total_reviews > 0 else 0
    neu_p = (neu_count / total_reviews) * 100 if total_reviews > 0 else 0
    neg_p = (neg_count / total_reviews) * 100 if total_reviews > 0 else 0
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    col_m1.markdown(f'<div class="metric-box total-metric">Total Reviews<br><b>{total_reviews}</b></div>', unsafe_allow_html=True)
    col_m2.markdown(f'<div class="metric-box pos-metric">{POSITIVE_EMOJI} Positive Rate<br><b class="pos-text">{pos_p:.1f}%</b></div>', unsafe_allow_html=True)
    col_m3.markdown(f'<div class="metric-box">{NEUTRAL_EMOJI} Neutral Count<br><b class="neu-text">{neu_count}</b></div>', unsafe_allow_html=True)
    col_m4.markdown(f'<div class="metric-box neg-metric">{NEGATIVE_EMOJI} Negative Count<br><b class="neg-text">{neg_count}</b></div>', unsafe_allow_html=True)


    # 4. Time Series, Keywords, and Trend Forecast
    st.markdown("---")
    st.subheader("Time Trend & Forecasting")
    
    product_reviews_copy = product_reviews.copy()
    product_reviews_copy['date'] = product_reviews_copy['timestamp'].dt.normalize()
    # Ensure fill_value=0 is used after unstack to handle dates with no reviews
    time_series = product_reviews_copy.groupby(['date', 'sentiment']).size().unstack(fill_value=0).reset_index()

    ema_df = calculate_ema_trend(time_series)
    
    col_time, col_forecast = st.columns(2)
    
    with col_time:
        st.markdown("#### Daily Sentiment Trend")
        
        # Ensure 'Positive', 'Neutral', 'Negative' columns exist for plotting, filling with 0 if missing
        plot_cols = ['Positive', 'Neutral', 'Negative']
        for col in plot_cols:
            if col not in time_series.columns:
                time_series[col] = 0
                
        df_plot = time_series.melt(id_vars='date', value_vars=plot_cols, var_name='sentiment', value_name='count')
        
        fig_time = px.line(df_plot, x='date', y='count', color='sentiment',
                           color_discrete_map={'Positive':'#059669','Neutral':'#f59e0b','Negative':'#dc2626'})
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col_forecast:
        st.markdown("#### Positive Rate Trend Forecast (EMA)")
        fig_forecast = go.Figure()
        
        # Historical EMA Line
        fig_forecast.add_trace(go.Scatter(
            x=ema_df[~ema_df['is_forecast']]['date'], 
            y=ema_df[~ema_df['is_forecast']]['EMA_Pos_Rate'],
            mode='lines',
            name='Historical Trend (EMA)',
            line=dict(color='#4338ca', width=3)
        ))
        
        # Forecast Line
        fig_forecast.add_trace(go.Scatter(
            x=ema_df['date'], 
            y=ema_df['EMA_Pos_Rate'],
            mode='lines',
            name='Forecast',
            line=dict(color='#a5b4fc', width=2, dash='dash')
        ))

        fig_forecast.update_layout(
            title='7-Day Positive Sentiment Rate Trend & Forecast',
            yaxis_title='Positive Rate (%)',
            xaxis_title='Date'
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    # 5. Review Section (including voting and replies)
    st.markdown("---")
    st.subheader("Individual Reviews")
    
    # Sort by upvotes/newest
    product_reviews = product_reviews.sort_values(by=['upvotes', 'timestamp'], ascending=[False, False])

    for index, review_row in product_reviews.iterrows():
        col_type, col_meta = st.columns([1, 4])
        
        # Reviewer Type Badge
        type_color = "#34d399" if review_row['reviewer_type'] == 'Verified Buyer' else "#f59e0b"
        col_type.markdown(f"<span style='background-color: {type_color}; color: black; padding: 4px 8px; border-radius: 8px; font-weight: bold; font-size: 0.8em;'>{review_row['reviewer_type']}</span>", unsafe_allow_html=True)

        sentiment_color = "#059669" if review_row['sentiment'] == 'Positive' else "#dc2626" if review_row['sentiment'] == 'Negative' else "#f59e0b"
        col_meta.markdown(f"**Sentiment:** <span style='color: {sentiment_color}; font-weight: bold;'>{review_row['sentiment']}</span> | *{review_row['timestamp'].strftime('%Y-%m-%d')}*", unsafe_allow_html=True)
        
        # Display review with keyword highlighting
        highlighted_review_html = highlight_review_keywords(review_row['review'], product_id)
        st.markdown(f"<p style='margin-top: 10px; font-size: 1.1em;'>{highlighted_review_html}</p>", unsafe_allow_html=True)
        
        # Voting Buttons
        col_up, col_down, col_reply_btn, col_spacer = st.columns([1, 1, 1.5, 6.5])
        
        col_up.button(f"👍 {review_row['upvotes']}", key=f"up_{index}", on_click=handle_vote, args=(index, 'up',))
        col_down.button(f"👎 {review_row['downvotes']}", key=f"down_{index}", on_click=handle_vote, args=(index, 'down',))
        
        # Manager Reply logic
        is_manager = st.session_state['current_role'] in ['Admin', 'Manager']
        
        if review_row['manager_reply']:
            st.info(f"Brand Manager Reply: {review_row['manager_reply']}")
        elif is_manager:
            # Initialize reply form state if not exists
            if f'show_reply_form_{index}' not in st.session_state:
                st.session_state[f'show_reply_form_{index}'] = False
                
            if col_reply_btn.button("Add Reply", key=f"reply_btn_{index}"):
                st.session_state[f'show_reply_form_{index}'] = not st.session_state.get(f'show_reply_form_{index}', False)
            
            if st.session_state.get(f'show_reply_form_{index}'):
                with st.container(border=True):
                    reply_text = st.text_area("Your Manager Response:", key=f"reply_text_{index}", height=50)
                    if st.button("Post Reply", key=f"post_reply_{index}"):
                        if reply_text.strip():
                            handle_manager_reply(index, reply_text)
                        else:
                            st.warning("Reply cannot be empty.")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
# ----------------------------
# Main Application Flow
# ----------------------------

if not st.session_state['logged_in']:
    main_login_screen()
else:
    # --- Sidebar Controls ---
    st.sidebar.markdown(f"### 👋 Welcome, **{st.session_state['current_role']}**!")
    
    # NOTE: Dark Mode Toggle removed here to enforce Light Mode via CSS
    
    st.sidebar.markdown("---")
    
    st.session_state['auto_refresh'] = st.sidebar.toggle("🔄 Auto-Refresh Every 10s", value=st.session_state.get('auto_refresh', False), key="auto_refresh_toggle")
    
    if st.sidebar.button("Logout", key="logout_btn"):
        logout()
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Products", len(df_products))
    st.sidebar.metric("Total Reviews", len(df_reviews))
    st.sidebar.markdown("---")

    if st.session_state['show_detail_id'] is not None:
        show_product_detail(st.session_state['show_detail_id'])
        
    else:
        st.title("🛒 E-Commerce Platform with Interactive Sentiment Analytics")

        # --- ADMIN PANEL ---
        if st.session_state['current_role'] == "Admin":
            with st.expander("👑 Administrator Tools & Alerts", expanded=False):
                st.markdown("### Product Catalog Inline Editor")
                st.info("Edit, add, or delete products directly in the table below. Changes are saved automatically when you confirm.")
                
                # Inline Editing for Admin
                edited_df = st.data_editor(
                    st.session_state['df_products'],
                    key="product_catalog_editor",
                    num_rows="dynamic",
                    hide_index=True,
                    column_config={
                        "id": st.column_config.NumberColumn("Product ID", disabled=True),
                        "price": st.column_config.NumberColumn("Price (₹)", format="%.2f"),
                        "category": st.column_config.SelectboxColumn("Category", options=list(CATEGORY_ICONS.keys()))
                    }
                )
                
                if st.button("Confirm & Save Catalog Changes", key="save_catalog_btn"):
                    # Process and save the edited DataFrame
                    
                    # 1. Handle new rows (assign new max ID)
                    new_rows = edited_df[edited_df['id'].isnull()]
                    current_max_id = st.session_state['df_products']['id'].max() if not st.session_state['df_products'].empty else 0
                    
                    for index, row in new_rows.iterrows():
                        current_max_id += 1
                        edited_df.loc[index, 'id'] = current_max_id
                    
                    # 2. Update session state and save
                    st.session_state['df_products'] = edited_df.dropna(subset=['name', 'price']) # remove rows with no name/price
                    save_products()
                    st.success("Product Catalog updated successfully.")
                    st.rerun()

                st.markdown("---")
                st.markdown("### Review Data Management")
                col_up, col_alert, col_export = st.columns(3)
                
                # Bulk Review Upload
                with col_up:
                    uploaded_file = st.file_uploader("Upload New Reviews (.csv)", type="csv")
                    if uploaded_file is not None:
                        try:
                            new_reviews_df = pd.read_csv(uploaded_file)
                            
                            # Standardize columns and add missing ones
                            required_cols = ['product_id', 'review', 'sentiment']
                            if all(col in new_reviews_df.columns for col in required_cols):
                                
                                new_reviews_df['timestamp'] = datetime.now()
                                new_reviews_df['upvotes'] = 0
                                new_reviews_df['downvotes'] = 0
                                new_reviews_df['reviewer_type'] = 'Guest'
                                new_reviews_df['manager_reply'] = None
                                
                                # Ensure correct data types
                                new_reviews_df['product_id'] = pd.to_numeric(new_reviews_df['product_id'], errors='coerce').fillna(0).astype('Int64')
                                
                                st.session_state['df_reviews'] = pd.concat([st.session_state['df_reviews'], new_reviews_df], ignore_index=True)
                                save_reviews()
                                st.success(f"Successfully imported {len(new_reviews_df)} new reviews!")
                                st.rerun()
                            else:
                                st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                        except Exception as e:
                            st.error(f"Failed to read CSV: {e}")
                            
                # Anomaly Alert Logic
                with col_alert:
                    st.markdown("#### Anomaly Alerts (Sentiment Drop)")
                    
                    today = datetime.now().date()
                    seven_days_ago = today - timedelta(days=7)
                    fourteen_days_ago = today - timedelta(days=14)
                    
                    # 1. Calculate Neg_Percent for Last 7 days
                    last_week_reviews = df_reviews[df_reviews['timestamp'].dt.date > seven_days_ago]
                    last_week_stats = last_week_reviews.groupby('product_id')['sentiment'].value_counts().unstack(fill_value=0)
                    last_week_stats['Total'] = last_week_stats.sum(axis=1)
                    last_week_stats['Neg_Rate_LW'] = (last_week_stats.get('Negative', 0) / last_week_stats['Total']) * 100
                    
                    # 2. Calculate Neg_Percent for Previous 7 days
                    prev_week_reviews = df_reviews[(df_reviews['timestamp'].dt.date <= seven_days_ago) & (df_reviews['timestamp'].dt.date > fourteen_days_ago)]
                    prev_week_stats = prev_week_reviews.groupby('product_id')['sentiment'].value_counts().unstack(fill_value=0)
                    prev_week_stats['Total'] = prev_week_stats.sum(axis=1)
                    prev_week_stats['Neg_Rate_PW'] = (prev_week_stats.get('Negative', 0) / prev_week_stats['Total']) * 100

                    # 3. Merge and Compare
                    anomaly_check = last_week_stats.merge(
                        prev_week_stats['Neg_Rate_PW'], 
                        left_index=True, 
                        right_index=True, 
                        how='inner'
                    ).fillna(0)
                    
                    anomaly_check['Neg_Rate_Change'] = anomaly_check['Neg_Rate_LW'] - anomaly_check['Neg_Rate_PW']
                    anomaly_check = anomaly_check[
                        (anomaly_check['Neg_Rate_Change'] > 15) & # Negative Rate increased by more than 15%
                        (anomaly_check['Total'] >= 5) # Ensure sufficient review volume
                    ].sort_values(by='Neg_Rate_Change', ascending=False)
                    
                    if not anomaly_check.empty:
                        for product_id in anomaly_check.index.tolist():
                            product_name = df_products[df_products['id'] == product_id]['name'].iloc[0]
                            change = anomaly_check.loc[product_id, 'Neg_Rate_Change']
                            st.error(f"🚨 **{product_name}**: Negativity up {change:.1f}%!")
                    else:
                        st.success("No critical sentiment anomalies detected.")
                
                # Export to CSV
                with col_export:
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv_data = convert_df_to_csv(st.session_state['df_reviews'])
                    st.download_button(
                        label="Export All Reviews to CSV",
                        data=csv_data,
                        file_name=f'reviews_export_{datetime.now().strftime("%Y%m%d")}.csv',
                        mime='text/csv',
                    )


        st.header("🛍 Product Catalog")
        st.markdown("---")

        # --- Product Listing and Filters (Updated for Gauge and Better Metrics) ---
        
        # Calculate sentiment metrics once
        sentiment_groups = None 
        if not df_reviews.empty:
            sentiment_groups = df_reviews.groupby('product_id')['sentiment'].value_counts().unstack(fill_value=0)
            sentiment_groups['Total'] = sentiment_groups.sum(axis=1)
            sentiment_groups['Pos_Percent'] = (sentiment_groups.get('Positive', 0) / sentiment_groups['Total']) * 100
            sentiment_groups['Neg_Percent'] = (sentiment_groups.get('Negative', 0) / sentiment_groups['Total']) * 100
            
            # Merge sentiment into products
            display_products = st.session_state['df_products'].merge(
                sentiment_groups[['Pos_Percent', 'Neg_Percent', 'Total']].rename(columns={'Total': 'Review_Count'}), 
                left_on='id', 
                right_index=True, 
                how='left'
            ).fillna({'Pos_Percent': 0, 'Neg_Percent': 0, 'Review_Count': 0})
        else:
            display_products = st.session_state['df_products'].copy()
            display_products['Pos_Percent'] = 0
            display_products['Neg_Percent'] = 0
            display_products['Review_Count'] = 0

        # Interactive Filter and Search - Now 5 columns
        col_filter_region, col_filter_category, col_sort, col_sentiment, col_search = st.columns([1, 2, 1, 1, 2])
        
        with col_filter_region:
            region_filter = st.selectbox("🌎 Region", ["All"] + sorted(st.session_state['df_products']['region'].astype(str).unique().tolist()))

        with col_filter_category:
            unique_categories = st.session_state['df_products']['category'].astype(str).unique().tolist()
            def format_category_with_icon(cat):
                icon = CATEGORY_ICONS.get(cat, CATEGORY_ICONS['Uncategorized'])
                return f"{icon} {cat}"
            category_filter = st.selectbox(
                "🎯 Category", 
                ["All"] + sorted(unique_categories), 
                format_func=lambda x: format_category_with_icon(x) if x != "All" else "All Categories"
            )

        with col_sort:
            sort_option = st.selectbox("↕️ Sort By", ["ID", "Price (L-H)", "Price (H-L)", "Positive Rate"])
        
        with col_sentiment:
            min_pos_percent = st.slider("Min Pos. %", 0, 100, 0, step=5)
        
        with col_search:
            search_query = st.text_input("🔍 Search (Name/Desc)", "")
        
        
        # --- Apply Filters ---
        
        display_products = display_products[display_products['Pos_Percent'] >= min_pos_percent]

        if region_filter != "All":
            display_products = display_products[display_products['region'].astype(str) == region_filter]
        
        if category_filter != "All":
            display_products = display_products[display_products['category'].astype(str) == category_filter]


        if search_query:
            search_query = search_query.lower()
            display_products = display_products[
                display_products['name'].astype(str).str.lower().str.contains(search_query, na=False) |
                display_products['description'].astype(str).str.lower().str.contains(search_query, na=False)
            ]

        if sort_option == "Price (L-H)":
            display_products = display_products.sort_values(by='price', ascending=True)
        elif sort_option == "Price (H-L)":
            display_products = display_products.sort_values(by='price', ascending=False)
        elif sort_option == "Positive Rate":
            display_products = display_products.sort_values(by='Pos_Percent', ascending=False)
        else:
            display_products = display_products.sort_values(by='id')

        # --- Product Display ---
        st.markdown("<hr style='border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
        if display_products.empty:
            st.warning("No products match your current criteria.")
        
        cols_per_row = 3
        for i in range(0, len(display_products), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, (_, product) in enumerate(display_products.iloc[i:i+cols_per_row].iterrows()):
                product_id = str(product['id']).strip()

                
                with cols[j]:
                    total_reviews = int(product['Review_Count'])
                    pos_percent_val = product['Pos_Percent']
                    
                    # Create Sentiment Gauge Meter (Plotly Indicator)
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = pos_percent_val,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Positive Rate"},
                        number={'suffix': "%", 'font': {'size': 24}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "lightgray"},
                            'bar': {'color': "#4338ca"},
                            'bgcolor': "#e0e7ff",
                            'borderwidth': 0,
                            'steps': [
                                {'range': [0, 50], 'color': "#f87171"},  # Red
                                {'range': [50, 75], 'color': "#fcd34d"}, # Yellow
                                {'range': [75, 100], 'color': "#34d399"} # Green
                            ]}
                        ))
                    fig_gauge.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
                    pos_percent_val = product.get('Pos_Percent', 0)
                    neg_percent_val = product.get('Neg_Percent', 0)
                    neu_percent_val = 100 - pos_percent_val - neg_percent_val

                    pos_percent = f"{pos_percent_val:.0f}%"
                    neu_percent = f"{neu_percent_val:.0f}%"
                    neg_percent = f"{neg_percent_val:.0f}%"                    
                    category = product['category']
                    category_icon = CATEGORY_ICONS.get(category, CATEGORY_ICONS['Uncategorized'])
                                                                
                    st.markdown(f"""
                    <style>
                    .product-card {{
                        background: white;
                        border-radius: 16px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        transition: all 0.3s ease;
                        padding: 20px;
                        text-align: center;
                        border: 1px solid #f1f1f1;
                    }}
                    .product-card:hover {{
                        transform: translateY(-5px);
                        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
                    }}
                    .product-card img {{
                        border-radius: 12px;
                        margin-bottom: 15px;
                        width: 150px;
                        height: 150px;
                        object-fit: cover;
                        border: 1px solid #e5e7eb;
                    }}
                    .product-card h4 {{
                        font-size: 1.1em;
                        color: #4338ca;
                        font-weight: 700;
                        height: 45px;
                        overflow: hidden;
                        margin-bottom: 6px;
                    }}
                    .product-card p.category {{
                        font-size: 0.9em;
                        color: #6366f1;
                        font-weight: bold;
                        margin-bottom: 12px;
                    }}
                    .price-tag {{
                        font-size: 1.2em;
                        font-weight: 700;
                        color: #1f2937;
                        margin-bottom: 12px;
                    }}
                    .sentiment-bar {{
                        display: flex;
                        justify-content: space-around;
                        font-size: 0.85em;
                        margin-top: 15px;
                        padding: 10px;
                        background-color: #f9fafb;
                        border-radius: 10px;
                    }}
                    .pos-text {{ color: #22c55e; font-weight: 600; }}
                    .neu-text {{ color: #9ca3af; font-weight: 600; }}
                    .neg-text {{ color: #ef4444; font-weight: 600; }}
                    </style>

                    <div class="product-card">
                        <div class='card-content'>
                            <h4>{product['name']}</h4>
                            <p class='category'>{category_icon} {category}</p>
                            <img src="{product['image_url']}" 
                                 onerror="this.onerror=null;this.src='https://placehold.co/150x150/e5e7eb/000000?text=No+Image';">
                            <p class='price-tag'>₹{product['price']:.2f}</p>
                        </div>
                        <div class='sentiment-bar'>
                            <span class='pos-text'>{POSITIVE_EMOJI} {pos_percent}</span>
                            <span class='neu-text'>{NEUTRAL_EMOJI} {neu_percent}</span>
                            <span class='neg-text'>{NEGATIVE_EMOJI} {neg_percent}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


                    st.plotly_chart(
                        fig_gauge, 
                        use_container_width=True, 
                        config={'displayModeBar': False}, 
                        key=f"gauge_{product_id}"
                    )

                    st.markdown(
                        f"<p style='font-size: 0.8em; color: #6b7280; text-align:center; margin-top: 8px;'>⭐ {total_reviews} reviews analyzed</p>", 
                        unsafe_allow_html=True
                    )

                    st.markdown("""
                    <style>
                    div[data-testid="stButton"] button[kind="secondary"] {{
                        background: linear-gradient(90deg, #6366f1, #4338ca);
                        color: white !important;
                        font-weight: 600;
                        border: none;
                        border-radius: 10px;
                        padding: 10px 0;
                        width: 100%;
                        transition: all 0.3s ease;
                    }}
                    div[data-testid="stButton"] button[kind="secondary"]:hover {{
                        transform: translateY(-2px);
                        background: linear-gradient(90deg, #4f46e5, #3730a3);
                        box-shadow: 0 4px 12px rgba(99,102,241,0.4);
                    }}
                    .review-box {{
                        background-color: #f9fafb;
                        border: 1px solid #e5e7eb;
                        border-radius: 10px;
                        padding: 15px;
                        margin-top: 10px;
                    }}
                    </style>
                    """, unsafe_allow_html=True)

                    # --- Beautiful Single Button ---
                    if st.button("View Detailed Analytics", 
                                 key=f"detail_btn_{product_id}",
                                 on_click=lambda pid=product_id: st.session_state.update({'show_detail_id': pid, 'product_summary_cache': {}}),
                                 use_container_width=True):
                        pass

                    with st.expander(f"📝 Write a Review for {product['name']}", expanded=False):
                        st.markdown("<div class='review-box'>", unsafe_allow_html=True)
                        review_text = st.text_area("Your review here:", key=f"review_text_{product_id}")
                        submit_review = st.button("Submit Review & See Sentiment", key=f"submit_review_{product_id}")

                        if submit_review and review_text.strip() != "":
                            if model_ready:
                                sentiment = predict_sentiment(review_text, st.session_state['vectorizer'], st.session_state['clf'])
                                new_review = pd.DataFrame([
                                    {
                                        'product_id': product_id, 
                                        'review': review_text, 
                                        'sentiment': sentiment, 
                                        'timestamp': datetime.now(), 
                                        'upvotes': 0,
                                        'downvotes': 0,
                                        'reviewer_type': random.choice(['Verified Buyer', 'Guest']),
                                        'manager_reply': None
                                    }
                                ])
                                
                                st.session_state['df_reviews'] = pd.concat([st.session_state['df_reviews'], new_review], ignore_index=True)
                                save_reviews()
                                
                                emoji_result = POSITIVE_EMOJI if sentiment=="Positive" else NEUTRAL_EMOJI if sentiment=="Neutral" else NEGATIVE_EMOJI
                                st.success(f"✅ Review submitted! Predicted Sentiment: **{sentiment}** {emoji_result}")
                                st.cache_data.clear()
                                st.rerun() 
                            else:
                                st.error("⚠️ Cannot submit review: Sentiment model is not loaded.")
                        st.markdown("</div>", unsafe_allow_html=True)


        # ----------------------------
        # Dashboard Tabs 
        # ----------------------------
        st.markdown("---")
        st.header("📊 Global Sentiment Analytics Dashboard")

        if df_reviews.empty:
            st.info("No reviews have been submitted yet to generate the dashboard.")
        else:
            total_reviews = len(st.session_state['df_reviews'])
            positive_reviews = len(st.session_state['df_reviews'][st.session_state['df_reviews']['sentiment'] == 'Positive'])
            positive_rate = f"{ (positive_reviews / total_reviews) * 100 :.1f}%" if total_reviews > 0 else "0%"

            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            col_kpi1.metric("Total Reviews Analyzed", total_reviews)
            col_kpi2.metric("Overall Positive Rate", positive_rate)
            col_kpi3.metric("Total Products in Catalog", len(st.session_state['df_products']))

            tabs = st.tabs([
                "Category & Regional Heatmap", 
                "Product Performance Bubble",
                "Reviewer Type Breakdown",
                "Top/Worst Performers",
                "Raw Reviews & Voting"
            ])

            # Pre-calculate data for analytics tabs
            df_merged = df_products.merge(
                sentiment_groups.reset_index(), 
                left_on='id', 
                right_on='product_id', 
                how='left'
            ).fillna(0)
            
            # FIX: Rename 'Total' to 'Review_Count' to match aggregation name later
            df_merged['Review_Count'] = df_merged.get('Positive', 0) + df_merged.get('Neutral', 0) + df_merged.get('Negative', 0)
            df_merged['Pos_Percent'] = np.where(df_merged['Review_Count'] > 0, (df_merged.get('Positive', 0) / df_merged['Review_Count']) * 100, 0)

            # Tab 1: Category & Regional Heatmap
            with tabs[0]:
                st.subheader("🔥 Positive Sentiment Heatmap: Category vs. Region")
                
                # Group by region and category
                heatmap_data = df_merged.groupby(['region', 'category'])['Pos_Percent'].mean().reset_index()
                
                fig_heatmap = px.density_heatmap(heatmap_data, 
                                                x='region', 
                                                y='category', 
                                                z='Pos_Percent', 
                                                histfunc="avg", 
                                                title="Average Positive Rate (%) by Category and Region",
                                                color_continuous_scale=px.colors.sequential.Plotly3)
                
                fig_heatmap.update_xaxes(title="Region")
                fig_heatmap.update_yaxes(title="Category")
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Category-Level Overall Analytics")
                
                # THIS SECTION NOW WORKS because 'Review_Count' is available in df_merged
                cat_summary = df_merged.groupby('category').agg(
                    Avg_Pos_Rate=('Pos_Percent', 'mean'),
                    Total_Reviews=('Review_Count', 'sum'),
                    Total_Products=('id', 'count')
                ).reset_index().sort_values(by='Avg_Pos_Rate', ascending=False)
                
                cat_summary['Avg_Pos_Rate'] = cat_summary['Avg_Pos_Rate'].round(1).astype(str) + '%'
                
                st.dataframe(cat_summary, hide_index=True, use_container_width=True)


            # Tab 2: Product Performance Bubble Chart
            with tabs[1]:
                st.subheader("🎈 Product Review Volume vs. Positive Rate")
                
                # Filter for products with at least 1 review
                # Uses Review_Count
                bubble_data = df_merged[df_merged['Review_Count'] > 0].copy()
                
                fig_bubble = px.scatter(
                    bubble_data, 
                    x='price', 
                    y='Pos_Percent', 
                    size='Review_Count', 
                    color='category',
                    hover_name='name',
                    log_x=True,
                    size_max=60,
                    title="Product Performance: Price, Volume, and Sentiment"
                )
                fig_bubble.update_layout(
                    xaxis_title="Price (Log Scale)", 
                    yaxis_title="Positive Rate (%)",
                    height=600
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
            
            # Tab 3: Reviewer Type Breakdown
            with tabs[2]:
                st.subheader("👤 Sentiment Breakdown by Reviewer Type")
                
                # Group by reviewer type and sentiment
                type_counts = df_reviews.groupby(['reviewer_type', 'sentiment']).size().unstack(fill_value=0)
                type_counts = type_counts.reset_index().rename(columns={'reviewer_type': 'Reviewer Type'})
                
                fig_type = px.bar(
                    type_counts, 
                    x='Reviewer Type', 
                    y=['Positive', 'Neutral', 'Negative'], 
                    title="Sentiment Distribution by Reviewer Type",
                    color_discrete_map={'Positive':'#34D399','Neutral':'#FACC15','Negative':'#F87171'}
                )
                fig_type.update_layout(yaxis_title="Review Count", barmode='stack')
                st.plotly_chart(fig_type, use_container_width=True)


            # Tab 4: Top/Worst Performing Products
            with tabs[3]:
                st.subheader("🏆 Top and Worst Performing Products by Positive Rate")
                
                # Sort for visualization (Top 10 only for clarity)
                product_performance = df_merged[df_merged['Review_Count'] > 0].sort_values(by='Pos_Percent', ascending=False)
                
                col_top, col_worst = st.columns(2)

                with col_top:
                    st.markdown("#### Top 10 Products (Min 5 Reviews)")
                    # Uses Review_Count
                    top_10 = product_performance[product_performance['Review_Count'] >= 5].head(10)
                    if not top_10.empty:
                        fig_perf = px.bar(top_10, 
                                          x='name', 
                                          y='Pos_Percent',
                                          title="Highest Positive Sentiment Rate",
                                          color='Pos_Percent',
                                          color_continuous_scale=px.colors.sequential.Plotly3)
                        fig_perf.update_layout(yaxis_title="Positive Rate (%)", xaxis_title="")
                        st.plotly_chart(fig_perf, use_container_width=True)
                    else:
                        st.info("Not enough products with 5+ reviews to display Top 10.")
                
                with col_worst:
                    st.markdown("#### Bottom 10 Products (Min 5 Reviews)")
                    # Uses Review_Count
                    bottom_10 = product_performance[product_performance['Review_Count'] >= 5].tail(10).sort_values(by='Pos_Percent', ascending=True)
                    if not bottom_10.empty:
                        fig_worst = px.bar(bottom_10, 
                                          x='name', 
                                          y='Pos_Percent',
                                          title="Lowest Positive Sentiment Rate",
                                          color='Pos_Percent',
                                          color_continuous_scale=px.colors.sequential.Reds_r)
                        fig_worst.update_layout(yaxis_title="Positive Rate (%)", xaxis_title="")
                        st.plotly_chart(fig_worst, use_container_width=True)
                    else:
                        st.info("Not enough products with 5+ reviews to display Bottom 10.")


            # Tab 5: Raw Reviews table
            with tabs[4]:
                st.subheader("🔍 All Customer Reviews & Voting Status")
                
                df_reviews_merged = df_reviews.merge(
                    df_products[['id', 'name', 'category']], 
                    left_on='product_id', 
                    right_on='id', 
                    how='left'
                ).rename(columns={'name': 'Product Name', 'category': 'Category'})
                
                display_cols = ['Product Name', 'Category', 'review', 'sentiment', 'upvotes', 'downvotes', 'reviewer_type', 'timestamp']
                
                st.dataframe(
                    df_reviews_merged[display_cols], 
                    use_container_width=True,
                    column_config={
                        "review": st.column_config.TextColumn("Review Content", width="large"),
                        "upvotes": st.column_config.NumberColumn("👍 Upvotes", format="%d", help="How many users found this review helpful"),
                        "downvotes": st.column_config.NumberColumn("👎 Downvotes", format="%d"),
                        "timestamp": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm")
                    },
                    hide_index=True
                )
