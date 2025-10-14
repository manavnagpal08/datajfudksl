import streamlit as st
import pandas as pd
import pickle
import os
import json 
import plotly.express as px
from datetime import datetime
from collections import Counter
import re
import time
from math import ceil

# --- Configuration and Constants ---
PRODUCTS_FILE = "data/products.csv"
REVIEWS_FILE = "data/reviews.csv"
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
PRODUCTS_COLUMNS = ['id', 'name', 'price', 'region', 'image_url', 'description', 'category'] # Added 'category'

# Sentiment Emojis
POSITIVE_EMOJI = "✅"
NEGATIVE_EMOJI = "❌"
NEUTRAL_EMOJI = "🟡" 

# Custom Credentials provided by user
USERS = {
    "admin": {"password": "admin123", "role": "Admin"},
    "user": {"password": "user123", "role": "User"}
}

# --- Internal Review Synthesis Logic (API-FREE) ---
def get_top_sentiment_words(df_subset, sentiment, n=3):
    """Calculates top N words for a specific sentiment."""
    if df_subset.empty: return []
    
    # Standard stop words plus generic e-commerce terms
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
    words = re.findall(r'\b\w{3,}\b', text) # Only include words with 3+ characters
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    word_counts = Counter(filtered_words)
    return [word.capitalize() for word, count in word_counts.most_common(n)]

def generate_product_summary_internal(product_name, reviews_df):
    """
    Generates a synthesized product review summary based *only* on internal analytics.
    (This replaces the external LLM API call.)
    """
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
        f"<p style='font-size: 1.1em; line-height: 1.6;'><b>Overall Assessment: <span style='color:#3b82f6;'>{overall_sentiment}</span></b> ({pos_count} Positive reviews out of {total}). "
        f"{sentiment_description} "
        f"{strengths} "
        f"{weaknesses}</p>"
    )
    
    return final_summary


# --- Aesthetics (Custom CSS for a more beautiful UI) ---
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #f0f2f6; /* Light gray/blue background */
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #1f2937;
        font-weight: 700;
    }
    h1 {
        border-bottom: 3px solid #3b82f6; 
        padding-bottom: 15px;
        margin-top: 0;
    }
    
    /* Login Page Styling - More Polish */
    .login-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        width: 100vw;
        position: fixed;
        top: 0;
        left: 0;
        background: linear-gradient(135deg, #e0e5ec 0%, #f0f2f6 100%);
    }
    .login-box {
        max-width: 400px;
        width: 90%;
        padding: 50px 40px;
        border-radius: 16px;
        background-color: #ffffff;
        box-shadow: 0 25px 60px rgba(0,0,0,0.2); /* Deeper shadow */
        border: 1px solid #dcdcdc;
    }
    
    /* Product Card Styling */
    .product-card {
        border: none;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        min-height: 540px; 
        box-shadow: 0 6px 15px rgba(0,0,0,0.08); /* Softer shadow */
        background-color: #ffffff;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        display: flex;
        flex-direction: column;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
    }
    .card-content {
        flex-grow: 1;
    }

    /* Custom button styling (Primary action) */
    .stButton>button {
        border-radius: 8px;
        border: none;
        color: white !important;
        background-color: #3b82f6;
        padding: 10px 20px;
        font-weight: 600;
        transition: background-color 0.2s, transform 0.2s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
    }
    
    /* Sentiment Colors */
    .pos-text { color: #10B981; font-weight: bold; }
    .neg-text { color: #EF4444; font-weight: bold; }
    .neu-text { color: #FBBF24; font-weight: bold; }

    /* Custom Metrics Boxes (Enhanced Visuals) */
    .metric-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 15px;
        transition: all 0.2s ease-in-out;
    }
    .metric-box:hover {
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    }
    .pos-metric { border-left: 5px solid #10B981; }
    .neg-metric { border-left: 5px solid #EF4444; }
    .total-metric { border-left: 5px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Data Loading and Utility Functions
# ----------------------------

@st.cache_data(show_spinner="Loading Data...")
def load_initial_data():
    """Loads and initializes products and reviews DataFrames."""
    os.makedirs("data", exist_ok=True)
    
    # Load products, ensuring 'category' column exists
    df_products = pd.read_csv(PRODUCTS_FILE) if os.path.exists(PRODUCTS_FILE) else pd.DataFrame(columns=PRODUCTS_COLUMNS)
    df_products['id'] = pd.to_numeric(df_products['id'], errors='coerce').fillna(0).astype('Int64')
    
    # Ensure all required product columns are present
    for col in PRODUCTS_COLUMNS:
        if col not in df_products.columns:
            # Explicitly add missing column with default value
            default_value = 'Uncategorized' if col == 'category' else None
            df_products.loc[:, col] = default_value
    
    # Ensure 'category' is a string and fill any remaining NaNs
    df_products['category'] = df_products['category'].fillna('Uncategorized').astype(str)
    
    # Load reviews
    REVIEW_COLUMNS = ['product_id', 'review', 'sentiment', 'timestamp'] 
    df_reviews = pd.DataFrame(columns=REVIEW_COLUMNS)
    if os.path.exists(REVIEWS_FILE) and os.path.getsize(REVIEWS_FILE) > 0:
        try:
            loaded_df = pd.read_csv(REVIEWS_FILE)
            if not loaded_df.empty and all(col in loaded_df.columns for col in REVIEW_COLUMNS[:3]):
                df_reviews = loaded_df
        except:
            pass
            
    df_reviews['product_id'] = pd.to_numeric(df_reviews['product_id'], errors='coerce').fillna(0).astype('Int64')
    if 'timestamp' not in df_reviews.columns: df_reviews['timestamp'] = pd.NaT
    df_reviews['timestamp'] = pd.to_datetime(df_reviews['timestamp'], errors='coerce').fillna(pd.to_datetime('2024-01-01 00:00:00'))
    
    # Save files if they were just created or updated to ensure future loads are stable
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

def get_top_words(df_subset, n=20):
    """Calculates top N words from a DataFrame subset of reviews."""
    if df_subset.empty: return pd.DataFrame()
    stop_words = set([
        'the', 'a', 'an', 'is', 'it', 'and', 'but', 'or', 'to', 'of', 'in', 'for', 
        'with', 'on', 'this', 'that', 'i', 'was', 'my', 'had', 'have', 'very', 'not',
        'would', 'me', 'be', 'so', 'get', 'product', 'item', 'just', 'too', 'great', 
        'good', 'bad', 'best', 'worst', 'really', 'much', 'like', 'for', 'about', 'is', 'i'
    ])
    text = ' '.join(df_subset['review'].astype(str).str.lower().tolist())
    words = re.findall(r'\b\w+\b', text)
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    word_counts = Counter(filtered_words)
    return pd.DataFrame(word_counts.most_common(n), columns=['Word', 'Frequency'])


# ----------------------------
# Session State Initialization & Setup
# ----------------------------

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['current_role'] = 'Guest'
    st.session_state['show_detail_id'] = None 
    st.session_state['product_summary_cache'] = {} 

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


# ----------------------------
# Authentication
# ----------------------------

def main_login_screen():
    """Renders the central, polished login interface."""
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    with st.form("login_form"):
        # Enhanced title and header
        st.markdown("<h2 style='text-align: center; color: #3b82f6;'>📈 E-Commerce Analytics Hub</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #4b5563; margin-bottom: 25px;'>Securely access your data insights.</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        # User input fields
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
# Product Detail View Function
# ----------------------------

def show_product_detail(product_id):
    """Shows detailed analytics for a single product."""
    if df_products[df_products['id'] == product_id].empty:
        st.error("Product not found.")
        st.session_state.update({'show_detail_id': None})
        st.rerun()
        
    product = df_products[df_products['id'] == product_id].iloc[0]
    
    st.header(f"Product Detail Analysis: {product['name']} (ID: {product_id})")
    st.button("← Back to Catalog", on_click=lambda: st.session_state.update({'show_detail_id': None}))
    
    product_reviews = df_reviews[df_reviews['product_id'] == product_id]
    
    if product_reviews.empty:
        st.warning("No reviews available for detailed analysis yet.")
        return

    # Split layout for product info and summary
    col_img, col_summary = st.columns([1, 2])
    
    with col_img:
        # 1. Product Image and Details
        st.image(
            product['image_url'],
            caption=f"{product['name']} - {product['category']}",
            width=250,
            use_column_width='auto',
            output_format='PNG',
        )
        st.markdown(f"**Price:** ₹{product['price']:.2f}")
        st.markdown(f"**Region:** {product['region']}")
        st.markdown(f"**Category:** **{product['category']}**")
        st.markdown(f"**Description:** <span style='font-style: italic; font-size: 0.9em;'>{product['description']}</span>", unsafe_allow_html=True)

    with col_summary:
        # 2. Internal Generated Summary (Projected Sentiment Analysis)
        st.subheader("🤖 AI-Free Synthesis & Projected Sentiment Analysis")
        
        summary_placeholder = st.empty()
        if product_id not in st.session_state['product_summary_cache']:
            with summary_placeholder:
                with st.spinner("Analyzing all reviews and synthesizing summary..."):
                    summary = generate_product_summary_internal(product['name'], product_reviews)
                    st.session_state['product_summary_cache'][product_id] = summary
        
        summary_placeholder.markdown(st.session_state['product_summary_cache'][product_id], unsafe_allow_html=True)

    # 3. Product Summary and Metrics (Enhanced Visuals)
    st.markdown("---")
    st.subheader("Review Metrics Breakdown")

    total_reviews = len(product_reviews)
    sentiment_counts = product_reviews['sentiment'].value_counts(normalize=True).mul(100).round(1).to_dict()
    pos_p = sentiment_counts.get('Positive', 0)
    neu_p = sentiment_counts.get('Neutral', 0)
    neg_p = sentiment_counts.get('Negative', 0)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    # Display Metrics with custom HTML and Emojis
    col_m1.markdown(f'<div class="metric-box total-metric">Total Reviews<br><b>{total_reviews}</b></div>', unsafe_allow_html=True)
    col_m2.markdown(f'<div class="metric-box pos-metric">{POSITIVE_EMOJI} Positive Rate<br><b class="pos-text">{pos_p}%</b></div>', unsafe_allow_html=True)
    col_m3.markdown(f'<div class="metric-box">{NEUTRAL_EMOJI} Neutral Rate<br><b class="neu-text">{neu_p}%</b></div>', unsafe_allow_html=True)
    col_m4.markdown(f'<div class="metric-box neg-metric">{NEGATIVE_EMOJI} Negative Rate<br><b class="neg-text">{neg_p}%</b></div>', unsafe_allow_html=True)


    # 4. Time Series, Keywords, and NEW Insight (Review Length)
    st.markdown("---")
    st.subheader("Time Trend & Deeper Keyword Insights")
    
    col_time, col_key, col_length = st.columns([2, 1, 1])

    with col_time:
        product_reviews_copy = product_reviews.copy()
        product_reviews_copy['date'] = product_reviews_copy['timestamp'].dt.date
        time_series = product_reviews_copy.groupby(['date', 'sentiment']).size().reset_index(name='count')
        
        fig_time = px.line(time_series, x='date', y='count', color='sentiment',
                           title=f"Daily Sentiment Trend",
                           color_discrete_map={'Positive':'#10B981','Neutral':'#FBBF24','Negative':'#EF4444'})
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col_key:
        st.markdown("#### Top 5 Overall Keywords")
        all_words = get_top_words(product_reviews, n=5)
        st.dataframe(all_words, use_container_width=True, hide_index=True)

    with col_length:
        st.markdown("#### Review Length Distribution")
        # Ensure 'review_length' is calculated only once if possible, but recalculate here for safety if not in session state
        product_reviews['review_length'] = product_reviews['review'].str.len()
        
        fig_len = px.histogram(product_reviews, x='review_length', nbins=10, 
                               labels={'review_length': 'Review Length (Characters)', 'count': 'Number of Reviews'},
                               title='Review Length Distribution')
        fig_len.update_layout(showlegend=False, bargap=0.1)
        st.plotly_chart(fig_len, use_container_width=True)


# ----------------------------
# Main Application Flow
# ----------------------------

if not st.session_state['logged_in']:
    main_login_screen()
else:
    # --- Sidebar Metrics and Logout ---
    st.sidebar.markdown(f"### 👋 Welcome, **{st.session_state['current_role']}**!")
    if st.sidebar.button("Logout", key="logout_btn"):
        logout()
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Products", len(df_products))
    st.sidebar.metric("Total Reviews", len(df_reviews))
    st.sidebar.markdown("---")
    if st.sidebar.button("Hard Refresh Data"):
        st.cache_data.clear()
        st.session_state['df_products'], st.session_state['df_reviews'] = load_initial_data()
        st.session_state['vectorizer'], st.session_state['clf'] = load_model_and_vectorizer()
        st.rerun()

    if st.session_state['show_detail_id'] is not None:
        show_product_detail(st.session_state['show_detail_id'])
        
    else:
        st.title("🛒 E-Commerce Platform with Interactive Sentiment Analytics")

        if st.session_state['current_role'] == "Admin":
            with st.expander("👑 Administrator Panel: Product & Category Management"):
                
                # --- CATEGORY ASSIGNMENT ---
                st.markdown("##### 📝 Assign/Update Product Category")
                if not st.session_state['df_products'].empty:
                    with st.form("category_update_form"):
                        
                        # Get products for selection
                        product_options = st.session_state['df_products']['id'].tolist()
                        
                        prod_id_to_edit = st.selectbox(
                            "Select Product to Edit", 
                            options=product_options,
                            format_func=lambda x: f"{x} - {st.session_state['df_products'][st.session_state['df_products']['id']==x]['name'].iloc[0]}",
                            key="select_product_id"
                        )
                        
                        current_category = st.session_state['df_products'][st.session_state['df_products']['id']==prod_id_to_edit]['category'].iloc[0] if prod_id_to_edit is not None else "Uncategorized"
                        
                        # Get all existing unique categories
                        # Use a defensive check here as well
                        if 'category' in st.session_state['df_products'].columns:
                            all_categories = st.session_state['df_products']['category'].unique().tolist()
                        else:
                            all_categories = ['Uncategorized']

                        if 'Uncategorized' not in all_categories: all_categories.append('Uncategorized')
                        
                        st.markdown(f"**Current Category:** `{current_category}`")

                        # Allow user to select existing category
                        selected_existing = st.selectbox(
                            "Select an Existing Category", 
                            ["(Leave unchanged)"] + sorted(all_categories),
                            key="select_existing_category"
                        )
                        
                        # Allow user to type a new category
                        typed_new_category = st.text_input(
                            "OR Type a New Category Name", 
                            value="",
                            placeholder="e.g., Electronics, Apparel, Home Goods",
                            key="typed_new_category"
                        )

                        update_submitted = st.form_submit_button("Update Category")
                        
                        if update_submitted:
                            if prod_id_to_edit in st.session_state['df_products']['id'].tolist():
                                new_category = current_category
                                
                                if selected_existing != "(Leave unchanged)":
                                    new_category = selected_existing
                                elif typed_new_category.strip():
                                    new_category = typed_new_category.strip()
                                
                                if new_category != current_category:
                                    st.session_state['df_products'].loc[st.session_state['df_products']['id'] == prod_id_to_edit, 'category'] = new_category
                                    save_products()
                                    st.success(f"Category for Product ID {prod_id_to_edit} updated to **{new_category}**.")
                                    st.rerun()
                                else:
                                    st.warning("No change made or category input was empty.")
                            else:
                                st.error("Product ID not found.")
                else:
                    st.info("No products available to categorize.")

        st.header("🛍 Product Catalog")

        # Interactive Filter and Search - Now 5 columns
        col_filter_region, col_filter_category, col_sort, col_sentiment, col_search = st.columns([1, 1, 1, 1, 2])
        
        with col_filter_region:
            region_filter = st.selectbox("Filter by Region", ["All"] + sorted(st.session_state['df_products']['region'].astype(str).unique().tolist()))

        with col_filter_category:
            # --- FIX APPLIED HERE: Defensive check for 'category' column ---
            if 'category' in st.session_state['df_products'].columns:
                unique_categories = st.session_state['df_products']['category'].astype(str).unique().tolist()
            else:
                # Fallback if the column is missing to prevent KeyError
                unique_categories = ['Uncategorized']
                
            category_filter = st.selectbox("Filter by Category", ["All"] + sorted(unique_categories))
            # --- END FIX ---

        with col_sort:
            sort_option = st.selectbox("Sort By", ["ID", "Price (L-H)", "Price (H-L)"])
        
        with col_sentiment:
            min_pos_percent = st.slider("Min Pos. %", 0, 100, 0, step=5)
        
        with col_search:
            search_query = st.text_input("Search Product (Name/Desc)", "")
        
        
        # --- Data Preparation for Filtering ---
        display_products = st.session_state['df_products'].copy()
        
        sentiment_groups = None 

        if not df_reviews.empty:
            sentiment_groups = df_reviews.groupby('product_id')['sentiment'].value_counts().unstack(fill_value=0)
            sentiment_groups['Total'] = sentiment_groups.sum(axis=1)
            sentiment_groups['Pos_Percent'] = (sentiment_groups.get('Positive', 0) / sentiment_groups['Total']) * 100
            sentiment_groups['Neg_Percent'] = (sentiment_groups.get('Negative', 0) / sentiment_groups['Total']) * 100
            
            display_products = display_products.merge(
                sentiment_groups[['Pos_Percent', 'Neg_Percent']], 
                left_on='id', 
                right_index=True, 
                how='left'
            ).fillna({'Pos_Percent': 0, 'Neg_Percent': 0})
            
            display_products = display_products[display_products['Pos_Percent'] >= min_pos_percent]

        if region_filter != "All":
            display_products = display_products[display_products['region'].astype(str) == region_filter]
        
        # Check for category column existence before applying category filter
        if 'category' in display_products.columns and category_filter != "All":
            display_products = display_products[display_products['category'].astype(str) == category_filter]
        elif 'category' not in display_products.columns:
            st.warning("Category column missing for filtering. Please refresh data if this persists.")


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
        else:
            display_products = display_products.sort_values(by='id')

        # --- Product Display (Using Columns) ---
        if display_products.empty:
            st.warning("No products match your current criteria.")
        
        cols_per_row = 3
        for i in range(0, len(display_products), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, (_, product) in enumerate(display_products.iloc[i:i+cols_per_row].iterrows()):
                product_id = int(product['id'])
                
                with cols[j]:
                    total_reviews = len(df_reviews[df_reviews['product_id'] == product_id])
                    
                    pos_percent_val = product.get('Pos_Percent', 0)
                    neg_percent_val = product.get('Neg_Percent', 0)
                    neu_percent_val = 100 - pos_percent_val - neg_percent_val

                    pos_percent = f"{pos_percent_val:.0f}%"
                    neu_percent = f"{neu_percent_val:.0f}%"
                    neg_percent = f"{neg_percent_val:.0f}%"
                        
                    # Custom HTML for Card (Integrating Emojis and better layout)
                    st.markdown(f"""
                    <div class="product-card">
                    <div class='card-content'>
                        <h4 style="height: 40px; overflow: hidden;">{product['name']}</h4>
                        <p style='font-size: 0.9em; color: #3b82f6; font-weight: bold; margin-bottom: 10px;'>{product['category'] if 'category' in product else 'Uncategorized'}</p>
                        <img src="{product['image_url']}" onerror="this.onerror=null;this.src='https://via.placeholder.com/150/EEEEEE/000000?text=No+Image';" width="150" style="border-radius: 5px; margin-bottom: 15px; border: 1px solid #e0e0e0;">
                        <p style="height: 60px; overflow: hidden; font-size: 0.9em; color: #555;">{product['description']}</p>
                        <p><b>Price: ₹{product['price']:.2f}</b></p>
                        
                        <div style='display: flex; justify-content: space-around; font-size: 0.85em; margin-top: 15px; padding: 10px; background-color: #f7f7f7; border-radius: 8px;'>
                            <span class='pos-text'>{POSITIVE_EMOJI} {pos_percent}</span>
                            <span class='neu-text'>{NEUTRAL_EMOJI} {neu_percent}</span>
                            <span class='neg-text'>{NEGATIVE_EMOJI} {neg_percent}</span>
                        </div>
                        <p style='font-size: 0.75em; color: #888; margin-top: 5px;'>({total_reviews} reviews analyzed)</p>
                    </div>
                    <div style='height: 10px;'></div> 
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.button("View Detail Analytics", 
                              key=f"detail_btn_{product_id}",
                              on_click=lambda pid=product_id: st.session_state.update({'show_detail_id': pid, 'product_summary_cache': {}}), 
                              use_container_width=True)

                    with st.expander(f"Write a Review for {product['name']}"):
                        review_text = st.text_area("Your review here:", key=f"review_text_{product_id}")
                        submit_review = st.button("Submit Review & See Sentiment", key=f"submit_review_{product_id}")
                        
                        if submit_review and review_text.strip() != "":
                            if model_ready:
                                sentiment = predict_sentiment(review_text, st.session_state['vectorizer'], st.session_state['clf'])
                                new_review = pd.DataFrame([[product_id, review_text, sentiment, datetime.now()]],
                                                            columns=['product_id', 'review', 'sentiment', 'timestamp'])
                                
                                st.session_state['df_reviews'] = pd.concat([st.session_state['df_reviews'], new_review], ignore_index=True)
                                save_reviews()
                                
                                emoji_result = POSITIVE_EMOJI if sentiment=="Positive" else NEUTRAL_EMOJI if sentiment=="Neutral" else NEGATIVE_EMOJI
                                st.success(f"Review submitted! Predicted Sentiment: **{sentiment}** {emoji_result}")
                                st.cache_data.clear()
                                st.rerun() 
                            else:
                                st.error("Cannot submit review: Sentiment model is not loaded.")

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
                "Overall Breakdown & Regional View", 
                "Product Performance", 
                "Top/Worst Performers (Filtered by Category)", # Tab 2 updated
                "Price Quartile Analysis", 
                "Extreme Reviews",
                "Raw Reviews Table"
            ])

            # Tab 1: Overall sentiment & Regional Breakdown
            with tabs[0]:
                st.subheader("Global Sentiment Distribution and Regional Comparison")
                col_pie, col_region = st.columns(2)
                
                with col_pie:
                    st.markdown("#### Global Sentiment Mix")
                    fig = px.pie(st.session_state['df_reviews'], names='sentiment', title="Distribution of All Customer Feedback",
                                color='sentiment', 
                                color_discrete_map={'Positive':'#34D399','Neutral':'#FACC15','Negative':'#F87171'})
                    st.plotly_chart(fig, use_container_width=True)

                with col_region:
                    st.markdown("#### Regional Positive Sentiment Rate")
                    df_region = df_reviews.merge(df_products[['id', 'region']], left_on='product_id', right_on='id', how='left')
                    
                    # Calculate regional positive rate
                    region_counts = df_region.groupby('region')['sentiment'].value_counts().unstack(fill_value=0)
                    region_counts['Total'] = region_counts.sum(axis=1)
                    # Handle potential division by zero if a region has 0 reviews
                    region_counts['Pos_Rate'] = (region_counts.get('Positive', 0) / region_counts['Total']) * 100
                    region_counts = region_counts.reset_index().sort_values(by='Pos_Rate', ascending=False)
                    
                    fig_region = px.bar(region_counts, x='region', y='Pos_Rate',
                                        title='Positive Sentiment Rate by Region',
                                        color='Pos_Rate',
                                        color_continuous_scale=px.colors.sequential.Bluyl)
                    fig_region.update_layout(yaxis_title="Positive Rate (%)")
                    st.plotly_chart(fig_region, use_container_width=True)


            # Tab 2: Per product sentiment
            with tabs[1]:
                st.subheader("Sentiment Count Per Product")
                
                sentiment_summary = df_reviews.groupby(['product_id','sentiment']).size().unstack(fill_value=0)
                
                # Check for category column before merging
                product_cols_to_merge = ['name']
                if 'category' in df_products.columns:
                    product_cols_to_merge.append('category')

                sentiment_summary = sentiment_summary.join(
                    df_products.set_index('id')[product_cols_to_merge].rename({'name': 'Product Name', 'category': 'Category'}, axis=1)
                ).fillna(0).reset_index()
                
                for s in ['Positive', 'Neutral', 'Negative']:
                    if s not in sentiment_summary.columns: sentiment_summary[s] = 0

                if not sentiment_summary.empty:
                    hover_data = ['Category'] if 'Category' in sentiment_summary.columns else None
                    fig2 = px.bar(sentiment_summary, x='Product Name', y=['Positive','Neutral','Negative'],
                                  title="Sentiment Count per Product", 
                                  color_discrete_map={'Positive':'#34D399','Neutral':'#FACC15','Negative':'#F87171'},
                                  hover_data=hover_data) 
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Tab 3: Top/Worst Performing Products (with Category Filter)
            with tabs[2]:
                st.subheader("🏆 Top and Worst Performing Products by Positive Rate")
                
                # Defensive retrieval of categories
                if 'category' in st.session_state['df_products'].columns:
                    all_categories = st.session_state['df_products']['category'].astype(str).unique().tolist()
                else:
                    all_categories = ['Uncategorized']

                perf_category_filter = st.selectbox(
                    "Filter Performance by Category", 
                    ["All"] + sorted(all_categories), 
                    key="perf_cat_filter",
                    help="Analyze top/worst performers within a specific product segment."
                )

                # Use a merged dataframe that contains Pos_Percent, filtered by the Dashboard's general view requirements
                product_performance = display_products[display_products['Pos_Percent'].notna() & (display_products['Pos_Percent'] >= 0)].copy()
                
                # Apply category filter specifically for performance charts
                if 'category' in product_performance.columns and perf_category_filter != "All":
                    product_performance = product_performance[product_performance['category'].astype(str) == perf_category_filter]

                if product_performance.empty:
                    st.info(f"No products with review data match the criteria, or no products found in **{perf_category_filter}**.")
                else:
                    st.markdown(f"#### Results filtered for Category: **{perf_category_filter}**")

                    # Decide which columns to include in hover data
                    perf_hover_data = ['category'] if 'category' in product_performance.columns else None

                    # Sort for visualization (Top 10 only for clarity)
                    product_performance = product_performance.sort_values(by='Pos_Percent', ascending=False)
                    
                    # Top 10 Chart
                    fig_perf = px.bar(product_performance.head(10), 
                                      x='name', 
                                      y='Pos_Percent',
                                      title="Top 10 Products by Positive Sentiment Rate",
                                      color='Pos_Percent',
                                      color_continuous_scale=px.colors.sequential.Plotly3,
                                      hover_data=perf_hover_data)
                    fig_perf.update_layout(yaxis_title="Positive Rate (%)")
                    st.plotly_chart(fig_perf, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Bottom 10 Chart
                    fig_worst = px.bar(product_performance.tail(10), 
                                      x='name', 
                                      y='Pos_Percent',
                                      title="Bottom 10 Products by Positive Sentiment Rate",
                                      color='Pos_Percent',
                                      color_continuous_scale=px.colors.sequential.Reds_r, # Reverse color scale for bad performance
                                      hover_data=perf_hover_data)
                    fig_worst.update_layout(yaxis_title="Positive Rate (%)")
                    st.plotly_chart(fig_worst, use_container_width=True)


            # Tab 4: Price Quartile Analysis
            with tabs[3]:
                st.subheader("📈 Positive Sentiment Rate by Price Bracket")
                
                df_merged = df_products.merge(
                    sentiment_groups[['Pos_Percent']], 
                    left_on='id', 
                    right_index=True, 
                    how='left'
                ).fillna({'Pos_Percent': 0})

                if not df_merged.empty and len(df_merged) >= 3:
                    try:
                        df_merged['Price_Bracket'] = pd.qcut(df_merged['price'], q=3, labels=['Low Price', 'Medium Price', 'High Price'], duplicates='drop')
                        price_sentiment = df_merged.groupby('Price_Bracket')['Pos_Percent'].mean().reset_index()
                        
                        fig_quartile = px.bar(price_sentiment, x='Price_Bracket', y='Pos_Percent',
                                              title="Average Positive Review Rate across Product Price Brackets",
                                              color='Pos_Percent',
                                              color_continuous_scale=px.colors.sequential.Tealgrn)
                        fig_quartile.update_layout(yaxis_title="Average Positive Rate (%)")
                        st.plotly_chart(fig_quartile, use_container_width=True)
                    except ValueError:
                        st.warning("Not enough distinct prices to create 3 price brackets.")
                else:
                    st.info("Not enough product data to perform price quartile analysis.")


            # Tab 5: Top Extreme Reviews
            with tabs[4]:
                st.subheader("🔥 Top 5 Most Extreme Reviews")
                
                sentiment_scores = df_reviews['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
                df_reviews['sentiment_score'] = sentiment_scores
                df_reviews['review_length'] = df_reviews['review'].str.len()
                
                top_positive = df_reviews.sort_values(by=['sentiment_score', 'review_length'], ascending=[False, False]).head(5)
                top_negative = df_reviews.sort_values(by=['sentiment_score', 'review_length'], ascending=[True, False]).head(5)

                col_pos_extreme, col_neg_extreme = st.columns(2)
                
                with col_pos_extreme:
                    st.markdown("#### Top 5 Most Positive Reviews")
                    for _, row in top_positive.iterrows():
                        product_data = df_products[df_products['id'] == row['product_id']]
                        product_name = product_data['name'].iloc[0] if not product_data.empty else "Unknown Product"
                        product_category = product_data['category'].iloc[0] if 'category' in product_data.columns and not product_data.empty else "Uncategorized"

                        st.success(f"{POSITIVE_EMOJI} **{product_name}** ({product_category}) - *{row['sentiment']}*")
                        st.write(f"_{row['review']}_")
                        st.markdown("---")

                with col_neg_extreme:
                    st.markdown("#### Top 5 Most Negative Reviews")
                    for _, row in top_negative.iterrows():
                        product_data = df_products[df_products['id'] == row['product_id']]
                        product_name = product_data['name'].iloc[0] if not product_data.empty else "Unknown Product"
                        product_category = product_data['category'].iloc[0] if 'category' in product_data.columns and not product_data.empty else "Uncategorized"

                        st.error(f"{NEGATIVE_EMOJI} **{product_name}** ({product_category}) - *{row['sentiment']}*")
                        st.write(f"_{row['review']}_")
                        st.markdown("---")


            # Tab 6: Raw Reviews table
            with tabs[5]:
                st.subheader("🔍 All Customer Reviews (Interactive Filtering)")
                
                col_filt_sent, col_filt_date, col_filt_len = st.columns([1, 1, 1])
                
                review_filter = col_filt_sent.multiselect(
                    "Filter by Sentiment Type", 
                    options=['Positive', 'Neutral', 'Negative'], 
                    default=['Positive', 'Neutral', 'Negative'],
                    key="review_table_filter"
                )

                min_date_val = df_reviews['timestamp'].min().date() if not df_reviews.empty else datetime.now().date()
                min_date = col_filt_date.date_input("Filter from Date", 
                                                value=min_date_val,
                                                min_value=min_date_val
                                            )
                
                min_length = col_filt_len.slider("Min Review Length (Chars)", 0, 100, 10)


                filtered_reviews = df_reviews[df_reviews['sentiment'].isin(review_filter)].copy()
                
                filtered_reviews = filtered_reviews[filtered_reviews['timestamp'].dt.date >= min_date]
                # Calculate review length here just in case it wasn't done earlier
                filtered_reviews['review_length'] = filtered_reviews['review'].str.len()
                filtered_reviews = filtered_reviews[filtered_reviews['review_length'] >= min_length]
                
                # Conditional merge based on column existence
                cols_for_merge = ['name']
                if 'category' in df_products.columns:
                    cols_for_merge.append('category')

                filtered_reviews = filtered_reviews.join(
                    df_products.set_index('id')[cols_for_merge].rename({'name': 'Product Name', 'category': 'Category'}, axis=1),
                    on='product_id'
                )
                
                display_cols = ['Product Name', 'review', 'sentiment', 'product_id', 'timestamp']
                col_config = {
                    "review": st.column_config.TextColumn("Review Content", width="large"),
                    "sentiment": st.column_config.TextColumn("Predicted Sentiment", width="small"),
                    "Product Name": st.column_config.TextColumn("Product Name", width="medium"),
                    "product_id": "ID",
                    "timestamp": st.column_config.DatetimeColumn("Review Date", format="YYYY-MM-DD HH:mm")
                }
                
                if 'Category' in filtered_reviews.columns:
                    display_cols.insert(1, 'Category')
                    col_config["Category"] = st.column_config.TextColumn("Category", width="small")

                display_df = filtered_reviews[display_cols]
                
                st.dataframe(
                    display_df, 
                    use_container_width=True,
                    column_config=col_config
                )
