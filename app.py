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

# --- Configuration and Constants ---
PRODUCTS_FILE = "data/products.csv"
REVIEWS_FILE = "data/reviews.csv"
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

# Defined Credentials (Simple for demo)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"
USER_USERNAME = "user"
USER_PASSWORD = "123"

# --- Aesthetics (Custom CSS for a more beautiful UI) ---
st.markdown("""
<style>
    /* Light background and general container padding */
    .stApp {
        background-color: #f7f9fc;
    }
    /* Main header styling */
    h1 {
        color: #1f2937;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 10px;
        font-weight: 700;
    }
    h2, h3, h4 {
        color: #374151;
    }
    /* Product Card Styling */
    .product-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        min-height: 440px; /* Increased height for new metrics */
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        background-color: #ffffff;
        text-align: center;
        transition: transform 0.2s;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    /* Custom button styling */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #3b82f6;
        color: white !important;
        background-color: #3b82f6;
        padding: 8px 16px;
        transition: background-color 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    /* Centering content for login */
    .login-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 80vh;
    }
    .login-box {
        max-width: 400px;
        width: 100%;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Data Loading and Utility Functions
# ----------------------------

@st.cache_data(show_spinner="Loading Data...")
def load_initial_data():
    """Loads and initializes products and reviews DataFrames."""
    os.makedirs("data", exist_ok=True)

    # Load products
    df_products = pd.read_csv(PRODUCTS_FILE) if os.path.exists(PRODUCTS_FILE) else pd.DataFrame(columns=['id', 'name', 'price', 'region', 'image_url', 'description'])
    df_products['id'] = pd.to_numeric(df_products['id'], errors='coerce').fillna(0).astype('Int64')
    
    # Load reviews
    REVIEW_COLUMNS = ['product_id', 'review', 'sentiment', 'timestamp'] 
    df_reviews = pd.DataFrame(columns=REVIEW_COLUMNS)
    if os.path.exists(REVIEWS_FILE) and os.path.getsize(REVIEWS_FILE) > 0:
        try:
            loaded_df = pd.read_csv(REVIEWS_FILE)
            if not loaded_df.empty and all(col in loaded_df.columns for col in REVIEW_COLUMNS[:3]):
                df_reviews = loaded_df
        except pd.errors.EmptyDataError:
            pass
        except Exception as e:
            print(f"Error loading reviews file: {e}")

    df_reviews['product_id'] = pd.to_numeric(df_reviews['product_id'], errors='coerce').fillna(0).astype('Int64')
    if 'timestamp' not in df_reviews.columns:
        df_reviews['timestamp'] = pd.NaT
    df_reviews['timestamp'] = pd.to_datetime(df_reviews['timestamp'], errors='coerce').fillna(pd.to_datetime('2024-01-01 00:00:00'))
    
    if df_reviews.empty and (not os.path.exists(REVIEWS_FILE) or os.path.getsize(REVIEWS_FILE) == 0):
         df_reviews.to_csv(REVIEWS_FILE, index=False) 

    return df_products, df_reviews

def save_products():
    """Saves the products DataFrame from session state to CSV."""
    st.session_state['df_products'].to_csv(PRODUCTS_FILE, index=False)

def save_reviews():
    """Saves the reviews DataFrame from session state to CSV."""
    st.session_state['df_reviews'].to_csv(REVIEWS_FILE, index=False)

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
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Model Error"

def get_top_words(df_subset, n=20):
    """Calculates top N words from a DataFrame subset of reviews."""
    if df_subset.empty:
        return pd.DataFrame()

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
    top_n = word_counts.most_common(n)
    
    return pd.DataFrame(top_n, columns=['Word', 'Frequency'])


# ----------------------------
# Session State Initialization & Setup
# ----------------------------

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['current_role'] = 'Guest'

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
    """Renders the central login interface for Admin or User."""
    st.title("🛒 E-Commerce Platform Login")
    
    # Use custom HTML/CSS to center the form
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    with st.form("login_form"):
        st.subheader("Sign In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        st.markdown("---")
        st.markdown(f"**Admin:** `{ADMIN_USERNAME}` / `{ADMIN_PASSWORD}`")
        st.markdown(f"**User:** `{USER_USERNAME}` / `{USER_PASSWORD}`")
        
        if submitted:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state['logged_in'] = True
                st.session_state['current_role'] = 'Admin'
                st.success("Logged in as Admin!")
                time.sleep(0.5)
                st.rerun()
            elif username == USER_USERNAME and password == USER_PASSWORD:
                st.session_state['logged_in'] = True
                st.session_state['current_role'] = 'User'
                st.success("Logged in as User!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Invalid credentials. Please check your username and password.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
def logout():
    """Logs out the current user."""
    st.session_state['logged_in'] = False
    st.session_state['current_role'] = 'Guest'
    st.info("You have been logged out.")
    time.sleep(0.5)
    st.rerun()

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
        # Force reload data into session state
        st.session_state['df_products'], st.session_state['df_reviews'] = load_initial_data()
        st.session_state['vectorizer'], st.session_state['clf'] = load_model_and_vectorizer()
        st.rerun()

    st.title("🛒 E-Commerce Platform with Interactive Sentiment Analytics")

    # ----------------------------
    # Admin Section
    # ----------------------------
    if st.session_state['current_role'] == "Admin":
        st.header("👑 Administrator Panel")
        st.info("Manage products, perform maintenance, and review data integrity.")

        # --- Product Management (Simplified for brevity, same logic as before) ---
        st.subheader("➕ Add & Manage Products")
        
        col_add, col_delete = st.columns(2)
        
        with col_add:
            with st.expander("Add Single Product"):
                with st.form("add_product_form"):
                    name = st.text_input("Product Name", value="New Gadget")
                    price = st.number_input("Price (₹)", min_value=0.0, format="%.2f", value=499.99)
                    region = st.text_input("Region (e.g., North, South)", "Global")
                    image_url = st.text_input("Image URL", "https://via.placeholder.com/150/0000FF/FFFFFF?text=Product")
                    description = st.text_area("Description", "A description of the awesome new product.")
                    submitted = st.form_submit_button("Add Product")
                    
                    if submitted:
                        current_df = st.session_state['df_products']
                        new_id = current_df['id'].max() + 1 if not current_df.empty else 1
                        new_id = int(new_id) 
                        
                        new_row = pd.DataFrame([[new_id, name, price, region, image_url, description]],
                                                columns=current_df.columns)
                        
                        st.session_state['df_products'] = pd.concat([current_df, new_row], ignore_index=True)
                        save_products()
                        st.success(f"Product '{name}' (ID: {new_id}) added successfully! Hard refresh data in sidebar to see changes.")

        with col_delete:
            with st.expander("Delete Product by ID"):
                with st.form("delete_product_form"):
                    delete_id = st.number_input("Product ID to Delete", min_value=1, step=1, key="delete_id")
                    delete_submitted = st.form_submit_button("Delete Product")

                    if delete_submitted:
                        df_products_state = st.session_state['df_products']
                        df_reviews_state = st.session_state['df_reviews']

                        if delete_id in df_products_state['id'].values:
                            product_name = df_products_state[df_products_state['id'] == delete_id]['name'].iloc[0]
                            
                            st.session_state['df_products'] = df_products_state[df_products_state['id'] != delete_id]
                            save_products()
                            
                            reviews_deleted = len(df_reviews_state[df_reviews_state['product_id'] == delete_id])
                            st.session_state['df_reviews'] = df_reviews_state[df_reviews_state['product_id'] != delete_id]
                            save_reviews()
                            
                            st.success(f"Product '{product_name}' (ID: {int(delete_id)}) and its {reviews_deleted} reviews successfully deleted. Hard refresh data to update view.")
                            st.cache_data.clear()
                        else:
                            st.error(f"Product with ID {int(delete_id)} not found.")
        
        st.subheader("📦 Current Products Catalog")
        st.dataframe(st.session_state['df_products'], use_container_width=True)

        st.subheader("🛠 Sentiment Override / Correction")
        if df_reviews.empty:
            st.warning("No reviews available to override.")
        else:
            reviews_with_product_name = df_reviews.merge(
                df_products[['id', 'name']], 
                left_on='product_id', 
                right_on='id', 
                suffixes=('', '_prod')
            )
            reviews_with_product_name['display_name'] = (
                'ID ' + reviews_with_product_name.index.astype(str) + 
                ' | ' + reviews_with_product_name['name_prod'] + 
                ' | Current: ' + reviews_with_product_name['sentiment']
            )

            with st.form("sentiment_override_form"):
                col_select, col_new_sent = st.columns([3, 1])
                
                selected_review_index = col_select.selectbox(
                    "Select Review to Edit", 
                    options=reviews_with_product_name.index.tolist(),
                    format_func=lambda x: reviews_with_product_name.loc[x, 'display_name']
                )

                if selected_review_index in df_reviews.index:
                    current_sentiment = df_reviews.loc[selected_review_index, 'sentiment']
                    
                    col_select.text_area("Review Content:", df_reviews.loc[selected_review_index, 'review'], height=100)
                    
                    new_sentiment = col_new_sent.radio(
                        "New Sentiment:",
                        options=['Positive', 'Neutral', 'Negative'],
                        index=['Positive', 'Neutral', 'Negative'].index(current_sentiment)
                    )

                    override_submitted = st.form_submit_button("Override Sentiment")

                    if override_submitted:
                        st.session_state['df_reviews'].loc[selected_review_index, 'sentiment'] = new_sentiment
                        save_reviews()
                        st.success(f"Review index {selected_review_index} sentiment updated to **{new_sentiment}**.")
                        st.cache_data.clear()
                        st.rerun()


    # ----------------------------
    # User Section & Dashboard 
    # ----------------------------
    
    if st.session_state['current_role'] in ["User", "Admin"]:
        
        st.header("🛍 Product Catalog")

        # Interactive Filter and Search
        col_filter, col_sort, col_search = st.columns([1, 1, 2])
        
        with col_filter:
            region_filter = st.selectbox("Filter by Region", ["All"] + sorted(st.session_state['df_products']['region'].astype(str).unique().tolist()))

        with col_sort:
            sort_option = st.selectbox("Sort By", ["ID", "Price (Low to High)", "Price (High to Low)", "Region"])
        
        with col_search:
            search_query = st.text_input("Search Product (Name or Description)", "")
        
        
        # --- Data Filtering and Sorting ---
        display_products = st.session_state['df_products'] if region_filter == "All" else st.session_state['df_products'][st.session_state['df_products']['region'].astype(str) == region_filter]
        
        if search_query:
            search_query = search_query.lower()
            display_products = display_products[
                display_products['name'].astype(str).str.lower().str.contains(search_query, na=False) |
                display_products['description'].astype(str).str.lower().str.contains(search_query, na=False)
            ]

        if sort_option == "Price (Low to High)":
            display_products = display_products.sort_values(by='price', ascending=True)
        elif sort_option == "Price (High to Low)":
            display_products = display_products.sort_values(by='price', ascending=False)
        elif sort_option == "Region":
            display_products = display_products.sort_values(by='region')
        else:
            display_products = display_products.sort_values(by='id')

        # --- Product Display ---
        if display_products.empty:
            st.warning("No products match your current search criteria.")
        
        cols_per_row = 3
        for i in range(0, len(display_products), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, (_, product) in enumerate(display_products.iloc[i:i+cols_per_row].iterrows()):
                product_id = int(product['id'])
                
                with cols[j]:
                    product_reviews = st.session_state['df_reviews'][st.session_state['df_reviews']['product_id'] == product_id]
                    total_reviews = len(product_reviews)
                    
                    pos_percent = "0%"
                    neu_percent = "0%"
                    neg_percent = "0%"
                    
                    if total_reviews > 0:
                        pos_count = len(product_reviews[product_reviews['sentiment']=='Positive'])
                        neu_count = len(product_reviews[product_reviews['sentiment']=='Neutral'])
                        neg_count = len(product_reviews[product_reviews['sentiment']=='Negative'])
                        
                        pos_percent = f"{(pos_count / total_reviews) * 100:.0f}%"
                        neu_percent = f"{(neu_count / total_reviews) * 100:.0f}%"
                        neg_percent = f"{(neg_count / total_reviews) * 100:.0f}%"
                        
                        
                    # Use custom CSS class for card styling and new detailed metric display
                    st.markdown(f"""
                    <div class="product-card">
                    <h4 style="height: 40px; overflow: hidden;">{product['name']} (ID: {product_id})</h4>
                    <img src="{product['image_url']}" onerror="this.onerror=null;this.src='https://via.placeholder.com/150/EEEEEE/000000?text=No+Image';" width="150" style="border-radius: 5px; margin-bottom: 10px;">
                    <p style="height: 60px; overflow: hidden; font-size: 0.9em; color: #555;">{product['description']}</p>
                    <p><b>Price: ₹{product['price']:.2f}</b></p>
                    
                    <div style='display: flex; justify-content: space-around; font-size: 0.8em; margin-top: 10px;'>
                        <span style='color: #10B981; font-weight: bold;'>{pos_percent} Pos</span>
                        <span style='color: #FBBF24; font-weight: bold;'>{neu_percent} Neu</span>
                        <span style='color: #EF4444; font-weight: bold;'>{neg_percent} Neg</span>
                    </div>
                    <p style='font-size: 0.75em; color: #888;'>({total_reviews} reviews analyzed)</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"Write a Review for {product['name']}"):
                        review_text = st.text_area("Your review here (be specific!):", key=f"review_text_{product_id}")
                        submit_review = st.button("Submit Review & See Sentiment", key=f"submit_review_{product_id}")
                        
                        if submit_review and review_text.strip() != "":
                            if model_ready:
                                sentiment = predict_sentiment(review_text, st.session_state['vectorizer'], st.session_state['clf'])
                                
                                new_review = pd.DataFrame([[product_id, review_text, sentiment, datetime.now()]],
                                                            columns=['product_id', 'review', 'sentiment', 'timestamp'])
                                
                                st.session_state['df_reviews'] = pd.concat([st.session_state['df_reviews'], new_review], ignore_index=True)
                                save_reviews()
                                
                                emoji = "🤩" if sentiment=="Positive" else "🧐" if sentiment=="Neutral" else "😞"
                                st.success(f"Review submitted! Predicted Sentiment: **{sentiment}** {emoji}")
                                st.cache_data.clear()
                                st.rerun() 
                            else:
                                st.error("Cannot submit review: Sentiment model is not loaded.")

        # ----------------------------
        # Dashboard Tabs (Logic remains the same, but uses session state)
        # ----------------------------
        st.markdown("---")
        st.header("📊 Sentiment Analytics Dashboard")

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


            tabs = st.tabs(["Overall Sentiment Breakdown", "Product Performance", "Sentiment Over Time", "Regional Analysis", "Word Analysis", "Raw Reviews Table"])

            # Tab 1: Overall sentiment 
            with tabs[0]:
                st.subheader("Overall Sentiment Distribution")
                fig = px.pie(st.session_state['df_reviews'], names='sentiment', title="Distribution of All Customer Feedback",
                             color='sentiment', 
                             color_discrete_map={'Positive':'#34D399','Neutral':'#FACC15','Negative':'#F87171'})
                st.plotly_chart(fig, use_container_width=True)

            # Tab 2: Per product sentiment
            with tabs[1]:
                st.subheader("Sentiment Count Per Product")
                
                sentiment_summary = st.session_state['df_reviews'].groupby(['product_id','sentiment']).size().unstack(fill_value=0)
                
                sentiment_summary = sentiment_summary.join(
                    st.session_state['df_products'].set_index('id')['name'].rename('Product Name')
                ).fillna(0)
                sentiment_summary = sentiment_summary.reset_index()
                
                for s in ['Positive', 'Neutral', 'Negative']:
                    if s not in sentiment_summary.columns:
                        sentiment_summary[s] = 0

                if not sentiment_summary.empty:
                    fig2 = px.bar(sentiment_summary, x='Product Name', y=['Positive','Neutral','Negative'],
                                  title="Sentiment per Product", 
                                  color_discrete_map={'Positive':'#34D399','Neutral':'#FACC15','Negative':'#F87171'})
                    fig2.update_layout(xaxis_title="Product", yaxis_title="Number of Reviews")
                    st.plotly_chart(fig2, use_container_width=True)
                    
            # Tab 3: Sentiment Over Time
            with tabs[2]:
                st.subheader("Sentiment Trend Over Time (Daily)")
                
                df_reviews_copy = st.session_state['df_reviews'].copy()
                df_reviews_copy['date'] = df_reviews_copy['timestamp'].dt.date
                time_series = df_reviews_copy.groupby(['date', 'sentiment']).size().reset_index(name='count')
                
                fig_time = px.line(time_series, x='date', y='count', color='sentiment',
                                   title="Daily Review Sentiment Count",
                                   color_discrete_map={'Positive':'#34D399','Neutral':'#FACC15','Negative':'#F87171'})
                fig_time.update_layout(xaxis_title="Date", yaxis_title="Number of Reviews")
                st.plotly_chart(fig_time, use_container_width=True)

            # Tab 4: Regional Analysis
            with tabs[3]:
                st.subheader("Regional Sentiment Comparison")
                
                df_merged = st.session_state['df_reviews'].merge(st.session_state['df_products'][['id', 'region']], 
                                              left_on='product_id', 
                                              right_on='id', 
                                              how='left',
                                              suffixes=('_review', '_product'))
                
                regional_summary = df_merged.groupby(['region', 'sentiment']).size().reset_index(name='count')
                
                fig_region = px.bar(regional_summary, x='region', y='count', color='sentiment',
                                    title="Sentiment Distribution by Region",
                                    barmode='group',
                                    color_discrete_map={'Positive':'#34D399','Neutral':'#FACC15','Negative':'#F87171'})
                fig_region.update_layout(xaxis_title="Region", yaxis_title="Number of Reviews")
                st.plotly_chart(fig_region, use_container_width=True)

            # Tab 5: Word Analysis
            with tabs[4]:
                st.subheader("Word Frequency Analysis (Top 20)")
                col_pos_word, col_neg_word = st.columns(2)
                
                positive_words = get_top_words(st.session_state['df_reviews'][st.session_state['df_reviews']['sentiment'] == 'Positive'])
                with col_pos_word:
                    st.markdown("#### 👍 Top Positive Words")
                    if not positive_words.empty:
                        fig_pos = px.bar(positive_words, x='Frequency', y='Word', orientation='h',
                                         title='Most Frequent Words in Positive Reviews',
                                         color_discrete_sequence=['#34D399'])
                        fig_pos.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_pos, use_container_width=True)
                    else:
                         st.info("No positive reviews yet.")

                negative_words = get_top_words(st.session_state['df_reviews'][st.session_state['df_reviews']['sentiment'] == 'Negative'])
                with col_neg_word:
                    st.markdown("#### 👎 Top Negative Words")
                    if not negative_words.empty:
                        fig_neg = px.bar(negative_words, x='Frequency', y='Word', orientation='h',
                                         title='Most Frequent Words in Negative Reviews',
                                         color_discrete_sequence=['#F87171'])
                        fig_neg.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_neg, use_container_width=True)
                    else:
                        st.info("No negative reviews yet.")


            # Tab 6: Raw Reviews table
            with tabs[5]:
                st.subheader("🔍 All Customer Reviews")
                
                review_filter = st.multiselect(
                    "Filter by Sentiment Type", 
                    options=['Positive', 'Neutral', 'Negative'], 
                    default=['Positive', 'Neutral', 'Negative'],
                    key="review_table_filter"
                )

                filtered_reviews = st.session_state['df_reviews'][st.session_state['df_reviews']['sentiment'].isin(review_filter)].copy()
                
                filtered_reviews = filtered_reviews.join(
                    st.session_state['df_products'].set_index('id')['name'].rename('Product Name'), 
                    on='product_id'
                )
                
                display_df = filtered_reviews[['Product Name', 'review', 'sentiment', 'product_id', 'timestamp']]
                
                st.dataframe(
                    display_df, 
                    use_container_width=True,
                    column_config={
                        "review": st.column_config.TextColumn("Review Content", width="large"),
                        "sentiment": st.column_config.TextColumn("Predicted Sentiment", width="small"),
                        "Product Name": st.column_config.TextColumn("Product Name", width="medium"),
                        "product_id": "ID",
                        "timestamp": st.column_config.DatetimeColumn("Review Date", format="YYYY-MM-DD HH:mm")
                    }
                )
