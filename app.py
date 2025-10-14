import streamlit as st
import pandas as pd
import pickle
import os
import json # ADDED: Required for JSON bulk upload
import plotly.express as px
from datetime import datetime

# --- Configuration and Constants ---
PRODUCTS_FILE = "data/products.csv"
REVIEWS_FILE = "data/reviews.csv"
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"

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
        min-height: 420px;
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
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Data Loading and Utility Functions
# ----------------------------

@st.cache_data(show_spinner="Loading Data...")
def load_data():
    """Loads and initializes products and reviews DataFrames."""
    os.makedirs("data", exist_ok=True)

    # --- Products Initialization ---
    if os.path.exists(PRODUCTS_FILE):
        df_products = pd.read_csv(PRODUCTS_FILE)
    else:
        df_products = pd.DataFrame(columns=['id', 'name', 'price', 'region', 'image_url', 'description'])
        df_products['id'] = df_products['id'].astype('Int64')
        df_products.to_csv(PRODUCTS_FILE, index=False)
    
    df_products['id'] = pd.to_numeric(df_products['id'], errors='coerce').fillna(0).astype('Int64')
    
    # --- Reviews Initialization (Robust against KeyError) ---
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
            pass

    # Post-load cleanup: Ensure types and handle timestamp
    df_reviews['product_id'] = pd.to_numeric(df_reviews['product_id'], errors='coerce').fillna(0).astype('Int64')
    if 'timestamp' not in df_reviews.columns:
        df_reviews['timestamp'] = pd.NaT
        
    df_reviews['timestamp'] = pd.to_datetime(df_reviews['timestamp'], errors='coerce')
    # Fill missing timestamps with a default for consistent plotting
    df_reviews['timestamp'] = df_reviews['timestamp'].fillna(pd.to_datetime('2024-01-01 00:00:00'))
    
    if df_reviews.empty and (not os.path.exists(REVIEWS_FILE) or os.path.getsize(REVIEWS_FILE) == 0):
         df_reviews.to_csv(REVIEWS_FILE, index=False)

    return df_products, df_reviews

def save_products(df):
    """Saves the products DataFrame to CSV."""
    df.to_csv(PRODUCTS_FILE, index=False)

def save_reviews(df):
    """Saves the reviews DataFrame to CSV."""
    df.to_csv(REVIEWS_FILE, index=False)

def predict_sentiment(text):
    """Uses the loaded model to predict sentiment."""
    try:
        # Load model and vectorizer only once
        if 'vectorizer' not in st.session_state:
            st.session_state['vectorizer'] = pickle.load(open(VECTORIZER_PATH, "rb"))
            st.session_state['clf'] = pickle.load(open(MODEL_PATH, "rb"))
            
        X = st.session_state['vectorizer'].transform([text])
        return st.session_state['clf'].predict(X)[0]
    except FileNotFoundError:
        return "Model Error"
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Model Error"

# ----------------------------
# Authentication and Initialization
# ----------------------------

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['current_role'] = 'Guest'

# Check model availability
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    st.error("🚨 Sentiment Model Not Found! Prediction functionality is disabled.")
    st.stop()

# Load initial data
df_products, df_reviews = load_data()


def login_screen():
    """Renders the login/role selection interface."""
    st.sidebar.markdown("### 🔑 Login")
    
    selected_role = st.sidebar.radio("Choose Role", ["User (Guest)", "Admin"], index=0)
    
    if selected_role == "User (Guest)":
        if st.sidebar.button("Continue as User"):
            st.session_state['logged_in'] = True
            st.session_state['current_role'] = 'User'
            st.rerun()
    
    elif selected_role == "Admin":
        with st.sidebar.form("admin_login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Admin Login")
            
            if submitted:
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state['logged_in'] = True
                    st.session_state['current_role'] = 'Admin'
                    st.sidebar.success("Logged in as Admin!")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid Admin credentials.")
                    
def logout():
    """Logs out the current user."""
    st.session_state['logged_in'] = False
    st.session_state['current_role'] = 'Guest'
    st.info("You have been logged out.")
    st.rerun()

# ----------------------------
# Main Application Flow
# ----------------------------

st.title("🛒 E-Commerce Platform with Interactive Sentiment Analytics")

if not st.session_state['logged_in']:
    login_screen()
    st.info("Please select a role to access the application. Admin credentials: `admin` / `password123`")
else:
    # --- Sidebar Metrics and Logout ---
    st.sidebar.markdown(f"### 👋 Welcome, {st.session_state['current_role']}!")
    if st.sidebar.button("Logout", key="logout_btn"):
        logout()
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Products", len(df_products))
    st.sidebar.metric("Total Reviews", len(df_reviews))
    st.sidebar.markdown("---")
    if st.sidebar.button("Hard Refresh Data"):
        st.cache_data.clear()
        st.rerun()


    # ----------------------------
    # Admin Section
    # ----------------------------
    if st.session_state['current_role'] == "Admin":
        st.header("👑 Administrator Panel")
        st.info("Manage products, perform maintenance, and review data integrity.")

        # --- Product Management ---
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
                        new_id = df_products['id'].max() + 1 if not df_products.empty else 1
                        new_id = int(new_id) 
                        
                        new_row = pd.DataFrame([[new_id, name, price, region, image_url, description]],
                                                columns=df_products.columns)
                        global df_products
                        df_products = pd.concat([df_products, new_row], ignore_index=True)
                        save_products(df_products)
                        st.success(f"Product '{name}' (ID: {new_id}) added successfully! **Refresh to see changes.**")
        
        with col_delete:
            with st.expander("Delete Product by ID"):
                with st.form("delete_product_form"):
                    delete_id = st.number_input("Product ID to Delete", min_value=1, step=1, key="delete_id")
                    delete_submitted = st.form_submit_button("Delete Product")

                    if delete_submitted:
                        if delete_id in df_products['id'].values:
                            product_name = df_products[df_products['id'] == delete_id]['name'].iloc[0]
                            
                            global df_reviews
                            df_products = df_products[df_products['id'] != delete_id]
                            save_products(df_products)
                            
                            reviews_deleted = len(df_reviews[df_reviews['product_id'] == delete_id])
                            df_reviews = df_reviews[df_reviews['product_id'] != delete_id]
                            save_reviews(df_reviews)
                            
                            st.success(f"Product '{product_name}' (ID: {int(delete_id)}) and its {reviews_deleted} reviews successfully deleted. **Hard refresh to update view.**")
                            st.cache_data.clear()
                        else:
                            st.error(f"Product with ID {int(delete_id)} not found.")


        # --- Bulk Upload (CSV/JSON) ---
        with st.expander("🚀 Bulk Upload Products (CSV or JSON)"):
            st.markdown("Expected fields: `name`, `price`, `region`, `image_url`, `description`")
            uploaded_file = st.file_uploader("Upload File", type=["csv", "json"])
            
            if uploaded_file:
                try:
                    file_extension = uploaded_file.name.split('.')[-1]
                    
                    if file_extension == 'csv':
                        bulk_df = pd.read_csv(uploaded_file)
                    elif file_extension == 'json':
                        data = json.load(uploaded_file)
                        bulk_df = pd.DataFrame(data)
                    else:
                        st.error("Unsupported file type.")
                        bulk_df = pd.DataFrame() # Stop processing

                    required_cols = ['name', 'price', 'region', 'image_url', 'description']
                    
                    if bulk_df.empty or not all(col in bulk_df.columns for col in required_cols):
                        st.error(f"Uploaded file is empty or missing required columns: {required_cols}")
                    else:
                        start_id = df_products['id'].max() + 1 if not df_products.empty else 1
                        bulk_df['id'] = range(int(start_id), int(start_id) + len(bulk_df))
                        
                        bulk_df['price'] = pd.to_numeric(bulk_df['price'], errors='coerce')
                        bulk_df.dropna(subset=['price'], inplace=True)
                        
                        global df_products
                        df_products = pd.concat([df_products, bulk_df[['id'] + required_cols]], ignore_index=True)
                        save_products(df_products)
                        st.success(f"{len(bulk_df)} products added successfully from {file_extension.upper()} file! **Hard refresh to update view.**")
                        st.cache_data.clear()
                        
                except Exception as e:
                    st.error(f"An error occurred during bulk upload: {e}")

        st.subheader("📦 Current Products Catalog")
        st.dataframe(df_products, use_container_width=True)

        # --- Maintenance Section ---
        st.subheader("🧹 Database Maintenance")
        if st.button("🔴 Clear ALL Reviews (DANGER ZONE)", help="This action is irreversible."):
            if st.session_state.get('confirm_clear_reviews', False):
                global df_reviews
                df_reviews = pd.DataFrame(columns=['product_id', 'review', 'sentiment', 'timestamp'])
                df_reviews['product_id'] = df_reviews['product_id'].astype('Int64')
                save_reviews(df_reviews)
                st.success("✅ All reviews have been successfully cleared. **Hard refresh to update view.**")
                st.session_state['confirm_clear_reviews'] = False
                st.cache_data.clear()
            else:
                st.warning("Are you sure? Click again to confirm clearing ALL reviews.")
                st.session_state['confirm_clear_reviews'] = True
            
    # ----------------------------
    # User Section & Dashboard (Access for both Admin and User)
    # ----------------------------
    
    if st.session_state['current_role'] == "User" or st.session_state['current_role'] == "Admin":
        
        st.header("🛍 Product Catalog")

        # Interactive Filter and Search
        col_filter, col_sort, col_search = st.columns([1, 1, 2])
        
        with col_filter:
            region_filter = st.selectbox("Filter by Region", ["All"] + sorted(df_products['region'].astype(str).unique().tolist()))

        with col_sort:
            sort_option = st.selectbox("Sort By", ["ID", "Price (Low to High)", "Price (High to Low)", "Region"])
        
        with col_search:
            search_query = st.text_input("Search Product (Name or Description)", "")
        
        
        # --- Data Filtering and Sorting ---
        display_products = df_products if region_filter == "All" else df_products[df_products['region'].astype(str) == region_filter]
        
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
                    product_reviews = df_reviews[df_reviews['product_id'] == product_id]
                    
                    total_reviews = len(product_reviews)
                    if total_reviews > 0:
                        pos_count = len(product_reviews[product_reviews['sentiment']=='Positive'])
                        avg_pos = pos_count / total_reviews
                        avg_sent = f"{avg_pos*100:.0f}% Positive ({total_reviews} reviews)"
                        sentiment_color = 'green'
                    else:
                        avg_sent = "No Reviews Yet"
                        sentiment_color = 'gray'

                    # Use custom CSS class for card styling
                    st.markdown(f"""
                    <div class="product-card">
                    <h4 style="height: 40px; overflow: hidden;">{product['name']} (ID: {product_id})</h4>
                    <img src="{product['image_url']}" onerror="this.onerror=null;this.src='https://via.placeholder.com/150/EEEEEE/000000?text=No+Image';" width="150" style="border-radius: 5px; margin-bottom: 10px;">
                    <p style="height: 60px; overflow: hidden; font-size: 0.9em; color: #555;">{product['description']}</p>
                    <p><b>Price: ₹{product['price']:.2f}</b></p>
                    <p style='color:{sentiment_color};'><b>{avg_sent}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"Write a Review for {product['name']}"):
                        review_text = st.text_area("Your review here (be specific!):", key=f"review_text_{product_id}")
                        submit_review = st.button("Submit Review & See Sentiment", key=f"submit_review_{product_id}")
                        
                        if submit_review and review_text.strip() != "":
                            sentiment = predict_sentiment(review_text)
                            
                            new_review = pd.DataFrame([[product_id, review_text, sentiment, datetime.now()]],
                                                        columns=['product_id', 'review', 'sentiment', 'timestamp'])
                            
                            global df_reviews
                            df_reviews = pd.concat([df_reviews, new_review], ignore_index=True)
                            save_reviews(df_reviews)
                            
                            emoji = "🤩" if sentiment=="Positive" else "🧐" if sentiment=="Neutral" else "😞"
                            st.success(f"Review submitted! Predicted Sentiment: **{sentiment}** {emoji}")
                            st.cache_data.clear() # Clear cache to force reload and update dashboard

        # ----------------------------
        # Dashboard Tabs
        # ----------------------------
        st.markdown("---")
        st.header("📊 Sentiment Analytics Dashboard")

        if df_reviews.empty:
            st.info("No reviews have been submitted yet to generate the dashboard.")
        else:
            
            # --- KPIs ---
            total_reviews = len(df_reviews)
            positive_reviews = len(df_reviews[df_reviews['sentiment'] == 'Positive'])
            positive_rate = f"{ (positive_reviews / total_reviews) * 100 :.1f}%" if total_reviews > 0 else "0%"

            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            col_kpi1.metric("Total Reviews Analyzed", total_reviews)
            col_kpi2.metric("Overall Positive Rate", positive_rate)
            col_kpi3.metric("Total Products in Catalog", len(df_products))


            tabs = st.tabs(["Overall Sentiment Breakdown", "Product Performance", "Sentiment Over Time", "Regional Analysis", "Raw Reviews Table"])

            # Tab 1: Overall sentiment 
            with tabs[0]:
                st.subheader("Overall Sentiment Distribution")
                fig = px.pie(df_reviews, names='sentiment', title="Distribution of All Customer Feedback",
                             color='sentiment', 
                             color_discrete_map={'Positive':'#34D399','Neutral':'#FACC15','Negative':'#F87171'})
                st.plotly_chart(fig, use_container_width=True)

            # Tab 2: Per product sentiment
            with tabs[1]:
                st.subheader("Sentiment Count Per Product")
                
                sentiment_summary = df_reviews.groupby(['product_id','sentiment']).size().unstack(fill_value=0)
                
                sentiment_summary = sentiment_summary.join(
                    df_products.set_index('id')['name'].rename('Product Name')
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
                
                df_reviews['date'] = df_reviews['timestamp'].dt.date
                time_series = df_reviews.groupby(['date', 'sentiment']).size().reset_index(name='count')
                
                fig_time = px.line(time_series, x='date', y='count', color='sentiment',
                                   title="Daily Review Sentiment Count",
                                   color_discrete_map={'Positive':'#34D399','Neutral':'#FACC15','Negative':'#F87171'})
                fig_time.update_layout(xaxis_title="Date", yaxis_title="Number of Reviews")
                st.plotly_chart(fig_time, use_container_width=True)

            # Tab 4: Regional Analysis
            with tabs[3]:
                st.subheader("Regional Sentiment Comparison")
                
                df_merged = df_reviews.merge(df_products[['id', 'region']], 
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

            # Tab 5: Raw Reviews table
            with tabs[4]:
                st.subheader("🔍 All Customer Reviews")
                
                review_filter = st.multiselect(
                    "Filter by Sentiment Type", 
                    options=['Positive', 'Neutral', 'Negative'], 
                    default=['Positive', 'Neutral', 'Negative'],
                    key="review_table_filter"
                )

                filtered_reviews = df_reviews[df_reviews['sentiment'].isin(review_filter)].copy()
                
                filtered_reviews = filtered_reviews.join(
                    df_products.set_index('id')['name'].rename('Product Name'), 
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
