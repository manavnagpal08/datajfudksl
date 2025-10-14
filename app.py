import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px

# --- Configuration and Constants ---
PRODUCTS_FILE = "data/products.csv"
REVIEWS_FILE = "data/reviews.csv"
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

# ----------------------------
# Data Loading and Utility Functions
# ----------------------------

@st.cache_data(show_spinner=False)
def load_data():
    """Loads and initializes products and reviews DataFrames."""
    os.makedirs("data", exist_ok=True)

    # Initialize products CSV
    if os.path.exists(PRODUCTS_FILE):
        df_products = pd.read_csv(PRODUCTS_FILE)
    else:
        # Define specific dtypes to prevent Streamlit type inference warnings
        df_products = pd.DataFrame(columns=['id', 'name', 'price', 'region', 'image_url', 'description'])
        df_products['id'] = df_products['id'].astype(float)
        df_products.to_csv(PRODUCTS_FILE, index=False)

    # Initialize reviews CSV
    if os.path.exists(REVIEWS_FILE):
        df_reviews = pd.read_csv(REVIEWS_FILE)
    else:
        df_reviews = pd.DataFrame(columns=['product_id', 'review', 'sentiment'])
        df_reviews['product_id'] = df_reviews['product_id'].astype(float)
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
# Page Setup
# ----------------------------
st.set_page_config(page_title="E-Commerce Sentiment App", layout="wide")
st.title("🛒 E-Commerce Platform with Interactive Sentiment Analytics")

# Check model availability and load necessary dataframes
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    st.error("""
        **🚨 Sentiment Model Not Found!** Please ensure the required files (`sentiment_model.pkl` and `vectorizer.pkl`) 
        are present in the `model/` directory. The prediction functionality is disabled.
    """)
    st.stop()

# Load initial data
df_products, df_reviews = load_data()


# ----------------------------
# Role Selection
# ----------------------------
role = st.sidebar.radio("Select Role", ["User", "Admin"], index=0)

st.sidebar.markdown("---")
if 'df_products' in locals():
    st.sidebar.metric("Total Products", len(df_products))
if 'df_reviews' in locals():
    st.sidebar.metric("Total Reviews", len(df_reviews))


# ----------------------------
# Admin Section
# ----------------------------
if role == "Admin":
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
                    # Ensure ID is an integer for cleaner look
                    new_id = int(new_id) 
                    
                    new_row = pd.DataFrame([[new_id, name, price, region, image_url, description]],
                                            columns=df_products.columns)
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
                        
                        # Delete from products
                        df_products = df_products[df_products['id'] != delete_id]
                        save_products(df_products)
                        
                        # Delete associated reviews
                        global df_reviews # Need to modify the global reviews DF
                        reviews_deleted = len(df_reviews[df_reviews['product_id'] == delete_id])
                        df_reviews = df_reviews[df_reviews['product_id'] != delete_id]
                        save_reviews(df_reviews)
                        
                        st.success(f"Product '{product_name}' (ID: {int(delete_id)}) and its {reviews_deleted} reviews successfully deleted. **Refresh to see changes.**")
                        st.cache_data.clear() # Clear cache to force reload
                    else:
                        st.error(f"Product with ID {int(delete_id)} not found.")


    # --- Bulk Upload ---
    with st.expander("Bulk Upload Products (CSV)"):
        st.markdown("Expected CSV columns: `name`, `price`, `region`, `image_url`, `description`")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            try:
                bulk_df = pd.read_csv(uploaded_file)
                required_cols = ['name', 'price', 'region', 'image_url', 'description']
                if all(col in bulk_df.columns for col in required_cols):
                    start_id = df_products['id'].max() + 1 if not df_products.empty else 1
                    bulk_df['id'] = range(int(start_id), int(start_id) + len(bulk_df))
                    
                    # Ensure price is numeric
                    bulk_df['price'] = pd.to_numeric(bulk_df['price'], errors='coerce')
                    bulk_df.dropna(subset=['price'], inplace=True)
                    
                    df_products = pd.concat([df_products, bulk_df[['id'] + required_cols]], ignore_index=True)
                    save_products(df_products)
                    st.success(f"{len(bulk_df)} products added successfully! **Refresh to see changes.**")
                    st.cache_data.clear() # Clear cache to force reload
                else:
                    st.error(f"CSV must contain columns: {required_cols}")
            except Exception as e:
                st.error(f"An error occurred during bulk upload: {e}")

    st.subheader("📦 Current Products Catalog")
    st.dataframe(df_products, use_container_width=True)

    # --- Maintenance Section ---
    st.subheader("🧹 Database Maintenance")
    if st.button("🔴 Clear ALL Reviews (DANGER ZONE)", help="This action is irreversible."):
        if st.session_state.get('confirm_clear_reviews', False):
            # Final confirmation step
            global df_reviews
            df_reviews = pd.DataFrame(columns=['product_id', 'review', 'sentiment'])
            df_reviews['product_id'] = df_reviews['product_id'].astype(float)
            save_reviews(df_reviews)
            st.success("✅ All reviews have been successfully cleared.")
            st.session_state['confirm_clear_reviews'] = False
            st.cache_data.clear() # Clear cache to force reload
        else:
            st.warning("Are you sure? Click again to confirm clearing ALL reviews.")
            st.session_state['confirm_clear_reviews'] = True
            
# ----------------------------
# User Section
# ----------------------------
else: # Role == "User"
    st.header("🛍 Browse Products")

    # Interactive Filter and Search
    col_filter, col_search = st.columns([1, 2])
    
    with col_filter:
        region_filter = st.selectbox("Filter by Region", ["All"] + sorted(df_products['region'].unique().astype(str).tolist()))

    with col_search:
        search_query = st.text_input("Search Product (Name or Description)", "")
    
    # 1. Apply Region Filter
    display_products = df_products if region_filter == "All" else df_products[df_products['region'] == region_filter]
    
    # 2. Apply Search Filter
    if search_query:
        search_query = search_query.lower()
        display_products = display_products[
            display_products['name'].str.lower().str.contains(search_query, na=False) |
            display_products['description'].str.lower().str.contains(search_query, na=False)
        ]

    if display_products.empty:
        st.warning("No products match your current search criteria.")
    
    # Display products in columns
    cols_per_row = 3
    for i in range(0, len(display_products), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (_, product) in enumerate(display_products.iloc[i:i+cols_per_row].iterrows()):
            product_id = int(product['id'])
            
            with cols[j]:
                # Calculate avg sentiment
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

                st.markdown(f"""
                <div style="
                    border:1px solid #ccc;
                    border-radius:10px;
                    padding:15px;
                    margin-bottom: 20px;
                    min-height: 400px; /* Ensure uniform card height */
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    background-color:#ffffff;
                    text-align:center;
                    ">
                <h4 style="height: 40px; overflow: hidden;">{product['name']} (ID: {product_id})</h4>
                <img src="{product['image_url']}" onerror="this.onerror=null;this.src='https://via.placeholder.com/150/EEEEEE/000000?text=No+Image';" width="150" style="border-radius: 5px; margin-bottom: 10px;">
                <p style="height: 60px; overflow: hidden; font-size: 0.9em; color: #555;">{product['description']}</p>
                <p><b>Price: ₹{product['price']:.2f}</b></p>
                <p style='color:{sentiment_color};'><b>{avg_sent}</b></p>
                </div>
                """, unsafe_allow_html=True)

                # Expander for review submission
                with st.expander(f"Write a Review for {product['name']}"):
                    review_text = st.text_area("Your review here (be specific!):", key=f"review_text_{product_id}")
                    submit_review = st.button("Submit Review & See Sentiment", key=f"submit_review_{product_id}")
                    
                    if submit_review and review_text.strip() != "":
                        sentiment = predict_sentiment(review_text)
                        
                        new_review = pd.DataFrame([[product_id, review_text, sentiment]],
                                                    columns=['product_id', 'review', 'sentiment'])
                        
                        # Use global df_reviews to update the state
                        global df_reviews
                        df_reviews = pd.concat([df_reviews, new_review], ignore_index=True)
                        save_reviews(df_reviews)
                        
                        emoji = "🤩" if sentiment=="Positive" else "🧐" if sentiment=="Neutral" else "😞"
                        st.success(f"Review submitted! Predicted Sentiment: **{sentiment}** {emoji}")
                        st.cache_data.clear() # Clear cache to force reload the dataframes

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


        tabs = st.tabs(["Overall Sentiment Breakdown", "Product Performance Bar Chart", "Raw Reviews Table"])

        # Tab 1: Overall sentiment
        with tabs[0]:
            st.subheader("Overall Sentiment Distribution")
            fig = px.pie(df_reviews, names='sentiment', title="Distribution of All Customer Feedback",
                         color='sentiment', 
                         color_discrete_map={'Positive':'#34D399','Neutral':'#A0AEC0','Negative':'#F87171'})
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Per product sentiment
        with tabs[1]:
            st.subheader("Sentiment Count Per Product")
            
            # Aggregate sentiment counts and join with product name
            sentiment_summary = df_reviews.groupby(['product_id','sentiment']).size().unstack(fill_value=0)
            
            # Map product IDs to names
            sentiment_summary = sentiment_summary.join(
                df_products.set_index('id')['name'].rename('Product Name')
            ).fillna(0)
            sentiment_summary = sentiment_summary.reset_index()
            
            # Ensure all sentiment columns exist for consistent plotting
            for s in ['Positive', 'Neutral', 'Negative']:
                if s not in sentiment_summary.columns:
                    sentiment_summary[s] = 0

            if not sentiment_summary.empty:
                fig2 = px.bar(sentiment_summary, x='Product Name', y=['Positive','Neutral','Negative'],
                              title="Sentiment per Product", 
                              color_discrete_map={'Positive':'#34D399','Neutral':'#A0AEC0','Negative':'#F87171'})
                fig2.update_layout(xaxis_title="Product", yaxis_title="Number of Reviews")
                st.plotly_chart(fig2, use_container_width=True)

        # Tab 3: Interactive Reviews table
        with tabs[2]:
            st.subheader("🔍 All Customer Reviews")
            
            # Add interactive filtering
            review_filter = st.multiselect(
                "Filter by Sentiment Type", 
                options=['Positive', 'Neutral', 'Negative'], 
                default=['Positive', 'Neutral', 'Negative']
            )

            filtered_reviews = df_reviews[df_reviews['sentiment'].isin(review_filter)].copy()
            
            # Join with product name for better visibility
            filtered_reviews = filtered_reviews.join(
                df_products.set_index('id')['name'].rename('Product Name'), 
                on='product_id'
            )
            
            # Select and reorder columns for display
            display_df = filtered_reviews[['Product Name', 'review', 'sentiment', 'product_id']]
            
            st.dataframe(
                display_df, 
                use_container_width=True,
                column_config={
                    "review": st.column_config.TextColumn("Review Content", width="large"),
                    "sentiment": st.column_config.TextColumn("Predicted Sentiment", width="small"),
                    "Product Name": st.column_config.TextColumn("Product Name", width="medium"),
                    "product_id": "ID"
                }
            )

# Clear cache button for development/admin use
st.sidebar.markdown("---")
if st.sidebar.button("Hard Refresh Data"):
    st.cache_data.clear()
    st.rerun()
