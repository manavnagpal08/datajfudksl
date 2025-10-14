import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="E-Commerce Sentiment App", layout="wide")
st.title("🛒 E-Commerce Platform with Interactive Sentiment Analytics")

# ----------------------------
# Load Model
# ----------------------------
if not os.path.exists("model/sentiment_model.pkl") or not os.path.exists("model/vectorizer.pkl"):
    st.error("Sentiment model or vectorizer not found! Train the model first.")
    st.stop()

vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
clf = pickle.load(open("model/sentiment_model.pkl", "rb"))

def predict_sentiment(text):
    X = vectorizer.transform([text])
    return clf.predict(X)[0]

# ----------------------------
# Setup CSV Files
# ----------------------------
os.makedirs("data", exist_ok=True)

products_file = "data/products.csv"
reviews_file = "data/reviews.csv"

# Initialize products CSV
if os.path.exists(products_file):
    df_products = pd.read_csv(products_file)
else:
    df_products = pd.DataFrame(columns=['id', 'name', 'price', 'region', 'image_url', 'description'])
    df_products.to_csv(products_file, index=False)

# Initialize reviews CSV
if os.path.exists(reviews_file):
    df_reviews = pd.read_csv(reviews_file)
    for col in ['product_id', 'review', 'sentiment']:
        if col not in df_reviews.columns:
            df_reviews[col] = ''
else:
    df_reviews = pd.DataFrame(columns=['product_id', 'review', 'sentiment'])
    df_reviews.to_csv(reviews_file, index=False)

# ----------------------------
# Role Selection
# ----------------------------
role = st.radio("Select Role", ["Admin", "User"])

# ----------------------------
# Admin Section
# ----------------------------
if role == "Admin":
    st.header("➕ Add Products")
    with st.expander("Add Single Product"):
        with st.form("add_product_form"):
            name = st.text_input("Product Name")
            description = st.text_area("Description")
            price = st.number_input("Price", min_value=0)
            region = st.text_input("Region", "All")
            image_url = st.text_input("Image URL", "https://via.placeholder.com/150")
            submitted = st.form_submit_button("Add Product")
            if submitted:
                new_id = df_products['id'].max() + 1 if not df_products.empty else 1
                new_row = pd.DataFrame([[new_id, name, price, region, image_url, description]],
                                       columns=df_products.columns)
                df_products = pd.concat([df_products, new_row], ignore_index=True)
                df_products.to_csv(products_file, index=False)
                st.success(f"Product '{name}' added successfully!")

    # Bulk upload
    with st.expander("Bulk Upload Products (CSV)"):
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            bulk_df = pd.read_csv(uploaded_file)
            required_cols = ['name', 'price', 'region', 'image_url', 'description']
            if all(col in bulk_df.columns for col in required_cols):
                bulk_df['id'] = range(df_products['id'].max() + 1 if not df_products.empty else 1,
                                      df_products['id'].max() + 1 + len(bulk_df))
                df_products = pd.concat([df_products, bulk_df[['id'] + required_cols]], ignore_index=True)
                df_products.to_csv(products_file, index=False)
                st.success(f"{len(bulk_df)} products added successfully!")
            else:
                st.error(f"CSV must contain columns: {required_cols}")

    st.subheader("📦 Current Products")
    st.dataframe(df_products)

# ----------------------------
# User Section
# ----------------------------
else:
    st.header("🛍 Products")
    region_filter = st.selectbox("Filter by Region", ["All"] + sorted(df_products['region'].unique()))
    display_products = df_products if region_filter == "All" else df_products[df_products['region'] == region_filter]

    # Display products in columns
    cols_per_row = 3
    for i in range(0, len(display_products), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (_, product) in enumerate(display_products.iloc[i:i+cols_per_row].iterrows()):
            with cols[j]:
                # Calculate avg sentiment
                product_reviews = df_reviews[df_reviews['product_id'] == product['id']]
                if not product_reviews.empty:
                    avg_pos = len(product_reviews[product_reviews['sentiment']=='Positive']) / len(product_reviews)
                    avg_sent = f"{avg_pos*100:.0f}% Positive"
                else:
                    avg_sent = "No Reviews Yet"

                st.markdown(f"""
                <div style="
                    border:1px solid #ccc;
                    border-radius:10px;
                    padding:10px;
                    box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
                    background-color:#fefefe;
                    text-align:center;
                    ">
                <h4>{product['name']}</h4>
                <img src="{product['image_url']}" width="150">
                <p>{product['description']}</p>
                <p><b>Price: ₹{product['price']}</b></p>
                <p style='color:green;'><b>{avg_sent}</b></p>
                </div>
                """, unsafe_allow_html=True)

                # Expander for review submission
                with st.expander("Write a Review"):
                    review_text = st.text_area("Your review here:", key=f"review_{product['id']}")
                    submit_review = st.button("Submit Review", key=f"submit_{product['id']}")
                    if submit_review and review_text.strip() != "":
                        sentiment = predict_sentiment(review_text)
                        new_review = pd.DataFrame([[product['id'], review_text, sentiment]],
                                                  columns=['product_id', 'review', 'sentiment'])
                        df_reviews = pd.concat([df_reviews, new_review], ignore_index=True)
                        df_reviews.to_csv(reviews_file, index=False)
                        emoji = "😊" if sentiment=="Positive" else "😐" if sentiment=="Neutral" else "😞"
                        st.success(f"Review submitted! Sentiment: {sentiment} {emoji}")

    # ----------------------------
    # Dashboard Tabs
    # ----------------------------
    if not df_reviews.empty:
        st.header("📊 Sentiment Analytics Dashboard")
        tabs = st.tabs(["Overall Sentiment", "Per Product Sentiment", "Reviews"])

        # Tab 1: Overall sentiment
        with tabs[0]:
            fig = px.pie(df_reviews, names='sentiment', title="Overall Sentiment Distribution",
                         color='sentiment', color_discrete_map={'Positive':'green','Neutral':'gray','Negative':'red'})
            st.plotly_chart(fig)

        # Tab 2: Per product sentiment
        with tabs[1]:
            sentiment_summary = df_reviews.groupby(['product_id','sentiment']).size().unstack(fill_value=0)
            sentiment_summary = sentiment_summary.join(df_products.set_index('id')['name'])
            sentiment_summary = sentiment_summary.reset_index()
            if not sentiment_summary.empty:
                fig2 = px.bar(sentiment_summary, x='name', y=['Positive','Neutral','Negative'],
                              title="Sentiment per Product", color_discrete_map={'Positive':'green','Neutral':'gray','Negative':'red'})
                st.plotly_chart(fig2)

        # Tab 3: Reviews table
        with tabs[2]:
            st.subheader("All Reviews")
            st.dataframe(df_reviews[['product_id','review','sentiment']])
