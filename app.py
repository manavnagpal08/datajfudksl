import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px

st.set_page_config(page_title="E-Commerce Sentiment App", layout="wide")
st.title("🛒 E-Commerce Platform with Sentiment Analysis")

# ----------------------------
# Load Model and Vectorizer
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
# Files and Data Setup
# ----------------------------
os.makedirs("data", exist_ok=True)

# Products CSV
products_file = "data/products.csv"
if os.path.exists(products_file):
    df_products = pd.read_csv(products_file)
else:
    df_products = pd.DataFrame(columns=['id', 'name', 'price', 'region', 'image_url', 'description'])
    df_products.to_csv(products_file, index=False)

# Reviews CSV
reviews_file = "data/reviews.csv"
if os.path.exists(reviews_file):
    df_reviews = pd.read_csv(reviews_file)
else:
    df_reviews = pd.DataFrame(columns=['product_id', 'review', 'sentiment'])
    df_reviews.to_csv(reviews_file, index=False)

# ----------------------------
# Select Role
# ----------------------------
role = st.radio("Select Role", ["Admin", "User"])

# ----------------------------
# Admin Section
# ----------------------------
if role == "Admin":
    st.header("➕ Add New Product")
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

    st.subheader("📦 Current Products")
    st.dataframe(df_products)

# ----------------------------
# User Section
# ----------------------------
else:
    st.header("🛍 Products")
    region_filter = st.selectbox("Filter by Region", ["All"] + sorted(df_products['region'].unique()))
    display_products = df_products if region_filter == "All" else df_products[df_products['region'] == region_filter]

    for _, product in display_products.iterrows():
        st.subheader(product['name'])
        st.image(product['image_url'], width=150)
        st.write(product['description'])
        st.write(f"Price: ₹{product['price']}")

        # Review submission form
        with st.form(f"review_form_{product['id']}"):
            review_text = st.text_area("Write a review:", key=f"review_{product['id']}")
            submit_review = st.form_submit_button("Submit Review")
            if submit_review and review_text.strip() != "":
                sentiment = predict_sentiment(review_text)
                new_review = pd.DataFrame([[product['id'], review_text, sentiment]],
                                          columns=df_reviews.columns)
                df_reviews = pd.concat([df_reviews, new_review], ignore_index=True)
                df_reviews.to_csv(reviews_file, index=False)
                st.success(f"Review submitted! Predicted Sentiment: {sentiment}")

    # ----------------------------
    # Sentiment Dashboard
    # ----------------------------
    if not df_reviews.empty:
        st.header("📈 Sentiment Analysis Dashboard")
        fig = px.histogram(df_reviews, x='sentiment', color='sentiment',
                           title="Overall Sentiment Distribution")
        st.plotly_chart(fig)

        st.subheader("Reviews by Product")
        for _, product in display_products.iterrows():
            product_reviews = df_reviews[df_reviews['product_id'] == product['id']]
            if not product_reviews.empty:
                st.write(f"**{product['name']}** Reviews:")
                st.dataframe(product_reviews[['review', 'sentiment']])
