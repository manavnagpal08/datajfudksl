import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px

# ----------------------------
# Page Configuration
# ----------------------------
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
# Data Setup
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

    # Display products in columns
    cols_per_row = 3
    for i in range(0, len(display_products), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (_, product) in enumerate(display_products.iloc[i:i+cols_per_row].iterrows()):
            with cols[j]:
                st.markdown(f"""
                <div style="
                    border:1px solid #ccc;
                    border-radius:10px;
                    padding:10px;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    background-color:#f9f9f9;
                    ">
                <h4>{product['name']}</h4>
                <img src="{product['image_url']}" width="150">
                <p>{product['description']}</p>
                <p><b>Price: ₹{product['price']}</b></p>
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
                        st.success(f"Review submitted! Predicted Sentiment: {sentiment}")

    # ----------------------------
    # Sentiment Dashboard
    # ----------------------------
    if not df_reviews.empty:
        st.header("📈 Sentiment Analysis Dashboard")

        # Overall sentiment pie chart
        fig = px.pie(df_reviews, names='sentiment', title="Overall Sentiment Distribution",
                     color='sentiment', color_discrete_map={'Positive':'green','Neutral':'gray','Negative':'red'})
        st.plotly_chart(fig)

        # Reviews by Product
        st.subheader("Reviews by Product")
        for _, product in display_products.iterrows():
            product_reviews = df_reviews[df_reviews['product_id'] == product['id']]
            if not product_reviews.empty:
                st.markdown(f"**{product['name']}** Reviews:")
                st.dataframe(product_reviews[['review', 'sentiment']])
