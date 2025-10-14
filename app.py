
import streamlit as st
import pandas as pd
from textblob import TextBlob
import pickle
import plotly.express as px

st.set_page_config(page_title="E-Commerce Sentiment", layout="wide")
st.title("🛒 Simple E-Commerce Platform")

# Upload or load product data
uploaded_file = st.file_uploader("Upload Product CSV", type=['csv'])
if uploaded_file:
    df_products = pd.read_csv(uploaded_file)
else:
    df_products = pd.read_csv("data/products.csv")

region = st.selectbox("Select Region", ["All"] + sorted(df_products['region'].unique()))
if region != "All":
    df_products = df_products[df_products['region'] == region]

st.subheader("Products")
for _, row in df_products.iterrows():
    st.image(row['image_url'], width=150)
    st.write(f"**{row['name']}** - ₹{row['price']}")
    st.write(row['description'])

# Sentiment Analysis
st.header("📈 Sentiment Analysis Dashboard")

# Load or create sentiment data
try:
    df_reviews = pickle.load(open("model/sentiment_data.pkl", "rb"))
except:
    df_reviews = pd.read_csv("data/reviews.csv")
    df_reviews['sentiment'] = df_reviews['review'].apply(lambda x: 'Positive' if TextBlob(x).sentiment.polarity>0 else ('Negative' if TextBlob(x).sentiment.polarity<0 else 'Neutral'))
    pickle.dump(df_reviews, open("model/sentiment_data.pkl", "wb"))

fig = px.histogram(df_reviews, x='sentiment', color='sentiment', title="Sentiment Distribution")
st.plotly_chart(fig)
