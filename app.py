import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
from datetime import datetime

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="E-Commerce Sentiment App", layout="wide")
st.title("🛒 Interactive E-Commerce Platform with Sentiment Analytics")

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
    for col in ['product_id', 'review', 'sentiment', 'timestamp']:
        if col not in df_reviews.columns:
            df_reviews[col] = ''
else:
    df_reviews = pd.DataFrame(columns=['product_id', 'review', 'sentiment', 'timestamp'])
    df_reviews.to_csv(reviews_file, index=False)

# ----------------------------
# Login System
# ----------------------------
st.subheader("🔐 Login")
username = st.text_input("Username")
password = st.text_input("Password", type="password")
login_button = st.button("Login")

# Demo users
users = {
    "admin": {"password": "admin123", "role": "Admin"},
    "user": {"password": "user123", "role": "User"}
}

if login_button:
    if username in users and users[username]["password"] == password:
        role = users[username]["role"]
        st.success(f"Logged in as {role}: {username}")
    else:
        st.error("Invalid username or password")
        st.stop()
else:
    st.stop()

# ----------------------------
# Admin Section
# ----------------------------
if role == "Admin":
    st.header("➕ Product Management")

    # Add single product
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

    # Bulk upload products
    with st.expander("Bulk Upload Products (CSV/JSON)"):
        uploaded_file = st.file_uploader("Upload CSV or JSON file", type=["csv", "json"])
        if uploaded_file:
            try:
                if uploaded_file.type == "application/json" or uploaded_file.name.endswith(".json"):
                    bulk_df = pd.read_json(uploaded_file)
                else:
                    bulk_df = pd.read_csv(uploaded_file)

                required_cols = ['name', 'price', 'region', 'image_url', 'description']
                if all(col in bulk_df.columns for col in required_cols):
                    start_id = df_products['id'].max() + 1 if not df_products.empty else 1
                    bulk_df['id'] = range(start_id, start_id + len(bulk_df))
                    bulk_df = bulk_df[['id'] + required_cols]
                    df_products = pd.concat([df_products, bulk_df], ignore_index=True)
                    df_products.to_csv(products_file, index=False)
                    st.success(f"{len(bulk_df)} products added successfully!")
                else:
                    st.error(f"Uploaded file must have columns: {required_cols}")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    st.subheader("📦 Current Products")
    st.dataframe(df_products)

    st.subheader("📝 All Reviews")
    st.dataframe(df_reviews)

# ----------------------------
# User Section
# ----------------------------
else:
    st.header("🛍 Products")
    search_term = st.text_input("🔎 Search Products")
    region_filter = st.selectbox("Filter by Region", ["All"] + sorted(df_products['region'].unique()))

    display_products = df_products.copy()
    if region_filter != "All":
        display_products = display_products[display_products['region'] == region_filter]
    if search_term.strip() != "":
        display_products = display_products[display_products['name'].str.contains(search_term, case=False)]

    # Function to choose color based on sentiment
    def card_color(product_id):
        revs = df_reviews[df_reviews['product_id']==product_id]
        if revs.empty:
            return "#fefefe"
        pos = len(revs[revs['sentiment']=="Positive"])
        neg = len(revs[revs['sentiment']=="Negative"])
        total = len(revs)
        score = (pos - neg)/total
        if score > 0.3:
            return "#d4edda"  # green
        elif score < -0.3:
            return "#f8d7da"  # red
        else:
            return "#fff3cd"  # yellow

    # Display products in columns
    cols_per_row = 3
    for i in range(0, len(display_products), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (_, product) in enumerate(display_products.iloc[i:i+cols_per_row].iterrows()):
            with cols[j]:
                color = card_color(product['id'])
                product_reviews = df_reviews[df_reviews['product_id'] == product['id']]
                avg_sent = "No Reviews Yet" if product_reviews.empty else \
                    f"{(len(product_reviews[product_reviews['sentiment']=='Positive'])/len(product_reviews)*100):.0f}% Positive"
                st.markdown(f"""
                <div style="
                    border:1px solid #ccc;
                    border-radius:10px;
                    padding:10px;
                    box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
                    background-color:{color};
                    text-align:center;
                    ">
                <h4>{product['name']}</h4>
                <img src="{product['image_url']}" width="150">
                <p>{product['description']}</p>
                <p><b>Price: ₹{product['price']}</b></p>
                <p><b>{avg_sent}</b></p>
                </div>
                """, unsafe_allow_html=True)

                # Review submission
                with st.expander("Write a Review"):
                    review_text = st.text_area("Your review here:", key=f"review_{product['id']}")
                    submit_review = st.button("Submit Review", key=f"submit_{product['id']}")
                    if submit_review and review_text.strip() != "":
                        sentiment = predict_sentiment(review_text)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_review = pd.DataFrame([[product['id'], review_text, sentiment, timestamp]],
                                                  columns=['product_id', 'review', 'sentiment', 'timestamp'])
                        df_reviews = pd.concat([df_reviews, new_review], ignore_index=True)
                        df_reviews.to_csv(reviews_file, index=False)
                        emoji = "😊" if sentiment=="Positive" else "😐" if sentiment=="Neutral" else "😞"
                        st.success(f"Review submitted! Sentiment: {sentiment} {emoji}")

    # ----------------------------
    # Dashboard Tabs
    # ----------------------------
    if not df_reviews.empty:
        st.header("📊 Sentiment Analytics Dashboard")
        tabs = st.tabs(["Overall Sentiment", "Per Product Sentiment", "Reviews Timeline", "Top Products"])

        # Overall sentiment
        with tabs[0]:
            fig = px.pie(df_reviews, names='sentiment', title="Overall Sentiment Distribution",
                         color='sentiment', color_discrete_map={'Positive':'green','Neutral':'gray','Negative':'red'})
            st.plotly_chart(fig)

        # Sentiment per product
        with tabs[1]:
            sentiment_summary = df_reviews.groupby(['product_id','sentiment']).size().unstack(fill_value=0)
            sentiment_summary = sentiment_summary.join(df_products.set_index('id')['name'])
            sentiment_summary = sentiment_summary.reset_index()
            if not sentiment_summary.empty:
                fig2 = px.bar(sentiment_summary, x='name', y=['Positive','Neutral','Negative'],
                              title="Sentiment per Product", color_discrete_map={'Positive':'green','Neutral':'gray','Negative':'red'})
                st.plotly_chart(fig2)

        # Reviews timeline
        with tabs[2]:
            df_reviews['timestamp'] = pd.to_datetime(df_reviews['timestamp'])
            timeline = df_reviews.groupby(df_reviews['timestamp'].dt.date).size().reset_index(name='count')
            fig3 = px.line(timeline, x='timestamp', y='count', title="Number of Reviews Over Time")
            st.plotly_chart(fig3)

        # Top products
        with tabs[3]:
            top = df_reviews[df_reviews['sentiment']=="Positive"].groupby('product_id').size().reset_index(name='Positive Count')
            top = top.join(df_products.set_index('id')['name'], on='product_id')
            top = top.sort_values(by='Positive Count', ascending=False)
            st.dataframe(top[['name','Positive Count']])
