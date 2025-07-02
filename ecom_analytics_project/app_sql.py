import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sqlite3
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import openai
import os
from dotenv import load_dotenv

# Load OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect to SQLite database (or create one)
conn = sqlite3.connect("ecom_data.db")
cursor = conn.cursor()

# Create tables and load CSVs if they don't exist
def load_data():
    if not os.path.exists("data/mall_customers.csv") or not os.path.exists("data/ecommerce_sales.csv") or not os.path.exists("data/churn_data.csv"):
        st.error("Missing dataset files in data/ folder.")
        return
    cust_df = pd.read_csv("data/mall_customers.csv")
    sales_df = pd.read_csv("data/ecommerce_sales.csv")
    churn_df = pd.read_csv("data/churn_data.csv")
    cust_df.to_sql("customers", conn, if_exists="replace", index=False)
    sales_df.to_sql("sales", conn, if_exists="replace", index=False)
    churn_df.to_sql("churn", conn, if_exists="replace", index=False)

load_data()

# Streamlit UI
st.set_page_config(page_title="🧠 E-Commerce AI SQL Dashboard", layout="wide")
st.title("📊 AI-Powered E-Commerce Dashboard (with SQL)")

# Sidebar
page = st.sidebar.radio("Choose Analysis", ["Customer Segmentation", "Product Analysis", "Churn Prediction", "Ask the AI Bot", "SQL Console"])

# Load DataFrames
customers = pd.read_sql_query("SELECT * FROM customers", conn)
sales = pd.read_sql_query("SELECT * FROM sales", conn)
churn = pd.read_sql_query("SELECT * FROM churn", conn)

# CUSTOMER SEGMENTATION
if page == "Customer Segmentation":
    st.subheader("🧍‍♂️ Customer Segmentation")
    kmeans = KMeans(n_clusters=5, random_state=42)
    customers['Cluster'] = kmeans.fit_predict(customers[['Annual Income (k$)', 'Spending Score (1-100)']])

    fig, ax = plt.subplots()
    sns.scatterplot(data=customers, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', ax=ax)
    st.pyplot(fig)

# PRODUCT ANALYSIS
elif page == "Product Analysis":
    st.subheader("🛍️ Product Analysis")
    sales['Total'] = sales['Price'] * sales['Quantity']

    top_qty = sales.groupby("Product")["Quantity"].sum().sort_values(ascending=False).head(5)
    top_rev = sales.groupby("Product")["Total"].sum().sort_values(ascending=False).head(5)

    st.write("Top 5 Products by Quantity Sold")
    st.dataframe(top_qty)
    st.write("Top 5 Products by Revenue")
    st.dataframe(top_rev)

    kmeans = KMeans(n_clusters=5, random_state=42)
    customers['Cluster'] = kmeans.fit_predict(customers[['Annual Income (k$)', 'Spending Score (1-100)']])
    merged = pd.merge(sales, customers[['CustomerID', 'Cluster']], on='CustomerID')
    cluster_product = merged.groupby(['Cluster', 'Product'])['Quantity'].sum().reset_index()

    fig2, ax2 = plt.subplots()
    sns.barplot(data=cluster_product, x='Cluster', y='Quantity', hue='Product', ax=ax2)
    ax2.set_title("Product Preference by Segment")
    st.pyplot(fig2)

# CHURN PREDICTION
elif page == "Churn Prediction":
    st.subheader("⚠️ Churn Prediction")

    try:
        model = joblib.load('models/churn_model.pkl')
    except:
        X = churn.drop(['CustomerID', 'Churn'], axis=1)
        y = churn['Churn']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/churn_model.pkl')

    age = st.slider("Age", 18, 70, 30)
    income = st.slider("Income", 20000, 100000, 50000)
    purchases = st.slider("Purchases", 1, 50, 10)
    calls = st.slider("Support Calls", 0, 10, 1)

    input_data = pd.DataFrame([[age, income, purchases, calls]], columns=['Age', 'Income', 'Purchases', 'SupportCalls'])
    prediction = model.predict(input_data)[0]
    st.success("✅ Likely to Stay" if prediction == 0 else "❌ Likely to Churn")

# CHATBOT SECTION
elif page == "Ask the AI Bot":
    st.subheader("🤖 Ask the AI Bot")
    user_input = st.text_input("Ask me about your sales, customers, or churn...")

    prompt = (
        "You are a helpful business analyst assistant. You have access to customer segmentation, product sales, and churn analysis. "
        "Provide business insights clearly and suggest data-backed strategies."
    )

    if user_input:
        with st.spinner("Analyzing..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            st.write("AI Bot:")
            st.success(response['choices'][0]['message']['content'])

# SQL QUERY TOOL
elif page == "SQL Console":
    st.subheader("🧾 SQL Query Console")
    query = st.text_area("Write your SQL query below", "SELECT * FROM customers LIMIT 5")

    if st.button("Run Query"):
        try:
            result = pd.read_sql_query(query, conn)
            st.dataframe(result)
        except Exception as e:
            st.error(f"Error: {e}")
