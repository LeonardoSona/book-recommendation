import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample dataset of books
data = {
    "Book Title": [
        "The Lean Startup",
        "Measure What Matters",
        "Data Science for Business",
        "Storytelling with Data",
        "Thinking, Fast and Slow",
        "Deep Learning",
        "The Art of Statistics",
    ],
    "Category": [
        "Product Management",
        "Product Management",
        "Data Science",
        "Data Science",
        "Product Management",
        "Data Science",
        "Data Science",
    ],
    "Description": [
        "How constant innovation creates successful businesses.",
        "Objectives and key results to drive growth.",
        "Fundamentals of data mining and predictive analytics.",
        "How to tell compelling stories with data.",
        "The science of decision-making and behavior.",
        "Comprehensive introduction to neural networks.",
        "Statistics explained in a clear and practical way.",
    ],
}

# Create DataFrame
df = pd.DataFrame(data)

# Sidebar user selection
st.sidebar.title("Book Categories")
selected_categories = st.sidebar.multiselect(
    "Select categories:", options=df["Category"].unique(), default=df["Category"].unique()
)

# Filter dataset based on selected categories
filtered_df = df[df["Category"].isin(selected_categories)]

# Display filtered books
st.title("Recommended Books")
if filtered_df.empty:
    st.write("No books match the selected categories.")
else:
    st.write(filtered_df[["Book Title", "Category"]])

# Text preprocessing and vectorization
if not filtered_df.empty:
    vectorizer = TfidfVectorizer()
    book_embeddings = vectorizer.fit_transform(filtered_df["Description"])

# Recommendation function
def recommend_books(book_title, top_n=3):
    try:
        book_idx = filtered_df[filtered_df["Book Title"] == book_title].index[0]
        similarities = cosine_similarity(book_embeddings[book_idx], book_embeddings).flatten()
        similar_indices = similarities.argsort()[-top_n - 1 : -1][::-1]
        recommendations = filtered_df.iloc[similar_indices]
        return recommendations
    except IndexError:
        return pd.DataFrame([])

# User selects a book
st.subheader("Select a book you like:")
selected_book = st.selectbox("Book Title", options=filtered_df["Book Title"] if not filtered_df.empty else [])

# Show recommendations
if st.button("Recommend Similar Books") and not filtered_df.empty:
    recommendations = recommend_books(selected_book)
    if not recommendations.empty:
        st.write("We recommend:")
        st.write(recommendations[["Book Title", "Category"]])
    else:
        st.write("No similar books found.")

# Evaluation Metrics Placeholder
st.subheader("Evaluation Metrics")
st.write("Metrics like Precision, Recall, and NDCG can be calculated with actual user feedback data.")
