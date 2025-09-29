import streamlit as st
import pandas as pd
import numpy as np

# Page settings
st.set_page_config(page_title="My First Streamlit App", layout="wide")

# Title
st.title("🚀 Hello, Streamlit!")

# Text input
name = st.text_input("What's your name?", "Guest")
st.write(f"Hello, {name}! 👋")

# Number input
num_points = st.slider("Select number of points", 10, 500, 50)

# Create random data
data = pd.DataFrame({
    "x": np.random.randn(num_points),
    "y": np.random.randn(num_points)
})

# Show table
st.subheader("Random Data")
st.dataframe(data)

# Show chart
st.subheader("Scatter Plot")
st.scatter_chart(data)
