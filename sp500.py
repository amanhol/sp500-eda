# app.py

import streamlit as st
import yfinance as yf

import time
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import statsmodels.api as sm

#importing the data through a scriot
import prep
import map_data as md
import machine_learning_data as ml

# Load S&P 500 data from Wikipedia
@st.cache_data
def load_data():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_df = pd.read_html(sp500_url)[0]
    sp500_df['Date added'] = pd.to_datetime(sp500_df['Date first added'], errors='coerce')
    sp500_df['Founded'] = pd.to_numeric(sp500_df['Founded'], errors='coerce')
    return sp500_df

# App Title
st.title("S&P 500 Stock Analysis")

# Load Data
sp500_df = load_data()

# Sidebar Filters
st.sidebar.header("Filters")

# Date Slider
date_range = st.sidebar.slider(
    "Date Added",
    min_value=sp500_df['Date added'].min(),
    max_value=sp500_df['Date added'].max(),
    value=(sp500_df['Date added'].min(), sp500_df['Date added'].max())
)
streamlit run app.py
# Founded Year Slider
founded_range = st.sidebar.slider(
    "Founded Year",
    min_value=int(sp500_df['Founded'].min()),
    max_value=int(sp500_df['Founded'].max()),
    value=(int(sp500_df['Founded'].min()), int(sp500_df['Founded'].max()))
)

# GICS Sector Filter
selected_sector = st.sidebar.multiselect(
    "Select GICS Sector",
    options=sp500_df['GICS Sector'].unique(),
    default=sp500_df['GICS Sector'].unique()
)

# Security Filter
selected_security = st.sidebar.multiselect(
    "Select Security",
    options=sp500_df['Security'].unique(),
    default=sp500_df['Security'].unique()
)

# Filter Data
filtered_df = sp500_df[
    (sp500_df['Date added'].between(*date_range)) &
    (sp500_df['Founded'].between(*founded_range)) &
    (sp500_df['GICS Sector'].isin(selected_sector)) &
    (sp500_df['Security'].isin(selected_security))
]

# Display Data Table
st.dataframe(filtered_df)

# Plot 1: Average Market Cap by Sector
fig1 = px.bar(
    filtered_df.groupby('GICS Sector').size().reset_index(name='Count'),
    x='GICS Sector',
    y='Count',
    color='GICS Sector',
    title='Number of Companies per GICS Sector'
)
st.plotly_chart(fig1, use_container_width=True)

# Plot 2: Stock Prices Line Chart (Dummy Data)
st.subheader("Stock Price Analysis (Sample)")
selected_tickers = filtered_df['Symbol'].unique()

# Download Sample Stock Prices
if len(selected_tickers) > 0:
    data = yf.download(selected_tickers.tolist(), start='2023-01-01', end='2023-12-31', group_by="ticker")

    stock_data = pd.concat(
        [data[ticker]['Close'].reset_index().assign(Ticker=ticker) for ticker in selected_tickers],
        ignore_index=True
    )

    fig2 = px.line(
        stock_data, 
        x='Date', 
        y='Close', 
        color='Ticker', 
        title='Stock Prices Over Time'
    )
    st.plotly_chart(fig2, use_container_width=True)

# Plot 3: Headquarters Locations Map
fig3 = px.scatter_mapbox(
    filtered_df, 
    lat=[37.7749]*len(filtered_df),  # Dummy Latitudes
    lon=[-122.4194]*len(filtered_df),  # Dummy Longitudes
    hover_name='Security',
    hover_data=['Headquarters Location', 'Founded'],
    color='GICS Sector',
    title='Headquarters Locations',
    mapbox_style="carto-positron",
    zoom=3
)
st.plotly_chart(fig3, use_container_width=True)