import streamlit as st
# Data manipulation
import numpy as np
import pandas as pd
# Visualization and plotting

from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load dataset only once
# @st.cache_data

df = pd.read_csv("Bank/bank_data.csv")
  


# Tab selector (use radio instead of st.tabs)
selected_tab = st.radio("🧭 Navigator", ["🏠 Home", "📊 Demographics", "👥 Customer Segmentation"], horizontal=True)

if selected_tab == "🏠 Home":
    st.markdown("<h3 style='color: black;'>🏦 BankTrust Customer Segmentation Interface</h3>", unsafe_allow_html=True)
    st.image("Bank/bank_seg.webp", width=500)  

    st.markdown("""
    ### 📌 Project Objective

    The goal of this project is to segment customers based on their **behavioral data** and develop data-driven strategies to:

    - 📉 **Reduce customer churn** by identifying at-risk customers.
    - 📨 **Improve personalization** by understanding spending habits.
    - 📈 **Optimize marketing efficiency** by targeting high-value customers.

    This dashboard will allow you to explore demographic trends, visualize transactional behavior, and apply 
                segmentation techniques such as RFM(Recency, Frequency, Monetary) and clustering.""")

elif selected_tab == "📊 Demographics":
    # st.header("📊 Customer Demographics")
   
    # Sidebar filters (only show here!)
    with st.sidebar:
        st.header("Filter Options")
        # Age filter
    min_age = int(df['Age'].min())
    max_age = int(df['Age'].max())
    age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

    # Gender filter
    gender_options = st.sidebar.multiselect(
        "Select Gender",
        options=df["CustGender"].dropna().unique(),
        default=df["CustGender"].dropna().unique()
    )

    # Monetary filter
    min_amt = int(df["Monetary"].min())
    max_amt = int(df["Monetary"].max())
    amt_range = st.sidebar.slider("Monetary", min_amt, max_amt, (min_amt, max_amt))

    # Apply filters
    filtered_df = df[
        (df["Age"] >= age_range[0]) &
        (df["Age"] <= age_range[1]) &
        (df["CustGender"].isin(gender_options)) &
        (df["Monetary"] >= amt_range[0]) &
        (df["Monetary"] <= amt_range[1])] 
    
     #  Show metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{filtered_df.shape[0]:,}")
    col2.metric("Avg Age", f"{filtered_df['Age'].mean():.0f}")
    col3.metric("Avg Monetary", f"₹{filtered_df['Monetary'].mean():,.2f}")

    # Custom-styled label
    st.markdown("<h4 style='font-weight: bold; color: black;'>🔍 Enter Customer ID to filter by each customer!</h4>", unsafe_allow_html=True)
    input_id = st.text_input("")
    if input_id:
        filtered_df = filtered_df[filtered_df["CustomerID"] == input_id]


    unique_customers = filtered_df["CustomerID"].unique()

    # # Customer ID filter (optional)
    # st.subheader(f"📊 Transaction Trend for Customer ID: {input_id}")

    if len(unique_customers) == 1:
        selected_customer = unique_customers[0]
        st.subheader(f"📊 Transaction Trend for Customer ID: {selected_customer}")

    
    st.subheader("📊 Daily Transaction Trend")

 # Optional: Aggregate if there are many entries per day
    daily_trend = (
    filtered_df.groupby("TransactionDate")["Monetary"]
    .sum()
    .reset_index()
    )

 # Create the interactive Plotly line chart
    fig = px.line(
    daily_trend,
    x="TransactionDate",
    y="Monetary",
    
    markers=False,
    )

    fig.update_layout( 
    xaxis_title=dict(
        text=" Date",
        font=dict(size=20, color="black", family="Arial")
    ),
    yaxis_title=dict(
        text=" Transaction (₹)",
        font=dict(size=20, color="black", family="Arial")
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(tickformat=",")
    )

    st.plotly_chart(fig, use_container_width=True)



    st.subheader("🏆 Top 5 Customers")

    # Group and get top 5 customers by total transaction amount
    top_customers = filtered_df.groupby('CustomerID')['Monetary'].sum().nlargest(5).reset_index()

    # Create interactive bar chart
    fig = px.bar(
        top_customers,
        x='CustomerID',y='Monetary',
        # title="Top 5 Customers by Transaction Amount",
    # color='Monetary',
        color='CustomerID',
        text='Monetary')
    # Customize layout
    fig.update_layout(
    xaxis_title=dict(
        text="Customer ID",
        font=dict(size=20, color="black", family="Arial")),
    yaxis_title=dict(
        text="Total Transaction (₹)",
        font=dict(size=20, color="black", family="Arial")),
    showlegend=False,
    template="plotly_white")

    fig.update_traces(
    texttemplate='₹%{text:,.0f}',
    textposition='outside'
        )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)



    st.subheader("🎂 Age Distribution")

    # Create interactive histogram with Plotly
    fig2 = px.histogram(
    filtered_df,
    x='Age',
    nbins=10,  # adjust bin count as needed
    color='CustGender',  
    marginal='box',      
    # title='Interactive Age Distribution',
    color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig2.update_layout(
    xaxis_title=dict(
        text='Age',
        font=dict(size=20, color="black", family="Arial")),

    yaxis_title=dict(
        text='Frequency',
        font=dict(size=20, color="black", family="Arial")),
    bargap=0.1,
    plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig2, use_container_width=True)


    st.subheader("🌍 Top 10 Locations by Customer Count")

    top_locations = filtered_df["CustLocation"].value_counts().nlargest(10).reset_index()
    top_locations.columns = ["CustLocation", "Count"]

    fig = px.bar(
    top_locations,
    x="CustLocation",
    y="Count",
    # color="Count",
   color='CustLocation',
    text="Count")

    fig.update_layout(
    xaxis_title=dict(
        text="Customer Location",
        font=dict(size=20, color="black", family="Arial")),
    yaxis_title=dict(
        text="Count",
        font=dict(size=20, color="black", family="Arial")),
    xaxis_tickangle=-45,
    showlegend=False,
    template="plotly_white")

    fig.update_traces(texttemplate='%{text:,}', textposition='outside')

    # Display the plot 
    st.plotly_chart(fig, use_container_width=True)



    st.subheader("🧑‍🤝‍🧑 Gender Distribution")
    gender_count = filtered_df['CustGender'].value_counts()
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', startangle=180)
    ax3.set_title("Gender Split", fontweight='bold')
    st.pyplot(fig3)

    # st.write(filtered_df)

elif selected_tab == "👥 Customer Segmentation":
   
    st.markdown("#### Customer Segmentation Metrics")
    # Sidebar filters with optional selection


    # Add a 'None' option for initial state
    cluster_options = ["All"] + sorted(df["Cluster"].unique().tolist())
    segment_options = ["All", "Loyal", "Stable", "At Risk ","Inactive"]

    selected_cluster = st.sidebar.selectbox("Select Cluster", options=cluster_options, key="cluster_selector")
    selected_segment = st.sidebar.selectbox("Select Customer Segment", options=segment_options, key="segment_selector")

        # Apply filters only if specific values are selected
    filtered_data = df.copy()

    if selected_cluster != "All":
        filtered_data = filtered_data[filtered_data["Cluster"] == selected_cluster]

    if selected_segment != "All":
        filtered_data = filtered_data[filtered_data["segments"] == selected_segment]

    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{filtered_data.shape[0]:,}")
    col2.metric("Avg Recency (days)", f"{filtered_data['Recency'].mean():.0f}")
    col3.metric("Avg Frequency", f"{filtered_data['Frequency'].mean():.2f}")
    col4.metric("Avg Monetary", f"₹{filtered_data['Monetary'].mean():,.2f}")

    # Plot interactive histogram
    fig = px.histogram(
    filtered_data,
    x="RFM_score",
    color="Cluster",
    title="RFM Score Distribution", 
    color_discrete_sequence=px.colors.qualitative.Bold,
    barmode='group'
    )

    fig.update_layout(
    yaxis=dict(
        range=[10000, None],  # Start at 10K
        title=dict(
            text="Customer Count by Cluster",
            font=dict(size=16, color="black", family="Arial")
        )
    ),
    xaxis=dict(
        title=dict(
            text="RFM Score",
            font=dict(size=16, color="black", family="Arial")
        )
    ),
    plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)





# Prepare the funnel data
    funnel_data = filtered_data["segments"].value_counts().reset_index()
    funnel_data.columns = ["segments", "CustomerID"]
    funnel_data = funnel_data.sort_values(by="CustomerID", ascending=False)

# Create funnel chart with single color
    fig = px.funnel(
    funnel_data,
    y="segments",
    x="CustomerID",
    color_discrete_sequence=["#1f77b4"],  # Set to a single color (blue, adjust if needed)
    title="🧩 Customer Distribution by Segment"
    )

# Style layout for a clean "3D-like" look
    fig.update_layout(
    xaxis_title=dict(
        text="Number of Customers",
        font=dict(size=16, color="black", family="Arial")
    ),
    yaxis_title=dict(
        text="Cluster",
        font=dict(size=16, color="black", family="Arial")
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="white",
    showlegend=False
    )

    # Add labels to each bar
    fig.update_traces(texttemplate="%{x:,}", textposition="outside", marker_line_width=1.5)

    # Show chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # Prepare the data
    funnel_data = filtered_data["Cluster"].value_counts().reset_index()
    funnel_data.columns = ["Cluster", "CustomerID"]

    # Convert Cluster to string to match color keys
    funnel_data["Cluster"] = funnel_data["Cluster"].astype(str)

    # Create pie chart
    fig = px.pie(
    funnel_data,
    names="Cluster",
    values="CustomerID",
    color="Cluster",
    color_discrete_map={
        "0": "#B0E0E6",  # Powder Blue
        "1": "#FF0000",  # Red
        "2": "#4169E1"   # Royal Blue
    },
    title="🧩 Customer Distribution by Cluster"
    )

    # Customize appearance
    fig.update_traces(
    textinfo="percent",
    textfont_size=14
    )
    fig.update_layout(
    legend_title_text="Cluster",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=True
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)




    st.subheader("🔍 Explore Top Customers")

    with st.expander("📋 Click here to view the list of top 20 customers per cluster & Segment."):
        top_customers_per_cluster = (
        filtered_data.groupby("Cluster")
        .apply(lambda df: df.nlargest(20, "Monetary"))
        .reset_index(drop=True)[["CustomerID", "Cluster","segments" , "Recency", "Frequency", "Monetary", "CustGender", "CustLocation", "Age"]]
        )

        st.dataframe(top_customers_per_cluster.style.format({"Monetary": "₹{:.0f}"}))





#     top_locations = filtered_data["CustLocation"].value_counts().nlargest(10).reset_index()
#     top_locations.columns = ["CustLocation", "Count"]

#     st.subheader("🌍 Top 10 Locations by Customer Count")

#     fig = px.bar(
#     top_locations,
#     x="CustLocation",
#     y="Count",
#     # color="Count",
#    color='CustLocation',
#     text="Count")

#     fig.update_layout(
#     xaxis_title="Customer Location",
#     yaxis_title="Count",
#     xaxis_tickangle=-45,
#     showlegend=False,
#     template="plotly_white")

#     fig.update_traces(texttemplate='%{text:,}', textposition='outside')

# # Display the plot in Streamlit
#     st.plotly_chart(fig, use_container_width=True)



    


