import pandas as pd
import streamlit as st
import streamlit as st
# from data_provider1 import get_cleaned_bank_data
# Data manipulation
import numpy as np
import pandas as pd

# Visualization and plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FuncFormatter

import datetime as dt
from datetime import datetime

import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yellowbrick
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA


df = pd.read_csv("bank_data_C.csv")

    # Convert date columns
df["CustomerDOB"] = pd.to_datetime(df["CustomerDOB"], dayfirst=True, errors="coerce")
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], dayfirst=True, errors="coerce")

    # Drop rows with zero transaction amount
df = df[df["TransactionAmount (INR)"] != 0]

    # Adjust DOBs with future years (e.g., 2099)
def adjust_year(date):
        if pd.notnull(date) and date.year > 2016:
            return date.replace(year=date.year - 100)
        return date

df["CustomerDOB"] = df["CustomerDOB"].apply(adjust_year)

    # Replace DOBs before 1900 with median DOB
def replace_age_outlier(df):
        threshold = 1900
        outliers = df[df["CustomerDOB"].dt.year < threshold].index
        median_dob = df.loc[~df.index.isin(outliers), "CustomerDOB"].median()
        df.loc[outliers, "CustomerDOB"] = median_dob
        return df

df = replace_age_outlier(df)

    # Recalculate Age
df["Age"] = df["TransactionDate"].dt.year - df["CustomerDOB"].dt.year
day = df["TransactionDate"].max() #obtain maximum date
day = pd.to_datetime(day)           #convert to date-time

df["CustGender"] = df["CustGender"].replace('T','M')
# new_df = df.drop(df[df["TransactionAmount (INR)"] == 0].index, axis=0, inplace=True)
# Remove rows where TransactionAmount is 0 
df[df["TransactionAmount (INR)"] != 0].copy()


 
recency = df.groupby("CustomerID").agg({"TransactionDate": lambda x: (day - x.max()).days + 1})

frequency = df.groupby("CustomerID")["TransactionID"].count()
monetary = df.groupby("CustomerID")["TransactionAmount (INR)"].sum()

RFM_table = pd.concat([recency, frequency, monetary], axis=1)

RFM_table = RFM_table.rename(columns={
    "TransactionDate": "Recency", 
    "TransactionID": "Frequency", 
    "TransactionAmount (INR)": "Monetary"
})

quartiles = RFM_table[['Recency', 'Frequency', 'Monetary']].quantile(q=[0.25, 0.5, 0.75]).to_dict()

def assign_R_score(x, feature):
    """this function is used to assign recency score
    the more recent a customer's latest transaction is, the higher the recency score"""

    if x <= quartiles[feature][0.25]:
        return 4
    elif x <= quartiles[feature][0.5]:
        return 3
    elif x <= quartiles[feature][0.75]:
        return 2
    else:
        return 1

def assign_M_score(x, feature):
    """This function is used for assigning monetary score.
    the higher the monetary value, the higher the monetary score"""

    if x <= quartiles[feature][0.25]:
        return 1
    elif x <= quartiles[feature][0.5]:
        return 2
    elif x <= quartiles[feature][0.75]:
        return 3
    else:
        return 4
    
def custom_frequency_score(x):
    """This function is used for assigning frequency score.
    frequency of 1,2 and 3 are assgned scores of 1,2 and 3 respectively
    and frequency of 4,5,and 6 are assigned scores of 4"""

    if x <= 3:
        return x
    else:
        return 4
    
# Assign quartile scores for recency
RFM_table['R_score'] = RFM_table['Recency'].apply(lambda x: assign_R_score(x, 'Recency'))

# Assign custom Frequency scores
RFM_table['F_score'] = RFM_table['Frequency'].apply(custom_frequency_score)

# Assign quartile scores for monetary component
RFM_table['M_score'] = RFM_table['Monetary'].apply(lambda x: assign_M_score(x, 'Monetary'))

RFM_table['RFM_group'] = RFM_table['R_score'].astype(str) + RFM_table['F_score'].astype(str) + RFM_table['M_score'].astype(str)
RFM_table["RFM_score"] = RFM_table[['R_score', 'M_score', 'F_score']].sum(axis = 1)


    
def assign_segments(x):
    if x <= 3:
        return "Inactive" #churned
    elif x  <= 5:
        return "At Risk " #Low
    elif x <= 8:
        return "Stable"
    else:
        return "Loyal" #high
    
# st.dataframe(RFM_table.head())
RFM_table["segments"] = RFM_table["RFM_score"].apply(lambda x: assign_segments(x))
st.dataframe(RFM_table.head())


RFM_table['weighted_score'] = (RFM_table['R_score'] * 2) + (RFM_table["F_score"] * 1) + (RFM_table['M_score'] * 1)


RFM_table["weighted_segments"] = RFM_table["weighted_score"].apply(lambda x: assign_segments(x))
st.dataframe(RFM_table.head())
RFM_table.to_csv("RFM_table.csv", index=True)
RFM_table.drop(["RFM_group", "weighted_score", "weighted_segments"], axis = 1)
RFM_data = RFM_table.drop(["RFM_group", "weighted_score", "weighted_segments"], axis = 1)
RFM_data.head(2)

# RFM_data = pd.concat([df["CustomerID"], RFM_table[["Recency", "Frequency", "Monetary", "R_score", "F_score", "M_score", "RFM_score"]]], axis=1)

# RFM_data.to_csv("RFM_data_C_2.csv", index=False)


features_for_clustering = RFM_table[["Recency", "Frequency", "Monetary","R_score","F_score","M_score","RFM_score"]].copy()
# features_for_clustering = pd.concat([df["CustomerID"], RFM_table[["Recency", "Frequency", "Monetary", "R_score", "F_score", "M_score", "RFM_score"]]], axis=1)


# # Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features_for_clustering)

# Fit KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(features_for_clustering)

# Assign clusters back to original RFM_table (DO NOT drop text columns!)
features_for_clustering["Cluster"] = kmeans.labels_

# Now weighted_segments and other metadata are intact for filtering



scaler = StandardScaler()
scaled_data = scaler.fit_transform(features_for_clustering)


model = KMeans(random_state = 1)

final_model = KMeans(random_state = 1, n_clusters = 3)
final_model.fit(scaled_data)
cluster_assignment = final_model.labels_
cluster_assignment

features_for_clustering["Cluster"] = cluster_assignment
# RFM_data.head()

# plt.figure(figsize=(10, 10))
# sns.scatterplot(data=RFM_data, x="Recency", y="Monetary", hue="Cluster", palette='viridis')
# plt.xlabel("Recency", fontweight='bold')
# plt.ylabel("Monetary", fontweight='bold')

# plt.title("Recency vs Monetary by Cluster", fontweight='bold')  
# plt.legend(title="Cluster")
# plt.tight_layout()
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# plt.savefig("Recency vs Monetary by Cluster.png", format='png', dpi=300)
# st.pyplot(plt)
# new_data = pd.concat([
#     new_df[["CustomerID", "CustGender", "CustLocation", "Age"]],
#     features_for_clustering
# ], axis=1)

features_for_clustering.to_csv("features_for_clustering.csv", index=True)
print("✅ Cleaned data saved as RFM_data_C.csv")
st.write("DONE")
