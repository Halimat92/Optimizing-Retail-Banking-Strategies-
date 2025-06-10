🏦 RFM-Based Customer Segmentation for BankTrust
This project focuses on optimizing retail banking strategies through RFM (Recency, Frequency, Monetary) analysis. Using customer transaction data, we segment users based on behavioral patterns to help BankTrust reduce churn, improve personalization, and enhance marketing efficiency.
________________________________________
📌 Project Objective
To identify key customer segments using transaction history and clustering techniques. This segmentation enables BankTrust to develop data-driven strategies to retain valuable customers, re-engage inactive ones, and tailor marketing campaigns effectively.
________________________________________
🧰 Key Features
•	Data Cleaning & Preparation:
o	Processed banking transaction records
o	Handled missing values, fixed data types, removed duplicates
o	Engineered RFM metrics:
	Recency – Days since last transaction
	Frequency – Total number of transactions
	Monetary – Total value of transactions
•	RFM Scoring & Segmentation:
o	Assigned quantile-based scores (1–4) to R, F, and M
o	Combined scores into labeled segments (e.g., Best Customers, At Risk, Need Attention)
•	Unsupervised Learning with KMeans:
o	Applied KMeans clustering to discover customer groups
o	Used the Elbow Method and Silhouette Score to find the optimal number of clusters
o	Analyzed cluster profiles to understand customer behavior patterns
•	Business Insights & Strategy:
o	Interpreted each segment's RFM profile
o	Suggested actions like retention campaigns, upselling to loyal customers, or re-engagement strategies for at-risk groups
•	Streamlit Dashboard (Bonus):
o	Built an interactive dashboard to:
	Visualize RFM segments
	Filter customers by cluster, gender, age group, or location
	Simulate “what-if” scenarios (e.g., increasing frequency among dormant users)
o	Deployed the app via Streamlit Cloud
________________________________________
🛠️ Tech Stack
•	Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
•	Streamlit for interactive dashboard and deployment
________________________________________
📊 Outcome
The project successfully created meaningful customer clusters based on transaction behavior, offering actionable insights for personalized marketing and customer retention strategies in the banking sector.

