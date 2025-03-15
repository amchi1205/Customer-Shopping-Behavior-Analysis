# Customer-Shopping-Behavior-Analysis

[Decoding Customer Trends.pdf](https://github.com/user-attachments/files/19259218/Decoding.Customer.Trends.pdf)

## Overview
This project analyzes customer shopping behavior using a publicly available dataset from Kaggle. The goal is to segment customers based on their shopping patterns and predict subscription behavior using machine learning techniques.

## Key Features
- **Data Cleaning**: Handled missing values, removed outliers, and standardized data.
- **Exploratory Data Analysis (EDA)**: Visualized trends and relationships in the data.
- **Unsupervised Learning**: Applied KMeans clustering to segment customers.
- **Supervised Learning**: Built a Decision Tree model to predict subscription behavior.

## Dataset
The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/customer-shopping-latest-trends). It includes variables such as:
- `Customer_ID`
- `Age`
- `Gender`
- `Annual_Income`
- `Spending_Score`
- `Product_Category`
- `Purchase_Amount`

## Tools and Technologies
- Python
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Pytrends
- Jupyter Notebook

## Results:
- Customer Segmentation: Identified 4 distinct customer segments using KMeans clustering.
- Subscription Prediction: Achieved 70% accuracy in predicting subscription behavior using a Decision Tree model.

## Future Work:

- Deploy the model as a web app using Flask or FastAPI.
- Explore additional machine learning models (e.g., Random Forest, XGBoost).

## Author:
- Andrew Chi
- Email: chia@oregonstate.edu
- GitHub: amchi1205
