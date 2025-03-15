#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 23:21:33 2025

@author: andrewchi
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE  # For handling class imbalance

# Load dataset
file_path = "/Users/andrewchi/Desktop/CS 399/shopping_trends_cleaned.csv"
df = pd.read_csv(file_path)

# Convert categorical target (Subscription Status) into numeric if not already
if df["Subscription Status"].dtype == "object":
    df["Subscription Status"] = df["Subscription Status"].map({"No": 0, "Yes": 1})

# Select features and target
features = ["Age", "Purchase Amount (USD)", "Frequency of Purchases"]
X = df[features]
y = df["Subscription Status"]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Visualization 1: Boxplot - Purchase Amount vs Subscription Status
plt.figure(figsize=(8, 6))
sns.boxplot(x=y, y=df["Purchase Amount (USD)"])
plt.title("Purchase Amount vs Subscription Status")
plt.xlabel("Subscription Status")
plt.ylabel("Purchase Amount (USD)")
plt.show()

# Visualization 2: Histogram - Age Distribution by Subscription Status
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x="Age", hue="Subscription Status", kde=True, bins=20)
plt.title("Age Distribution by Subscription Status")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Visualization 3: Scatterplot - Frequency of Purchases vs Purchase Amount
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Frequency of Purchases", y="Purchase Amount (USD)", hue="Subscription Status")
plt.title("Frequency of Purchases vs Purchase Amount")
plt.xlabel("Frequency of Purchases")
plt.ylabel("Purchase Amount (USD)")
plt.show()

# Optional Visualization 4: Count Plot - Distribution of Subscription Status
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Distribution of Subscription Status")
plt.xlabel("Subscription Status")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_res, y_train_res)

# Best model
best_dt_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Make predictions
y_pred = best_dt_model.predict(X_test)

# Evaluate model with Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Subscription", "Subscription"], yticklabels=["No Subscription", "Subscription"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print(classification_report(y_test, y_pred))

# ROC Curve
y_pred_proba = best_dt_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Visualize Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(best_dt_model, feature_names=features, class_names=["No Subscription", "Subscription"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Feature Importance
feature_importance = best_dt_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance")
plt.show()

# Cross-Validation
cv_scores = cross_val_score(best_dt_model, X_train_res, y_train_res, cv=5, scoring="accuracy")
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


print("Training Data:\n", X_train.head())
print("Testing Data:\n", X_test.head())