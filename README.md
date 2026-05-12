
Capstone Lab: Employee Promotion Prediction
Decision Tree vs SVM vs Random Forest — End-to-End ML Workflow
Business Problem
The HR Analytics team at a large organization wants to predict which employees are likely to be promoted in the next review cycle. The objective is to help leadership:

Proactively plan talent pipelines and succession
Identify patterns that lead to promotions
Reduce bias by using data-driven insights
Dataset Description
We will generate a synthetic HR dataset using the Faker library with the following 10 features and 1 target variable:

#	Feature	Type	Description
1	Age	Numeric	Employee age (22–60)
2	Gender	Categorical	Male / Female / Other
3	Department	Categorical	Sales, Engineering, Marketing, HR, Finance, Operations
4	Education	Ordinal	Bachelor / Master / PhD
5	YearsAtCompany	Numeric	Tenure in years (0–30)
6	TrainingHoursLastYear	Numeric	Training hours completed (0–200)
7	PerformanceRating	Numeric	Annual rating (1–5)
8	NumProjectsCompleted	Numeric	Projects delivered (0–20)
9	AvgMonthlyHours	Numeric	Average working hours/month (120–250)
10	PreviousPromotions	Numeric	Past promotions received (0–5)
Target	Promoted	Binary	0 = Not Promoted, 1 = Promoted
Note: The dataset will intentionally contain ~5–8% missing values across several columns to simulate real-world data and practice imputation techniques.

Learning Objectives
By the end of this capstone, you will be able to:

Generate realistic synthetic data using Faker with missing values
Perform comprehensive Exploratory Data Analysis (EDA)
Handle missing values, encode categoricals, and scale features — complete preprocessing pipeline
Train, tune, and evaluate Decision Tree, SVM, and Random Forest classifiers
Compare models using Accuracy, F1-Score, AUC-ROC, Confusion Matrix
Understand the key differences between DT, SVM, and RF and when to use which
Algorithm Quick Comparison (Theory Recap)
Criteria	Decision Tree 🌳	SVM (SVC) 📐	Random Forest 🌲🌲🌲
How it works	Recursive feature splitting using Gini/Entropy	Finds optimal hyperplane with maximum margin	Ensemble of many decision trees (Bagging)
Scaling needed?	❌ No	✅ Yes (distance-based)	❌ No
Handles missing values?	⚠️ Partially (sklearn: No)	❌ No	⚠️ Partially (sklearn: No)
Interpretability	⭐⭐⭐⭐⭐ (Very High — visual tree)	⭐⭐ (Low — black box)	⭐⭐⭐ (Moderate — feature importance)
Overfitting risk	⭐⭐⭐⭐⭐ (Very High)	⭐⭐ (Low with right C/γ)	⭐⭐ (Low — averaging reduces variance)
Key hyperparameters	max_depth, min_samples_split, criterion	C, gamma, kernel	n_estimators, max_depth, max_features
sklearn Class	DecisionTreeClassifier	SVC	RandomForestClassifier
Step 1: Install & Import Libraries
Lab Task: Run the cell below to install and import all required libraries.

# Install required packages
!pip install -q faker numpy pandas matplotlib seaborn scikit-learn

# Core Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Data Generation
from faker import Faker

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Models
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score, precision_recall_curve
)
from sklearn.inspection import permutation_importance

# Timing
import time

# Plot settings
COLORS = ['#2ecc71', '#e74c3c']
COLORS_3 = ['#3498db', '#e67e22', '#2ecc71']
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("All libraries imported successfully!")
print(f"   NumPy: {np.__version__}")
print(f"   Pandas: {pd.__version__}")
Step 2: Generate Synthetic HR Dataset Using Faker
Lab Task: Generate a realistic HR dataset with 2500 employees, 10 features, and intentional missing values (~5–8%).

Data Generation Logic:
Promotion probability is realistically linked to features:
Higher performance rating → higher chance
More training hours → higher chance
PhD education → slight advantage
More projects → higher chance
~5–8% missing values injected into: Age, TrainingHoursLastYear, PerformanceRating, AvgMonthlyHours, NumProjectsCompleted
# =============================================================
# STEP 2: Generate Synthetic HR Dataset
# =============================================================
fake = Faker()
Faker.seed(42)
np.random.seed(42)

n_samples = 2500

# Generate features
data = {
    'EmpID': [f'EMP{str(i).zfill(5)}' for i in range(1, n_samples + 1)],
    'Age': np.random.randint(22, 61, n_samples),
    'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
    'Department': np.random.choice(
        ['Sales', 'Engineering', 'Marketing', 'HR', 'Finance', 'Operations'],
        n_samples, p=[0.20, 0.25, 0.15, 0.10, 0.15, 0.15]
    ),
    'Education': np.random.choice(['Bachelor', 'Master', 'PhD'], n_samples, p=[0.50, 0.35, 0.15]),
    'YearsAtCompany': np.random.randint(0, 31, n_samples),
    'TrainingHoursLastYear': np.random.randint(0, 201, n_samples),
    'PerformanceRating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.40, 0.30, 0.10]),
    'NumProjectsCompleted': np.random.randint(0, 21, n_samples),
    'AvgMonthlyHours': np.random.randint(120, 251, n_samples),
    'PreviousPromotions': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.30, 0.30, 0.20, 0.12, 0.05, 0.03]),
}

df = pd.DataFrame(data)

# Generate realistic promotion target based on feature values
promotion_score = (
    df['PerformanceRating'] * 0.30 +
    (df['TrainingHoursLastYear'] / 200) * 0.15 +
    (df['NumProjectsCompleted'] / 20) * 0.15 +
    (df['YearsAtCompany'] / 30) * 0.10 +
    (df['Education'].map({'Bachelor': 0, 'Master': 0.5, 'PhD': 1.0})) * 0.10 +
    (df['AvgMonthlyHours'] / 250) * 0.10 +
    (df['PreviousPromotions'] / 5) * 0.10
)

# Normalize to probability and add noise
promo_prob = (promotion_score - promotion_score.min()) / (promotion_score.max() - promotion_score.min())
promo_prob = promo_prob * 0.6 + 0.05  # Scale to 5%-65% range
noise = np.random.normal(0, 0.1, n_samples)
promo_prob = np.clip(promo_prob + noise, 0.02, 0.95)

df['Promoted'] = (np.random.random(n_samples) < promo_prob).astype(int)

# -------------------------------------------------------
# INJECT MISSING VALUES (~5-8% per column)
# -------------------------------------------------------
cols_with_missing = ['Age', 'TrainingHoursLastYear', 'PerformanceRating', 'AvgMonthlyHours', 'NumProjectsCompleted']

for col in cols_with_missing:
    missing_pct = np.random.uniform(0.05, 0.08)  # 5-8%
    missing_idx = np.random.choice(df.index, size=int(n_samples * missing_pct), replace=False)
    df.loc[missing_idx, col] = np.nan

# Convert affected columns to float for NaN support
for col in cols_with_missing:
    df[col] = df[col].astype(float)

# -------------------------------------------------------
# Display Results
# -------------------------------------------------------
print("=" * 80)
print("SYNTHETIC HR DATASET GENERATED SUCCESSFULLY")
print("=" * 80)
print(f"\n Dataset Shape: {df.shape}")
print(f"\n First 5 rows:")
print(df.head().to_string())

print(f"\n\n⚠️  Missing Values:")
print("-" * 40)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0].to_string())
print(f"\nTotal Missing Cells: {df.isnull().sum().sum()} / {df.shape[0] * df.shape[1]} ({df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%)")

print(f"\n\n Target Distribution:")
promo_counts = df['Promoted'].value_counts()
print(f"   Not Promoted (0): {promo_counts[0]} ({promo_counts[0]/len(df)*100:.1f}%)")
print(f"   Promoted (1)    : {promo_counts[1]} ({promo_counts[1]/len(df)*100:.1f}%)")
Step 3: Exploratory Data Analysis (EDA)
Lab Task: Explore the dataset through visualizations to understand feature distributions, relationships, and missing data patterns.

# =============================================================
# 3.1 Dataset Overview
# =============================================================
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)

print("\n Data Types:")
print(df.dtypes.to_string())

print("\n\n Statistical Summary (Numerical):")
print(df.describe().round(2).to_string())

print("\n\nStatistical Summary (Categorical):")
print(df.describe(include='object').to_string())
# =============================================================
# 3.2 Missing Values Visualization
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Missing values bar chart
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
ax1 = axes[0]
bars = ax1.bar(missing.index, missing.values, color='#e74c3c', edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, missing.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_title('Missing Values Count by Column', fontsize=14, fontweight='bold')
ax1.set_ylabel('Count')
ax1.set_xlabel('Column')

# Missing values heatmap (sample of first 100 rows for visibility)
ax2 = axes[1]
cols_to_check = ['Age', 'TrainingHoursLastYear', 'PerformanceRating', 'AvgMonthlyHours', 'NumProjectsCompleted']
sns.heatmap(df[cols_to_check].head(200).isnull(), cbar=True, yticklabels=False,
            cmap=['#2ecc71', '#e74c3c'], ax=ax2)
ax2.set_title('Missing Values Heatmap (First 200 Rows)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_missing_values.png', dpi=150, bbox_inches='tight')
plt.show()
print("Missing values visualization saved!")
# =============================================================
# 3.3 Target Distribution
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Countplot
promo_counts = df['Promoted'].value_counts()
ax1 = axes[0]
bars = ax1.bar(['Not Promoted (0)', 'Promoted (1)'], promo_counts.values, color=COLORS, edgecolor='white', linewidth=2)
for bar, val in zip(bars, promo_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{val} ({val/len(df)*100:.1f}%)', ha='center', fontsize=12, fontweight='bold')
ax1.set_title('🎯 Target Distribution — Promoted vs Not Promoted', fontsize=14, fontweight='bold')
ax1.set_ylabel('Count')

# Pie chart
ax2 = axes[1]
ax2.pie(promo_counts.values, labels=['Not Promoted', 'Promoted'],
        colors=COLORS, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 13},
        explode=(0, 0.05), shadow=True)
ax2.set_title('Target Split', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
# =============================================================
# 3.4 Numerical Feature Distributions by Promotion Status
# =============================================================
num_cols = ['Age', 'YearsAtCompany', 'TrainingHoursLastYear', 'PerformanceRating',
            'NumProjectsCompleted', 'AvgMonthlyHours', 'PreviousPromotions']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    for promoted, color, label in zip([0, 1], COLORS, ['Not Promoted', 'Promoted']):
        subset = df[df['Promoted'] == promoted][col].dropna()
        ax.hist(subset, bins=20, alpha=0.6, color=color, label=label, edgecolor='white')
    ax.set_title(f'{col}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)

axes[-1].axis('off')  # Hide last empty subplot
fig.suptitle('📊 Feature Distributions by Promotion Status', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
# =============================================================
# 3.5 Boxplots — Key Features by Promotion Status
# =============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

box_cols = ['PerformanceRating', 'TrainingHoursLastYear', 'NumProjectsCompleted',
            'YearsAtCompany', 'AvgMonthlyHours', 'Age']

for i, col in enumerate(box_cols):
    sns.boxplot(data=df, x='Promoted', y=col, ax=axes[i], palette=COLORS, width=0.5)
    axes[i].set_title(f'{col} by Promotion', fontsize=12, fontweight='bold')
    axes[i].set_xticklabels(['Not Promoted', 'Promoted'])

fig.suptitle( 'Boxplots — Features by Promotion Status', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
# =============================================================
# 3.6 Categorical Features vs Promoted
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

cat_cols = ['Gender', 'Department', 'Education']

for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df['Promoted'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, ax=axes[i], color=COLORS, edgecolor='white', linewidth=1)
    axes[i].set_title(f'{col} vs Promotion Rate', fontsize=13, fontweight='bold')
    axes[i].set_ylabel('Percentage (%)')
    axes[i].set_xlabel('')
    axes[i].legend(['Not Promoted', 'Promoted'], fontsize=9)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

plt.suptitle('📊 Categorical Features — Promotion Rate', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_categorical_features.png', dpi=150, bbox_inches='tight')
plt.show()
# =============================================================
# 3.7 Correlation Heatmap
# =============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Select numeric columns only
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=1, ax=ax, vmin=-1, vmax=1)
ax.set_title('🔗 Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n💡 Key Observations:")
promo_corr = corr['Promoted'].drop('Promoted').sort_values(ascending=False)
for feat, val in promo_corr.items():
    direction = "📈 Positive" if val > 0 else "📉 Negative"
    print(f"   {feat:30s} → Promoted: {val:+.3f} ({direction})")
Step 4: Data Preprocessing — Complete ML Pipeline
Lab Task: Clean the data, handle missing values, encode categoricals, split and scale the data.

Preprocessing Steps:
Drop EmpID (non-informative)
Impute missing values (Median for numerical, Mode for categorical)
Encode categorical variables (OneHot for Gender/Department, Ordinal for Education)
Split into Train/Test (80/20, stratified)
Scale features for SVM (DT and RF don't need scaling!)
# =============================================================
# STEP 4: DATA PREPROCESSING
# =============================================================
print("=" * 80)
print("⚙️  DATA PREPROCESSING PIPELINE")
print("=" * 80)

# 4.1 Drop EmpID
df_processed = df.drop('EmpID', axis=1)
print("\nStep 4.1: Dropped 'EmpID' column")

# 4.2 Handle Missing Values
print("\nStep 4.2: Handling Missing Values")
print(f"   Before — Total missing: {df_processed.isnull().sum().sum()}")

# Numerical columns — Median imputation
num_cols_missing = ['Age', 'TrainingHoursLastYear', 'PerformanceRating', 'AvgMonthlyHours', 'NumProjectsCompleted']
num_imputer = SimpleImputer(strategy='median')
df_processed[num_cols_missing] = num_imputer.fit_transform(df_processed[num_cols_missing])

print(f"   After  — Total missing: {df_processed.isnull().sum().sum()}")

# 4.3 Encode Categorical Variables
print("\nStep 4.3: Encoding Categorical Variables")

# Ordinal encoding for Education
edu_mapping = {'Bachelor': 0, 'Master': 1, 'PhD': 2}
df_processed['Education'] = df_processed['Education'].map(edu_mapping)
print("   Education → Ordinal (Bachelor=0, Master=1, PhD=2)")

# One-Hot Encoding for Gender and Department
df_processed = pd.get_dummies(df_processed, columns=['Gender', 'Department'], drop_first=True, dtype=int)
print("   Gender → OneHot (drop_first=True)")
print("   Department → OneHot (drop_first=True)")

# 4.4 Feature-Target Split
X = df_processed.drop('Promoted', axis=1)
y = df_processed['Promoted']
print(f"\nStep 4.4: Feature-Target Split")
print(f"   Features (X): {X.shape}")
print(f"   Target (y)  : {y.shape}")
print(f"   Feature names: {list(X.columns)}")

# 4.5 Train-Test Split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nStep 4.5: Train-Test Split (80/20, stratified)")
print(f"   Train: {X_train.shape[0]} samples | Promoted: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"   Test : {X_test.shape[0]} samples  | Promoted: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# 4.6 Feature Scaling (for SVM only!)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
print(f"\nStep 4.6: Feature Scaling (StandardScaler)")
print("   IMPORTANT: Scaled data ONLY used for SVM!")
print("   DT and RF use UNSCALED data (they are tree-based, not distance-based)")

print(f"\n{'='*80}")
print("PREPROCESSING COMPLETE!")
print(f"{'='*80}")
Step 5: Model 1 — Decision Tree Classifier
🔬 Lab Task: Train a Decision Tree, visualize the tree structure, extract feature importance, and tune hyperparameters.

Key Concepts:
Gini Impurity: Measures probability of incorrect classification at a node → Gini = 1 - Σ pᵢ²
Entropy/Information Gain: Measures reduction in uncertainty → IG = H(parent) - Σ H(child)
Pre-pruning: max_depth, min_samples_split, min_samples_leaf to prevent overfitting
Feature Importance: Based on total reduction in impurity by each feature across all splits
# =============================================================
# 5.1 Decision Tree — Baseline Model
# =============================================================
print("=" * 80)
print("🌳 MODEL 1: DECISION TREE — BASELINE")
print("=" * 80)

start = time.time()
dt_base = DecisionTreeClassifier(random_state=42)
dt_base.fit(X_train, y_train)
dt_base_time = time.time() - start

y_pred_dt_base = dt_base.predict(X_test)
y_prob_dt_base = dt_base.predict_proba(X_test)[:, 1]

print(f"\n⏱️  Training Time: {dt_base_time:.4f} seconds")
print(f"📊 Tree Depth   : {dt_base.get_depth()}")
print(f"🍃 Total Leaves : {dt_base.get_n_leaves()}")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_dt_base, target_names=['Not Promoted', 'Promoted']))

acc_dt_base = accuracy_score(y_test, y_pred_dt_base)
f1_dt_base = f1_score(y_test, y_pred_dt_base)
auc_dt_base = roc_auc_score(y_test, y_prob_dt_base)
print(f"✅ Accuracy: {acc_dt_base:.4f} | F1-Score: {f1_dt_base:.4f} | AUC-ROC: {auc_dt_base:.4f}")
# =============================================================
# 5.2 Visualize Decision Tree (max_depth=4 for readability)
# =============================================================
fig, ax = plt.subplots(figsize=(25, 12))
plot_tree(dt_base, max_depth=4, feature_names=X.columns, class_names=['Not Promoted', 'Promoted'],
          filled=True, rounded=True, fontsize=8, ax=ax,
          impurity=True, proportion=True)
ax.set_title('Decision Tree Visualization (Top 4 Levels)', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_tree_visualization.png', dpi=120, bbox_inches='tight')
plt.show()
print(" Decision Tree visualization saved!")
# =============================================================
# 5.3 Feature Importance — Decision Tree
# =============================================================
importance_dt = pd.Series(dt_base.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(importance_dt.index, importance_dt.values, color='#27ae60', edgecolor='white', linewidth=1)
for bar, val in zip(bars, importance_dt.values):
    if val > 0.01:
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=9, fontweight='bold')
ax.set_title('🌳 Feature Importance — Decision Tree (Baseline)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance (Gini)')
plt.tight_layout()
plt.savefig('dt_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
# =============================================================
# 5.4 Hyperparameter Tuning — Decision Tree (GridSearchCV)
# =============================================================
print("=" * 80)
print("🔧 DECISION TREE — HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 80)

param_grid_dt = {
    'max_depth': [3, 5, 7, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt, cv=cv, scoring='f1', n_jobs=-1, verbose=0
)

start = time.time()
grid_dt.fit(X_train, y_train)
dt_tune_time = time.time() - start

print(f"\n⏱️  Grid Search Time: {dt_tune_time:.2f} seconds")
print(f"🏆 Best Parameters : {grid_dt.best_params_}")
print(f"📊 Best CV F1-Score: {grid_dt.best_score_:.4f}")

# Evaluate tuned model
dt_tuned = grid_dt.best_estimator_
y_pred_dt_tuned = dt_tuned.predict(X_test)
y_prob_dt_tuned = dt_tuned.predict_proba(X_test)[:, 1]

print(f"\n📋 Tuned Classification Report:")
print(classification_report(y_test, y_pred_dt_tuned, target_names=['Not Promoted', 'Promoted']))

acc_dt_tuned = accuracy_score(y_test, y_pred_dt_tuned)
f1_dt_tuned = f1_score(y_test, y_pred_dt_tuned)
auc_dt_tuned = roc_auc_score(y_test, y_prob_dt_tuned)
print(f"✅ Tuned — Accuracy: {acc_dt_tuned:.4f} | F1: {f1_dt_tuned:.4f} | AUC: {auc_dt_tuned:.4f}")

# Improvement
print(f"\n📈 Improvement over Baseline:")
print(f"   Accuracy: {acc_dt_base:.4f} → {acc_dt_tuned:.4f} ({(acc_dt_tuned-acc_dt_base)*100:+.2f}%)")
print(f"   F1-Score: {f1_dt_base:.4f} → {f1_dt_tuned:.4f} ({(f1_dt_tuned-f1_dt_base)*100:+.2f}%)")
Step 6: Model 2 — Support Vector Classifier (SVM)
*Lab Task: Train an SVM on **scaled data, tune hyperparameters, and evaluate.

Key Concepts:
Hyperplane: Decision boundary that maximizes the margin between classes
Support Vectors: Data points closest to the hyperplane
Kernel Trick: Maps data to higher dimensions (RBF, Polynomial)
C parameter: Trade-off between margin width and misclassification tolerance
Gamma (γ): Controls influence of a single training example — low γ = far reach, high γ = close reach
IMPORTANT: SVM is a distance-based algorithm — feature scaling is MANDATORY! Using X_train_scaled and X_test_scaled.

# =============================================================
# 6.1 SVM — Baseline Model (using SCALED data!)
# =============================================================
print("=" * 80)
print(" MODEL 2: SVM (SVC) — BASELINE (Scaled Data)")
print("=" * 80)

start = time.time()
svm_base = SVC(kernel='rbf', probability=True, random_state=42)
svm_base.fit(X_train_scaled, y_train)
svm_base_time = time.time() - start

y_pred_svm_base = svm_base.predict(X_test_scaled)
y_prob_svm_base = svm_base.predict_proba(X_test_scaled)[:, 1]

print(f"\n⏱️  Training Time: {svm_base_time:.4f} seconds")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_svm_base, target_names=['Not Promoted', 'Promoted']))

acc_svm_base = accuracy_score(y_test, y_pred_svm_base)
f1_svm_base = f1_score(y_test, y_pred_svm_base)
auc_svm_base = roc_auc_score(y_test, y_prob_svm_base)
print(f"Accuracy: {acc_svm_base:.4f} | F1-Score: {f1_svm_base:.4f} | AUC-ROC: {auc_svm_base:.4f}")
# =============================================================
# 6.2 Hyperparameter Tuning — SVM (GridSearchCV)
# =============================================================
print("=" * 80)
print(" SVM — HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 80)

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['linear', 'rbf']
}

grid_svm = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid_svm, cv=cv, scoring='f1', n_jobs=-1, verbose=0
)

start = time.time()
grid_svm.fit(X_train_scaled, y_train)
svm_tune_time = time.time() - start

print(f"\n⏱️  Grid Search Time: {svm_tune_time:.2f} seconds")
print(f"🏆 Best Parameters : {grid_svm.best_params_}")
print(f"Best CV F1-Score: {grid_svm.best_score_:.4f}")

# Evaluate tuned model
svm_tuned = grid_svm.best_estimator_
y_pred_svm_tuned = svm_tuned.predict(X_test_scaled)
y_prob_svm_tuned = svm_tuned.predict_proba(X_test_scaled)[:, 1]

print(f"\nTuned Classification Report:")
print(classification_report(y_test, y_pred_svm_tuned, target_names=['Not Promoted', 'Promoted']))

acc_svm_tuned = accuracy_score(y_test, y_pred_svm_tuned)
f1_svm_tuned = f1_score(y_test, y_pred_svm_tuned)
auc_svm_tuned = roc_auc_score(y_test, y_prob_svm_tuned)
print(f"Tuned — Accuracy: {acc_svm_tuned:.4f} | F1: {f1_svm_tuned:.4f} | AUC: {auc_svm_tuned:.4f}")

print(f"\nImprovement over Baseline:")
print(f"   Accuracy: {acc_svm_base:.4f} → {acc_svm_tuned:.4f} ({(acc_svm_tuned-acc_svm_base)*100:+.2f}%)")
print(f"   F1-Score: {f1_svm_base:.4f} → {f1_svm_tuned:.4f} ({(f1_svm_tuned-f1_svm_base)*100:+.2f}%)")
Step 7: Model 3 — Random Forest Classifier
Lab Task: Train a Random Forest with OOB scoring, compare feature importance with Decision Tree, and tune.

Key Concepts:
Bagging (Bootstrap Aggregating): Each tree trained on a random bootstrap sample
Random Feature Selection: At each split, only √n features considered
Out-of-Bag (OOB) Error: ~36.8% data left out per tree → free validation!
Ensemble Voting: Final prediction = majority vote of all trees
Reduced Overfitting: Averaging many diverse trees reduces variance
# =============================================================
# 7.1 Random Forest — Baseline Model
# =============================================================
print("=" * 80)
print(" MODEL 3: RANDOM FOREST — BASELINE")
print("=" * 80)

start = time.time()
rf_base = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
rf_base_time = time.time() - start

y_pred_rf_base = rf_base.predict(X_test)
y_prob_rf_base = rf_base.predict_proba(X_test)[:, 1]

print(f"\n⏱️  Training Time   : {rf_base_time:.4f} seconds")
print(f"Number of Trees  : {rf_base.n_estimators}")
print(f"OOB Score        : {rf_base.oob_score_:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_rf_base, target_names=['Not Promoted', 'Promoted']))

acc_rf_base = accuracy_score(y_test, y_pred_rf_base)
f1_rf_base = f1_score(y_test, y_pred_rf_base)
auc_rf_base = roc_auc_score(y_test, y_prob_rf_base)
print(f" Accuracy: {acc_rf_base:.4f} | F1-Score: {f1_rf_base:.4f} | AUC-ROC: {auc_rf_base:.4f}")
# =============================================================
# 7.2 Feature Importance — Random Forest vs Decision Tree
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Decision Tree importance
imp_dt = pd.Series(dt_tuned.feature_importances_, index=X.columns).sort_values(ascending=True)
axes[0].barh(imp_dt.index, imp_dt.values, color='#27ae60', edgecolor='white')
axes[0].set_title('Decision Tree (Tuned)\nFeature Importance', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Importance')

# Random Forest importance
imp_rf = pd.Series(rf_base.feature_importances_, index=X.columns).sort_values(ascending=True)
axes[1].barh(imp_rf.index, imp_rf.values, color='#2980b9', edgecolor='white')
axes[1].set_title('Random Forest (100 Trees)\nFeature Importance', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Importance')

plt.suptitle('Feature Importance Comparison: DT vs RF', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Key Insight: Random Forest distributes importance more evenly across features,")
print("   while Decision Tree concentrates on fewer features (more prone to overfitting).")
# =============================================================
# 7.3 Hyperparameter Tuning — Random Forest (GridSearchCV)
# =============================================================
print("=" * 80)
print("🔧 RANDOM FOREST — HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 80)

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, oob_score=True, n_jobs=-1),
    param_grid_rf, cv=cv, scoring='f1', n_jobs=-1, verbose=0
)

start = time.time()
grid_rf.fit(X_train, y_train)
rf_tune_time = time.time() - start

print(f"\nGrid Search Time: {rf_tune_time:.2f} seconds")
print(f"🏆 Best Parameters : {grid_rf.best_params_}")
print(f"Best CV F1-Score: {grid_rf.best_score_:.4f}")

# Evaluate tuned model
rf_tuned = grid_rf.best_estimator_
y_pred_rf_tuned = rf_tuned.predict(X_test)
y_prob_rf_tuned = rf_tuned.predict_proba(X_test)[:, 1]

print(f"\n Tuned Classification Report:")
print(classification_report(y_test, y_pred_rf_tuned, target_names=['Not Promoted', 'Promoted']))

acc_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)
f1_rf_tuned = f1_score(y_test, y_pred_rf_tuned)
auc_rf_tuned = roc_auc_score(y_test, y_prob_rf_tuned)
print(f"Tuned — Accuracy: {acc_rf_tuned:.4f} | F1: {f1_rf_tuned:.4f} | AUC: {auc_rf_tuned:.4f}")

print(f"\nImprovement over Baseline:")
print(f"   Accuracy: {acc_rf_base:.4f} → {acc_rf_tuned:.4f} ({(acc_rf_tuned-acc_rf_base)*100:+.2f}%)")
print(f"   F1-Score: {f1_rf_base:.4f} → {f1_rf_tuned:.4f} ({(f1_rf_tuned-f1_rf_base)*100:+.2f}%)")
Step 8: Final Model Comparison — DT vs SVM vs RF
🔬 Lab Task: Compare all 6 models (3 base + 3 tuned) across Accuracy, F1-Score, AUC-ROC and visualize.

# =============================================================
# 8.1 Comprehensive Comparison Table
# =============================================================
results = pd.DataFrame({
    'Model': ['DT (Base)', 'DT (Tuned)', 'SVM (Base)', 'SVM (Tuned)', 'RF (Base)', 'RF (Tuned)'],
    'Accuracy': [acc_dt_base, acc_dt_tuned, acc_svm_base, acc_svm_tuned, acc_rf_base, acc_rf_tuned],
    'F1-Score': [f1_dt_base, f1_dt_tuned, f1_svm_base, f1_svm_tuned, f1_rf_base, f1_rf_tuned],
    'AUC-ROC': [auc_dt_base, auc_dt_tuned, auc_svm_base, auc_svm_tuned, auc_rf_base, auc_rf_tuned],
    'Train Time (s)': [dt_base_time, dt_tune_time, svm_base_time, svm_tune_time, rf_base_time, rf_tune_time]
})

print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)
print()
print(results.to_string(index=False, float_format='{:.4f}'.format))

# Highlight best model
best_idx = results['F1-Score'].idxmax()
print(f"\n🥇 Best Model (by F1-Score): {results.loc[best_idx, 'Model']}")
print(f"   Accuracy: {results.loc[best_idx, 'Accuracy']:.4f}")
print(f"   F1-Score: {results.loc[best_idx, 'F1-Score']:.4f}")
print(f"   AUC-ROC : {results.loc[best_idx, 'AUC-ROC']:.4f}")
# =============================================================
# 8.2 Grouped Bar Chart — Accuracy, F1, AUC Comparison
# =============================================================
fig, ax = plt.subplots(figsize=(16, 7))

x = np.arange(len(results))
width = 0.25

bars1 = ax.bar(x - width, results['Accuracy'], width, label='Accuracy', color='#3498db', edgecolor='white')
bars2 = ax.bar(x, results['F1-Score'], width, label='F1-Score', color='#e67e22', edgecolor='white')
bars3 = ax.bar(x + width, results['AUC-ROC'], width, label='AUC-ROC', color='#2ecc71', edgecolor='white')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('🏆 Model Performance Comparison — DT vs SVM vs RF', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results['Model'], fontsize=10)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison_bar.png', dpi=150, bbox_inches='tight')
plt.show()
# =============================================================
# 8.3 ROC Curves — All Models Overlay
# =============================================================
fig, ax = plt.subplots(figsize=(10, 8))

models_roc = [
    ('DT (Base)', y_prob_dt_base, '#bdc3c7'),
    ('DT (Tuned)', y_prob_dt_tuned, '#27ae60'),
    ('SVM (Base)', y_prob_svm_base, '#d4a574'),
    ('SVM (Tuned)', y_prob_svm_tuned, '#e67e22'),
    ('RF (Base)', y_prob_rf_base, '#85c1e9'),
    ('RF (Tuned)', y_prob_rf_tuned, '#2980b9'),
]

for name, y_prob, color in models_roc:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    lw = 3 if 'Tuned' in name else 1.5
    ls = '-' if 'Tuned' in name else '--'
    ax.plot(fpr, tpr, color=color, lw=lw, ls=ls, label=f'{name} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('📈 ROC Curves — All Models Comparison', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig('roc_curves_all_models.png', dpi=150, bbox_inches='tight')
plt.show()
# =============================================================
# 8.4 Confusion Matrices — Tuned Models Side by Side
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

tuned_models = [
    ('🌳 Decision Tree (Tuned)', y_pred_dt_tuned),
    ('📐 SVM (Tuned)', y_pred_svm_tuned),
    ('🌲 Random Forest (Tuned)', y_pred_rf_tuned),
]

for ax, (name, y_pred) in zip(axes, tuned_models):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Not Promoted', 'Promoted'],
                yticklabels=['Not Promoted', 'Promoted'],
                annot_kws={'size': 16, 'fontweight': 'bold'})
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.suptitle('Confusion Matrices — Tuned Models Comparison', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
Step 9: Key Differences — Decision Tree vs SVM vs Random Forest
Comprehensive Comparison Table
#	Criteria	Decision Tree 🌳	SVM (SVC) 📐	Random Forest 🌲🌲🌲
1	Algorithm Type	Single tree, recursive partitioning	Margin-based, finds optimal hyperplane	Ensemble of N decision trees (Bagging)
2	Decision Boundary	Axis-parallel (rectangular regions)	Linear/Non-linear (kernel-dependent)	Complex, averaged over many trees
3	Feature Scaling	❌ NOT needed	✅ MANDATORY (distance-based)	❌ NOT needed
4	Missing Value Handling	⚠️ Needs imputation in sklearn	⚠️ Needs imputation	⚠️ Needs imputation in sklearn
5	Interpretability	⭐⭐⭐⭐⭐ Highest (visual tree!)	⭐⭐ Low (black box)	⭐⭐⭐ Moderate (feature importance)
6	Overfitting Risk	🔴 Very High (deep trees memorize)	🟢 Low (regularization via C, γ)	🟢 Low (averaging reduces variance)
7	Training Speed	⚡ Very Fast	🐢 Slow (especially large datasets)	⚡ Moderate (parallelizable)
8	Prediction Speed	⚡ Very Fast	⚡ Fast	⚡ Moderate (N trees to traverse)
9	Feature Importance	✅ Built-in (Gini/Entropy-based)	❌ Not directly available	✅ Built-in (averaged across trees)
10	Handles Large Data	✅ Good	❌ Poor (O(n²) to O(n³))	✅ Excellent (parallelizable)
11	Key Hyperparameters	max_depth, min_samples_split, criterion	C, gamma, kernel	n_estimators, max_depth, max_features
12	Best Use Case	Explainable AI, quick baselines, small data	Small-medium data, high-dimensional, clear margins	General purpose, tabular data, production systems
🧭 When to Use Which? — Decision Guide
Use Decision Tree when:
✅ You need a fully interpretable model (regulatory/compliance)
✅ You want a quick baseline before trying complex models
✅ Data has non-linear relationships with clear feature thresholds
✅ Stakeholders need to understand every decision the model makes
Use SVM when:
✅ Data has clear margin of separation between classes
✅ Dataset is small to medium sized (< 10,000 samples)
✅ You have high-dimensional data (many features)
✅ You need strong generalization with good regularization
❌ Avoid for very large datasets (training is O(n²) to O(n³))
Use Random Forest when:
✅ You want a robust, production-ready model out of the box
✅ Dataset is medium to large with mixed feature types
✅ You need feature importance for business insights
✅ You want low overfitting without extensive tuning
✅ This is your go-to algorithm for tabular classification problems
💡 Pro Tip: Always start with Random Forest as your baseline for tabular data. Then try DT for interpretability or SVM for small, high-dimensional datasets.

Step 10: Conclusion & Capstone Summary
What We Achieved
Step	Task	Status
1	Generated 2500-row synthetic HR dataset with Faker	✅
2	Injected ~5-8% realistic missing values	✅
3	Comprehensive EDA with 8+ visualizations	✅
4	End-to-end preprocessing (imputation, encoding, scaling)	✅
5	Trained & tuned Decision Tree with tree visualization	✅
6	Trained & tuned SVM on scaled data	✅
7	Trained & tuned Random Forest with OOB score	✅
8	Compared all 6 models (ROC, confusion matrix, bar charts)	✅
9	Documented key differences and when-to-use guide	✅
Key Insights from the Lab
Random Forest typically provides the best balance of accuracy and generalization for tabular data
Decision Tree (without pruning) overfits badly — tuning max_depth is critical
SVM requires feature scaling and is slower to train, but generalizes well with proper C/γ tuning
Feature Importance is a powerful benefit of tree-based models for business interpretability
PerformanceRating is consistently the most important predictor of promotion across all models
 
