# telecom-customer-classification
# Telecommunications Customer Category Prediction

This project uses machine learning to predict customer service categories for a telecommunications provider. By analyzing demographic and behavioral data, the model classifies customers into four distinct service groups, allowing the company to personalize marketing and service offerings.

## Purpose
The primary goal of this project is to build a classification model that can accurately assign customers to one of the following categories:
1. **Basic Service**
2. **E-Service**
3. **Plus Service**
4. **Total Service**

By identifying the patterns behind service selection, the business can optimize its customer acquisition strategy and improve resource allocation.

## Dataset
The dataset used is the [Telecommunications Dataset](https://www.kaggle.com/datasets/navins7/telecommunications/data). It includes 1,000 customer records with the following features:
- **Demographics:** Region, Age, Marital Status, Gender, Education Level (`ed`), and Household Size (`reside`).
- **Behavioral/Stability:** Tenure (months with company), Address (years at current location), Employment (years with employer), and Retirement status.
- **Financial:** Annual Income.

## Procedure

### 1. Data Exploration & Visualization
- Analyzed descriptive statistics to understand the scale of features like income and age.
- Visualized the distribution of **Education Levels** to see the demographic spread.
- Created histograms and box plots for **Income**, identifying a right-skewed distribution and outliers that required transformation.

### 2. Preprocessing & Feature Engineering
- **Log Transformation:** Applied `np.log1p` to the 'income' column to handle skewness and normalize the data.
- **Feature Selection:** Selected the most relevant predictors: `tenure`, `age`, `marital`, `address`, `income_log`, `ed`, `employ`, and `retire`.
- **Standardization:** Scaled all features using `StandardScaler` to ensure the model isn't biased toward features with larger magnitudes (like income vs. education level).

### 3. Model Building
- **Algorithm:** Support Vector Machine (SVM).
- **Hyperparameters:** The model was tuned using an RBF (Radial Basis Function) kernel with `C=10` and `gamma=0.001`.
- **Train/Test Split:** The data was split into 80% training and 20% testing sets to evaluate performance on unseen data.

### 4. Evaluation
The model performance was evaluated using a Classification Report, measuring Precision, Recall, and F1-Score for each of the four customer categories.

## Installation
To run this project, you need Python 3.x and the following libraries:
```bash
pip install pandas numpy matplotlib scikit-learn
