# Fitness Recommendation Machine Learning Project

## Overview

This project implements a multi-class machine learning system to recommend personalized workout types based on individual fitness profiles and physiological metrics. The system classifies users into three primary workout categories: strength training, high-intensity interval training (HIIT), and flexibility-focused exercises.

## Executive Summary

This research demonstrates that machine learning models can effectively predict optimal workout types for individuals by analyzing their physical characteristics, fitness levels, and behavioral patterns. The Random Forest classifier emerged as the best-performing model, outperforming baseline approaches and providing actionable insights for personalized fitness recommendations.

## Methodology

### Step 1: Environment Setup and Library Installation
The project begins by installing all required dependencies including data processing libraries (pandas, numpy), machine learning frameworks (scikit-learn, xgboost), statistical analysis tools (scipy), and visualization libraries (matplotlib, seaborn).

### Step 2: Data Import and Initial Exploration
The workout dataset is loaded and subjected to comprehensive exploratory data analysis. This stage includes examining data shape, identifying missing values, reviewing data types, and obtaining statistical summaries to understand the baseline data quality.

### Step 3: Dataset Copying and Preparation
A working copy of the raw dataset is created to preserve the original data while enabling iterative preprocessing and transformations throughout the analysis pipeline.

### Step 4: Feature Column Pruning
Unnecessary columns are systematically removed from the dataset, including row identifiers, timestamps, data source indicators, workout names, muscle group categories, equipment requirements, and sleep hour metrics. These columns are excluded because they either contain redundant information or are not relevant predictors for the multi-class classification task.

### Step 5: Target Variable Encoding
The target variable (workout type) is extracted and separated from feature columns. Target classes include three workout categories: strength training, high-intensity interval training (HIIT), and flexibility-focused workouts.

### Step 6: Missing Value Imputation
Missing values in numeric columns are addressed using median imputation strategy, while categorical columns utilize most frequent value imputation. This ensures complete dataset coverage without introducing significant bias from deletion strategies.

### Step 7: Outlier Detection and Removal
Statistical outlier detection is performed on key physiological metrics (user age, BMI, resting heart rate, target duration) using the interquartile range (IQR) method with a 1.5x IQR threshold. This step enhances model robustness by removing anomalous data points.

### Step 8: Feature Engineering
Novel features are engineered to capture complex relationships between existing variables:

- **Cardio Capacity Index**: Ratio of estimated VO2 maximum to resting heart rate, indicating cardiovascular efficiency
- **Heart Rate Efficiency**: Normalized measure of heart rate reserve relative to maximum heart rate
- **Fitness Experience Ratio**: Interaction term between fitness level and years of experience, representing cumulative fitness development

### Step 9: Target Variable Numeric Encoding
Categorical workout type labels are converted to numeric values: Strength (0), HIIT (1), Flexibility (2). This encoding enables compatibility with machine learning algorithms while preserving ordinal relationships.

### Step 10: Feature Selection
A curated set of 16 features is selected based on domain knowledge and relevance to workout recommendation:

- User Demographics: age, BMI
- Fitness Assessment Metrics: fitness level, experience years, VO2 max estimate
- Physiological Indicators: resting heart rate, heart rate reserve, maximum heart rate
- Behavioral Factors: motivation score, diet quality, hydration intake
- Performance Metrics: workout adherence percentage, time available, weekly consistency
- Engineered Features: cardio capacity index, heart rate efficiency, fitness experience ratio

### Step 11: Data Partitioning
The dataset is split into training and testing subsets using stratified random sampling (80-20 split) with random state 42. Stratification ensures consistent class distribution across both partitions.

### Step 12: Feature Scaling
StandardScaler normalization is applied to the feature space. This step is critical for distance-based algorithms (Logistic Regression, KNN) to prevent features with larger scales from dominating the model learning process.

### Step 13: Model Definition
Five machine learning algorithms are configured with optimized hyperparameters:

- **Logistic Regression**: Linear baseline model with 1000 iteration maximum
- **K-Nearest Neighbors**: Distance-based classifier with 7 neighbors
- **Random Forest**: Ensemble model with 200 trees and maximum depth of 10
- **Gradient Boosting**: Sequential ensemble classifier with default parameters
- **XGBoost**: Advanced gradient boosting implementation with 100 estimators and 0.1 learning rate

### Step 14: Model Training and Evaluation
All models are trained on the training partition and evaluated on the held-out test set. Performance is measured using multiple metrics:

- **Accuracy**: Overall correct classification rate
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean balancing precision and recall

### Step 15: Results Aggregation
Model performance metrics are compiled into a comparison dataframe, sorted by F1-score to identify the best overall performer.

### Step 16: Best Model Identification
The model achieving the highest weighted F1-score is identified for detailed analysis and implementation.

### Step 17: Final Model Training
The Random Forest classifier is retrained on the complete training dataset with optimized parameters (200 estimators, maximum depth 10) to maximize model capacity.

### Step 18: Confusion Matrix Analysis
A confusion matrix is generated and visualized to examine classification patterns across the three workout categories, revealing potential misclassification patterns and model strengths.

### Step 19: Feature Importance Extraction
The Random Forest model provides feature importance values indicating each feature's contribution to classification decisions, enabling identification of the most influential predictive variables.

### Step 20: Feature Importance Visualization
Importance scores are visualized in a horizontal bar plot, ranked from highest to lowest, facilitating easy interpretation of feature contributions.

### Step 21: Cross-Validation Assessment
Five-fold stratified cross-validation is performed using the F1-weighted metric to assess model stability and generalization performance across different data partitions.

### Step 22: Statistical Significance Testing
A paired t-test compares the Random Forest model against the Logistic Regression baseline across cross-validation folds, determining whether performance differences are statistically significant.

### Step 23: Comprehensive Findings Summary
Final findings document key discoveries and model implications.

## Key Findings

1. **Model Performance**: The Random Forest classifier achieved superior performance compared to baseline models, demonstrating the effectiveness of ensemble learning for multi-class workout recommendation tasks.

2. **Data Quality Impact**: Systematic preprocessing including outlier removal, missing value imputation, and feature engineering substantially enhanced model discriminative ability and generalization.

3. **Feature Relevance**: Behavioral and physiological features emerged as strong predictors of optimal workout type, indicating that personal characteristics significantly influence workout suitability.

4. **Feature Engineering Value**: Engineered features derived from existing metrics contributed measurably to model performance, demonstrating the importance of domain-informed feature creation.

5. **Classification Approach**: Multi-class classification proved more practical and realistic than binary classification approaches, accommodating the diversity of fitness needs and preferences.

## Dataset Information

The dataset comprises physiological measurements, fitness assessments, and behavioral metrics collected from fitness tracking sources. Features include:

- Demographic variables (age, BMI)
- Fitness metrics (fitness level, experience, VO2 max)
- Cardiovascular indicators (resting heart rate, heart rate reserve)
- Lifestyle factors (motivation, diet quality, hydration)
- Performance measures (adherence, available time, consistency)

The dataset represents a diverse population with varied fitness profiles and preferences.

## Usage Instructions

1. Ensure Python 3.7+ is installed with required dependencies
2. Install dependencies: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy imbalanced-learn joblib`
3. Place the notebook and workout_dataset.csv in the same directory
4. Execute the notebook cells sequentially from top to bottom
5. Review visualizations and model outputs for insights
6. Examine the final findings section for conclusions

## Project Structure

```
Fitness_recommendation_ML_Project/
├── README.md
├── Machine Project notebook.ipynb
└── workout_dataset.csv
```

## Contact Information

For further information regarding the scraped dataset, data collection methodology, or technical questions about the analysis, please contact:

Email: ahmadyazied144@gmail.com

## Requirements

- Python 3.7 or higher
- pandas >= 1.0
- numpy >= 1.18
- scikit-learn >= 0.24
- xgboost >= 1.3
- matplotlib >= 3.3
- seaborn >= 0.11
- scipy >= 1.5
- imbalanced-learn >= 0.8
- joblib >= 1.0

## License

This project is provided for research and educational purposes.