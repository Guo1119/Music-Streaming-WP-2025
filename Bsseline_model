# 1. Load Data

import pandas as pd
import numpy as np
import duckdb as ddb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
print("="*80)
print("STEP 1: CREATE OBSERVATION TABLES")
print("="*80)

con = ddb.connect("/Users/michaelmlr/Desktop/Master/Data/ncm-research-data.duckdb")

# Create observation_label table
print("\nCreating observation_label table...")
con.sql("""
CREATE OR REPLACE TABLE main.observation_label AS 
WITH first_act AS (
    SELECT
        userId,
        MIN(dt) AS first_act_dt
    FROM main.impression i 
    GROUP BY 1
    HAVING MIN(dt) <= 7
),
act_date AS (
    SELECT
        userId,
        dt
    FROM main.impression i
    GROUP BY 1, 2
)
SELECT 
    fa.userId,
    fa.first_act_dt,
    CASE
        WHEN COUNT(ad.dt) > 0 THEN 0
        ELSE 1
    END AS is_inactive
FROM first_act fa
LEFT JOIN act_date ad ON fa.userId = ad.userId
    AND ad.dt >= fa.first_act_dt + 14 
GROUP BY 1, 2
ORDER BY 1
""")
print("✓ observation_label table created")

# Create observation_impression table
print("\nCreating observation_impression table...")
con.sql("""
CREATE OR REPLACE TABLE main.observation_impression AS
SELECT 
    i.userId,
    dt,
    EPOCH_MS(impressTime) AS impressTimeStamp,
    impressPosition,
    mlogId,
    isClick,
    isComment,
    isLike,
    isIntoPersonalHomepage,
    isShare,
    isViewComment,
    mlogViewTime,
    detailMlogInfoList 
FROM main.observation_label ol 
LEFT JOIN main.impression i 
ON ol.userId = i.userId AND i.dt <= ol.first_act_dt + 7
ORDER BY i.userId, impressTime, impressPosition 
""")
print("✓ observation_impression table created")

# Create observation_demographics table
print("\nCreating observation_demographics table...")
con.sql("""
CREATE OR REPLACE TABLE main.observation_demographics AS 
WITH observations AS (
    SELECT
        userId,
        MIN(dt) AS first_act_dt
    FROM
        main.impression
    GROUP BY 1
    HAVING MIN(dt) <= 7
),
mode_province AS (
    SELECT 
        province,
        COUNT(*)
    FROM main.user_demographics
    GROUP BY 1
    ORDER BY COUNT(*) DESC
    LIMIT 1
),
medians AS (
    SELECT
        MEDIAN(age) AS median_age,
        MEDIAN(registeredMonthCnt) AS median_registeredMonthCnt,
        MEDIAN(followCnt) AS median_followCnt,
        MEDIAN(level) AS median_level
    FROM main.user_demographics
)
SELECT
    o.userId,
    COALESCE(province, (SELECT province FROM mode_province)) AS province,
    COALESCE(age, (SELECT median_age FROM medians)) AS age,
    COALESCE(gender, 'unknown') AS gender,
    IF(age IS NULL, 1, 0) AS is_age_missing,
    COALESCE(registeredMonthCnt, (SELECT median_registeredMonthCnt FROM medians)) AS registeredMonthCnt,
    COALESCE(followCnt, (SELECT median_followCnt FROM medians)) AS followCnt,
    COALESCE(level, (SELECT median_level FROM medians)) AS level
FROM
    observations o
LEFT JOIN main.user_demographics ud ON o.userId = ud.userId
ORDER BY o.userId
""")
print("✓ observation_demographics table created")
## EXTRACT FEATURES FROM DATABASE
print("\n" + "="*80)
print("STEP 2: EXTRACT FEATURES")
print("="*80)

query = """
WITH aggregated AS (
    SELECT
        userId,
        COUNT(DISTINCT dt) AS active_days,
        COUNT(mlogId) AS total_impressions,
        COUNT(IF(isClick = 1, mlogId, NULL)) AS total_clicks,
        COUNT(IF(isLike = 1, mlogId, NULL)) AS total_likes,
        COUNT(IF(isShare = 1, mlogId, NULL)) AS total_shares,
        COUNT(IF(isComment = 1, mlogId, NULL)) AS total_comments,
        COUNT(IF(isIntoPersonalHomepage = 1, mlogId, NULL)) AS total_into_personal_pages,
        COUNT(IF(isViewComment = 1, mlogId, NULL)) AS total_view_comments,
        COALESCE(
            SUM(mlogViewTime) / 60,
            0
        ) AS total_mlog_watchtime,
        COUNT(IF(isClick = 1, mlogId, NULL)) / COUNT(mlogId) AS impression_ctr,
        COUNT(IF(isLike = 1 OR isShare = 1 OR isComment = 1 OR isIntoPersonalHomepage = 1 OR isViewComment = 1, mlogId, NULL)) / COUNT(mlogId) AS interaction_rate,
        COALESCE(
            COUNT(IF(isLike = 1, mlogId, NULL)) * 1.0 / NULLIF(
                COUNT(IF(isClick = 1, mlogId, NULL)),
                0
            ),
            0
        )AS like_rate,
        COALESCE(
            COUNT(IF(isShare = 1, mlogId, NULL)) * 1.0 / NULLIF(
                COUNT(IF(isClick = 1, mlogId, NULL)),
                0
            ),
            0
        )AS share_rate,
        COALESCE(
            COUNT(IF(isComment = 1, mlogId, NULL)) * 1.0 / NULLIF(
                COUNT(IF(isClick = 1, mlogId, NULL)),
                0
            ),
            0
        )AS comment_rate
    FROM
        main.observation_impression oi
    GROUP BY
        userId
),
per_day AS (
    SELECT 
        userId,
        COUNT(mlogId) / COUNT(DISTINCT dt) AS avg_daily_impressions,
        COUNT(IF(isClick = 1, mlogId, NULL)) / COUNT(DISTINCT dt) AS avg_daily_clicks,
        COUNT(IF(isLike = 1, mlogId, NULL)) / COUNT(DISTINCT dt) AS avg_daily_likes,
        COUNT(IF(isShare = 1, mlogId, NULL)) / COUNT(DISTINCT dt) AS avg_daily_shares,
        COUNT(IF(isComment = 1, mlogId, NULL)) / COUNT(DISTINCT dt) AS avg_daily_comments
    FROM
        main.observation_impression oi
    GROUP BY
        userId
),
preference AS (
    SELECT
        userId,
        AVG(impressPosition) AS avg_impress_position,
        COUNT(IF(detailMlogInfoList IS NOT NULL, oi.mlogId, NULL)) / COUNT(oi.mlogId) AS swipe_down_rate,
        MODE(
            DATE_PART(
                'hour',
                impressTimeStamp
            )
        ) AS favorite_hour,
        CASE
            WHEN COUNT(IF(TYPE = 1, oi.mlogId, NULL)) >= COUNT(IF(TYPE = 2, oi.mlogId, NULL)) THEN 1
            ELSE 2
        END AS favorite_format,
        COALESCE(
            MODE(creatorType) FILTER (
                isLike = 1
                OR isShare = 1
                OR isComment = 1
            ),
            -1
        ) AS favorite_creator_type,
        COALESCE(
            AVG(userLikeCount) FILTER(
                isClick = 1
            ),
            0
        ) AS avg_user_likes_in_clicked,
        COALESCE(
            AVG(userShareCount) FILTER(
                isClick = 1
            ),
            0
        ) AS avg_user_shares_in_clicked,
        COALESCE(
            AVG(userCommentCount) FILTER(
                isClick = 1
            ),
            0
        ) AS avg_user_comments_in_clicked,
        COALESCE(
            AVG(followeds) FILTER (
                isLike = 1
                OR isShare = 1
                OR isComment = 1
            ),
            0
        ) AS avg_creator_followers
    FROM
        main.observation_impression oi
    LEFT JOIN main.mlog_demographics md ON
        oi.mlogId = md.mlogId
    LEFT JOIN main.mlog_stats ms ON
        oi.mlogId = ms.mlogId
        AND oi.dt = ms.dt
    LEFT JOIN main.creator_demographics cd ON
        md.creatorId = cd.creatorId
    GROUP BY
        userId
)
SELECT 
    is_inactive,
    ol.userId,
    province,
    gender,
    age,
    is_age_missing,
    registeredMonthCnt AS registered_month_count,
    followCnt AS follow_count,
    active_days,
    total_impressions,
    total_clicks,
    total_likes,
    total_shares,
    total_comments,
    total_into_personal_pages,
    total_view_comments,
    ROUND(
        total_mlog_watchtime,
        2
    ) AS total_mlog_watchtime,
    ROUND(
        impression_ctr,
        4
    ) AS impression_ctr,
    ROUND(
        interaction_rate,
        4
    ) AS interaction_rate,
    ROUND(
        like_rate,
        4
    ) AS like_rate,
    ROUND(
        share_rate,
        4
    ) AS share_rate,
    ROUND(
        comment_rate,
        4
    ) AS comment_rate,
    ROUND(
        avg_daily_impressions,
        2
    ) AS avg_daily_impressions,
    ROUND(
        avg_daily_clicks,
        2
    ) AS avg_daily_clicks,
    ROUND(
        avg_daily_likes,
        2
    ) AS avg_daily_likes,
    ROUND(
        avg_daily_shares,
        2
    ) AS avg_daily_shares,
    ROUND(
        avg_daily_comments,
        2
    ) AS avg_daily_comments,
    ROUND(
        avg_impress_position,
        2
    ) AS avg_impress_position,
    ROUND(
        swipe_down_rate,
        4
    ) AS swipe_down_rate,
    favorite_hour,
    favorite_format,
    favorite_creator_type,
    ROUND(
        avg_user_likes_in_clicked,
        2
    ) AS avg_user_likes_in_clicked,
    ROUND(
        avg_user_shares_in_clicked,
        2
    ) AS avg_user_shares_in_clicked,
    ROUND(
        avg_user_comments_in_clicked,
        2
    ) AS avg_user_comments_in_clicked,
    ROUND(
        avg_creator_followers,
        2
    ) AS avg_creator_followers
FROM 
    main.observation_label ol
LEFT JOIN main.observation_demographics od ON
    ol.userId = od.userId
LEFT JOIN aggregated a ON
    ol.userId = a.userId
LEFT JOIN per_day pd ON
    ol.userId = pd.userId
LEFT JOIN preference p ON
    ol.userId = p.userId
ORDER BY
    ol.userId
"""

print("\nExecuting feature extraction query...")
df = con.sql(query).to_df()
con.close()

print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nTarget Variable Distribution:")
print(df['is_inactive'].value_counts())
print(f"Inactive User Ratio: {df['is_inactive'].mean():.2%}")
## Feature Preparation
print("\n" + "="*80)
print("STEP 3: FEATURE PREPARATION")
print("="*80)

# Create a copy for feature engineering
df_features = df.copy()

# One-Hot Encoding for province
province_dummies = pd.get_dummies(df_features['province'], prefix='province', drop_first=True)

# One-Hot Encoding for gender
gender_dummies = pd.get_dummies(df_features['gender'], prefix='gender', drop_first=True)

# Combine all features
df_features = pd.concat([df_features, province_dummies, gender_dummies], axis=1)

# Define numeric feature columns (36 base features from screenshot)
numeric_features = [
    'age', 'is_age_missing', 'registered_month_count', 'follow_count',
    'active_days', 'total_impressions', 'total_clicks', 'total_likes',
    'total_shares', 'total_comments', 'total_into_personal_pages',
    'total_view_comments', 'total_mlog_watchtime', 'impression_ctr',
    'interaction_rate', 'like_rate', 'share_rate', 'comment_rate',
    'avg_daily_impressions', 'avg_daily_clicks', 'avg_daily_likes',
    'avg_daily_shares', 'avg_daily_comments', 'avg_impress_position',
    'swipe_down_rate', 'favorite_hour', 'favorite_format',
    'favorite_creator_type', 'avg_user_likes_in_clicked',
    'avg_user_shares_in_clicked', 'avg_user_comments_in_clicked',
    'avg_creator_followers'
]

# Combine numeric features with encoded categorical features
feature_cols = numeric_features + list(province_dummies.columns) + list(gender_dummies.columns)

X = df_features[feature_cols]
y = df_features['is_inactive']

# Handle missing values
X = X.fillna(0)

# Replace infinite values
X = X.replace([np.inf, -np.inf], 0)

print(f"\n✓ Final Feature Matrix: {X.shape}")
print(f"  - Numeric Features: {len(numeric_features)}")
print(f"  - Province Dummies: {len(province_dummies.columns)}")
print(f"  - Gender Dummies: {len(gender_dummies.columns)}")
print(f"  - Total Features: {len(feature_cols)}")
## TRAIN-TEST SPLIT
print("\n" + "="*80)
print("STEP 4: TRAIN-TEST SPLIT")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Training Set: {X_train.shape[0]} samples ({y_train.mean():.2%} inactive)")
print(f"✓ Test Set: {X_test.shape[0]} samples ({y_test.mean():.2%} inactive)")
## XGBOOST BASELINE MODEL
print("\n" + "="*80)
print("STEP 5: TRAIN BASELINE MODEL")
print("="*80)

# Baseline model parameters
baseline_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist'
}

# Train baseline model
print("\nTraining baseline XGBoost model...")
baseline_model = xgb.XGBClassifier(**baseline_params)
baseline_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Baseline predictions
y_pred_baseline = baseline_model.predict(X_test)
y_pred_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

# Baseline evaluation
baseline_train_acc = accuracy_score(y_train, baseline_model.predict(X_train))
baseline_test_acc = accuracy_score(y_test, y_pred_baseline)
baseline_train_auc = roc_auc_score(y_train, baseline_model.predict_proba(X_train)[:, 1])
baseline_test_auc = roc_auc_score(y_test, y_pred_proba_baseline)

print(f"\n✓ Baseline Model Performance:")
print(f"  Training Accuracy: {baseline_train_acc:.4f}")
print(f"  Test Accuracy:     {baseline_test_acc:.4f}")
print(f"  Training ROC-AUC:  {baseline_train_auc:.4f}")
print(f"  Test ROC-AUC:      {baseline_test_auc:.4f}")
## HYPERPARAMETER TUNING
print("\n" + "="*80)
print("STEP 6: HYPERPARAMETER TUNING")
print("="*80)

# Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

# Create base model for GridSearch
base_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    tree_method='hist'
)

# Perform GridSearchCV
total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\nPerforming Grid Search...")
print(f"Testing {total_combinations} parameter combinations (this may take a while)...")

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters
print(f"\n✓ Best Parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Train final model with best parameters
print("\nTraining final model with best parameters...")
best_model = grid_search.best_estimator_

# Final predictions
y_pred_final = best_model.predict(X_test)
y_pred_proba_final = best_model.predict_proba(X_test)[:, 1]
## Model Evaluation 
print("\n" + "="*80)
print("STEP 7: MODEL EVALUATION")
print("="*80)

# Final model metrics
final_train_acc = accuracy_score(y_train, best_model.predict(X_train))
final_test_acc = accuracy_score(y_test, y_pred_final)
final_train_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
final_test_auc = roc_auc_score(y_test, y_pred_proba_final)

# Comparison table
comparison_df = pd.DataFrame({
    'Metric': ['Train Accuracy', 'Test Accuracy', 'Train ROC-AUC', 'Test ROC-AUC'],
    'Baseline': [baseline_train_acc, baseline_test_acc, baseline_train_auc, baseline_test_auc],
    'Tuned': [final_train_acc, final_test_acc, final_train_auc, final_test_auc]
})
comparison_df['Improvement'] = comparison_df['Tuned'] - comparison_df['Baseline']

print("\nPerformance Comparison:")
print(comparison_df.to_string(index=False))

# Detailed classification report
print("\n" + "-"*80)
print("Classification Report - Tuned Model:")
print("-"*80)
print(classification_report(y_test, y_pred_final, target_names=['Active', 'Inactive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives (Active correctly predicted):  {cm[0,0]:,}")
print(f"False Positives (Active predicted as Inactive): {cm[0,1]:,}")
print(f"False Negatives (Inactive predicted as Active): {cm[1,0]:,}")
print(f"True Positives (Inactive correctly predicted):  {cm[1,1]:,}")
## Feature Importance 
print("\n" + "="*80)
print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Feature importance DataFrame
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 25 most important features:")
print(feature_importance.head(25).to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(12, 10))
top_features = feature_importance.head(25)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=9)
plt.xlabel('Importance Score', fontsize=11)
plt.title('Top 25 Feature Importance - XGBoost Tuned Model', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_tuned.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature Importance plot saved: feature_importance_tuned.png")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Active (0)', 'Inactive (1)'],
            yticklabels=['Active (0)', 'Inactive (1)'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Tuned Model', fontsize=13, fontweight='bold')
plt.ylabel('True Label', fontsize=11)
plt.xlabel('Predicted Label', fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix_tuned.png', dpi=300, bbox_inches='tight')
print("✓ Confusion Matrix plot saved: confusion_matrix_tuned.png")
## Summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"""
Dataset Information:
  - Total Users: {len(df):,}
  - Inactive Users: {(y == 1).sum():,} ({y.mean():.2%})
  - Active Users: {(y == 0).sum():,} ({(1-y.mean()):.2%})
  
Feature Engineering:
  - Numeric Features: {len(numeric_features)}
  - Province Dummies: {len(province_dummies.columns)}
  - Gender Dummies: {len(gender_dummies.columns)}
  - Total Features: {len(feature_cols)}
  
Baseline Model Performance:
  - Test Accuracy: {baseline_test_acc:.4f}
  - Test ROC-AUC:  {baseline_test_auc:.4f}
  
Tuned Model Performance:
  - Test Accuracy: {final_test_acc:.4f} ({'+' if final_test_acc > baseline_test_acc else ''}{(final_test_acc - baseline_test_acc):.4f})
  - Test ROC-AUC:  {final_test_auc:.4f} ({'+' if final_test_auc > baseline_test_auc else ''}{(final_test_auc - baseline_test_auc):.4f})
  - Generalization: {'⚠️ Potential overfitting' if (final_train_acc - final_test_acc) > 0.05 else '✓ Good generalization'}
  
Top 5 Most Important Features:
""")

for i, row in enumerate(feature_importance.head(5).itertuples(), 1):
    print(f"  {i}. {row.feature}: {row.importance:.4f}")

print("\n" + "="*80)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nGenerated files:")
print("  - xgboost_tuned_model.pkl")
print("  - feature_names.pkl")
print("  - feature_importance_tuned.png")
print("  - confusion_matrix_tuned.png")
