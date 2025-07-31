import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings('ignore')

def label_user(row, avg_margin):
    # Example rule-based labeling logic for training
    if row['days_since_last_order'] > 180:
        return 2  # ₹100 coupon
    elif row['sentiment_score'] is not None and row['sentiment_score'] >= 0.9 or avg_margin >= 0.25:
        return 1  # ₹50 coupon
    else:
        return 0  # No coupon

def train_coupon_model():
    # Load train and test data
    with open('train_Userlog.json', 'r') as f:
        train_data = json.load(f)
    with open('test_Userlog.json', 'r') as f:
        test_data = json.load(f)
    prices_df = pd.read_csv('final_scrap_prices.csv')

    train_df = pd.json_normalize(train_data['user_logs'])
    test_df = pd.json_normalize(test_data['user_logs'])

    # --- Feature Engineering ---
    category_to_item_map = {
        'Paper': 'Newspaper',
        'Cardboard': 'Cardboard',
        'Plastic': 'PET Bottles/Other Plastic',
        'Metals': 'Iron',
        'Electronics': 'Metal E-waste',
        'Vehicles': 'Iron'
    }
    prices_df['Name of Item'] = prices_df['Name of Item'].str.strip()
    item_profit_margin = prices_df.set_index('Name of Item')['Profit Margin'].to_dict()

    def calculate_avg_profit_margin(categories):
        if not isinstance(categories, list) or not categories:
            return 0
        margins = [item_profit_margin.get(category_to_item_map.get(cat)) for cat in categories if category_to_item_map.get(cat) in item_profit_margin]
        return np.mean(margins) if margins else 0

    train_df['avg_margin'] = train_df['preferred_categories'].apply(calculate_avg_profit_margin)
    test_df['avg_margin'] = test_df['preferred_categories'].apply(calculate_avg_profit_margin)

    # Market Competitiveness
    competitor_cols = [col for col in prices_df.columns if 'Competitor' in col]
    prices_df['avg_competitor_price'] = prices_df[competitor_cols].mean(axis=1, skipna=True)
    prices_df['competitiveness_score'] = np.where(prices_df['avg_competitor_price'] > 0, prices_df['Rate'] / prices_df['avg_competitor_price'], 1)
    item_competitiveness = prices_df.set_index('Name of Item')['competitiveness_score'].to_dict()

    def calculate_avg_competitiveness(categories):
        if not isinstance(categories, list) or not categories:
            return 1
        scores = [item_competitiveness.get(category_to_item_map.get(cat)) for cat in categories if category_to_item_map.get(cat) in item_competitiveness]
        return np.mean(scores) if scores else 1

    train_df['avg_competitiveness'] = train_df['preferred_categories'].apply(calculate_avg_competitiveness)
    test_df['avg_competitiveness'] = test_df['preferred_categories'].apply(calculate_avg_competitiveness)

    # --- Generate and Visualize Feature Correlation Matrix ---
    print("\n🔥 Feature Correlation Matrix:")

    # Select only numerical features for the correlation matrix
    numerical_features = train_df.select_dtypes(include=np.number).columns
    correlation_matrix = train_df[numerical_features].corr()

    # Plotting the heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Feature Correlation Matrix')
    plt.show()

    # Encode user_type
    for df in [train_df, test_df]:
        df['user_type_encoded'] = df['user_type'].astype('category').cat.codes

    # Fill missing sentiment_score and app_rating
    for df in [train_df, test_df]:
        df['sentiment_score'] = df['sentiment_score'].fillna(df['sentiment_score'].mean())
        df['app_rating'] = df['app_rating'].fillna(df['app_rating'].mean())

    # Date Features
    for df in [train_df, test_df]:
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        today = pd.to_datetime('2025-07-30')
        df['account_age_days'] = (today - df['registration_date']).dt.days
        df['last_order_date'] = pd.to_datetime(df['last_order_date'])
        df['days_since_last_order'].fillna((today - df['registration_date']).dt.days, inplace=True)

    # --- Labeling for Training ---
    train_df['label'] = train_df.apply(lambda row: label_user(row, row['avg_margin']), axis=1)

    # --- Data Preparation ---
    features_to_drop = [
        'user_id', 'name', 'phone', 'email', 'registration_date',
        'last_login', 'last_order_date', 'preferred_categories',
        'location.city', 'location.area', 'location.pincode',
        'churn_risk', 'coupon_tier', 'assigned_coupon'
    ]
    X_train = train_df.drop(columns=features_to_drop + ['label'], errors='ignore')
    y_train = train_df['label']

    X_test = test_df.drop(columns=features_to_drop, errors='ignore')

    categorical_features = X_train.select_dtypes(include=['object', 'bool']).columns
    numerical_features = X_train.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # --- Visualize Explained Variance by PCA Components ---
    print("\n📊 PCA Analysis:")
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)

    pca_test = PCA().fit(X_train_processed)

    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by PCA Components')
    plt.grid(True)
    plt.show()

    # Determine optimal number of components (where curve starts to flatten)
    cumulative_variance = np.cumsum(pca_test.explained_variance_ratio_)
    optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Optimal number of PCA components: {optimal_components} (95% variance explained)")

    # --- Model Training with Multiple Algorithms ---
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        xgb_available = False

    # Define models with and without PCA
    models_without_pca = [
        ('RandomForest', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)),
        ('LogisticRegression', LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0))
    ]
    
    models_with_pca = [
        ('RandomForest_PCA', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)),
        ('LogisticRegression_PCA', LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0))
    ]
    
    if xgb_available:
        models_without_pca.append(('XGBoost', XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')))
        models_with_pca.append(('XGBoost_PCA', XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')))

    # Map test assigned_coupon/coupon_tier to numeric labels for comparison
    coupon_map = {
        'SCRAP100': 2, 'BIGSAVE200': 2, 'HVU175': 2, 'COMEBACK150': 2,
        'SCRAP50': 1, 'SAVE75': 1, 'RECYCLE50': 1,
        'THANKYOU': 0, 'FUTUREOFFER': 0, 'WELCOME10': 0
    }
    
    if 'coupon_tier' in test_df.columns:
        tier_map = {'High': 2, 'Mid': 1, 'Low': 0}
        y_true = test_df['coupon_tier'].map(tier_map).fillna(0).astype(int)
    elif 'assigned_coupon' in test_df.columns:
        y_true = test_df['assigned_coupon'].map(coupon_map).fillna(0).astype(int)
    else:
        y_true = np.zeros(len(test_df), dtype=int)

    best_acc = 0
    best_model = None
    best_name = ''
    best_pred = None

    # Test models without PCA
    print("\n🔍 Testing Models Without PCA:")
    for name, clf in models_without_pca:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_true, y_pred)
        print(f"{name} Accuracy: {acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            best_model = pipe
            best_name = name
            best_pred = y_pred

    # Test models with PCA
    print(f"\n🔍 Testing Models With PCA ({optimal_components} components):")
    for name, clf in models_with_pca:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=optimal_components)),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_true, y_pred)
        print(f"{name} Accuracy: {acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            best_model = pipe
            best_name = name
            best_pred = y_pred

    # Save the best model
    joblib.dump(best_model, 'coupon_classifier_model.joblib')
    print(f"\n💾 Best model saved: {best_name} (Accuracy: {best_acc:.2%})")

    # --- Final Evaluation ---
    print(f"\n🏆 Final Results:")
    print(f"Best Model: {best_name}")
    print(f"Accuracy: {best_acc:.2%}")
    
    print("\nClassification Report (Performance per Tier):")
    print(classification_report(y_true, best_pred, labels=[2,1,0], target_names=['High','Mid','Low']))

    print("\n📋 Example Predictions vs. Actual Values:")
    result_df = test_df.copy()
    result_df['Predicted_Tier'] = best_pred
    result_df['Actual_Tier'] = y_true
    print(result_df[['user_id', 'Actual_Tier', 'assigned_coupon', 'Predicted_Tier']].head(10))

    # Confusion Matrix
    cm = confusion_matrix(y_true, best_pred, labels=[2,1,0])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['High', 'Mid', 'Low'], 
                yticklabels=['High', 'Mid', 'Low'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == '__main__':
    train_coupon_model()