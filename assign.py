import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
    # For test set, use assigned_coupon or coupon_tier for evaluation only, not for training

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
    # For test, use assigned_coupon/coupon_tier for evaluation

    categorical_features = X_train.select_dtypes(include=['object', 'bool']).columns
    numerical_features = X_train.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )


    # Try multiple models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        xgb_available = False

    models = [
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ('LogisticRegression', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ]
    if xgb_available:
        models.append(('XGBoost', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')))


    # Map test assigned_coupon/coupon_tier to numeric labels for comparison
    # Example mapping: Class 2 (High/₹100), Class 1 (Mid/₹50), Class 0 (Low/No coupon)
    coupon_map = {
        'SCRAP100': 2, 'BIGSAVE200': 2, 'HVU175': 2, 'COMEBACK150': 2,
        'SCRAP50': 1, 'SAVE75': 1, 'RECYCLE50': 1,
        'THANKYOU': 0, 'FUTUREOFFER': 0, 'WELCOME10': 0
    }
    # If coupon_tier is present, use that; else use assigned_coupon
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
    for name, clf in models:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_true, y_pred)
        print(f"\n{name} Accuracy: {acc:.2%}")
        print(classification_report(y_true, y_pred, labels=[2,1,0], target_names=['High','Mid','Low']))
        if acc > best_acc:
            best_acc = acc
            best_model = pipe
            best_name = name
            best_pred = y_pred

    joblib.dump(best_model, 'coupon_classifier_model.joblib')
    print(f"\n💾 Best model saved: {best_name} (Accuracy: {best_acc:.2%})")

    # --- Prediction & Evaluation ---
    y_pred = best_pred

    print("\nOverall Model Accuracy: {:.2%}".format(accuracy_score(y_true, y_pred)))
    print("\nClassification Report (Performance per Tier):")
    print(classification_report(y_true, y_pred, labels=[2,1,0], target_names=['High','Mid','Low']))

    print("\n📋 Example Predictions vs. Actual Values:")
    result_df = test_df.copy()
    result_df['Predicted_Tier'] = y_pred
    result_df['Actual_Tier'] = y_true
    print(result_df[['user_id', 'Actual_Tier', 'assigned_coupon', 'Predicted_Tier']].head(10))

if __name__ == '__main__':
    train_coupon_model()