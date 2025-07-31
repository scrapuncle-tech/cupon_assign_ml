import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
import random
from datetime import datetime, timedelta

# --- Import for handling class imbalance & data generation ---
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    from sklearn.pipeline import Pipeline as ImbPipeline # Fallback
    IMBLEARN_AVAILABLE = False

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def generate_synthetic_data(num_users=200):
    """
    Generates a synthetic dataset of user logs with varied classes.
    """
    if not FAKER_AVAILABLE:
        print("❌ Faker library not found. Cannot generate data. Please run: pip install Faker")
        return []

    fake = Faker()
    user_logs = []
    
    # Define potential values
    coupon_tiers = {
        'High': 'SCRAP100',
        'Mid': 'SAVE75',
        'Low': 'WELCOME10',
        'None': 'No Coupon'
    }
    tier_keys = list(coupon_tiers.keys())
    
    categories = ['Paper', 'Cardboard', 'Plastic', 'Metals', 'Electronics', 'Vehicles']

    for i in range(num_users):
        reg_date = fake.date_time_between(start_date='-2y', end_date='-1M')
        last_order_date = fake.date_time_between(start_date=reg_date, end_date='now')
        
        # Create a varied distribution of coupon tiers
        if i < num_users * 0.2: # 20% High tier
            tier = 'High'
        elif i < num_users * 0.5: # 30% Mid tier
            tier = 'Mid'
        else: # 50% Low/No tier
            tier = random.choice(['Low', 'None'])

        user_logs.append({
            'user_id': f'USER_{1001 + i}',
            'name': fake.name(),
            'phone': fake.phone_number(),
            'email': fake.email(),
            'registration_date': reg_date.isoformat(),
            'last_order_date': last_order_date.isoformat(),
            'total_orders': random.randint(1, 80),
            'preferred_categories': random.sample(categories, k=random.randint(1, 4)),
            'location': {
                'city': fake.city(),
                'area': fake.street_name(),
                'pincode': fake.postcode()
            },
            'app_rating': round(random.uniform(2.5, 5.0), 1),
            'sentiment_score': round(random.uniform(0.4, 1.0), 2),
            'days_since_last_order': (datetime.now() - last_order_date).days,
            'assigned_coupon': coupon_tiers[tier],
            'user_type': random.choice(['Residential', 'Commercial'])
        })
    return user_logs


def main():
    """
    Main function to run the entire coupon prediction pipeline.
    """
    # --- Generate synthetic data if files are missing ---
    for filename, count in [('train_Userlog.json', 200), ('test_Userlog.json', 50)]:
        if not os.path.exists(filename):
            print(f"📄 '{filename}' not found. Generating new synthetic data...")
            synthetic_data = generate_synthetic_data(num_users=count)
            if synthetic_data:
                with open(filename, 'w') as f:
                    json.dump({'user_logs': synthetic_data}, f, indent=4)
                print(f"✅ New '{filename}' created successfully.")

    # 1. Load data
    train_df, test_df, prices_df = load_data()
    if train_df is None:
        return

    # 2. Engineer features for both datasets
    today = pd.to_datetime('now', utc=True).normalize()
    train_df = engineer_features(train_df, prices_df, today)
    test_df = engineer_features(test_df, prices_df, today)

    # 3. Label data and define features for the model
    y_train = map_true_labels(train_df)
    y_true = map_true_labels(test_df)

    features_to_use = [
        'total_orders', 'app_rating', 'sentiment_score', 'days_since_last_order',
        'user_type', 'avg_margin', 'avg_competitiveness', 'account_age_days'
    ]

    X_train = train_df[features_to_use]
    X_test = test_df[features_to_use]

    # 4. Train models and find the best one
    best_model, best_name, best_pred, best_acc = train_and_evaluate(X_train, y_train, X_test, y_true)

    # 5. Save the best model and show final results
    if best_model:
        joblib.dump(best_model, 'coupon_classifier_model.joblib')
        print(f"\n💾 Best model saved: '{best_name}' with accuracy: {best_acc:.2%}")
        visualize_results(y_true, best_pred, best_name, test_df)
    else:
        print("\nCould not determine the best model. No model was saved.")


def load_data():
    """
    Loads training, testing, and pricing data from local files.
    """
    print("🔄 Loading data...")
    try:
        if not os.path.exists('final_scrap_prices.csv'):
            # Create a dummy prices file if it doesn't exist
            pd.DataFrame({
                'Name of Item': ['Newspaper', 'Cardboard', 'PET Bottles/Other Plastic', 'Iron', 'Metal E-waste'],
                'Rate': [15, 8, 12, 25, 18], 'Profit Margin': [0.1, 0.15, 0.12, 0.2, 0.25],
                'Competitor 1 Price': [14, 7, 11, 26, 17], 'Competitor 2 Price': [16, 8, 13, 24, 19]
            }).to_csv('final_scrap_prices.csv', index=False)

        with open('train_Userlog.json', 'r') as f:
            train_data = json.load(f)
        with open('test_Userlog.json', 'r') as f:
            test_data = json.load(f)

        train_key = 'userLogs' if 'userLogs' in train_data else 'user_logs'
        test_key = 'userLogs' if 'userLogs' in test_data else 'user_logs'

        train_df = pd.json_normalize(train_data[train_key])
        test_df = pd.json_normalize(test_data[test_key])
        prices_df = pd.read_csv('final_scrap_prices.csv')
        print("✅ Data loaded successfully.")
        return train_df, test_df, prices_df

    except (FileNotFoundError, KeyError) as e:
        print(f"❌ Error during data loading: {e}.")
        return None, None, None


def engineer_features(df, prices_df, today):
    """
    Creates new features for the model from the raw data.
    """
    category_to_item_map = {
        'Paper': 'Newspaper', 'Cardboard': 'Cardboard', 'Plastic': 'PET Bottles/Other Plastic',
        'Metals': 'Iron', 'Electronics': 'Metal E-waste', 'Vehicles': 'Iron'
    }
    prices_df['Name of Item'] = prices_df['Name of Item'].str.strip()
    item_profit_margin = prices_df.set_index('Name of Item')['Profit Margin'].to_dict()

    competitor_cols = [col for col in prices_df.columns if 'Competitor' in col]
    prices_df['avg_competitor_price'] = prices_df[competitor_cols].mean(axis=1, skipna=True)
    prices_df['competitiveness_score'] = np.where(prices_df['avg_competitor_price'] > 0, prices_df['Rate'] / prices_df['avg_competitor_price'], 1)
    item_competitiveness = prices_df.set_index('Name of Item')['competitiveness_score'].to_dict()

    def calculate_avg_metric(categories, metric_dict, default_val):
        if not isinstance(categories, list) or not categories: return default_val
        metrics = [metric_dict.get(category_to_item_map.get(cat)) for cat in categories if category_to_item_map.get(cat) in metric_dict]
        return np.mean(metrics) if metrics else default_val

    df['avg_margin'] = df['preferred_categories'].apply(lambda cats: calculate_avg_metric(cats, item_profit_margin, 0))
    df['avg_competitiveness'] = df['preferred_categories'].apply(lambda cats: calculate_avg_metric(cats, item_competitiveness, 1))

    df['sentiment_score'].fillna(df['sentiment_score'].mean(), inplace=True)
    df['app_rating'].fillna(df['app_rating'].mean(), inplace=True)

    df['registration_date'] = pd.to_datetime(df['registration_date'], format='ISO8601', utc=True)
    df['account_age_days'] = (today - df['registration_date']).dt.days
    df['days_since_last_order'].fillna((today - df['registration_date']).dt.days, inplace=True)

    return df


def map_true_labels(df):
    """
    Maps coupon names or tiers from the dataframe to numeric labels.
    """
    coupon_map = {
        'SCRAP100': 2, 'BIGSAVE200': 2, 'HVU175': 2, 'COMEBACK150': 2, # High
        'SCRAP50': 1, 'SAVE75': 1, 'RECYCLE50': 1, # Mid
        'THANKYOU': 0, 'FUTUREOFFER': 0, 'WELCOME10': 0, 'No Coupon': 0 # Low
    }
    if 'assigned_coupon' in df.columns:
        return df['assigned_coupon'].map(coupon_map).fillna(0).astype(int)
    return np.zeros(len(df), dtype=int)


def train_and_evaluate(X_train, y_train, X_test, y_true):
    """
    Defines, trains, and evaluates multiple models to find the best one.
    """
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced')
    }
    
    best_acc, best_model, best_name, best_pred = -1, None, "", None

    print("\n🔍 Model Training and Evaluation...")
    if not IMBLEARN_AVAILABLE:
        print("Warning: 'imbalanced-learn' not found. Skipping SMOTE. To install: pip install imbalanced-learn")

    for name, model in models.items():
        pipeline_steps = [('preprocessor', preprocessor)]
        if IMBLEARN_AVAILABLE:
             pipeline_steps.append(('smote', SMOTE(random_state=42)))
        pipeline_steps.append(('classifier', model))

        full_pipe = ImbPipeline(pipeline_steps)
        full_pipe.fit(X_train, y_train)
        y_pred = full_pipe.predict(X_test)
        acc = accuracy_score(y_true, y_pred)
        print(f"{name} Accuracy: {acc:.2%}")

        if acc > best_acc:
            best_acc, best_model, best_name, best_pred = acc, full_pipe, name, y_pred

    return best_model, best_name, best_pred, best_acc


def visualize_results(y_true, y_pred, model_name, test_df):
    """
    Generates and displays the final evaluation metrics and plots.
    """
    print(f"\n{'-'*20}\n🏆 Final Evaluation for {model_name}\n{'-'*20}")
    print("\nClassification Report (Performance per Tier):")
    print(classification_report(y_true, y_pred, labels=[2, 1, 0], target_names=['High', 'Mid', 'Low'], zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[2, 1, 0])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['High', 'Mid', 'Low'], yticklabels=['High', 'Mid', 'Low'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    print("\n📋 Example Predictions vs. Actual Values:")
    result_df = test_df[['user_id', 'assigned_coupon']].copy()
    result_df['Actual_Tier'] = y_true
    result_df['Predicted_Tier'] = y_pred
    tier_map_rev = {2: 'High', 1: 'Mid', 0: 'Low'}
    result_df['Predicted_Tier_Name'] = result_df['Predicted_Tier'].map(tier_map_rev)
    print(result_df[['user_id', 'assigned_coupon', 'Actual_Tier', 'Predicted_Tier_Name']].head(10))


if __name__ == '__main__':
    main()