# ScrapUncle Coupon Classification System

This project predicts which coupon tier (High, Mid, Low) should be offered to users of ScrapUncle based on their activity, preferences, and market data.

## Files
- `assign.py`: Main training and evaluation script. Trains a machine learning model to classify users into coupon tiers using user logs and scrap price data.
- `train_Userlog.json` / `test_Userlog.json`: Artificial user log datasets for training and testing.
- `final_scrap_prices.csv`: Scrap item prices, profit margins, and competitor prices.
- `coupon_classifier_model.joblib`: Saved best model after training.

## How It Works
1. **Feature Engineering**
   - Extracts user features (activity, sentiment, app rating, etc.)
   - Calculates average profit margin and market competitiveness for each user's preferred categories.
   - Encodes categorical features and fills missing values.
2. **Labeling**
   - Assigns training labels using a rule-based function (e.g., high churn or long inactivity → High coupon).
3. **Model Training**
   - Tries multiple models: RandomForest, LogisticRegression, and XGBoost (if available).
   - Selects and saves the best-performing model based on test accuracy.
4. **Evaluation**
   - Prints accuracy and classification report for each model.
   - Shows example predictions vs. actual coupon tiers.

## Usage
1. Place all data files in the same directory.
2. Run the script:
   ```
   python assign.py
   ```
3. The best model will be saved as `coupon_classifier_model.joblib`.

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, joblib
- (Optional) xgboost for best accuracy

## Customization
- You can tune the labeling logic, add more features, or adjust model parameters in `assign.py` for better results.

## Output
- Model accuracy and classification report for coupon tier prediction.
- Example predictions for test users.

---
For questions or improvements, contact the project maintainer.
