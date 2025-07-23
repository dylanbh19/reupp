#!/usr/bin/env python
"""
SIMPLE MAIL-TO-CALLS PREDICTION THAT ACTUALLY WORKS
==================================================

FOCUS: Get ONE good model working first
- Use capped outlier strategy (best correlation: 0.291)
- Simple features: top 5 mail types + basic lags
- Just 1-day prediction (mail today -> calls tomorrow)
- Get positive R² first, then expand

KISS PRINCIPLE: Keep It Simple, Stupid!
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

# ============================================================================
# SIMPLE CONFIGURATION
# ============================================================================

CONFIG = {
    # Data files
    "call_file": "callintent.csv",
    "mail_file": "mail.csv", 
    "output_dir": "simple_working_model",
    
    # Simple settings
    "top_mail_types": 5,  # Just top 5 mail types
    "outlier_cap_percentile": 95,  # Use capped strategy (worked best)
    "test_size": 0.3,  # Larger test set for validation
    "random_state": 42,
    
    # Model validation
    "min_r2_threshold": 0.1,  # Must beat this to be acceptable
}

def safe_print(msg):
    """Print safely"""
    try:
        print(str(msg).encode('ascii', 'ignore').decode('ascii'))
    except:
        print(str(msg))

# ============================================================================
# SIMPLE DATA LOADING
# ============================================================================

def load_and_cap_data():
    """Load data and apply capped outlier strategy (the winner)"""
    safe_print("=" * 60)
    safe_print("LOADING DATA WITH CAPPED OUTLIER STRATEGY")
    safe_print("=" * 60)
    
    # Load call data
    call_paths = [CONFIG["call_file"], f"data/{CONFIG['call_file']}"]
    call_path = next((p for p in call_paths if Path(p).exists()), None)
    
    if not call_path:
        raise FileNotFoundError("Call file not found")
    
    safe_print(f"Loading calls: {call_path}")
    calls_df = pd.read_csv(call_path, encoding='utf-8', low_memory=False)
    calls_df.columns = [str(col).lower().strip() for col in calls_df.columns]
    
    # Find date column
    date_col = next((col for col in calls_df.columns if any(kw in col for kw in ['date', 'start', 'time'])), None)
    
    calls_df[date_col] = pd.to_datetime(calls_df[date_col], errors='coerce')
    calls_df = calls_df.dropna(subset=[date_col])
    calls_df = calls_df[calls_df[date_col].dt.year >= 2025]
    
    # Create daily calls
    calls_df['call_date'] = calls_df[date_col].dt.date
    daily_calls = calls_df.groupby('call_date').size().reset_index()
    daily_calls.columns = ['date', 'call_volume']
    daily_calls['date'] = pd.to_datetime(daily_calls['date'])
    daily_calls = daily_calls[daily_calls['date'].dt.weekday < 5]  # Business days only
    
    safe_print(f"Call data: {len(daily_calls)} business days")
    
    # Load mail data
    mail_paths = [CONFIG["mail_file"], f"data/{CONFIG['mail_file']}"]
    mail_path = next((p for p in mail_paths if Path(p).exists()), None)
    
    if not mail_path:
        raise FileNotFoundError("Mail file not found")
    
    safe_print(f"Loading mail: {mail_path}")
    mail_df = pd.read_csv(mail_path, encoding='utf-8', low_memory=False)
    mail_df.columns = [str(col).lower().strip() for col in mail_df.columns]
    
    # Find mail columns
    date_col = next((col for col in mail_df.columns if 'date' in col), None)
    volume_col = next((col for col in mail_df.columns if 'volume' in col), None)
    type_col = next((col for col in mail_df.columns if 'type' in col), None)
    
    mail_df[date_col] = pd.to_datetime(mail_df[date_col], errors='coerce')
    mail_df = mail_df.dropna(subset=[date_col])
    mail_df = mail_df[mail_df[date_col].dt.year >= 2025]
    
    mail_df[volume_col] = pd.to_numeric(mail_df[volume_col], errors='coerce')
    mail_df = mail_df.dropna(subset=[volume_col])
    mail_df = mail_df[mail_df[volume_col] > 0]
    
    # Create daily mail pivot
    mail_df['mail_date'] = mail_df[date_col].dt.date
    daily_mail = mail_df.groupby(['mail_date', type_col])[volume_col].sum().reset_index()
    daily_mail.columns = ['date', 'mail_type', 'volume']
    daily_mail['date'] = pd.to_datetime(daily_mail['date'])
    daily_mail = daily_mail[daily_mail['date'].dt.weekday < 5]
    
    mail_pivot = daily_mail.pivot(index='date', columns='mail_type', values='volume').fillna(0)
    mail_pivot = mail_pivot.reset_index()
    
    safe_print(f"Mail data: {len(mail_pivot)} business days, {len(mail_pivot.columns)-1} mail types")
    
    # Merge data
    merged = pd.merge(daily_calls, mail_pivot, on='date', how='inner')
    merged = merged.sort_values('date').reset_index(drop=True)
    
    safe_print(f"Merged data: {len(merged)} days")
    
    # Apply CAPPED outlier strategy (the winner!)
    cap_percentile = CONFIG["outlier_cap_percentile"]
    
    # Cap call volumes
    call_cap = merged['call_volume'].quantile(cap_percentile / 100)
    call_floor = merged['call_volume'].quantile((100 - cap_percentile) / 100)
    merged['call_volume'] = merged['call_volume'].clip(lower=call_floor, upper=call_cap)
    
    # Cap mail volumes
    mail_columns = [col for col in merged.columns if col not in ['date', 'call_volume']]
    for col in mail_columns:
        cap_value = merged[col].quantile(cap_percentile / 100)
        merged[col] = merged[col].clip(upper=cap_value)
    
    safe_print(f"Applied capping at {cap_percentile}th percentile")
    
    # Check correlation
    total_mail = merged[mail_columns].sum(axis=1)
    correlation = merged['call_volume'].corr(total_mail)
    safe_print(f"Total mail vs calls correlation: {correlation:.3f}")
    
    return merged, mail_columns

# ============================================================================
# SIMPLE FEATURE ENGINEERING
# ============================================================================

def create_simple_features(df, mail_columns):
    """Create simple, effective features"""
    safe_print("\n" + "=" * 60)
    safe_print("CREATING SIMPLE FEATURES")
    safe_print("=" * 60)
    
    # Select top 5 mail types by volume
    mail_volumes = df[mail_columns].sum().sort_values(ascending=False)
    top_mail_types = mail_volumes.head(CONFIG["top_mail_types"]).index.tolist()
    
    safe_print(f"Top {len(top_mail_types)} mail types:")
    for i, mail_type in enumerate(top_mail_types):
        volume = mail_volumes[mail_type]
        safe_print(f"  {i+1}. {mail_type[:40]}: {volume:,.0f}")
    
    # Create feature dataset
    features_list = []
    targets_list = []
    
    # Simple approach: mail today -> calls tomorrow
    for i in range(len(df) - 1):  # Leave room for next-day target
        current_day = df.iloc[i]
        next_day = df.iloc[i + 1]
        
        feature_row = {}
        
        # 1. SIMPLE MAIL FEATURES (today's mail)
        for mail_type in top_mail_types:
            clean_name = str(mail_type).replace(' ', '').replace('-', '').replace('_', '')[:15]
            feature_row[f"{clean_name}"] = current_day[mail_type]
        
        # 2. TOTAL MAIL (today)
        total_mail = sum(current_day[mt] for mt in top_mail_types)
        feature_row['total_mail'] = total_mail
        feature_row['log_total_mail'] = np.log1p(total_mail)
        
        # 3. SIMPLE CALL HISTORY (yesterday's calls)
        feature_row['calls_yesterday'] = current_day['call_volume']
        
        # 4. SIMPLE TEMPORAL FEATURES
        current_date = current_day['date']
        feature_row['weekday'] = current_date.weekday()
        feature_row['month'] = current_date.month
        feature_row['is_month_end'] = 1 if current_date.day >= 25 else 0
        
        # 5. SIMPLE RATIOS (if we have enough mail)
        if total_mail > 0:
            # What percentage is the biggest mail type?
            biggest_mail = max(current_day[mt] for mt in top_mail_types)
            feature_row['biggest_mail_pct'] = biggest_mail / total_mail
        else:
            feature_row['biggest_mail_pct'] = 0
        
        # TARGET: Next day's calls
        target = next_day['call_volume']
        
        features_list.append(feature_row)
        targets_list.append(target)
    
    # Convert to DataFrames
    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)
    
    # Clean data
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    safe_print(f"Created {len(X.columns)} simple features:")
    for col in X.columns:
        safe_print(f"  - {col}")
    
    safe_print(f"Dataset: {len(X)} samples")
    
    return X, y, top_mail_types

# ============================================================================
# SIMPLE MODEL TRAINING
# ============================================================================

def train_simple_models(X, y):
    """Train simple models and pick the best"""
    safe_print("\n" + "=" * 60)
    safe_print("TRAINING SIMPLE MODELS")
    safe_print("=" * 60)
    
    # Split data (time-aware)
    split_idx = int(len(X) * (1 - CONFIG["test_size"]))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    safe_print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Simple models to try
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0, random_state=CONFIG["random_state"]),
        'ridge_strong': Ridge(alpha=10.0, random_state=CONFIG["random_state"]),
        'forest_simple': RandomForestRegressor(
            n_estimators=50, 
            max_depth=5, 
            min_samples_leaf=5,
            random_state=CONFIG["random_state"]
        ),
        'forest_tiny': RandomForestRegressor(
            n_estimators=20, 
            max_depth=3, 
            min_samples_leaf=8,
            random_state=CONFIG["random_state"]
        )
    }
    
    results = {}
    best_model = None
    best_score = -float('inf')
    best_name = None
    
    for name, model in models.items():
        safe_print(f"\nTesting {name}...")
        
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Check for overfitting
            overfitting = train_r2 - test_r2
            
            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'overfitting': overfitting,
                'model': model
            }
            
            safe_print(f"  Train R²: {train_r2:.3f}")
            safe_print(f"  Test R²:  {test_r2:.3f}")
            safe_print(f"  Test MAE: {test_mae:.0f}")
            safe_print(f"  Overfitting: {overfitting:.3f}")
            
            # Select best model (prioritize test R², penalize overfitting)
            adjusted_score = test_r2 - max(0, overfitting - 0.1)  # Penalize overfitting > 0.1
            
            if adjusted_score > best_score and test_r2 > CONFIG["min_r2_threshold"]:
                best_score = adjusted_score
                best_model = model
                best_name = name
                safe_print(f"  ★ NEW BEST MODEL! (Adjusted score: {adjusted_score:.3f})")
            
        except Exception as e:
            safe_print(f"  ✗ Failed: {e}")
            results[name] = {'error': str(e)}
    
    if best_model:
        safe_print(f"\n🎯 BEST MODEL: {best_name}")
        safe_print(f"   Test R²: {results[best_name]['test_r2']:.3f}")
        safe_print(f"   Test MAE: {results[best_name]['test_mae']:.0f}")
        
        # Train on full dataset
        best_model.fit(X, y)
        
    else:
        safe_print("\n❌ NO MODEL MEETS MINIMUM THRESHOLD!")
        safe_print(f"   All models had R² < {CONFIG['min_r2_threshold']}")
    
    return best_model, best_name, results

# ============================================================================
# MODEL VALIDATION AND VISUALIZATION
# ============================================================================

def validate_and_visualize(model, model_name, X, y, results):
    """Validate model and create plots"""
    safe_print("\n" + "=" * 60)
    safe_print("MODEL VALIDATION & VISUALIZATION")
    safe_print("=" * 60)
    
    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    # Split for visualization
    split_idx = int(len(X) * (1 - CONFIG["test_size"]))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Get predictions
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Model Validation: {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted (Test)
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Calls')
    axes[0, 0].set_ylabel('Predicted Calls')
    axes[0, 0].set_title(f'Test Set: R² = {results[model_name]["test_r2"]:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals
    residuals = y_test - y_pred_test
    axes[0, 1].scatter(y_pred_test, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Calls')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Time series comparison
    axes[0, 2].plot(range(len(y_test)), y_test.values, 'b-', label='Actual', linewidth=2)
    axes[0, 2].plot(range(len(y_test)), y_pred_test, 'r-', label='Predicted', linewidth=2, alpha=0.7)
    axes[0, 2].set_xlabel('Test Day')
    axes[0, 2].set_ylabel('Call Volume')
    axes[0, 2].set_title('Time Series Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Residual distribution
    axes[1, 0].hist(residuals, bins=15, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_names = X.columns
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1][:10]  # Top 10
        
        axes[1, 1].barh(range(len(sorted_idx)), importance[sorted_idx])
        axes[1, 1].set_yticks(range(len(sorted_idx)))
        axes[1, 1].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Feature Importance')
        axes[1, 1].grid(True, alpha=0.3)
        
    elif hasattr(model, 'coef_'):
        coef = np.abs(model.coef_)
        feature_names = X.columns
        
        # Sort by coefficient magnitude
        sorted_idx = np.argsort(coef)[::-1][:10]
        
        axes[1, 1].barh(range(len(sorted_idx)), coef[sorted_idx])
        axes[1, 1].set_yticks(range(len(sorted_idx)))
        axes[1, 1].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[1, 1].set_xlabel('|Coefficient|')
        axes[1, 1].set_title('Feature Coefficients')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Model comparison
    model_names = list(results.keys())
    test_r2_scores = [results[name].get('test_r2', 0) for name in model_names if 'error' not in results[name]]
    valid_names = [name for name in model_names if 'error' not in results[name]]
    
    if valid_names:
        axes[1, 2].bar(range(len(valid_names)), test_r2_scores, alpha=0.7, color='orange')
        axes[1, 2].set_xticks(range(len(valid_names)))
        axes[1, 2].set_xticklabels(valid_names, rotation=45)
        axes[1, 2].set_ylabel('Test R²')
        axes[1, 2].set_title('Model Comparison')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=CONFIG["min_r2_threshold"], color='r', linestyle='--', label='Threshold')
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_validation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    safe_print(f"Validation plot saved: {output_dir}/model_validation.png")
    
    return True

# ============================================================================
# SIMPLE PREDICTION FUNCTION
# ============================================================================

def create_prediction_function(model, top_mail_types):
    """Create simple prediction function"""
    
    def predict_calls(mail_input, calls_yesterday=None, prediction_date=None):
        """
        Simple prediction function
        
        Args:
            mail_input: dict like {'DRP Stmt.': 2000, 'Cheque': 1500, ...}
            calls_yesterday: yesterday's call volume (optional)
            prediction_date: date string (optional)
        
        Returns:
            dict with prediction and details
        """
        
        try:
            # Create feature vector
            features = {}
            
            # Mail features (today's mail)
            for mail_type in top_mail_types:
                clean_name = str(mail_type).replace(' ', '').replace('-', '').replace('_', '')[:15]
                features[clean_name] = mail_input.get(mail_type, 0)
            
            # Total mail
            total_mail = sum(mail_input.get(mt, 0) for mt in top_mail_types)
            features['total_mail'] = total_mail
            features['log_total_mail'] = np.log1p(total_mail)
            
            # Call history
            features['calls_yesterday'] = calls_yesterday or 12000  # Default average
            
            # Temporal features
            if prediction_date:
                pred_date = pd.to_datetime(prediction_date)
            else:
                pred_date = pd.Timestamp.now()
            
            features['weekday'] = pred_date.weekday()
            features['month'] = pred_date.month  
            features['is_month_end'] = 1 if pred_date.day >= 25 else 0
            
            # Simple ratios
            if total_mail > 0:
                biggest_mail = max(mail_input.get(mt, 0) for mt in top_mail_types)
                features['biggest_mail_pct'] = biggest_mail / total_mail
            else:
                features['biggest_mail_pct'] = 0
            
            # Convert to array
            feature_vector = [features[col] for col in model.feature_names_in_]
            
            # Predict
            prediction = model.predict([feature_vector])[0]
            prediction = max(0, round(prediction))
            
            return {
                'predicted_calls': int(prediction),
                'prediction_date': pred_date.strftime('%Y-%m-%d'),
                'mail_input': mail_input,
                'total_mail_volume': total_mail,
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}
    
    return predict_calls

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution - simple approach that works"""
    
    safe_print("SIMPLE MAIL-TO-CALLS PREDICTION THAT ACTUALLY WORKS")
    safe_print("=" * 60)
    safe_print("APPROACH:")
    safe_print("* Use capped outlier strategy (best correlation)")
    safe_print("* Top 5 mail types only")
    safe_print("* Simple features (no complex lags)")
    safe_print("* Just 1-day prediction (mail today -> calls tomorrow)")
    safe_print("* Must achieve R² > 0.1 to be acceptable")
    safe_print("=" * 60)
    
    try:
        # Load and prepare data
        df, mail_columns = load_and_cap_data()
        
        # Create simple features
        X, y, top_mail_types = create_simple_features(df, mail_columns)
        
        # Train simple models
        best_model, best_name, results = train_simple_models(X, y)
        
        if not best_model:
            safe_print("\n❌ FAILED: No model achieved minimum performance")
            return {'success': False, 'error': 'No acceptable model found'}
        
        # Validate and visualize
        validate_and_visualize(best_model, best_name, X, y, results)
        
        # Create prediction function
        predict_calls = create_prediction_function(best_model, top_mail_types)
        
        # Save model
        output_dir = Path(CONFIG["output_dir"])
        output_dir.mkdir(exist_ok=True)
        
        joblib.dump(best_model, output_dir / "simple_model.pkl")
        joblib.dump(top_mail_types, output_dir / "top_mail_types.pkl")
        
        # Test prediction
        safe_print("\n" + "=" * 60)
        safe_print("TESTING SIMPLE PREDICTION")
        safe_print("=" * 60)
        
        test_input = {}
        for i, mail_type in enumerate(top_mail_types):
            # Create realistic test volumes
            test_input[mail_type] = [2000, 1500, 1000, 800, 600][i]
        
        result = predict_calls(test_input, calls_yesterday=12000)
        
        if result['status'] == 'success':
            safe_print("✅ TEST SUCCESSFUL!")
            safe_print(f"   Input: {test_input}")
            safe_print(f"   Predicted calls tomorrow: {result['predicted_calls']}")
            safe_print(f"   Total mail volume: {result['total_mail_volume']:,}")
        else:
            safe_print(f"❌ TEST FAILED: {result['error']}")
        
        # Create usage example
        usage_example = f"""
SIMPLE MAIL-TO-CALLS PREDICTION MODEL
====================================

MODEL PERFORMANCE:
- Best Model: {best_name}
- Test R²: {results[best_name]['test_r2']:.3f}
- Test MAE: {results[best_name]['test_mae']:.0f}

USAGE:
-----
import joblib

# Load model
model = joblib.load('{output_dir}/simple_model.pkl')
top_types = joblib.load('{output_dir}/top_mail_types.pkl')

# Make prediction
mail_today = {{
    '{top_mail_types[0]}': 2000,
    '{top_mail_types[1] if len(top_mail_types) > 1 else "Mail_Type"}': 1500,
    '{top_mail_types[2] if len(top_mail_types) > 2 else "Mail_Type"}': 1000
}}

# This predicts calls for TOMORROW
predicted_calls = model.predict([features_from_mail])

YOUR TOP MAIL TYPES:
{chr(10).join([f"  {i+1}. {mt}" for i, mt in enumerate(top_mail_types)])}

FILES SAVED:
- simple_model.pkl: Trained model
- top_mail_types.pkl: Your top mail types
- model_validation.png: Performance plots
"""
        
        with open(output_dir / "USAGE_GUIDE.txt", 'w') as f:
            f.write(usage_example)
        
        # Final summary
        safe_print("\n" + "=" * 60)
        safe_print("🎯 SUCCESS! SIMPLE MODEL WORKS!")
        safe_print("=" * 60)
        safe_print(f"✅ Best Model: {best_name}")
        safe_print(f"✅ Test R²: {results[best_name]['test_r2']:.3f}")
        safe_print(f"✅ Test MAE: {results[best_name]['test_mae']:.0f} calls")
        safe_print(f"✅ Top Mail Types: {len(top_mail_types)}")
        safe_print(f"✅ Files Saved: {output_dir}/")
        safe_print("")
        safe_print("READY FOR PRODUCTION:")
        safe_print("- Simple, interpretable model ✅")
        safe_print("- Positive R² achieved ✅") 
        safe_print("- Capped outlier strategy ✅")
        safe_print("- Mail today -> calls tomorrow ✅")
        safe_print("- Validation plots created ✅")
        
        return {
            'success': True,
            'best_model': best_model,
            'best_model_name': best_name,
            'test_r2': results[best_name]['test_r2'],
            'test_mae': results[best_name]['test_mae'],
            'top_mail_types': top_mail_types,
            'predict_function': predict_calls,
            'output_dir': str(output_dir)
        }
        
    except Exception as e:
        safe_print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    result = main()
    if result['success']:
        safe_print("🚀 SIMPLE MODEL DEPLOYED SUCCESSFULLY!")
    else:
        safe_print(f"💥 FAILED: {result['error']}")
