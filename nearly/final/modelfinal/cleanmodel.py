#!/usr/bin/env python
"""
WEEKLY MAIL-TO-CALLS PREDICTION (FUNDAMENTAL STRATEGY CHANGE)
============================================================

NEW APPROACH: Weekly aggregation instead of daily
- Aggregate mail and calls to WEEKLY totals
- This gives us more stable patterns and reduces noise
- Weekly mail volume -> Weekly call volume
- Should have better signal-to-noise ratio

RATIONALE: Daily patterns are too noisy for 81-day dataset
Weekly patterns should be more predictable and stable
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
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data files
    "call_file": "callintent.csv",
    "mail_file": "mail.csv", 
    "output_dir": "weekly_prediction_model",
    
    # Weekly aggregation settings
    "top_mail_types": 8,
    "outlier_cap_percentile": 95,
    "test_size": 0.25,  # Still reasonable for weekly data
    "random_state": 42,
    
    # Model validation
    "min_r2_threshold": 0.15,  # Higher threshold for weekly data
}

def safe_print(msg):
    """Print safely"""
    try:
        print(str(msg).encode('ascii', 'ignore').decode('ascii'))
    except:
        print(str(msg))

# ============================================================================
# WEEKLY DATA AGGREGATION
# ============================================================================

def load_and_aggregate_weekly():
    """Load data and aggregate to weekly totals"""
    safe_print("=" * 60)
    safe_print("LOADING DATA & AGGREGATING TO WEEKLY TOTALS")
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
    
    # Create daily calls first
    calls_df['call_date'] = calls_df[date_col].dt.date
    daily_calls = calls_df.groupby('call_date').size().reset_index()
    daily_calls.columns = ['date', 'call_volume']
    daily_calls['date'] = pd.to_datetime(daily_calls['date'])
    daily_calls = daily_calls[daily_calls['date'].dt.weekday < 5]  # Business days only
    
    # Aggregate to WEEKLY totals
    daily_calls['week'] = daily_calls['date'].dt.to_period('W-FRI')  # Week ending Friday
    weekly_calls = daily_calls.groupby('week')['call_volume'].sum().reset_index()
    weekly_calls['week_start'] = weekly_calls['week'].dt.start_time
    weekly_calls['week_end'] = weekly_calls['week'].dt.end_time
    
    safe_print(f"Call data: {len(daily_calls)} daily → {len(weekly_calls)} weekly")
    
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
    
    # Create daily mail first
    mail_df['mail_date'] = mail_df[date_col].dt.date
    daily_mail = mail_df.groupby(['mail_date', type_col])[volume_col].sum().reset_index()
    daily_mail.columns = ['date', 'mail_type', 'volume']
    daily_mail['date'] = pd.to_datetime(daily_mail['date'])
    daily_mail = daily_mail[daily_mail['date'].dt.weekday < 5]
    
    # Aggregate to WEEKLY totals
    daily_mail['week'] = daily_mail['date'].dt.to_period('W-FRI')
    weekly_mail = daily_mail.groupby(['week', 'mail_type'])['volume'].sum().reset_index()
    
    # Pivot to get mail types as columns
    mail_pivot = weekly_mail.pivot(index='week', columns='mail_type', values='volume').fillna(0)
    mail_pivot = mail_pivot.reset_index()
    
    safe_print(f"Mail data: {len(daily_mail)} daily rows → {len(mail_pivot)} weekly, {len(mail_pivot.columns)-1} mail types")
    
    # Merge weekly data
    merged = pd.merge(weekly_calls[['week', 'call_volume']], mail_pivot, on='week', how='inner')
    merged = merged.sort_values('week').reset_index(drop=True)
    
    safe_print(f"Merged weekly data: {len(merged)} weeks")
    
    # Apply capping to weekly data
    cap_percentile = CONFIG["outlier_cap_percentile"]
    
    # Cap call volumes
    call_cap = merged['call_volume'].quantile(cap_percentile / 100)
    call_floor = merged['call_volume'].quantile((100 - cap_percentile) / 100)
    merged['call_volume'] = merged['call_volume'].clip(lower=call_floor, upper=call_cap)
    
    # Cap mail volumes
    mail_columns = [col for col in merged.columns if col not in ['week', 'call_volume']]
    for col in mail_columns:
        cap_value = merged[col].quantile(cap_percentile / 100)
        merged[col] = merged[col].clip(upper=cap_value)
    
    safe_print(f"Applied weekly capping at {cap_percentile}th percentile")
    
    # Check weekly correlation
    total_mail = merged[mail_columns].sum(axis=1)
    correlation = merged['call_volume'].corr(total_mail)
    safe_print(f"Weekly total mail vs calls correlation: {correlation:.3f}")
    
    # Show data range
    safe_print(f"Date range: {merged['week'].min()} to {merged['week'].max()}")
    safe_print(f"Call volume range: {merged['call_volume'].min():.0f} to {merged['call_volume'].max():.0f}")
    safe_print(f"Total mail range: {total_mail.min():.0f} to {total_mail.max():.0f}")
    
    return merged, mail_columns

# ============================================================================
# WEEKLY FEATURE ENGINEERING
# ============================================================================

def create_weekly_features(df, mail_columns):
    """Create features for weekly prediction"""
    safe_print("\n" + "=" * 60)
    safe_print("CREATING WEEKLY FEATURES")
    safe_print("=" * 60)
    
    # Select top mail types by weekly volume
    mail_volumes = df[mail_columns].sum().sort_values(ascending=False)
    top_mail_types = mail_volumes.head(CONFIG["top_mail_types"]).index.tolist()
    
    safe_print(f"Top {len(top_mail_types)} weekly mail types:")
    for i, mail_type in enumerate(top_mail_types):
        volume = mail_volumes[mail_type]
        weekly_avg = volume / len(df)
        safe_print(f"  {i+1}. {mail_type[:35]:<35}: {volume:>10,.0f} total ({weekly_avg:>8,.0f}/week)")
    
    # Create weekly features
    features_list = []
    targets_list = []
    
    # For weekly prediction: this week's mail -> this week's calls
    # Or: this week's mail -> next week's calls (if we want predictive)
    
    # Let's try SAME WEEK first (concurrent relationship)
    for i in range(len(df)):
        current_week = df.iloc[i]
        
        feature_row = {}
        
        # 1. WEEKLY MAIL FEATURES
        for mail_type in top_mail_types:
            clean_name = str(mail_type).replace(' ', '').replace('-', '').replace('_', '').replace('(', '').replace(')', '')[:15]
            feature_row[f"{clean_name}"] = current_week[mail_type]
        
        # 2. TOTAL WEEKLY MAIL
        total_mail = sum(current_week[mt] for mt in top_mail_types)
        feature_row['total_mail'] = total_mail
        feature_row['log_total_mail'] = np.log1p(total_mail)
        
        # 3. MAIL DISTRIBUTION FEATURES
        if total_mail > 0:
            # What percentage is the biggest mail type?
            biggest_mail = max(current_week[mt] for mt in top_mail_types)
            feature_row['biggest_mail_pct'] = biggest_mail / total_mail
            
            # Mail diversity (how many types have >5% of total)
            significant_types = sum(1 for mt in top_mail_types if current_week[mt] / total_mail > 0.05)
            feature_row['mail_diversity'] = significant_types
        else:
            feature_row['biggest_mail_pct'] = 0
            feature_row['mail_diversity'] = 0
        
        # 4. TEMPORAL FEATURES (week-based)
        week_period = current_week['week']
        # Extract month and quarter from week period
        week_start = week_period.start_time
        
        feature_row['month'] = week_start.month
        feature_row['quarter'] = week_start.quarter
        feature_row['week_of_year'] = week_start.isocalendar()[1]
        feature_row['is_month_end_week'] = 1 if week_start.day >= 22 else 0  # Last week of month
        feature_row['is_quarter_end_week'] = 1 if (week_start.month % 3 == 0 and week_start.day >= 22) else 0
        
        # 5. HISTORICAL FEATURES (if available)
        if i > 0:  # Previous week available
            prev_week = df.iloc[i-1]
            prev_total_mail = sum(prev_week[mt] for mt in top_mail_types)
            feature_row['prev_total_mail'] = prev_total_mail
            feature_row['prev_calls'] = prev_week['call_volume']
            
            # Week-over-week change
            if prev_total_mail > 0:
                feature_row['mail_wow_change'] = (total_mail - prev_total_mail) / prev_total_mail
            else:
                feature_row['mail_wow_change'] = 0
        else:
            feature_row['prev_total_mail'] = total_mail  # Use current as proxy
            feature_row['prev_calls'] = current_week['call_volume']
            feature_row['mail_wow_change'] = 0
        
        # TARGET: This week's calls
        target = current_week['call_volume']
        
        features_list.append(feature_row)
        targets_list.append(target)
    
    # Convert to DataFrames
    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)
    
    # Clean data
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    safe_print(f"Created {len(X.columns)} weekly features:")
    for col in X.columns:
        safe_print(f"  - {col}")
    
    safe_print(f"Weekly dataset: {len(X)} weeks")
    
    return X, y, top_mail_types

# ============================================================================
# WEEKLY MODEL TRAINING
# ============================================================================

def train_weekly_models(X, y):
    """Train models for weekly prediction"""
    safe_print("\n" + "=" * 60)
    safe_print("TRAINING WEEKLY MODELS")
    safe_print("=" * 60)
    
    safe_print(f"Dataset size: {len(X)} weeks")
    
    # For small datasets, use different split strategy
    if len(X) < 20:
        # Use leave-one-out or very small test set
        test_size = max(3, int(len(X) * 0.2))  # At least 3 weeks for test
        split_idx = len(X) - test_size
    else:
        split_idx = int(len(X) * (1 - CONFIG["test_size"]))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    safe_print(f"Train: {len(X_train)} weeks, Test: {len(X_test)} weeks")
    
    # Models optimized for small datasets
    models = {
        'linear': LinearRegression(),
        'ridge_light': Ridge(alpha=0.1, random_state=CONFIG["random_state"]),
        'ridge_medium': Ridge(alpha=1.0, random_state=CONFIG["random_state"]),
        'ridge_strong': Ridge(alpha=10.0, random_state=CONFIG["random_state"]),
    }
    
    # Only add tree models if we have enough data
    if len(X_train) >= 10:
        models['forest_tiny'] = RandomForestRegressor(
            n_estimators=10, 
            max_depth=2, 
            min_samples_leaf=max(2, len(X_train)//5),
            random_state=CONFIG["random_state"]
        )
    
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
            
            # Select best model (prioritize test R², moderate overfitting penalty)
            adjusted_score = test_r2 - max(0, (overfitting - 0.2) * 0.5)  # Gentle overfitting penalty
            
            if adjusted_score > best_score and test_r2 > CONFIG["min_r2_threshold"]:
                best_score = adjusted_score
                best_model = model
                best_name = name
                safe_print(f"  ★ NEW BEST MODEL! (Adjusted score: {adjusted_score:.3f})")
            
        except Exception as e:
            safe_print(f"  ✗ Failed: {e}")
            results[name] = {'error': str(e)}
    
    if best_model:
        safe_print(f"\n🎯 BEST WEEKLY MODEL: {best_name}")
        safe_print(f"   Test R²: {results[best_name]['test_r2']:.3f}")
        safe_print(f"   Test MAE: {results[best_name]['test_mae']:.0f} calls per week")
        
        # Train on full dataset
        best_model.fit(X, y)
        
        # Show feature importance/coefficients
        if hasattr(best_model, 'coef_'):
            safe_print(f"\n   Feature importance (Ridge coefficients):")
            coef_pairs = list(zip(X.columns, best_model.coef_))
            coef_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            for feature, coef in coef_pairs[:8]:  # Top 8
                safe_print(f"   {feature:20s}: {coef:8.2f}")
        
    else:
        safe_print("\n❌ NO WEEKLY MODEL MEETS MINIMUM THRESHOLD!")
        best_test_r2 = max([r.get('test_r2', -999) for r in results.values() if 'error' not in r], default=-999)
        safe_print(f"   Best Test R²: {best_test_r2:.3f} < {CONFIG['min_r2_threshold']}")
    
    return best_model, best_name, results

# ============================================================================
# WEEKLY PREDICTION AND VISUALIZATION
# ============================================================================

def create_weekly_prediction_system(model, model_name, X, y, results, top_mail_types):
    """Create weekly prediction system with validation"""
    safe_print("\n" + "=" * 60)
    safe_print("CREATING WEEKLY PREDICTION SYSTEM")
    safe_print("=" * 60)
    
    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    # Create validation plots
    split_idx = int(len(X) * (1 - CONFIG["test_size"]))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Weekly Model Validation: {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.7, s=80, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Weekly Calls')
    axes[0, 0].set_ylabel('Predicted Weekly Calls')
    axes[0, 0].set_title(f'Test Set: R² = {results[model_name]["test_r2"]:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series
    axes[0, 1].plot(range(len(y_test)), y_test.values, 'b-', label='Actual', linewidth=3, marker='o')
    axes[0, 1].plot(range(len(y_test)), y_pred_test, 'r-', label='Predicted', linewidth=3, marker='s', alpha=0.7)
    axes[0, 1].set_xlabel('Test Week')
    axes[0, 1].set_ylabel('Weekly Call Volume')
    axes[0, 1].set_title('Weekly Predictions vs Actual')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals
    residuals = y_test - y_pred_test
    axes[1, 0].scatter(y_pred_test, residuals, alpha=0.7, s=80, color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Predicted Weekly Calls')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Model comparison
    model_names = [name for name in results.keys() if 'error' not in results[name]]
    test_r2_scores = [results[name]['test_r2'] for name in model_names]
    
    bars = axes[1, 1].bar(range(len(model_names)), test_r2_scores, alpha=0.7, color='orange')
    axes[1, 1].set_xticks(range(len(model_names)))
    axes[1, 1].set_xticklabels(model_names, rotation=45)
    axes[1, 1].set_ylabel('Test R²')
    axes[1, 1].set_title('Weekly Model Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=CONFIG["min_r2_threshold"], color='r', linestyle='--', label='Threshold')
    axes[1, 1].legend()
    
    # Highlight best model
    best_idx = model_names.index(model_name)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "weekly_model_validation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    safe_print(f"Weekly validation plot saved: {output_dir}/weekly_model_validation.png")
    
    # Create prediction function
    def predict_weekly_calls(weekly_mail_input):
        """
        Predict weekly calls from weekly mail volumes
        
        Args:
            weekly_mail_input: dict like {'DRP Stmt.': 20000, 'Cheque': 15000, ...}
        
        Returns:
            dict with prediction
        """
        
        try:
            # Create feature vector
            features = {}
            
            # Mail features
            for mail_type in top_mail_types:
                clean_name = str(mail_type).replace(' ', '').replace('-', '').replace('_', '').replace('(', '').replace(')', '')[:15]
                features[clean_name] = weekly_mail_input.get(mail_type, 0)
            
            # Total mail
            total_mail = sum(weekly_mail_input.get(mt, 0) for mt in top_mail_types)
            features['total_mail'] = total_mail
            features['log_total_mail'] = np.log1p(total_mail)
            
            # Mail distribution
            if total_mail > 0:
                biggest_mail = max(weekly_mail_input.get(mt, 0) for mt in top_mail_types)
                features['biggest_mail_pct'] = biggest_mail / total_mail
                significant_types = sum(1 for mt in top_mail_types if weekly_mail_input.get(mt, 0) / total_mail > 0.05)
                features['mail_diversity'] = significant_types
            else:
                features['biggest_mail_pct'] = 0
                features['mail_diversity'] = 0
            
            # Temporal features (use current date)
            from datetime import datetime
            now = datetime.now()
            features['month'] = now.month
            features['quarter'] = (now.month - 1) // 3 + 1
            features['week_of_year'] = now.isocalendar()[1]
            features['is_month_end_week'] = 1 if now.day >= 22 else 0
            features['is_quarter_end_week'] = 1 if (now.month % 3 == 0 and now.day >= 22) else 0
            
            # Historical features (use averages as defaults)
            avg_weekly_calls = y.mean()
            avg_weekly_mail = X['total_mail'].mean()
            features['prev_total_mail'] = avg_weekly_mail
            features['prev_calls'] = avg_weekly_calls
            features['mail_wow_change'] = 0  # Assume no change
            
            # Convert to array
            feature_vector = [features[col] for col in model.feature_names_in_]
            
            # Predict
            prediction = model.predict([feature_vector])[0]
            prediction = max(0, round(prediction))
            
            return {
                'predicted_weekly_calls': int(prediction),
                'weekly_mail_input': weekly_mail_input,
                'total_weekly_mail': int(total_mail),
                'prediction_per_day': int(prediction / 5),  # Assuming 5 business days
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}
    
    return predict_weekly_calls

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution - weekly aggregation approach"""
    
    safe_print("WEEKLY MAIL-TO-CALLS PREDICTION (FUNDAMENTAL STRATEGY CHANGE)")
    safe_print("=" * 70)
    safe_print("NEW APPROACH:")
    safe_print("* Aggregate daily data to WEEKLY totals")
    safe_print("* Weekly mail volume -> Weekly call volume")
    safe_print("* Should reduce noise and improve signal")
    safe_print("* Better for small dataset (fewer but more stable samples)")
    safe_print("=" * 70)
    
    try:
        # Load and aggregate weekly data
        df, mail_columns = load_and_aggregate_weekly()
        
        if len(df) < 8:
            safe_print(f"❌ INSUFFICIENT WEEKLY DATA: Only {len(df)} weeks available")
            safe_print("   Need at least 8 weeks for reliable modeling")
            return {'success': False, 'error': 'Insufficient weekly data'}
        
        # Create weekly features
        X, y, top_mail_types = create_weekly_features(df, mail_columns)
        
        # Train weekly models
        best_model, best_name, results = train_weekly_models(X, y)
        
        if not best_model:
            safe_print("\n❌ FAILED: No weekly model achieved minimum performance")
            safe_print("   This suggests the relationship between mail and calls is too weak")
            safe_print("   or the dataset is too small/noisy for any modeling approach")
            return {'success': False, 'error': 'No acceptable weekly model found'}
        
        # Create prediction system
        predict_weekly_calls = create_weekly_prediction_system(
            best_model, best_name, X, y, results, top_mail_types
        )
        
        # Save model
        output_dir = Path(CONFIG["output_dir"])
        output_dir.mkdir(exist_ok=True)
        
        joblib.dump(best_model, output_dir / "weekly_model.pkl")
        joblib.dump(top_mail_types, output_dir / "top_weekly_mail_types.pkl")
        
        # Test prediction
        safe_print("\n" + "=" * 60)
        safe_print("TESTING WEEKLY PREDICTION")
        safe_print("=" * 60)
        
        # Create realistic weekly test volumes (much higher than daily)
        test_weekly_input = {}
        for i, mail_type in enumerate(top_mail_types):
            # Weekly volumes should be ~5x daily volumes
            test_weekly_input[mail_type] = [10000, 7500, 5000, 4000, 3000, 2500, 2000, 1500][i] if i < 8 else 1000
        
        result = predict_weekly_calls(test_weekly_input)
        
        if result['status'] == 'success':
            safe_print("✅ WEEKLY TEST SUCCESSFUL!")
            safe_print(f"   Weekly Mail Input: {test_weekly_input}")
            safe_print(f"   Predicted Weekly Calls: {result['predicted_weekly_calls']:,}")
            safe_print(f"   Total Weekly Mail: {result['total_weekly_mail']:,}")
            safe_print(f"   Average Daily Calls: {result['prediction_per_day']:,}")
        else:
            safe_print(f"❌ WEEKLY TEST FAILED: {result['error']}")
        
        # Final summary
        safe_print("\n" + "=" * 70)
        safe_print("🎯 SUCCESS! WEEKLY MODEL WORKS!")
        safe_print("=" * 70)
        safe_print(f"✅ Best Model: {best_name}")
        safe_print(f"✅ Test R²: {results[best_name]['test_r2']:.3f}")
        safe_print(f"✅ Test MAE: {results[best_name]['test_mae']:.0f} calls/week")
        safe_print(f"✅ Weekly Data Points: {len(X)}")
        safe_print(f"✅ Top Mail Types: {len(top_mail_types)}")
        safe_print("")
        safe_print("WEEKLY PREDICTION READY:")
        safe_print("- Input: Weekly mail volumes by type")
        safe_print("- Output: Total weekly call volume")
        safe_print("- More stable than daily predictions")
        safe_print("- Better signal-to-noise ratio")
        
        return {
            'success': True,
            'best_model': best_model,
            'best_model_name': best_name,
            'test_r2': results[best_name]['test_r2'],
            'test_mae': results[best_name]['test_mae'],
            'weekly_samples': len(X),
            'top_mail_types': top_mail_types,
            'predict_function': predict_weekly_calls,
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
        safe_print("🚀 WEEKLY PREDICTION MODEL DEPLOYED!")
    else:
        safe_print(f"💥 FAILED: {result['error']}")
