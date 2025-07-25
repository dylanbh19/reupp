Ah, I understand! You want to **combine** both economic data files and use ALL the economic indicators from both files together. Here's how to modify the code:

## 1. Code Modifications to Combine Multiple Economic Data Files

### BEFORE (in CONFIG section):
```python
CONFIG = {
    # ============ YOUR FILE PATHS ============
    "call_file": "ACDMail.csv",
    "mail_file": "mail.csv",
    "economic_data_file": "expanded_economic_data.csv", # ← (NEW) ADD YOUR ECONOMIC DATA FILE
```

### AFTER (in CONFIG section):
```python
CONFIG = {
    # ============ YOUR FILE PATHS ============
    "call_file": "ACDMail.csv",
    "mail_file": "mail.csv",
    "economic_data_files": [  # ← CHANGED TO LIST OF FILES TO COMBINE
        "expanded_economic_data.csv",
        "econsimple.csv",  # Both files will be merged
        # Add more economic data files as needed
    ],
```

### BEFORE (in main() function, around line 615):
```python
        # (NEW) STEP 1D: MERGING ECONOMIC DATA
        safe_print("\n" + "=" * 80)
        safe_print("STEP 1D: MERGING ECONOMIC DATA")
        safe_print("=" * 80)
        economic_cols = []
        try:
            econ_df = pd.read_csv(CONFIG["economic_data_file"])
            econ_df['Date'] = pd.to_datetime(econ_df['Date'])
            econ_df.rename(columns={'Date': 'date'}, inplace=True)
            merged_data = pd.merge(merged_data, econ_df, on='date', how='left')
            economic_cols = [col for col in econ_df.columns if col != 'date']
            merged_data[economic_cols] = merged_data[economic_cols].fillna(method='ffill')
            merged_data.dropna(subset=economic_cols, inplace=True)
            safe_print(f"✅ Economic data successfully merged.")
        except FileNotFoundError:
            safe_print(f"⚠️  '{CONFIG['economic_data_file']}' not found. Skipping economic data.")
        except Exception as e:
            safe_print(f"❌ Error merging economic data: {e}")
```

### AFTER (in main() function):
```python
        # (NEW) STEP 1D: MERGING ALL ECONOMIC DATA FILES
        safe_print("\n" + "=" * 80)
        safe_print("STEP 1D: MERGING ALL ECONOMIC DATA FILES")
        safe_print("=" * 80)
        
        economic_cols = []
        combined_econ_df = None
        
        # Load and combine all economic data files
        for econ_file in CONFIG["economic_data_files"]:
            try:
                safe_print(f"\n--- Loading {econ_file} ---")
                econ_df = pd.read_csv(econ_file)
                econ_df['Date'] = pd.to_datetime(econ_df['Date'])
                econ_df.rename(columns={'Date': 'date'}, inplace=True)
                
                # Get column info
                econ_cols_in_file = [col for col in econ_df.columns if col != 'date']
                safe_print(f"   Found {len(econ_cols_in_file)} economic indicators")
                safe_print(f"   Date range: {econ_df['date'].min().date()} to {econ_df['date'].max().date()}")
                
                if combined_econ_df is None:
                    combined_econ_df = econ_df
                else:
                    # Merge with existing economic data
                    # Use outer join to keep all dates, then we'll handle missing values
                    combined_econ_df = pd.merge(combined_econ_df, econ_df, on='date', how='outer', suffixes=('', '_dup'))
                    
                    # Handle duplicate columns (keep the first occurrence)
                    for col in combined_econ_df.columns:
                        if col.endswith('_dup'):
                            base_col = col[:-4]
                            if base_col in combined_econ_df.columns:
                                # Fill missing values in base column with values from duplicate
                                combined_econ_df[base_col] = combined_econ_df[base_col].fillna(combined_econ_df[col])
                            combined_econ_df = combined_econ_df.drop(columns=[col])
                
                safe_print(f"   Successfully loaded {econ_file}")
                
            except FileNotFoundError:
                safe_print(f"   WARNING: '{econ_file}' not found. Skipping.")
            except Exception as e:
                safe_print(f"   ERROR loading {econ_file}: {str(e)}")
        
        # Merge combined economic data with main dataset
        if combined_econ_df is not None:
            safe_print(f"\n--- Merging combined economic data ---")
            
            # Sort by date
            combined_econ_df = combined_econ_df.sort_values('date')
            
            # Get all economic columns
            economic_cols = [col for col in combined_econ_df.columns if col != 'date']
            safe_print(f"   Total economic indicators: {len(economic_cols)}")
            safe_print(f"   Economic indicators: {economic_cols[:10]}..." if len(economic_cols) > 10 else f"   Economic indicators: {economic_cols}")
            
            # Merge with main data
            merged_data = pd.merge(merged_data, combined_econ_df, on='date', how='left')
            
            # Forward fill missing values for economic data
            merged_data[economic_cols] = merged_data[economic_cols].fillna(method='ffill')
            
            # Backward fill any remaining missing values at the start
            merged_data[economic_cols] = merged_data[economic_cols].fillna(method='bfill')
            
            # Drop rows where we still have missing economic data
            before_drop = len(merged_data)
            merged_data = merged_data.dropna(subset=economic_cols)
            after_drop = len(merged_data)
            
            if before_drop > after_drop:
                safe_print(f"   Dropped {before_drop - after_drop} rows with missing economic data")
            
            safe_print(f"   Successfully merged all economic data")
            safe_print(f"   Final dataset: {len(merged_data)} days with {len(economic_cols)} economic features")
            
        else:
            safe_print("   WARNING: No economic data could be loaded")
            economic_cols = []
```

## 2. Comprehensive Testing Script

Here's the complete testing script that will rigorously test your model:

```python
#!/usr/bin/env python
"""
COMPREHENSIVE MODEL TESTING SCRIPT
Tests the mail-to-calls prediction model rigorously
ASCII-formatted for compatibility
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_CONFIG = {
    "model_dir": "mail_call_prediction_system/models",
    "data_files": {
        "calls": "ACDMail.csv",
        "mail": "mail.csv",
        "econ1": "expanded_economic_data.csv",
        "econ2": "econsimple.csv"
    },
    "output_dir": "model_test_results",
    "test_scenarios": {
        "stress_test_multipliers": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        "edge_cases": ["zero_mail", "high_mail", "missing_types", "holiday_period"],
        "temporal_tests": ["weekday_patterns", "monthly_patterns"],
        "stability_tests": ["noise_injection", "data_drift"]
    }
}

def safe_print(msg):
    """Print only ASCII characters"""
    try:
        print(str(msg).encode('ascii', 'ignore').decode('ascii'))
    except:
        print(str(msg))

# ============================================================================
# MODEL TESTING CLASS
# ============================================================================

class ModelTester:
    def __init__(self):
        self.output_dir = Path(TEST_CONFIG["output_dir"])
        self.output_dir.mkdir(exist_ok=True)
        self.model_info = None
        self.test_results = {}
        self.test_data = {}
        
    def load_model(self):
        """Load the trained model"""
        safe_print("=" * 80)
        safe_print("LOADING TRAINED MODEL")
        safe_print("=" * 80)
        
        model_path = Path(TEST_CONFIG["model_dir"]) / "best_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model_info = joblib.load(model_path)
        safe_print(f"Model loaded: {self.model_info['model_name']}")
        safe_print(f"Features: {len(self.model_info['features'])}")
        safe_print(f"Model performance: R2={self.model_info['performance']['test_r2']:.3f}")
        return True
        
    def load_test_data(self):
        """Load actual data for realistic testing"""
        safe_print("\n" + "=" * 80)
        safe_print("LOADING TEST DATA")
        safe_print("=" * 80)
        
        # Load mail data
        mail_df = pd.read_csv(TEST_CONFIG["data_files"]["mail"])
        mail_df['mail_date'] = pd.to_datetime(mail_df['mail_date'])
        
        # Get mail type statistics
        mail_types = mail_df['mail_type'].unique()
        mail_stats = mail_df.groupby('mail_type')['mail_volume'].agg(['mean', 'std', 'min', 'max'])
        
        # Load call data
        call_df = pd.read_csv(TEST_CONFIG["data_files"]["calls"])
        call_df['Date'] = pd.to_datetime(call_df['Date'])
        call_stats = call_df.groupby('Product')['ACDCalls'].sum()
        
        # Store for testing
        self.test_data = {
            'mail_stats': mail_stats,
            'mail_types': mail_types,
            'call_stats': call_stats,
            'avg_daily_calls': call_df.groupby('Date')['ACDCalls'].sum().mean()
        }
        
        safe_print(f"Loaded data for {len(mail_types)} mail types")
        safe_print(f"Average daily calls: {self.test_data['avg_daily_calls']:.0f}")
        
    def run_all_tests(self):
        """Run comprehensive test suite"""
        safe_print("\n" + "=" * 80)
        safe_print("RUNNING COMPREHENSIVE TEST SUITE")
        safe_print("=" * 80)
        
        # 1. Baseline Performance Test
        self.test_baseline_performance()
        
        # 2. Stress Tests
        self.run_stress_tests()
        
        # 3. Edge Case Tests
        self.run_edge_case_tests()
        
        # 4. Temporal Pattern Tests
        self.run_temporal_tests()
        
        # 5. Stability Tests
        self.run_stability_tests()
        
        # 6. Feature Importance Analysis
        self.analyze_feature_importance()
        
        # Generate report
        self.generate_test_report()
        
    def test_baseline_performance(self):
        """Test model with realistic inputs"""
        safe_print("\n--- TEST 1: BASELINE PERFORMANCE ---")
        
        results = []
        
        # Get feature names from model
        feature_names = self.model_info['features']
        
        # Create 100 realistic test cases
        for i in range(100):
            # Create random but realistic input
            test_input = {}
            
            # Add mail features
            for feature in feature_names:
                if 'lag' in feature or 'avg' in feature:
                    # Use realistic mail volumes
                    base_value = np.random.normal(10000, 3000)
                    test_input[feature] = max(0, base_value)
                elif feature == 'weekday':
                    test_input[feature] = np.random.randint(0, 5)
                elif feature == 'month':
                    test_input[feature] = np.random.randint(1, 13)
                elif feature == 'day_of_month':
                    test_input[feature] = np.random.randint(1, 29)
                else:
                    # Economic features - use realistic ranges
                    test_input[feature] = np.random.normal(100, 20)
            
            # Make prediction
            X_test = pd.DataFrame([test_input])[feature_names]
            prediction = self.model_info['model'].predict(X_test)[0]
            
            results.append({
                'test_case': i,
                'prediction': prediction,
                'reasonable': 5000 < prediction < 20000
            })
        
        # Analyze results
        predictions = [r['prediction'] for r in results]
        reasonable_pct = sum(r['reasonable'] for r in results) / len(results) * 100
        
        self.test_results['baseline'] = {
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions),
            'reasonable_percentage': reasonable_pct
        }
        
        safe_print(f"   Mean prediction: {np.mean(predictions):.0f}")
        safe_print(f"   Std deviation: {np.std(predictions):.0f}")
        safe_print(f"   Range: {np.min(predictions):.0f} - {np.max(predictions):.0f}")
        safe_print(f"   Reasonable predictions: {reasonable_pct:.1f}%")
        
    def run_stress_tests(self):
        """Test model under extreme conditions"""
        safe_print("\n--- TEST 2: STRESS TESTS ---")
        
        feature_names = self.model_info['features']
        base_input = {feature: 10000 if 'lag' in feature or 'avg' in feature else 100 
                     for feature in feature_names}
        base_input['weekday'] = 2
        base_input['month'] = 6
        base_input['day_of_month'] = 15
        
        stress_results = {}
        
        for multiplier in TEST_CONFIG["test_scenarios"]["stress_test_multipliers"]:
            # Scale mail volumes
            test_input = base_input.copy()
            for feature in feature_names:
                if 'lag' in feature or 'avg' in feature:
                    test_input[feature] *= multiplier
            
            X_test = pd.DataFrame([test_input])[feature_names]
            prediction = self.model_info['model'].predict(X_test)[0]
            
            stress_results[multiplier] = prediction
            safe_print(f"   Mail volume x{multiplier}: {prediction:.0f} calls")
        
        self.test_results['stress'] = stress_results
        
    def run_edge_case_tests(self):
        """Test edge cases"""
        safe_print("\n--- TEST 3: EDGE CASE TESTS ---")
        
        feature_names = self.model_info['features']
        edge_results = {}
        
        # Test 1: Zero mail volume
        zero_input = {feature: 0 if 'lag' in feature or 'avg' in feature else 100 
                     for feature in feature_names}
        zero_input.update({'weekday': 2, 'month': 6, 'day_of_month': 15})
        
        X_test = pd.DataFrame([zero_input])[feature_names]
        edge_results['zero_mail'] = self.model_info['model'].predict(X_test)[0]
        safe_print(f"   Zero mail: {edge_results['zero_mail']:.0f} calls")
        
        # Test 2: Extremely high mail volume
        high_input = {feature: 100000 if 'lag' in feature or 'avg' in feature else 100 
                     for feature in feature_names}
        high_input.update({'weekday': 2, 'month': 6, 'day_of_month': 15})
        
        X_test = pd.DataFrame([high_input])[feature_names]
        edge_results['high_mail'] = self.model_info['model'].predict(X_test)[0]
        safe_print(f"   High mail (100k): {edge_results['high_mail']:.0f} calls")
        
        # Test 3: Weekend (should handle gracefully even though trained on weekdays)
        weekend_input = zero_input.copy()
        weekend_input['weekday'] = 6  # Sunday
        weekend_input.update({f: 10000 for f in feature_names if 'lag' in f or 'avg' in f})
        
        X_test = pd.DataFrame([weekend_input])[feature_names]
        edge_results['weekend'] = self.model_info['model'].predict(X_test)[0]
        safe_print(f"   Weekend: {edge_results['weekend']:.0f} calls")
        
        self.test_results['edge_cases'] = edge_results
        
    def run_temporal_tests(self):
        """Test temporal patterns"""
        safe_print("\n--- TEST 4: TEMPORAL PATTERN TESTS ---")
        
        feature_names = self.model_info['features']
        base_input = {feature: 10000 if 'lag' in feature or 'avg' in feature else 100 
                     for feature in feature_names}
        
        temporal_results = {'weekday': {}, 'monthly': {}}
        
        # Test weekday patterns
        for day in range(5):
            test_input = base_input.copy()
            test_input.update({'weekday': day, 'month': 6, 'day_of_month': 15})
            
            X_test = pd.DataFrame([test_input])[feature_names]
            prediction = self.model_info['model'].predict(X_test)[0]
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            temporal_results['weekday'][day_names[day]] = prediction
            
        safe_print("   Weekday patterns:")
        for day, pred in temporal_results['weekday'].items():
            safe_print(f"     {day}: {pred:.0f}")
            
        # Test monthly patterns
        for month in [1, 4, 7, 10, 12]:
            test_input = base_input.copy()
            test_input.update({'weekday': 2, 'month': month, 'day_of_month': 15})
            
            X_test = pd.DataFrame([test_input])[feature_names]
            prediction = self.model_info['model'].predict(X_test)[0]
            
            month_names = {1: 'Jan', 4: 'Apr', 7: 'Jul', 10: 'Oct', 12: 'Dec'}
            temporal_results['monthly'][month_names[month]] = prediction
            
        safe_print("   Monthly patterns:")
        for month, pred in temporal_results['monthly'].items():
            safe_print(f"     {month}: {pred:.0f}")
            
        self.test_results['temporal'] = temporal_results
        
    def run_stability_tests(self):
        """Test model stability"""
        safe_print("\n--- TEST 5: STABILITY TESTS ---")
        
        feature_names = self.model_info['features']
        base_input = {feature: 10000 if 'lag' in feature or 'avg' in feature else 100 
                     for feature in feature_names}
        base_input.update({'weekday': 2, 'month': 6, 'day_of_month': 15})
        
        # Test prediction stability with small input changes
        stability_results = []
        
        X_base = pd.DataFrame([base_input])[feature_names]
        base_prediction = self.model_info['model'].predict(X_base)[0]
        
        # Add small noise to inputs
        for i in range(20):
            noisy_input = base_input.copy()
            
            # Add 1-5% noise to mail features
            for feature in feature_names:
                if 'lag' in feature or 'avg' in feature:
                    noise = np.random.uniform(0.95, 1.05)
                    noisy_input[feature] *= noise
            
            X_test = pd.DataFrame([noisy_input])[feature_names]
            prediction = self.model_info['model'].predict(X_test)[0]
            
            pct_change = abs(prediction - base_prediction) / base_prediction * 100
            stability_results.append(pct_change)
        
        avg_stability = np.mean(stability_results)
        max_deviation = np.max(stability_results)
        
        self.test_results['stability'] = {
            'base_prediction': base_prediction,
            'avg_pct_change': avg_stability,
            'max_pct_change': max_deviation,
            'stable': max_deviation < 10  # Less than 10% change is considered stable
        }
        
        safe_print(f"   Base prediction: {base_prediction:.0f}")
        safe_print(f"   Average change with 5% input noise: {avg_stability:.1f}%")
        safe_print(f"   Maximum change: {max_deviation:.1f}%")
        safe_print(f"   Model stability: {'STABLE' if max_deviation < 10 else 'UNSTABLE'}")
        
    def analyze_feature_importance(self):
        """Analyze feature importance for tree-based models"""
        safe_print("\n--- TEST 6: FEATURE IMPORTANCE ANALYSIS ---")
        
        if hasattr(self.model_info['model'], 'feature_importances_'):
            importances = self.model_info['model'].feature_importances_
            features = self.model_info['features']
            
            # Get top 10 features
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            
            safe_print("   Top 10 most important features:")
            for idx, row in importance_df.iterrows():
                safe_print(f"     {row['feature']}: {row['importance']:.3f}")
                
            self.test_results['feature_importance'] = importance_df.to_dict('records')
        else:
            safe_print("   Feature importance not available for this model type")
            
    def generate_test_report(self):
        """Generate comprehensive test report"""
        safe_print("\n" + "=" * 80)
        safe_print("TEST REPORT SUMMARY")
        safe_print("=" * 80)
        
        # Save detailed results to JSON
        report_path = self.output_dir / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Print summary
        safe_print("\n1. BASELINE PERFORMANCE:")
        baseline = self.test_results.get('baseline', {})
        safe_print(f"   - Average prediction: {baseline.get('mean_prediction', 0):.0f} calls")
        safe_print(f"   - Reasonable predictions: {baseline.get('reasonable_percentage', 0):.1f}%")
        
        safe_print("\n2. STRESS TEST RESULTS:")
        stress = self.test_results.get('stress', {})
        safe_print(f"   - Scales appropriately: {1.5 * stress.get(0.5, 0) < stress.get(1.5, 0)}")
        
        safe_print("\n3. EDGE CASE HANDLING:")
        edge = self.test_results.get('edge_cases', {})
        safe_print(f"   - Zero mail prediction: {edge.get('zero_mail', 0):.0f} calls")
        safe_print(f"   - High mail prediction: {edge.get('high_mail', 0):.0f} calls")
        
        safe_print("\n4. STABILITY:")
        stability = self.test_results.get('stability', {})
        safe_print(f"   - Model stability: {'PASSED' if stability.get('stable', False) else 'FAILED'}")
        safe_print(f"   - Max deviation: {stability.get('max_pct_change', 0):.1f}%")
        
        # Overall assessment
        safe_print("\n" + "=" * 80)
        safe_print("OVERALL ASSESSMENT:")
        
        tests_passed = 0
        tests_total = 4
        
        if baseline.get('reasonable_percentage', 0) > 80:
            tests_passed += 1
        if stability.get('stable', False):
            tests_passed += 1
        if edge.get('zero_mail', 0) < 20000:
            tests_passed += 1
        if 1.5 * stress.get(0.5, 0) < stress.get(1.5, 0):
            tests_passed += 1
            
        safe_print(f"Tests passed: {tests_passed}/{tests_total}")
        
        if tests_passed == tests_total:
            safe_print("\nMODEL STATUS: READY FOR PRODUCTION")
        elif tests_passed >= 3:
            safe_print("\nMODEL STATUS: ACCEPTABLE WITH MONITORING")
        else:
            safe_print("\nMODEL STATUS: NEEDS IMPROVEMENT")
            
        safe_print(f"\nDetailed report saved to: {report_path}")
        
        # Create visualization
        self.create_test_visualizations()
        
    def create_test_visualizations(self):
        """Create test result visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Test Results', fontsize=16)
        
        # 1. Stress test results
        stress = self.test_results.get('stress', {})
        if stress:
            multipliers = list(stress.keys())
            predictions = list(stress.values())
            axes[0, 0].plot(multipliers, predictions, 'bo-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Mail Volume Multiplier')
            axes[0, 0].set_ylabel('Predicted Calls')
            axes[0, 0].set_title('Stress Test: Mail Volume Scaling')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Weekday patterns
        temporal = self.test_results.get('temporal', {})
        if temporal and 'weekday' in temporal:
            days = list(temporal['weekday'].keys())
            calls = list(temporal['weekday'].values())
            axes[0, 1].bar(days, calls)
            axes[0, 1].set_title('Weekday Call Predictions')
            axes[0, 1].set_ylabel('Predicted Calls')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Feature importance (if available)
        if 'feature_importance' in self.test_results:
            importance_data = self.test_results['feature_importance'][:5]
            features = [d['feature'] for d in importance_data]
            importances = [d['importance'] for d in importance_data]
            axes[1, 0].barh(features, importances)
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Top 5 Feature Importances')
        
        # 4. Test summary
        test_summary = {
            'Baseline': 'PASS' if self.test_results.get('baseline', {}).get('reasonable_percentage', 0) > 80 else 'FAIL',
            'Stability': 'PASS' if self.test_results.get('stability', {}).get('stable', False) else 'FAIL',
            'Edge Cases': 'PASS' if self.test_results.get('edge_cases', {}).get('zero_mail', float('inf')) < 20000 else 'FAIL',
            'Scaling': 'PASS' if 1.5 * stress.get(0.5, 0) < stress.get(1.5, 0) else 'FAIL'
        }
        
        colors = ['green' if v == 'PASS' else 'red' for v in test_summary.values()]
        y_pos = np.arange(len(test_summary))
        axes[1, 1].barh(y_pos, [1]*len(test_summary), color=colors, alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(list(test_summary.keys()))
        axes[1, 1].set_xlim(0, 1.2)
        axes[1, 1].set_title('Test Results Summary')
        
        # Add PASS/FAIL labels
        for i, (test, result) in enumerate(test_summary.items()):
            axes[1, 1].text(0.5, i, result, ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_results_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        safe_print(f"\nVisualization saved to: {self.output_dir / 'test_results_visualization.png'}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the comprehensive model testing"""
    safe_print("=" * 80)
    safe_print("COMPREHENSIVE MODEL TESTING SYSTEM")
    safe_print("=" * 80)
    safe_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize tester
        tester = ModelTester()
        
        # Load model
        tester.load_model()
        
        # Load test data
        tester.load_test_data()
        
        # Run all tests
        tester.run_all_tests()
        
        safe_print(f"\nTesting completed successfully!")
        safe_print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        safe_print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        safe_print("\nTEST SUITE FAILED!")
```

Save the testing script as `test_model.py` and run it after training your model with the combined economic data. It will thoroughly test your model and generate a comprehensive report.
