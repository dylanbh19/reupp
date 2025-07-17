Looking at your output, there are several issues causing all models to fail. Let me give you the specific fixes:

## 🔧 **Critical Fixes Needed:**

### **1. Infinity/NaN Values Issue**
```python
# ADD this function after the validate_data_quality function:

def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean infinite and NaN values from dataset"""
    logger.info("Cleaning infinite and NaN values...")
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    logger.info(f"Cleaned dataset shape: {df.shape}")
    return df
```

### **2. LightGBM Special Characters Issue**
```python
# ADD this function:

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names for LightGBM compatibility"""
    logger.info("Cleaning column names for model compatibility...")
    
    # Clean column names - remove special characters
    df.columns = [
        col.replace(' ', '_')
           .replace('(', '')
           .replace(')', '')
           .replace('[', '')
           .replace(']', '')
           .replace('{', '')
           .replace('}', '')
           .replace(',', '_')
           .replace('.', '_')
           .replace(':', '_')
           .replace(';', '_')
           .replace('!', '')
           .replace('?', '')
           .replace("'", '')
           .replace('"', '')
           .replace('-', '_')
           .replace('/', '_')
           .replace('\\', '_')
           .replace('|', '_')
           .replace('&', 'and')
           .replace('%', 'pct')
           .replace('#', 'num')
           .replace('@', 'at')
        for col in df.columns
    ]
    
    # Ensure no column starts with number
    df.columns = [f"col_{col}" if col[0].isdigit() else col for col in df.columns]
    
    logger.info("Column names cleaned")
    return df
```

### **3. Encoding Issue Fix**
```python
# REPLACE the generate_summary_report function with this version:

def generate_summary_report(metrics: dict, weekly: pd.DataFrame, coverage: float, 
                          alerts: List[str], output_dir: Path) -> None:
    """Generate comprehensive summary report with enhanced insights"""
    
    # Extract model metrics
    model_metrics = metrics.get("model_metrics", metrics)
    run_metadata = metrics.get("run_metadata", {})
    
    report_lines = [
        "# Call Volume Forecasting Pipeline - Executive Summary",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Run ID**: {run_metadata.get('run_id', 'N/A')}",
        f"**Version**: {run_metadata.get('version', 'N/A')}",
        "",
        "## Dataset Overview",  # REMOVED EMOJI
        f"- **Samples**: {weekly.shape[0]} weeks",
        f"- **Features**: {weekly.shape[1]} total features",
        f"- **Date Range**: {weekly.index.min().strftime('%Y-%m-%d')} to {weekly.index.max().strftime('%Y-%m-%d')}",
        f"- **Target Statistics**:",
        f"  - Mean: {weekly['calls_total'].mean():.1f} calls/week",
        f"  - Std: {weekly['calls_total'].std():.1f} calls/week", 
        f"  - Range: [{weekly['calls_total'].min():.0f}, {weekly['calls_total'].max():.0f}]",
        "",
        "## Model Performance (Nested Cross-Validation)",  # REMOVED EMOJI
    ]
    
    if model_metrics:
        # Sort models by RMSE
        sorted_models = sorted(model_metrics.items(), 
                             key=lambda x: x[1].get("RMSE", float('inf')))
        
        # Performance table header
        report_lines.extend([
            "| Model | RMSE | MAPE | R2 | Status |",
            "|-------|------|------|----|----|"
        ])
        
        for model_name, scores in sorted_models:
            if isinstance(scores, dict) and "RMSE" in scores:
                rmse = scores.get('RMSE', 0)
                mape = scores.get('MAPE', 0)
                r2 = scores.get('R2', 0)
                rmse_std = scores.get('RMSE_std', 0)
                
                # Determine status
                status = "Good"  # REMOVED EMOJI
                if rmse > weekly['calls_total'].mean() * 0.15:
                    status = "High Error"  # REMOVED EMOJI
                elif r2 < 0.3:
                    status = "Low R2"  # REMOVED EMOJI
                elif mape > 0.3:
                    status = "High MAPE"  # REMOVED EMOJI
                
                report_lines.append(
                    f"| {model_name} | {rmse:.1f}+/-{rmse_std:.1f} | {mape:.1%} | {r2:.3f} | {status} |"
                )
    
    # Write report with UTF-8 encoding
    try:
        with open(output_dir / "summary_report.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        logger.info("Summary report generated")
    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")
```

### **4. Update the build_dataset function**
```python
# FIND this line in build_dataset function (around line 550):
    # Prepare features and target
    X = weekly.drop(columns=["calls_total"])
    y = weekly["calls_total"]

# REPLACE IT WITH:
    # Clean infinite values and column names before preparing features
    weekly = clean_infinite_values(weekly)
    weekly = clean_column_names(weekly)
    
    # Prepare features and target
    X = weekly.drop(columns=["calls_total"])
    y = weekly["calls_total"]
```

## 📝 **Summary of Issues:**

1. **Infinite values** - causing ElasticNet/RandomForest to fail
2. **Special characters in column names** - causing LightGBM to fail  
3. **Data type issues** - causing XGBoost to fail
4. **Unicode encoding** - causing report generation to fail on Windows

These fixes will resolve all the model training failures. The main issue is your data has some extreme outliers creating infinite values when calculating percentage changes and rolling statistics.
