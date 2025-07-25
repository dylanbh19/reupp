PS C:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel> C:\Users\BhungarD\python.exe "c:/Users/BhungarD/OneDrive - Computershare/Desktop/acdmodel/model_foundry.py"
================================================================================
COMPREHENSIVE MODEL TESTING SYSTEM
================================================================================
Start time: 2025-07-25 10:37:44
================================================================================
LOADING TRAINED MODEL
================================================================================
Model loaded: forest_simple
Features: 72
Model performance: R2=0.667

================================================================================
LOADING TEST DATA
================================================================================
Loaded data for 231 mail types
Average daily calls: 6877

================================================================================
RUNNING COMPREHENSIVE TEST SUITE
================================================================================

--- TEST 1: BASELINE PERFORMANCE ---
   Mean prediction: 9453
   Std deviation: 1374
   Range: 5572 - 12103
   Reasonable predictions: 100.0%

--- TEST 2: STRESS TESTS ---
   Mail volume x0.5: 8091 calls
   Mail volume x0.8: 8139 calls
   Mail volume x1.0: 8310 calls
   Mail volume x1.2: 10953 calls
   Mail volume x1.5: 11396 calls
   Mail volume x2.0: 11362 calls

--- TEST 3: EDGE CASE TESTS ---
   Zero mail: 7919 calls
   High mail (100k): 11406 calls
   Weekend: 8448 calls

--- TEST 4: TEMPORAL PATTERN TESTS ---
   Weekday patterns:
     Monday: 8400
     Tuesday: 8379
     Wednesday: 8310
     Thursday: 8231
     Friday: 8448
   Monthly patterns:
     Jan: 9470
     Apr: 9410
     Jul: 8310
     Oct: 7156
     Dec: 7156

--- TEST 5: STABILITY TESTS ---
   Base prediction: 8310
   Average change with 5% input noise: 0.4%
   Maximum change: 1.2%
   Model stability: STABLE

--- TEST 6: FEATURE IMPORTANCE ANALYSIS ---
   Top 10 most important features:
     calls_avg7: 0.278
     calls_avg3: 0.249
     month: 0.133
     calls_lag3: 0.027
     High_Dividend: 0.027
     REITs: 0.023
     calls_lag1: 0.021
     Dividend_Aristocrats: 0.016
     VIX: 0.015
     weekday: 0.012

================================================================================
TEST REPORT SUMMARY
================================================================================

1. BASELINE PERFORMANCE:
   - Average prediction: 9453 calls
   - Reasonable predictions: 100.0%

2. STRESS TEST RESULTS:
   - Scales appropriately: False

3. EDGE CASE HANDLING:
   - Zero mail prediction: 7919 calls
   - High mail prediction: 11406 calls

4. STABILITY:
   - Model stability: PASSED
   - Max deviation: 1.2%

================================================================================
OVERALL ASSESSMENT:
Tests passed: 3/4

MODEL STATUS: ACCEPTABLE WITH MONITORING

Detailed report saved to: model_test_results\test_report.json

Visualization saved to: model_test_results\test_results_visualization.png

Testing completed successfully!
End time: 2025-07-25 10:37:50
PS C:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel> 
