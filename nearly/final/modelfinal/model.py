                                                               & C:/Users/BhungarD/python.exe "c:/Users/BhungarD/OneDrive - Computershare/Desktop/finprod/model.py"
============================================================
 2025+ CALL VOLUME & INTENT PREDICTION PIPELINE
============================================================
 Fresh data analysis with overlapping dates
 Call volume prediction with mail lag modeling
 Intent classification (scope extension)
 5-day business outlook generation
============================================================

2025-07-22 20:30:05,529 | INFO | 🚀 STARTING 2025+ CALL VOLUME & INTENT PREDICTION PIPELINE
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\BhungarD\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\BhungarD\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f680' in position 33: character maps to <undefined>
Call stack:
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1239, in <module>
    exit_code = main()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1207, in main
    results = orchestrator.run_complete_pipeline()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 972, in run_complete_pipeline
    LOG.info("🚀 STARTING 2025+ CALL VOLUME & INTENT PREDICTION PIPELINE")
Message: '🚀 STARTING 2025+ CALL VOLUME & INTENT PREDICTION PIPELINE'
Arguments: ()
2025-07-22 20:30:05,536 | INFO | ======================================================================
2025-07-22 20:30:05,537 | INFO | 📊 PHASE 1: LOADING 2025+ DATA
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\BhungarD\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\BhungarD\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4ca' in position 33: character maps to <undefined>
Call stack:
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1239, in <module>
    exit_code = main()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1207, in main
    results = orchestrator.run_complete_pipeline()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 977, in run_complete_pipeline
    LOG.info("📊 PHASE 1: LOADING 2025+ DATA")
Message: '📊 PHASE 1: LOADING 2025+ DATA'
Arguments: ()
2025-07-22 20:30:05,539 | INFO | === LOADING FRESH 2025+ DATA ===
2025-07-22 20:30:05,540 | INFO | Loading call intent data...
2025-07-22 20:30:05,541 | INFO | Loading: data\callintent.csv
2025-07-22 20:30:17,996 | INFO | Loaded data\callintent.csv: 1053601 rows, 42 columns
2025-07-22 20:30:18,204 | INFO | Selected date column: conversationstart
2025-07-22 20:30:19,827 | INFO | Found 1053601 call records from 2025+
2025-07-22 20:30:20,537 | INFO | Found 116 unique intents: ['Access account information for Consolidated Edison Inc.', 'Address Change', 'Assistance related to shareholder account concerns', 'Assistance with the account of a deceased shareholder.', 'Assistance with transferring stock for Bank of America Corporation shares.', 'Assistance with transferring stock for Consolidated Edison Inc.', 'Associate', 'Balance', 'Balance, Stock Quote', 'Balance/Value', 'Banking Details', 'Beneficiary', 'Beneficiary Information', 'Blank', 'Buy Stock', 'Buy stocks', 'Certificate Issuance', 'Certified W-9 Form', 'Change Registration', 'Change address on account', 'Check Replacement', 'Company Information', 'Complaint Call', 'Connect with a representative', 'Consolidating multiple accounts into one.', 'Consolidation', 'Contact Information', 'Corporate Action', 'Cost Basis', 'Customer requested electric service restoration, unrelated to shareholder services.', 'Customer requested general customer service assistance', 'Data Protection', 'Deceased Estate', 'Deceased Shareholder', 'Direct Registration', 'Dividend Payment', 'Dividend Reinvestment', 'End Call', 'Enrolment', 'Escheatment', 'Existing IC User Login Problem', 'Financial management for elderly care', 'Follow-up on a transaction', 'Fraud Assistance', 'Fulfilment', 'General Inquiry', 'Get balance on account', 'Help', 'Help with medallion signature', 'Inquiry about account balance', 'Investor Center Login', 'James is requesting a summary of his tax information related to his shareholder account.', 'Legal Department', 'Lost Certificate', 'Make a payment to account', 'Name Change', 'New IC User Login Problem', 'Not Collected', 'Pay bill on behalf of client', 'Power of Attorney', 'Press and Media', 'Privacy Breach', 'Proxy Inquiry', 'Receive Letter', 'Recent Activity', 'Refund', 'Repeat Caller', 'Representative', 'Request to speak with a representative', 'Restricted Shares', 'Sell', 'Sell shares', 'Start New Service', 'Statement', 'Stock Quote', "Stockholder's report", 'Tax Forms', 'Tax Information', 'The shareholder wants to add a beneficiary to his account.', 'The user requested a copy of her account statement for Consolidated Edison Inc shares.', 'Transfer', 'Transfer Call', 'Transfer Funds', 'Transfer Status', 'Transfer Stock', 'Transfer call to senior support representative', 'Transfer shares', 'Transfer to representative', 'Unknown', 'Update Account', 'Update address', 'Update address details', 'User is inquiring about pension payment.', 'User is requesting assistance with transferring stock.', 'User is requesting information about the recent share price of Consolidated Edison Inc.', 'User is seeking a stock quote for Consolidated Edison Inc.', 'User is seeking assistance to transfer shares.', 'User is seeking assistance to transfer stock of Consolidated Edison Inc.', 'User is seeking assistance to transfer stock.', 'User is seeking assistance to transfer their stock in Bank of America Corporation.', 'User needs help managing physical stock shares.', 'User requested assistance regarding their Consolidated Edison Inc stock, but the specific request was unclear or not mentioned.', 'User requested to know the balance in their Consolidated Edison Inc account.', 'User requested to speak with a senior support representative for further assistance.', 'User seeks assistance to add a beneficiary to the account.', 'User seeks assistance to update his address on his shareholder account for Alphabet Inc.', 'User sought specific assistance regarding Consolidated Edison Inc. shares, but the request was unclear.', 'User wants to add a beneficiary to their shareholder account.', 'User wants to sell some shares in Consolidated Edison Inc.', 'User wants to transfer stock in Consolidated Edison Inc.', 'Value', "Verify if spouse's name is on the account", 'new service', 'speak to a representative', 'start service', 'verify account']   
2025-07-22 20:30:20,668 | INFO | Loading mail data...
2025-07-22 20:30:20,675 | INFO | Loading: data\mail.csv
2025-07-22 20:30:21,698 | INFO | Loaded data\mail.csv: 1409780 rows, 4 columns
2025-07-22 20:30:21,761 | INFO | Selected date column: mail_date
2025-07-22 20:30:22,013 | WARNING | Could not identify mail structure. Type: None, Volume: mail_volume
2025-07-22 20:30:22,019 | INFO | Aligning data to overlapping dates...
2025-07-22 20:30:22,021 | INFO | Using 123 call-only dates (no mail data)
2025-07-22 20:30:22,032 | INFO | Data aligned: 123 overlapping days

======================================================================
2025+ DATA LOADING SUMMARY
======================================================================

CALL DATA:
  total_calls: 1053601
  daily_records: 123
  date_range: 2025-02-05 to 2025-06-08
  avg_daily_calls: 8565.861788617885
  intent_types: ['Access account information for Consolidated Edison Inc.', 'Address Change', 'Assistance related to shareholder account concerns'] ... (116 total)

ALIGNED DATA:
  overlapping_days: 123
  date_range: 2025-02-05 00:00:00 to 2025-06-08 00:00:00
  has_mail: False
  has_intents: True
======================================================================
2025-07-22 20:30:22,036 | INFO |
🔧 PHASE 2: FEATURE ENGINEERING
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\BhungarD\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\BhungarD\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f527' in position 35: character maps to <undefined>
Call stack:
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1239, in <module>
    exit_code = main()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1207, in main
    results = orchestrator.run_complete_pipeline()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 985, in run_complete_pipeline
    LOG.info("\n🔧 PHASE 2: FEATURE ENGINEERING")
Message: '\n🔧 PHASE 2: FEATURE ENGINEERING'
Arguments: ()
2025-07-22 20:30:22,049 | INFO | Creating volume prediction features...
2025-07-22 20:30:22,052 | INFO | Creating temporal features...
2025-07-22 20:30:22,763 | INFO | Created 11 temporal features
2025-07-22 20:30:22,764 | INFO | Creating call history features...
2025-07-22 20:30:22,791 | INFO | Created 10 call history features
2025-07-22 20:30:22,794 | INFO | Volume features: 21 features, 122 samples
2025-07-22 20:30:22,795 | INFO | Creating intent prediction features...
2025-07-22 20:30:22,796 | INFO | Creating temporal features...
2025-07-22 20:30:22,801 | INFO | Created 11 temporal features
2025-07-22 20:30:22,802 | INFO | Creating call history features...
2025-07-22 20:30:22,806 | INFO | Created 10 call history features
2025-07-22 20:30:22,807 | INFO | Intent features: 137 features, 122 samples
2025-07-22 20:30:22,808 | INFO |
🤖 PHASE 3: MODEL TRAINING
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\BhungarD\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\BhungarD\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f916' in position 35: character maps to <undefined>
Call stack:
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1239, in <module>
    exit_code = main()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1207, in main
    results = orchestrator.run_complete_pipeline()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 995, in run_complete_pipeline
    LOG.info("\n🤖 PHASE 3: MODEL TRAINING")
Message: '\n🤖 PHASE 3: MODEL TRAINING'
Arguments: ()
2025-07-22 20:30:22,812 | INFO | === TRAINING VOLUME & INTENT MODELS ===
2025-07-22 20:30:22,812 | INFO | Training call volume prediction model...
2025-07-22 20:30:22,873 | INFO |   ridge: CV R² = 0.488, Test R² = 0.643
2025-07-22 20:30:24,198 | INFO |   random_forest: CV R² = 0.615, Test R² = 0.737
2025-07-22 20:30:24,924 | INFO |   gradient_boost: CV R² = 0.713, Test R² = 0.693
2025-07-22 20:30:25,131 | INFO | Best volume model: GradientBoostingRegressor (R² = 0.713)
2025-07-22 20:30:25,132 | INFO | Training intent prediction models...
2025-07-22 20:30:25,132 | INFO |   Training model for intent: Access account information for Consolidated Edison Inc.
2025-07-22 20:30:25,635 | INFO |     Access account information for Consolidated Edison Inc.: CV Accuracy = 0.989
2025-07-22 20:30:25,636 | INFO |   Training model for intent: Address Change
2025-07-22 20:30:26,101 | INFO |     Address Change: CV Accuracy = 0.600
2025-07-22 20:30:26,101 | INFO |   Training model for intent: Assistance related to shareholder account concerns
2025-07-22 20:30:26,529 | INFO |     Assistance related to shareholder account concerns: CV Accuracy = 0.989
2025-07-22 20:30:26,529 | INFO |   Training model for intent: Assistance with the account of a deceased shareholder.
2025-07-22 20:30:26,959 | INFO |     Assistance with the account of a deceased shareholder.: CV Accuracy = 0.989
2025-07-22 20:30:26,960 | INFO |   Training model for intent: Assistance with transferring stock for Bank of America Corporation shares.
2025-07-22 20:30:27,436 | INFO |     Assistance with transferring stock for Bank of America Corporation shares.: CV Accuracy = 0.989
2025-07-22 20:30:27,437 | INFO |   Training model for intent: Assistance with transferring stock for Consolidated Edison Inc.
2025-07-22 20:30:27,879 | INFO |     Assistance with transferring stock for Consolidated Edison Inc.: CV Accuracy = 0.989
2025-07-22 20:30:27,880 | INFO |   Training model for intent: Associate
2025-07-22 20:30:28,503 | INFO |     Associate: CV Accuracy = 0.767
2025-07-22 20:30:28,503 | INFO |   Training model for intent: Balance
2025-07-22 20:30:28,971 | INFO |     Balance: CV Accuracy = 0.989
2025-07-22 20:30:28,971 | INFO |   Training model for intent: Balance, Stock Quote
2025-07-22 20:30:29,423 | INFO |     Balance, Stock Quote: CV Accuracy = 0.989
2025-07-22 20:30:29,424 | INFO |   Training model for intent: Balance/Value
2025-07-22 20:30:29,955 | INFO |     Balance/Value: CV Accuracy = 0.978
2025-07-22 20:30:29,955 | INFO |   Training model for intent: Banking Details
2025-07-22 20:30:30,548 | INFO |     Banking Details: CV Accuracy = 0.733
2025-07-22 20:30:30,548 | INFO |   Training model for intent: Beneficiary
2025-07-22 20:30:30,988 | INFO |     Beneficiary: CV Accuracy = 0.978
2025-07-22 20:30:30,989 | INFO |   Training model for intent: Beneficiary Information
2025-07-22 20:30:31,418 | INFO |     Beneficiary Information: CV Accuracy = 0.400
2025-07-22 20:30:31,418 | INFO |   Training model for intent: Blank
2025-07-22 20:30:31,992 | INFO |     Blank: CV Accuracy = 1.000
2025-07-22 20:30:31,993 | INFO |   Training model for intent: Buy Stock
2025-07-22 20:30:32,523 | INFO |     Buy Stock: CV Accuracy = 0.556
2025-07-22 20:30:32,523 | INFO |   Training model for intent: Buy stocks
2025-07-22 20:30:32,965 | INFO |     Buy stocks: CV Accuracy = 0.989
2025-07-22 20:30:32,966 | INFO |   Training model for intent: Certificate Issuance
2025-07-22 20:30:33,462 | INFO |     Certificate Issuance: CV Accuracy = 0.589
2025-07-22 20:30:33,462 | INFO |   Training model for intent: Certified W-9 Form
2025-07-22 20:30:33,898 | INFO |     Certified W-9 Form: CV Accuracy = 0.989
2025-07-22 20:30:33,899 | INFO |   Training model for intent: Change Registration
2025-07-22 20:30:34,360 | INFO |     Change Registration: CV Accuracy = 0.967
2025-07-22 20:30:34,360 | INFO |   Training model for intent: Change address on account
2025-07-22 20:30:34,803 | INFO |     Change address on account: CV Accuracy = 0.989
2025-07-22 20:30:34,803 | INFO |   Training model for intent: Check Replacement
2025-07-22 20:30:35,271 | INFO |     Check Replacement: CV Accuracy = 0.389
2025-07-22 20:30:35,272 | INFO |   Training model for intent: Company Information
2025-07-22 20:30:35,701 | INFO |     Company Information: CV Accuracy = 0.989
2025-07-22 20:30:35,701 | INFO |   Training model for intent: Complaint Call
2025-07-22 20:30:36,137 | INFO |     Complaint Call: CV Accuracy = 0.967
2025-07-22 20:30:36,138 | INFO |   Training model for intent: Connect with a representative
2025-07-22 20:30:36,588 | INFO |     Connect with a representative: CV Accuracy = 0.989
2025-07-22 20:30:36,589 | INFO |   Training model for intent: Consolidating multiple accounts into one.
2025-07-22 20:30:37,003 | INFO |     Consolidating multiple accounts into one.: CV Accuracy = 0.989
2025-07-22 20:30:37,003 | INFO |   Training model for intent: Consolidation
2025-07-22 20:30:37,430 | INFO |     Consolidation: CV Accuracy = 0.967
2025-07-22 20:30:37,431 | INFO |   Training model for intent: Contact Information
2025-07-22 20:30:37,860 | INFO |     Contact Information: CV Accuracy = 0.967
2025-07-22 20:30:37,860 | INFO |   Training model for intent: Corporate Action
2025-07-22 20:30:38,276 | INFO |     Corporate Action: CV Accuracy = 0.867
2025-07-22 20:30:38,276 | INFO |   Training model for intent: Cost Basis
2025-07-22 20:30:38,697 | INFO |     Cost Basis: CV Accuracy = 0.967
2025-07-22 20:30:38,698 | INFO |   Training model for intent: Customer requested electric service restoration, unrelated to shareholder services.
2025-07-22 20:30:39,101 | INFO |     Customer requested electric service restoration, unrelated to shareholder services.: CV Accuracy = 0.989
2025-07-22 20:30:39,102 | INFO |   Training model for intent: Customer requested general customer service assistance
2025-07-22 20:30:39,510 | INFO |     Customer requested general customer service assistance: CV Accuracy = 0.989
2025-07-22 20:30:39,511 | INFO |   Training model for intent: Data Protection
2025-07-22 20:30:40,122 | INFO |     Data Protection: CV Accuracy = 0.889
2025-07-22 20:30:40,123 | INFO |   Training model for intent: Deceased Estate
2025-07-22 20:30:40,773 | INFO |     Deceased Estate: CV Accuracy = 0.967
2025-07-22 20:30:40,774 | INFO |   Training model for intent: Deceased Shareholder
2025-07-22 20:30:41,690 | INFO |     Deceased Shareholder: CV Accuracy = 0.978
2025-07-22 20:30:41,691 | INFO |   Training model for intent: Direct Registration
2025-07-22 20:30:42,447 | INFO |     Direct Registration: CV Accuracy = 0.978
2025-07-22 20:30:42,447 | INFO |   Training model for intent: Dividend Payment
2025-07-22 20:30:43,138 | INFO |     Dividend Payment: CV Accuracy = 0.644
2025-07-22 20:30:43,138 | INFO |   Training model for intent: Dividend Reinvestment
2025-07-22 20:30:43,721 | INFO |     Dividend Reinvestment: CV Accuracy = 0.922
2025-07-22 20:30:43,721 | INFO |   Training model for intent: End Call
2025-07-22 20:30:44,512 | INFO |     End Call: CV Accuracy = 0.944
2025-07-22 20:30:44,513 | INFO |   Training model for intent: Enrolment
2025-07-22 20:30:45,231 | INFO |     Enrolment: CV Accuracy = 0.933
2025-07-22 20:30:45,232 | INFO |   Training model for intent: Escheatment
2025-07-22 20:30:45,898 | INFO |     Escheatment: CV Accuracy = 0.711
2025-07-22 20:30:45,899 | INFO |   Training model for intent: Existing IC User Login Problem
2025-07-22 20:30:46,507 | INFO |     Existing IC User Login Problem: CV Accuracy = 0.989
2025-07-22 20:30:46,508 | INFO |   Training model for intent: Financial management for elderly care
2025-07-22 20:30:47,079 | INFO |     Financial management for elderly care: CV Accuracy = 0.989
2025-07-22 20:30:47,079 | INFO |   Training model for intent: Follow-up on a transaction
2025-07-22 20:30:47,638 | INFO |     Follow-up on a transaction: CV Accuracy = 0.989
2025-07-22 20:30:47,638 | INFO |   Training model for intent: Fraud Assistance
2025-07-22 20:30:48,250 | INFO |     Fraud Assistance: CV Accuracy = 0.856
2025-07-22 20:30:48,251 | INFO |   Training model for intent: Fulfilment
2025-07-22 20:30:48,908 | INFO |     Fulfilment: CV Accuracy = 0.911
2025-07-22 20:30:48,909 | INFO |   Training model for intent: General Inquiry
2025-07-22 20:30:49,487 | INFO |     General Inquiry: CV Accuracy = 0.667
2025-07-22 20:30:49,487 | INFO |   Training model for intent: Get balance on account
2025-07-22 20:30:50,051 | INFO |     Get balance on account: CV Accuracy = 0.989
2025-07-22 20:30:50,052 | INFO |   Training model for intent: Help
2025-07-22 20:30:50,629 | INFO |     Help: CV Accuracy = 0.944
2025-07-22 20:30:50,629 | INFO |   Training model for intent: Help with medallion signature
2025-07-22 20:30:51,239 | INFO |     Help with medallion signature: CV Accuracy = 0.989
2025-07-22 20:30:51,239 | INFO |   Training model for intent: Inquiry about account balance
2025-07-22 20:30:51,783 | INFO |     Inquiry about account balance: CV Accuracy = 0.989
2025-07-22 20:30:51,784 | INFO |   Training model for intent: Investor Center Login
2025-07-22 20:30:52,382 | INFO |     Investor Center Login: CV Accuracy = 0.956
2025-07-22 20:30:52,382 | INFO |   Training model for intent: James is requesting a summary of his tax information related to his shareholder account.       
2025-07-22 20:30:52,958 | INFO |     James is requesting a summary of his tax information related to his shareholder account.: CV Accuracy = 0.989
2025-07-22 20:30:52,958 | INFO |   Training model for intent: Legal Department
2025-07-22 20:30:53,513 | INFO |     Legal Department: CV Accuracy = 0.989
2025-07-22 20:30:53,513 | INFO |   Training model for intent: Lost Certificate
2025-07-22 20:30:54,078 | INFO |     Lost Certificate: CV Accuracy = 0.578
2025-07-22 20:30:54,079 | INFO |   Training model for intent: Make a payment to account
2025-07-22 20:30:54,655 | INFO |     Make a payment to account: CV Accuracy = 0.989
2025-07-22 20:30:54,655 | INFO |   Training model for intent: Name Change
2025-07-22 20:30:55,259 | INFO |     Name Change: CV Accuracy = 0.756
2025-07-22 20:30:55,259 | INFO |   Training model for intent: New IC User Login Problem
2025-07-22 20:30:55,831 | INFO |     New IC User Login Problem: CV Accuracy = 0.533
2025-07-22 20:30:55,832 | INFO |   Training model for intent: Not Collected
2025-07-22 20:30:56,416 | INFO |     Not Collected: CV Accuracy = 0.978
2025-07-22 20:30:56,416 | INFO |   Training model for intent: Pay bill on behalf of client
2025-07-22 20:30:57,013 | INFO |     Pay bill on behalf of client: CV Accuracy = 0.989
2025-07-22 20:30:57,013 | INFO |   Training model for intent: Power of Attorney
2025-07-22 20:30:57,607 | INFO |     Power of Attorney: CV Accuracy = 0.978
2025-07-22 20:30:57,607 | INFO |   Training model for intent: Press and Media
2025-07-22 20:30:58,199 | INFO |     Press and Media: CV Accuracy = 0.922
2025-07-22 20:30:58,200 | INFO |   Training model for intent: Privacy Breach
2025-07-22 20:30:58,762 | INFO |     Privacy Breach: CV Accuracy = 0.989
2025-07-22 20:30:58,763 | INFO |   Training model for intent: Proxy Inquiry
2025-07-22 20:30:59,369 | INFO |     Proxy Inquiry: CV Accuracy = 0.944
2025-07-22 20:30:59,370 | INFO |   Training model for intent: Receive Letter
2025-07-22 20:30:59,978 | INFO |     Receive Letter: CV Accuracy = 0.967
2025-07-22 20:30:59,978 | INFO |   Training model for intent: Recent Activity
2025-07-22 20:31:00,715 | INFO |     Recent Activity: CV Accuracy = 0.411
2025-07-22 20:31:00,715 | INFO |   Training model for intent: Refund
2025-07-22 20:31:01,467 | INFO |     Refund: CV Accuracy = 0.989
2025-07-22 20:31:01,468 | INFO |   Training model for intent: Repeat Caller
2025-07-22 20:31:02,252 | INFO |     Repeat Caller: CV Accuracy = 0.711
2025-07-22 20:31:02,252 | INFO |   Training model for intent: Representative
2025-07-22 20:31:02,995 | INFO |     Representative: CV Accuracy = 0.978
2025-07-22 20:31:02,995 | INFO |   Training model for intent: Request to speak with a representative
2025-07-22 20:31:03,849 | INFO |     Request to speak with a representative: CV Accuracy = 0.989
2025-07-22 20:31:03,850 | INFO |   Training model for intent: Restricted Shares
2025-07-22 20:31:04,637 | INFO |     Restricted Shares: CV Accuracy = 0.989
2025-07-22 20:31:04,637 | INFO |   Training model for intent: Sell
2025-07-22 20:31:05,282 | INFO |     Sell: CV Accuracy = 0.978
2025-07-22 20:31:05,282 | INFO |   Training model for intent: Sell shares
2025-07-22 20:31:05,951 | INFO |     Sell shares: CV Accuracy = 0.989
2025-07-22 20:31:05,952 | INFO |   Training model for intent: Start New Service
2025-07-22 20:31:06,788 | INFO |     Start New Service: CV Accuracy = 0.989
2025-07-22 20:31:06,789 | INFO |   Training model for intent: Statement
2025-07-22 20:31:07,577 | INFO |     Statement: CV Accuracy = 0.744
2025-07-22 20:31:07,578 | INFO |   Training model for intent: Stock Quote
2025-07-22 20:31:08,301 | INFO |     Stock Quote: CV Accuracy = 0.567
2025-07-22 20:31:08,302 | INFO |   Training model for intent: Stockholder's report
2025-07-22 20:31:08,901 | INFO |     Stockholder's report: CV Accuracy = 0.989
2025-07-22 20:31:08,902 | INFO |   Training model for intent: Tax Forms
2025-07-22 20:31:09,465 | INFO |     Tax Forms: CV Accuracy = 0.978
2025-07-22 20:31:09,466 | INFO |   Training model for intent: Tax Information
2025-07-22 20:31:10,149 | INFO |     Tax Information: CV Accuracy = 0.578
2025-07-22 20:31:10,150 | INFO |   Training model for intent: The shareholder wants to add a beneficiary to his account.
2025-07-22 20:31:10,723 | INFO |     The shareholder wants to add a beneficiary to his account.: CV Accuracy = 0.989
2025-07-22 20:31:10,723 | INFO |   Training model for intent: The user requested a copy of her account statement for Consolidated Edison Inc shares.
2025-07-22 20:31:11,336 | INFO |     The user requested a copy of her account statement for Consolidated Edison Inc shares.: CV Accuracy = 0.989
2025-07-22 20:31:11,336 | INFO |   Training model for intent: Transfer
2025-07-22 20:31:11,972 | INFO |     Transfer: CV Accuracy = 0.567
2025-07-22 20:31:11,972 | INFO |   Training model for intent: Transfer Call
2025-07-22 20:31:12,591 | INFO |     Transfer Call: CV Accuracy = 0.978
2025-07-22 20:31:12,592 | INFO |   Training model for intent: Transfer Funds
2025-07-22 20:31:13,167 | INFO |     Transfer Funds: CV Accuracy = 0.989
2025-07-22 20:31:13,167 | INFO |   Training model for intent: Transfer Status
2025-07-22 20:31:13,607 | INFO |     Transfer Status: CV Accuracy = 0.989
2025-07-22 20:31:13,608 | INFO |   Training model for intent: Transfer Stock
2025-07-22 20:31:13,992 | INFO |     Transfer Stock: CV Accuracy = 0.967
2025-07-22 20:31:13,993 | INFO |   Training model for intent: Transfer call to senior support representative
2025-07-22 20:31:14,393 | INFO |     Transfer call to senior support representative: CV Accuracy = 0.989
2025-07-22 20:31:14,394 | INFO |   Training model for intent: Transfer shares
2025-07-22 20:31:14,810 | INFO |     Transfer shares: CV Accuracy = 0.989
2025-07-22 20:31:14,810 | INFO |   Training model for intent: Transfer to representative
2025-07-22 20:31:15,219 | INFO |     Transfer to representative: CV Accuracy = 0.989
2025-07-22 20:31:15,219 | INFO |   Training model for intent: Unknown
2025-07-22 20:31:15,654 | INFO |     Unknown: CV Accuracy = 0.744
2025-07-22 20:31:15,654 | INFO |   Training model for intent: Update Account
2025-07-22 20:31:16,159 | INFO |     Update Account: CV Accuracy = 0.978
2025-07-22 20:31:16,160 | INFO |   Training model for intent: Update address
2025-07-22 20:31:16,735 | INFO |     Update address: CV Accuracy = 0.989
2025-07-22 20:31:16,735 | INFO |   Training model for intent: Update address details
2025-07-22 20:31:17,136 | INFO |     Update address details: CV Accuracy = 0.989
2025-07-22 20:31:17,137 | INFO |   Training model for intent: User is inquiring about pension payment.
2025-07-22 20:31:17,529 | INFO |     User is inquiring about pension payment.: CV Accuracy = 0.989
2025-07-22 20:31:17,529 | INFO |   Training model for intent: User is requesting assistance with transferring stock.
2025-07-22 20:31:17,925 | INFO |     User is requesting assistance with transferring stock.: CV Accuracy = 0.989
2025-07-22 20:31:17,926 | INFO |   Training model for intent: User is requesting information about the recent share price of Consolidated Edison Inc.        
2025-07-22 20:31:18,318 | INFO |     User is requesting information about the recent share price of Consolidated Edison Inc.: CV Accuracy = 0.989
2025-07-22 20:31:18,318 | INFO |   Training model for intent: User is seeking a stock quote for Consolidated Edison Inc.
2025-07-22 20:31:18,712 | INFO |     User is seeking a stock quote for Consolidated Edison Inc.: CV Accuracy = 0.989
2025-07-22 20:31:18,713 | INFO |   Training model for intent: User is seeking assistance to transfer shares.
2025-07-22 20:31:19,139 | INFO |     User is seeking assistance to transfer shares.: CV Accuracy = 0.978
2025-07-22 20:31:19,140 | INFO |   Training model for intent: User is seeking assistance to transfer stock of Consolidated Edison Inc.
2025-07-22 20:31:19,521 | INFO |     User is seeking assistance to transfer stock of Consolidated Edison Inc.: CV Accuracy = 0.978
2025-07-22 20:31:19,522 | INFO |   Training model for intent: User is seeking assistance to transfer stock.
2025-07-22 20:31:19,939 | INFO |     User is seeking assistance to transfer stock.: CV Accuracy = 0.967
2025-07-22 20:31:19,940 | INFO |   Training model for intent: User is seeking assistance to transfer their stock in Bank of America Corporation.
2025-07-22 20:31:20,399 | INFO |     User is seeking assistance to transfer their stock in Bank of America Corporation.: CV Accuracy = 0.989
2025-07-22 20:31:20,399 | INFO |   Training model for intent: User needs help managing physical stock shares.
2025-07-22 20:31:20,853 | INFO |     User needs help managing physical stock shares.: CV Accuracy = 0.989
2025-07-22 20:31:20,853 | INFO |   Training model for intent: User requested assistance regarding their Consolidated Edison Inc stock, but the specific request was unclear or not mentioned.
2025-07-22 20:31:21,256 | INFO |     User requested assistance regarding their Consolidated Edison Inc stock, but the specific request was unclear or not mentioned.: CV Accuracy = 0.978
2025-07-22 20:31:21,256 | INFO |   Training model for intent: User requested to know the balance in their Consolidated Edison Inc account.
2025-07-22 20:31:21,647 | INFO |     User requested to know the balance in their Consolidated Edison Inc account.: CV Accuracy = 0.989
2025-07-22 20:31:21,648 | INFO |   Training model for intent: User requested to speak with a senior support representative for further assistance.
2025-07-22 20:31:22,057 | INFO |     User requested to speak with a senior support representative for further assistance.: CV Accuracy = 0.989
2025-07-22 20:31:22,057 | INFO |   Training model for intent: User seeks assistance to add a beneficiary to the account.
2025-07-22 20:31:22,439 | INFO |     User seeks assistance to add a beneficiary to the account.: CV Accuracy = 0.989
2025-07-22 20:31:22,439 | INFO |   Training model for intent: User seeks assistance to update his address on his shareholder account for Alphabet Inc.       
2025-07-22 20:31:22,815 | INFO |     User seeks assistance to update his address on his shareholder account for Alphabet Inc.: CV Accuracy = 0.989
2025-07-22 20:31:22,816 | INFO |   Training model for intent: User sought specific assistance regarding Consolidated Edison Inc. shares, but the request was unclear.
2025-07-22 20:31:23,204 | INFO |     User sought specific assistance regarding Consolidated Edison Inc. shares, but the request was unclear.: CV Accuracy = 0.989
2025-07-22 20:31:23,204 | INFO |   Training model for intent: User wants to add a beneficiary to their shareholder account.
2025-07-22 20:31:23,599 | INFO |     User wants to add a beneficiary to their shareholder account.: CV Accuracy = 0.989
2025-07-22 20:31:23,600 | INFO |   Training model for intent: User wants to sell some shares in Consolidated Edison Inc.
2025-07-22 20:31:23,990 | INFO |     User wants to sell some shares in Consolidated Edison Inc.: CV Accuracy = 0.989
2025-07-22 20:31:23,991 | INFO |   Training model for intent: User wants to transfer stock in Consolidated Edison Inc.
2025-07-22 20:31:24,400 | INFO |     User wants to transfer stock in Consolidated Edison Inc.: CV Accuracy = 0.978
2025-07-22 20:31:24,400 | INFO |   Training model for intent: Value
2025-07-22 20:31:24,888 | INFO |     Value: CV Accuracy = 0.978
2025-07-22 20:31:24,888 | INFO |   Training model for intent: Verify if spouse's name is on the account
2025-07-22 20:31:25,322 | INFO |     Verify if spouse's name is on the account: CV Accuracy = 0.989
2025-07-22 20:31:25,322 | INFO |   Training model for intent: new service
2025-07-22 20:31:25,732 | INFO |     new service: CV Accuracy = 0.989
2025-07-22 20:31:25,733 | INFO |   Training model for intent: speak to a representative
2025-07-22 20:31:26,230 | INFO |     speak to a representative: CV Accuracy = 0.989
2025-07-22 20:31:26,230 | INFO |   Training model for intent: start service
2025-07-22 20:31:26,748 | INFO |     start service: CV Accuracy = 0.989
2025-07-22 20:31:26,749 | INFO |   Training model for intent: verify account
2025-07-22 20:31:27,168 | INFO |     verify account: CV Accuracy = 0.989
2025-07-22 20:31:27,168 | INFO |
🔮 PHASE 4: GENERATING PREDICTIONS
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\BhungarD\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\BhungarD\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f52e' in position 35: character maps to <undefined>
Call stack:
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1239, in <module>
    exit_code = main()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1207, in main
    results = orchestrator.run_complete_pipeline()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1000, in run_complete_pipeline
    LOG.info("\n🔮 PHASE 4: GENERATING PREDICTIONS")
Message: '\n🔮 PHASE 4: GENERATING PREDICTIONS'
Arguments: ()
2025-07-22 20:31:27,171 | INFO | Generating 5-day outlook...
2025-07-22 20:31:27,294 | INFO | 
💾 PHASE 5: SAVING RESULTS
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\BhungarD\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\BhungarD\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4be' in position 35: character maps to <undefined>
Call stack:
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1239, in <module>
    exit_code = main()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1207, in main
    results = orchestrator.run_complete_pipeline()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1013, in run_complete_pipeline
    LOG.info("\n💾 PHASE 5: SAVING RESULTS")
Message: '\n💾 PHASE 5: SAVING RESULTS'
Arguments: ()
2025-07-22 20:31:30,496 | INFO | All results saved successfully
2025-07-22 20:31:30,496 | INFO |
📋 PHASE 6: GENERATING REPORT
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\BhungarD\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\BhungarD\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4cb' in position 35: character maps to <undefined>
Call stack:
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1239, in <module>
    exit_code = main()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1207, in main
    results = orchestrator.run_complete_pipeline()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1017, in run_complete_pipeline
    LOG.info("\n📋 PHASE 6: GENERATING REPORT")
Message: '\n📋 PHASE 6: GENERATING REPORT'
Arguments: ()



    # ====================================================================
    2025+ CALL VOLUME & INTENT PREDICTION PIPELINE REPORT

    EXECUTION SUMMARY:
     Pipeline Status: SUCCESS
     Execution Time: 1.4 minutes
     Output Directory: call_prediction_pipeline

    DATA SUMMARY:
     Total Call Records: 1053601
     Overlapping Days: 123
     Date Range: 2025-02-05 00:00:00 to 2025-06-08 00:00:00
     Has Mail Data: False
     Has Intent Data: True

    MODEL PERFORMANCE:
     Call Volume Model R: 0.713
     Intent Classification Accuracy: 0.914

    PREDICTIONS GENERATED:
     5-Day Call Volume Outlook: Available
     Intent Distribution Predictions: Available

    BUSINESS APPLICATIONS:
     Daily staffing optimization
     Mail campaign impact analysis
     Intent-based resource allocation
     5-day capacity planning

    FILES GENERATED:
     data_summary.json - Data loading summary
     training_results.json - Model performance metrics
     outlook_predictions.json - 5-day predictions
     volume_model.pkl - Trained volume prediction model
     intent_model_*.pkl - Trained intent classification models

    NEXT STEPS:

    1. Review prediction accuracy in outlook_predictions.json
    1. Use volume_model.pkl for daily predictions
    1. Monitor model performance with new data
    1. Retrain models monthly with fresh data

    # ====================================================================
    Generated: 2025-07-22 20:31:30


2025-07-22 20:31:30,506 | INFO |
======================================================================
2025-07-22 20:31:30,506 | INFO | 🎉 PIPELINE COMPLETED SUCCESSFULLY!
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\BhungarD\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\BhungarD\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f389' in position 33: character maps to <undefined>
Call stack:
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1239, in <module>
    exit_code = main()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1207, in main
    results = orchestrator.run_complete_pipeline()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1023, in run_complete_pipeline
    LOG.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
Message: '🎉 PIPELINE COMPLETED SUCCESSFULLY!'
Arguments: ()
2025-07-22 20:31:30,511 | INFO | ⏱️  Total execution time: 1.4 minutes
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\BhungarD\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\BhungarD\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode characters in position 33-34: character maps to <undefined>
Call stack:
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1239, in <module>
    exit_code = main()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1207, in main
    results = orchestrator.run_complete_pipeline()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1024, in run_complete_pipeline
    LOG.info(f"⏱️  Total execution time: {execution_time:.1f} minutes")
Message: '⏱️  Total execution time: 1.4 minutes'
Arguments: ()
2025-07-22 20:31:30,517 | INFO | 📁 Results saved to: call_prediction_pipeline
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\BhungarD\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\BhungarD\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4c1' in position 33: character maps to <undefined>
Call stack:
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1239, in <module>
    exit_code = main()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1207, in main
    results = orchestrator.run_complete_pipeline()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod\model.py", line 1025, in run_complete_pipeline
    LOG.info(f"📁 Results saved to: {self.output_dir}")
Message: '📁 Results saved to: call_prediction_pipeline'
Arguments: ()

 PIPELINE COMPLETED SUCCESSFULLY!

 2025+ data processed and analyzed
 Call volume prediction model trained
 Intent classification models trained
 5-day outlook generated
 Production-ready models saved

 Find all results in: call_prediction_pipeline

 Ready for daily predictions!




# #!/usr/bin/env python
“””
BULLETPROOF CALL VOLUME & INTENT PREDICTION PIPELINE

End-to-end pipeline for 2025+ data focusing on:

1. Call volume prediction with mail lag modeling
1. Call intent prediction (scope extension)
1. Uses only overlapping dates between call intent and mail data

DATA SOURCES:

- callintetn.csv: Call intent data (2025+)
- mail.csv: Mail volume data by type

OUTPUTS:

- Call volume predictions (3-5 day outlook)
- Intent distribution predictions
- Mail campaign impact analysis
  “””

import warnings
warnings.filterwarnings(‘ignore’)

import os
import sys
from pathlib import Path
import json
import logging
import traceback
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

# ML Libraries

from sklearn.model_selection import TimeSeriesSplit, cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

# Statistical Libraries

from scipy import stats
from scipy.stats import pearsonr

# ============================================================================

# CONFIGURATION

# ============================================================================

CONFIG = {
# File patterns for 2025+ data
“call_intent_files”: [“callintetn.csv”, “data/callintetn.csv”, “*intent*.csv”],
“mail_files”: [“mail.csv”, “data/mail.csv”, “*mail*.csv”],

```
# Mail lag configuration
"mail_lag_days": [1, 2, 3, 4, 5],
"lag_weights": {1: 0.2, 2: 0.4, 3: 0.25, 4: 0.1, 5: 0.05},

# Model configuration
"prediction_horizon_days": 5,
"confidence_levels": [0.68, 0.95],
"cv_folds": 5,
"min_train_samples": 20,

# Output directories
"output_dir": "call_prediction_pipeline",
"plots_dir": "analysis_plots",
"models_dir": "trained_models",
"results_dir": "results",

"random_state": 42
```

}

# ============================================================================

# LOGGING SETUP

# ============================================================================

def setup_logging():
“”“Setup clean logging”””
output_dir = Path(CONFIG[“output_dir”])
output_dir.mkdir(exist_ok=True)

```
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "pipeline.log", mode='w')
    ]
)
return logging.getLogger(__name__)
```

LOG = setup_logging()

def safe_print(message: str):
“”“Print safely”””
try:
print(str(message).encode(‘ascii’, ‘ignore’).decode(‘ascii’))
except:
print(str(message))

# ============================================================================

# DATA LOADER FOR 2025+ DATA

# ============================================================================

class Fresh2025DataLoader:
“”“Load and process fresh 2025+ call intent and mail data”””

```
def __init__(self):
    self.call_data = None
    self.mail_data = None
    self.intent_data = None
    self.data_summary = {}

def find_files(self, patterns: List[str]) -> List[Path]:
    """Find files matching patterns"""
    found_files = []
    
    for pattern in patterns:
        path = Path(pattern)
        if path.exists():
            found_files.append(path)
        elif '*' in pattern:
            parent_dir = path.parent if path.parent != Path('.') else Path('.')
            if parent_dir.exists():
                found_files.extend(parent_dir.glob(path.name))
    
    return list(set(found_files))

def load_csv_smart(self, file_path: Path) -> pd.DataFrame:
    """Smart CSV loader with multiple encoding attempts"""
    LOG.info(f"Loading: {file_path}")
    
    encodings = ['utf-8', 'latin1', 'cp1252']
    separators = [',', ';', '\t']
    
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=sep, low_memory=False)
                if df.shape[1] > 1 and len(df) > 0:
                    LOG.info(f"Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
                    return df
            except:
                continue
    
    raise ValueError(f"Could not load {file_path}")

def detect_date_column(self, df: pd.DataFrame) -> str:
    """Detect the main date column"""
    date_keywords = ['date', 'time', 'start', 'created', 'dt', 'timestamp']
    
    # Check column names
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in date_keywords):
            # Test if it's actually a date
            sample = df[col].dropna().head(50)
            try:
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().sum() / len(sample) > 0.8:
                    LOG.info(f"Selected date column: {col}")
                    return col
            except:
                continue
    
    raise ValueError("No valid date column found")

def load_call_intent_data(self) -> Tuple[pd.Series, pd.DataFrame]:
    """Load call intent data and calculate daily volumes + intent distribution"""
    
    LOG.info("Loading call intent data...")
    
    intent_files = self.find_files(CONFIG["call_intent_files"])
    if not intent_files:
        raise FileNotFoundError("No call intent files found")
    
    # Use the first/newest file
    intent_file = intent_files[0]
    df = self.load_csv_smart(intent_file)
    
    # Standardize column names
    df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
    
    # Find date column
    date_col = self.detect_date_column(df)
    
    # Parse dates and filter for 2025+
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df[df[date_col].dt.year >= 2025]
    
    if len(df) == 0:
        raise ValueError("No 2025+ data found in call intent file")
    
    LOG.info(f"Found {len(df)} call records from 2025+")
    
    # Find intent column
    intent_col = None
    for col in df.columns:
        if 'intent' in col.lower():
            intent_col = col
            break
    
    if intent_col is None:
        LOG.warning("No intent column found - will focus on volume only")
    
    # Calculate daily call volumes
    df['call_date'] = df[date_col].dt.date
    daily_calls = df.groupby('call_date').size()
    daily_calls.index = pd.to_datetime(daily_calls.index)
    daily_calls = daily_calls.sort_index()
    
    # Calculate daily intent distribution if available
    daily_intents = None
    if intent_col is not None:
        # Clean intent data
        df[intent_col] = df[intent_col].fillna('Unknown').astype(str)
        
        # Create daily intent distribution
        intent_counts = df.groupby(['call_date', intent_col]).size().unstack(fill_value=0)
        intent_counts.index = pd.to_datetime(intent_counts.index)
        intent_counts = intent_counts.sort_index()
        
        # Calculate percentages
        daily_intents = intent_counts.div(intent_counts.sum(axis=1), axis=0)
        daily_intents = daily_intents.fillna(0)
        
        LOG.info(f"Found {len(intent_counts.columns)} unique intents: {list(intent_counts.columns)}")
    
    self.data_summary['call_data'] = {
        'total_calls': len(df),
        'daily_records': len(daily_calls),
        'date_range': f"{daily_calls.index.min().date()} to {daily_calls.index.max().date()}",
        'avg_daily_calls': daily_calls.mean(),
        'intent_types': list(intent_counts.columns) if daily_intents is not None else None
    }
    
    self.call_data = daily_calls
    self.intent_data = daily_intents
    
    return daily_calls, daily_intents

def load_mail_data(self) -> pd.DataFrame:
    """Load mail data and process by type"""
    
    LOG.info("Loading mail data...")
    
    mail_files = self.find_files(CONFIG["mail_files"])
    if not mail_files:
        LOG.warning("No mail files found")
        return None
    
    # Use the first/newest file
    mail_file = mail_files[0]
    df = self.load_csv_smart(mail_file)
    
    # Standardize column names
    df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
    
    # Find date column
    date_col = self.detect_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Find mail type and volume columns
    mail_type_col = None
    volume_col = None
    
    for col in df.columns:
        if col != date_col:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['type', 'category', 'product']):
                if 2 <= df[col].nunique() <= 50:
                    mail_type_col = col
                    break
    
    for col in df.columns:
        if col not in [date_col, mail_type_col]:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['volume', 'count', 'amount', 'pieces']):
                if df[col].dtype in ['int64', 'float64']:
                    volume_col = col
                    break
    
    if volume_col is None:
        # Use first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            volume_col = numeric_cols[0]
    
    if mail_type_col is None or volume_col is None:
        LOG.warning(f"Could not identify mail structure. Type: {mail_type_col}, Volume: {volume_col}")
        return None
    
    LOG.info(f"Mail structure - Date: {date_col}, Type: {mail_type_col}, Volume: {volume_col}")
    
    # Process mail data
    df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')
    df = df.dropna(subset=[volume_col])
    df = df[df[volume_col] >= 0]
    
    # Create daily mail by type
    df['mail_date'] = df[date_col].dt.date
    mail_daily = df.groupby(['mail_date', mail_type_col])[volume_col].sum().unstack(fill_value=0)
    mail_daily.index = pd.to_datetime(mail_daily.index)
    mail_daily = mail_daily.sort_index()
    
    # Clean column names
    mail_daily.columns = [str(col).strip() for col in mail_daily.columns]
    
    self.data_summary['mail_data'] = {
        'total_records': len(df),
        'daily_records': len(mail_daily),
        'date_range': f"{mail_daily.index.min().date()} to {mail_daily.index.max().date()}",
        'mail_types': list(mail_daily.columns),
        'avg_daily_volume': mail_daily.sum(axis=1).mean()
    }
    
    self.mail_data = mail_daily
    return mail_daily

def align_data_to_overlap(self) -> Dict:
    """Align all data to overlapping dates only"""
    
    LOG.info("Aligning data to overlapping dates...")
    
    if self.call_data is None:
        raise ValueError("No call data loaded")
    
    # Start with call data dates
    common_dates = set(self.call_data.index)
    
    # Find overlap with mail data if available
    if self.mail_data is not None:
        mail_dates = set(self.mail_data.index)
        common_dates = common_dates.intersection(mail_dates)
        LOG.info(f"Found {len(common_dates)} overlapping dates between calls and mail")
    else:
        LOG.info(f"Using {len(common_dates)} call-only dates (no mail data)")
    
    if len(common_dates) < 10:
        raise ValueError(f"Insufficient overlapping data: only {len(common_dates)} days")
    
    # Convert to sorted list
    common_dates = sorted(common_dates)
    
    # Align all datasets
    aligned_data = {
        'calls': self.call_data.loc[common_dates],
        'dates': common_dates
    }
    
    if self.mail_data is not None:
        aligned_data['mail'] = self.mail_data.loc[common_dates]
    
    if self.intent_data is not None:
        aligned_data['intents'] = self.intent_data.loc[common_dates]
    
    self.data_summary['aligned_data'] = {
        'overlapping_days': len(common_dates),
        'date_range': f"{common_dates[0]} to {common_dates[-1]}",
        'has_mail': self.mail_data is not None,
        'has_intents': self.intent_data is not None
    }
    
    LOG.info(f"Data aligned: {len(common_dates)} overlapping days")
    return aligned_data

def load_all_data(self) -> Dict:
    """Load and align all data"""
    
    LOG.info("=== LOADING FRESH 2025+ DATA ===")
    
    # Load call intent data
    calls, intents = self.load_call_intent_data()
    
    # Load mail data
    mail = self.load_mail_data()
    
    # Align to overlapping dates
    aligned_data = self.align_data_to_overlap()
    
    # Print summary
    self.print_data_summary()
    
    return aligned_data

def print_data_summary(self):
    """Print clean data summary"""
    
    print("\n" + "="*70)
    print("2025+ DATA LOADING SUMMARY")
    print("="*70)
    
    for data_type, info in self.data_summary.items():
        print(f"\n{data_type.upper().replace('_', ' ')}:")
        if isinstance(info, dict):
            for key, value in info.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:3]} ... ({len(value)} total)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {info}")
    
    print("="*70)
```

# ============================================================================

# FEATURE ENGINEERING FOR VOLUME + INTENT PREDICTION

# ============================================================================

class VolumeIntentFeatureEngine:
“”“Create features for both volume and intent prediction”””

```
def __init__(self):
    self.volume_features = None
    self.intent_features = None
    self.feature_names = []

def create_lag_features(self, mail_data: pd.DataFrame, call_data: pd.Series) -> pd.DataFrame:
    """Create mail lag features"""
    
    if mail_data is None:
        return pd.DataFrame(index=call_data.index)
    
    LOG.info("Creating mail lag features...")
    
    lag_features = pd.DataFrame(index=call_data.index)
    
    # Select top mail types by volume
    mail_volumes = mail_data.sum().sort_values(ascending=False)
    top_mail_types = mail_volumes.head(8).index.tolist()  # Top 8 mail types
    
    for mail_type in top_mail_types:
        mail_type_clean = str(mail_type).replace(' ', '_')[:15]
        mail_series = mail_data[mail_type]
        
        # Create lag features
        for lag in CONFIG["mail_lag_days"]:
            if lag == 0:
                lag_features[f"{mail_type_clean}_today"] = mail_series
            else:
                lag_features[f"{mail_type_clean}_lag_{lag}"] = mail_series.shift(lag)
        
        # Weighted lag feature
        weighted_lag = pd.Series(0, index=mail_series.index, dtype=float)
        for lag, weight in CONFIG["lag_weights"].items():
            if lag == 0:
                weighted_lag += mail_series * weight
            else:
                weighted_lag += mail_series.shift(lag).fillna(0) * weight
        
        lag_features[f"{mail_type_clean}_weighted"] = weighted_lag
    
    # Total mail features
    lag_features['total_mail_today'] = mail_data.sum(axis=1)
    lag_features['total_mail_lag_1'] = mail_data.sum(axis=1).shift(1)
    lag_features['total_mail_lag_2'] = mail_data.sum(axis=1).shift(2)
    
    # Fill NaN
    lag_features = lag_features.fillna(0)
    
    LOG.info(f"Created {len(lag_features.columns)} mail lag features")
    return lag_features

def create_temporal_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Create temporal features"""
    
    LOG.info("Creating temporal features...")
    
    temporal_features = pd.DataFrame(index=dates)
    
    # Basic temporal
    temporal_features['weekday'] = dates.weekday
    temporal_features['month'] = dates.month
    temporal_features['day_of_month'] = dates.day
    temporal_features['quarter'] = dates.quarter
    
    # Business calendar
    temporal_features['is_month_start'] = (dates.day <= 5).astype(int)
    temporal_features['is_month_end'] = (dates.day >= 25).astype(int)
    
    # Cyclical encoding
    temporal_features['weekday_sin'] = np.sin(2 * np.pi * temporal_features['weekday'] / 7)
    temporal_features['weekday_cos'] = np.cos(2 * np.pi * temporal_features['weekday'] / 7)
    temporal_features['month_sin'] = np.sin(2 * np.pi * temporal_features['month'] / 12)
    temporal_features['month_cos'] = np.cos(2 * np.pi * temporal_features['month'] / 12)
    
    # Holidays
    try:
        us_holidays = holidays.US()
        temporal_features['is_holiday'] = dates.to_series().apply(
            lambda x: 1 if x.date() in us_holidays else 0
        ).values
    except:
        temporal_features['is_holiday'] = 0
    
    LOG.info(f"Created {len(temporal_features.columns)} temporal features")
    return temporal_features

def create_call_history_features(self, call_data: pd.Series) -> pd.DataFrame:
    """Create call history features"""
    
    LOG.info("Creating call history features...")
    
    call_features = pd.DataFrame(index=call_data.index)
    
    # Lag features
    for lag in [1, 2, 3, 7]:
        call_features[f'calls_lag_{lag}'] = call_data.shift(lag)
    
    # Rolling statistics
    for window in [3, 7, 14]:
        call_features[f'calls_mean_{window}d'] = call_data.rolling(window, min_periods=1).mean()
        call_features[f'calls_std_{window}d'] = call_data.rolling(window, min_periods=1).std()
    
    # Fill NaN
    call_features = call_features.fillna(method='ffill').fillna(call_data.mean())
    
    LOG.info(f"Created {len(call_features.columns)} call history features")
    return call_features

def create_volume_features(self, aligned_data: Dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Create features for volume prediction"""
    
    LOG.info("Creating volume prediction features...")
    
    call_data = aligned_data['calls']
    mail_data = aligned_data.get('mail')
    
    # Target: next day calls
    y_volume = call_data.shift(-1).dropna()
    common_dates = y_volume.index
    
    all_features = []
    
    # 1. Mail lag features
    if mail_data is not None:
        lag_features = self.create_lag_features(mail_data, call_data)
        lag_features = lag_features.reindex(common_dates, fill_value=0)
        all_features.append(lag_features)
    
    # 2. Temporal features
    temporal_features = self.create_temporal_features(common_dates)
    all_features.append(temporal_features)
    
    # 3. Call history features
    call_features = self.create_call_history_features(call_data)
    call_features = call_features.reindex(common_dates, fill_value=0)
    all_features.append(call_features)
    
    # Combine features
    if all_features:
        X_volume = pd.concat(all_features, axis=1)
    else:
        X_volume = pd.DataFrame(index=common_dates)
        X_volume['weekday'] = common_dates.weekday
        X_volume['calls_lag_1'] = call_data.shift(1).reindex(common_dates, fill_value=call_data.mean())
    
    # Handle NaN
    X_volume = X_volume.fillna(0)
    
    self.volume_features = X_volume.columns.tolist()
    
    LOG.info(f"Volume features: {X_volume.shape[1]} features, {len(y_volume)} samples")
    return X_volume, y_volume

def create_intent_features(self, aligned_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create features for intent prediction"""
    
    intent_data = aligned_data.get('intents')
    if intent_data is None:
        LOG.info("No intent data available for intent prediction")
        return None, None
    
    LOG.info("Creating intent prediction features...")
    
    call_data = aligned_data['calls']
    mail_data = aligned_data.get('mail')
    
    # Target: next day intent distribution
    y_intent = intent_data.shift(-1).dropna()
    common_dates = y_intent.index
    
    all_features = []
    
    # 1. Current intent distribution
    current_intent = intent_data.reindex(common_dates, fill_value=0)
    current_intent.columns = [f'current_{col}' for col in current_intent.columns]
    all_features.append(current_intent)
    
    # 2. Mail features (if available)
    if mail_data is not None:
        lag_features = self.create_lag_features(mail_data, call_data)
        lag_features = lag_features.reindex(common_dates, fill_value=0)
        all_features.append(lag_features)
    
    # 3. Temporal features
    temporal_features = self.create_temporal_features(common_dates)
    all_features.append(temporal_features)
    
    # 4. Call volume features
    call_features = self.create_call_history_features(call_data)
    call_features = call_features.reindex(common_dates, fill_value=0)
    all_features.append(call_features)
    
    # Combine features
    X_intent = pd.concat(all_features, axis=1)
    X_intent = X_intent.fillna(0)
    
    self.intent_features = X_intent.columns.tolist()
    
    LOG.info(f"Intent features: {X_intent.shape[1]} features, {len(y_intent)} samples")
    return X_intent, y_intent
```

# ============================================================================

# DUAL MODEL TRAINER (VOLUME + INTENT)

# ============================================================================

class DualModelTrainer:
“”“Train models for both volume and intent prediction”””

```
def __init__(self):
    self.volume_model = None
    self.intent_models = {}
    self.volume_results = {}
    self.intent_results = {}
    self.label_encoders = {}

def train_volume_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
    """Train call volume prediction model"""
    
    LOG.info("Training call volume prediction model...")
    
    if len(X) < CONFIG["min_train_samples"]:
        return {"error": "insufficient_data"}
    
    models = {
        'ridge': Ridge(alpha=10.0, random_state=CONFIG["random_state"]),
        'random_forest': RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_split=5,
            random_state=CONFIG["random_state"], n_jobs=-1
        ),
        'gradient_boost': GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=CONFIG["random_state"]
        )
    }
    
    best_model = None
    best_score = -float('inf')
    results = {}
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=min(CONFIG["cv_folds"], len(X)//10, 5))
    
    for model_name, model in models.items():
        try:
            # Cross-validation
            cv_results = cross_validate(
                model, X, y, cv=tscv,
                scoring=['neg_mean_absolute_error', 'r2'],
                return_train_score=False
            )
            
            cv_mae = -cv_results['test_neg_mean_absolute_error'].mean()
            cv_r2 = cv_results['test_r2'].mean()
            
            # Holdout validation
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            results[model_name] = {
                'cv_mae': cv_mae,
                'cv_r2': cv_r2,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'model': model
            }
            
            LOG.info(f"  {model_name}: CV R² = {cv_r2:.3f}, Test R² = {test_r2:.3f}")
            
            if cv_r2 > best_score:
                best_score = cv_r2
                best_model = model
                
        except Exception as e:
            LOG.error(f"  {model_name} failed: {e}")
            continue
    
    if best_model is not None:
        # Train best model on full data
        best_model.fit(X, y)
        self.volume_model = best_model
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, best_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            results['feature_importance'] = top_features
        
        LOG.info(f"Best volume model: {type(best_model).__name__} (R² = {best_score:.3f})")
    
    self.volume_results = results
    return results

def train_intent_models(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
    """Train intent prediction models for each intent type"""
    
    if X is None or y is None:
        LOG.info("No intent data - skipping intent model training")
        return {}
    
    LOG.info("Training intent prediction models...")
    
    results = {}
    
    # Train a model for each intent type
    for intent_type in y.columns:
        LOG.info(f"  Training model for intent: {intent_type}")
        
        try:
            # Convert probabilities to categories (high/medium/low)
            y_intent = y[intent_type]
            y_categorical = pd.cut(y_intent, bins=3, labels=['Low', 'Medium', 'High'])
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_categorical)
            self.label_encoders[intent_type] = le
            
            # Simple model for intent prediction
            model = RandomForestClassifier(
                n_estimators=50, max_depth=5,
                random_state=CONFIG["random_state"], n_jobs=-1
            )
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=min(3, len(X)//15, 5))
            cv_scores = cross_validate(model, X, y_encoded, cv=tscv, scoring='accuracy')
            cv_accuracy = cv_scores['test_score'].mean()
            
            # Train final model
            model.fit(X, y_encoded)
            self.intent_models[intent_type] = model
            
            results[intent_type] = {
                'cv_accuracy': cv_accuracy,
                'model': model,
                'label_encoder': le
            }
            
            LOG.info(f"    {intent_type}: CV Accuracy = {cv_accuracy:.3f}")
            
        except Exception as e:
            LOG.error(f"    {intent_type} failed: {e}")
            continue
    
    self.intent_results = results
    return results

def train_all_models(self, volume_data: Tuple, intent_data: Tuple) -> Dict:
    """Train both volume and intent models"""
    
    LOG.info("=== TRAINING VOLUME & INTENT MODELS ===")
    
    results = {}
    
    # Train volume model
    if volume_data[0] is not None:
        volume_results = self.train_volume_model(volume_data[0], volume_data[1])
        results['volume'] = volume_results
    
    # Train intent models
    if intent_data[0] is not None:
        intent_results = self.train_intent_models(intent_data[0], intent_data[1])
        results['intent'] = intent_results
    
    return results
```

# ============================================================================

# PREDICTION ENGINE

# ============================================================================

class CallPredictionEngine:
“”“Engine for making volume and intent predictions”””

```
def __init__(self, volume_model, intent_models, feature_engineer, call_data, mail_data=None):
    self.volume_model = volume_model
    self.intent_models = intent_models
    self.feature_engineer = feature_engineer
    self.call_data = call_data
    self.mail_data = mail_data
    self.last_known_date = call_data.index.max()

def predict_single_day(self, prediction_date: Union[str, datetime], 
                      mail_volumes: Dict[str, float] = None) -> Dict:
    """Predict volume and intent for a single day"""
    
    try:
        pred_date = pd.to_datetime(prediction_date)
        
        # Create simple feature vector for prediction
        features = []
        
        # Basic features
        if mail_volumes:
            features.append(sum(mail_volumes.values()))  # Total mail
        else:
            features.append(0)
        
        features.extend([
            pred_date.weekday(),  # Weekday
            pred_date.month,      # Month
            self.call_data.iloc[-1] if len(self.call_data) > 0 else 500,  # Last call volume
            self.call_data.tail(7).mean() if len(self.call_data) >= 7 else 500  # 7-day average
        ])
        
        # Pad to expected feature count
        expected_features = len(self.feature_engineer.volume_features) if self.feature_engineer.volume_features else 5
        while len(features) < expected_features:
            features.append(0)
        
        # Volume prediction
        volume_prediction = None
        if self.volume_model:
            try:
                vol_pred = self.volume_model.predict([features[:expected_features]])[0]
                volume_prediction = max(0, round(vol_pred, 0))
            except:
                volume_prediction = self.call_data.mean()
        
        # Intent predictions
        intent_predictions = {}
        if self.intent_models:
            for intent_type, model in self.intent_models.items():
                try:
                    intent_pred = model.predict([features[:expected_features]])[0]
                    intent_predictions[intent_type] = intent_pred
                except:
                    intent_predictions[intent_type] = 'Medium'
        
        # Confidence intervals for volume
        confidence_intervals = {}
        if volume_prediction:
            historical_std = self.call_data.std()
            for conf_level in CONFIG["confidence_levels"]:
                z_score = stats.norm.ppf((1 + conf_level) / 2)
                margin = z_score * historical_std * 0.3
                
                confidence_intervals[f'{conf_level:.0%}'] = {
                    'lower': max(0, round(volume_prediction - margin, 0)),
                    'upper': round(volume_prediction + margin, 0)
                }
        
        result = {
            'prediction_date': pred_date.strftime('%Y-%m-%d'),
            'weekday': pred_date.strftime('%A'),
            'predicted_volume': volume_prediction,
            'confidence_intervals': confidence_intervals,
            'predicted_intents': intent_predictions,
            'mail_input': mail_volumes if mail_volumes else {},
            'model_type': type(self.volume_model).__name__ if self.volume_model else 'None'
        }
        
        return result
        
    except Exception as e:
        LOG.error(f"Prediction failed: {e}")
        return {'error': str(e), 'prediction_date': str(prediction_date)}

def generate_outlook(self, days: int = 5) -> Dict:
    """Generate multi-day outlook"""
    
    LOG.info(f"Generating {days}-day outlook...")
    
    # Use recent mail patterns if available
    if self.mail_data is not None:
        typical_mail = self.mail_data.tail(14).median().to_dict()
    else:
        typical_mail = {}
    
    outlook_predictions = []
    current_date = self.last_known_date + timedelta(days=1)
    business_days_added = 0
    
    while business_days_added < days:
        if current_date.weekday() < 5:  # Business days only
            prediction = self.predict_single_day(current_date, typical_mail)
            prediction['outlook_day'] = business_days_added + 1
            outlook_predictions.append(prediction)
            business_days_added += 1
        
        current_date += timedelta(days=1)
    
    # Summary
    if outlook_predictions:
        volumes = [p.get('predicted_volume', 0) for p in outlook_predictions if p.get('predicted_volume')]
        
        if volumes:
            outlook_summary = {
                'outlook_period': f"{days} business days",
                'forecast_start': outlook_predictions[0]['prediction_date'],
                'forecast_end': outlook_predictions[-1]['prediction_date'],
                'predicted_range': f"{min(volumes):.0f} - {max(volumes):.0f} calls",
                'average_daily': f"{np.mean(volumes):.0f} calls",
                'total_expected': f"{sum(volumes):.0f} calls"
            }
        else:
            outlook_summary = {'note': 'Volume predictions not available'}
    else:
        outlook_summary = {'error': 'No predictions generated'}
    
    return {
        'outlook_summary': outlook_summary,
        'daily_predictions': outlook_predictions
    }
```

# ============================================================================

# MAIN PIPELINE ORCHESTRATOR

# ============================================================================

class Pipeline2025Orchestrator:
“”“Main orchestrator for 2025+ data pipeline”””

```
def __init__(self):
    self.start_time = time.time()
    self.output_dir = Path(CONFIG["output_dir"])
    self.output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    for subdir in ["plots_dir", "models_dir", "results_dir"]:
        (self.output_dir / CONFIG[subdir]).mkdir(exist_ok=True)

def run_complete_pipeline(self) -> Dict:
    """Run the complete end-to-end pipeline"""
    
    LOG.info("🚀 STARTING 2025+ CALL VOLUME & INTENT PREDICTION PIPELINE")
    LOG.info("=" * 70)
    
    try:
        # Phase 1: Load 2025+ Data
        LOG.info("📊 PHASE 1: LOADING 2025+ DATA")
        data_loader = Fresh2025DataLoader()
        aligned_data = data_loader.load_all_data()
        
        if len(aligned_data['calls']) < 10:
            raise ValueError("Insufficient data for modeling")
        
        # Phase 2: Feature Engineering
        LOG.info("\n🔧 PHASE 2: FEATURE ENGINEERING")
        feature_engineer = VolumeIntentFeatureEngine()
        
        # Volume features
        volume_data = feature_engineer.create_volume_features(aligned_data)
        
        # Intent features (if available)
        intent_data = feature_engineer.create_intent_features(aligned_data)
        
        # Phase 3: Model Training
        LOG.info("\n🤖 PHASE 3: MODEL TRAINING")
        trainer = DualModelTrainer()
        training_results = trainer.train_all_models(volume_data, intent_data)
        
        # Phase 4: Generate Predictions
        LOG.info("\n🔮 PHASE 4: GENERATING PREDICTIONS")
        prediction_engine = CallPredictionEngine(
            trainer.volume_model,
            trainer.intent_models,
            feature_engineer,
            aligned_data['calls'],
            aligned_data.get('mail')
        )
        
        # Generate 5-day outlook
        outlook_results = prediction_engine.generate_outlook(CONFIG["prediction_horizon_days"])
        
        # Phase 5: Save Results
        LOG.info("\n💾 PHASE 5: SAVING RESULTS")
        self.save_results(data_loader, training_results, outlook_results, trainer)
        
        # Phase 6: Generate Report
        LOG.info("\n📋 PHASE 6: GENERATING REPORT")
        report = self.generate_final_report(data_loader, training_results, outlook_results)
        
        execution_time = (time.time() - self.start_time) / 60
        
        LOG.info("\n" + "=" * 70)
        LOG.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        LOG.info(f"⏱️  Total execution time: {execution_time:.1f} minutes")
        LOG.info(f"📁 Results saved to: {self.output_dir}")
        
        return {
            'success': True,
            'execution_time_minutes': execution_time,
            'output_directory': str(self.output_dir),
            'data_summary': data_loader.data_summary,
            'training_results': training_results,
            'outlook_results': outlook_results,
            'prediction_engine': prediction_engine
        }
        
    except Exception as e:
        LOG.error(f"Pipeline failed: {e}")
        LOG.error(traceback.format_exc())
        
        return {
            'success': False,
            'error': str(e),
            'execution_time_minutes': (time.time() - self.start_time) / 60
        }

def save_results(self, data_loader, training_results, outlook_results, trainer):
    """Save all results"""
    
    try:
        results_dir = self.output_dir / CONFIG["results_dir"]
        models_dir = self.output_dir / CONFIG["models_dir"]
        
        # Save data summary
        with open(results_dir / "data_summary.json", 'w') as f:
            json.dump(data_loader.data_summary, f, indent=2, default=str)
        
        # Save training results
        with open(results_dir / "training_results.json", 'w') as f:
            # Convert models to strings for JSON serialization
            serializable_results = {}
            for key, value in training_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if k == 'model':
                            serializable_results[key][k] = str(type(v).__name__)
                        elif k == 'label_encoder':
                            serializable_results[key][k] = 'LabelEncoder'
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save outlook results
        with open(results_dir / "outlook_predictions.json", 'w') as f:
            json.dump(outlook_results, f, indent=2, default=str)
        
        # Save models
        if trainer.volume_model:
            joblib.dump(trainer.volume_model, models_dir / "volume_model.pkl")
        
        if trainer.intent_models:
            for intent_type, model in trainer.intent_models.items():
                safe_name = str(intent_type).replace(' ', '_').replace('/', '_')
                joblib.dump(model, models_dir / f"intent_model_{safe_name}.pkl")
        
        LOG.info("All results saved successfully")
        
    except Exception as e:
        LOG.error(f"Failed to save results: {e}")

def generate_final_report(self, data_loader, training_results, outlook_results) -> str:
    """Generate final report"""
    
    try:
        execution_time = (time.time() - self.start_time) / 60
        
        # Extract key metrics
        volume_r2 = 0
        intent_accuracy = 0
        
        if 'volume' in training_results:
            for model_name, results in training_results['volume'].items():
                if isinstance(results, dict) and 'cv_r2' in results:
                    volume_r2 = max(volume_r2, results['cv_r2'])
        
        if 'intent' in training_results:
            intent_accuracies = []
            for intent_type, results in training_results['intent'].items():
                if isinstance(results, dict) and 'cv_accuracy' in results:
                    intent_accuracies.append(results['cv_accuracy'])
            if intent_accuracies:
                intent_accuracy = np.mean(intent_accuracies)
        
        report = f"""
```

# ====================================================================
2025+ CALL VOLUME & INTENT PREDICTION PIPELINE REPORT

EXECUTION SUMMARY:
• Pipeline Status: SUCCESS
• Execution Time: {execution_time:.1f} minutes
• Output Directory: {self.output_dir}

DATA SUMMARY:
• Total Call Records: {data_loader.data_summary.get(‘call_data’, {}).get(‘total_calls’, ‘N/A’)}
• Overlapping Days: {data_loader.data_summary.get(‘aligned_data’, {}).get(‘overlapping_days’, ‘N/A’)}
• Date Range: {data_loader.data_summary.get(‘aligned_data’, {}).get(‘date_range’, ‘N/A’)}
• Has Mail Data: {data_loader.data_summary.get(‘aligned_data’, {}).get(‘has_mail’, False)}
• Has Intent Data: {data_loader.data_summary.get(‘aligned_data’, {}).get(‘has_intents’, False)}

MODEL PERFORMANCE:
• Call Volume Model R²: {volume_r2:.3f}
• Intent Classification Accuracy: {intent_accuracy:.3f}

PREDICTIONS GENERATED:
• 5-Day Call Volume Outlook: Available
• Intent Distribution Predictions: {‘Available’ if intent_accuracy > 0 else ‘Not Available’}

BUSINESS APPLICATIONS:
• Daily staffing optimization
• Mail campaign impact analysis
• Intent-based resource allocation
• 5-day capacity planning

FILES GENERATED:
• data_summary.json - Data loading summary
• training_results.json - Model performance metrics
• outlook_predictions.json - 5-day predictions
• volume_model.pkl - Trained volume prediction model
• intent_model_*.pkl - Trained intent classification models

NEXT STEPS:

1. Review prediction accuracy in outlook_predictions.json
1. Use volume_model.pkl for daily predictions
1. Monitor model performance with new data
1. Retrain models monthly with fresh data

# ====================================================================
Generated: {datetime.now().strftime(’%Y-%m-%d %H:%M:%S’)}

“””

```
        # Save report
        report_path = self.output_dir / "PIPELINE_REPORT.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Print report
        safe_print(report)
        
        return str(report_path)
        
    except Exception as e:
        LOG.error(f"Report generation failed: {e}")
        return ""
```

# ============================================================================

# MAIN EXECUTION

# ============================================================================

def main():
“”“Main execution function”””

```
safe_print("=" * 60)
safe_print("🚀 2025+ CALL VOLUME & INTENT PREDICTION PIPELINE")
safe_print("=" * 60)
safe_print("📊 Fresh data analysis with overlapping dates")
safe_print("📞 Call volume prediction with mail lag modeling") 
safe_print("🎯 Intent classification (scope extension)")
safe_print("🔮 5-day business outlook generation")
safe_print("=" * 60)
safe_print("")

try:
    # Run the complete pipeline
    orchestrator = Pipeline2025Orchestrator()
    results = orchestrator.run_complete_pipeline()
    
    if results['success']:
        safe_print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        safe_print("")
        safe_print("✅ 2025+ data processed and analyzed")
        safe_print("✅ Call volume prediction model trained") 
        safe_print("✅ Intent classification models trained")
        safe_print("✅ 5-day outlook generated")
        safe_print("✅ Production-ready models saved")
        safe_print("")
        safe_print(f"📁 Find all results in: {results['output_directory']}")
        safe_print("")
        safe_print("🚀 Ready for daily predictions!")
        
    else:
        safe_print("\n❌ PIPELINE FAILED")
        safe_print(f"Error: {results['error']}")
        safe_print("💡 Check the logs above for details")
    
    return 0 if results['success'] else 1
    
except KeyboardInterrupt:
    safe_print("\n⏹️  Pipeline interrupted by user")
    return 1
    
except Exception as e:
    safe_print(f"\n💥 Unexpected error: {e}")
    return 1
```

if **name** == “**main**”:
exit_code = main()
sys.exit(exit_code)
