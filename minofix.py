Why those “Invalid parameter … for estimator Pipeline(…)” errors appear

BayesSearchCV passes every hyper-parameter name verbatim to the
under-lying estimator.
When the estimator is wrapped in a sklearn.pipeline.Pipeline you must prefix
every parameter with <step_name>__ (double underscore).
In our pipeline the final model always sits in the step called
regressor, so the search-space keys have to look like:

regressor__n_estimators
regressor__max_depth
regressor__learning_rate
…

I accidentally left the prefixes off for the tree models – Ridge was fine,
the others blew up.

⸻

Quick patch – change only the search-space dictionaries

Open the file you are running (min.py in your log; if you copied my
rewrite it will be mail_calls_forecast.py).
Locate the def model_space(name: str): function and replace only the
three model dictionaries as shown below.

def model_space(name: str):
    if name == "Ridge":
        return {
            "feature_selector__k": Integer(5, CFG["max_features"]),
            "regressor__alpha": Real(0.01, 100.0, prior="log-uniform"),
        }

    # --------------  FIXED DICTIONARIES  -----------------
    if name == "RF":
        return {
            "feature_selector__k": Integer(5, CFG["max_features"]),
            "regressor__n_estimators": Integer(120, 300),
            "regressor__max_depth": Integer(3, 12),
            "regressor__min_samples_leaf": Integer(1, 8),
        }

    if name == "LGBM":
        return {
            "feature_selector__k": Integer(5, CFG["max_features"]),
            "regressor__n_estimators": Integer(120, 300),
            "regressor__num_leaves": Integer(20, 80),
            "regressor__learning_rate": Real(0.03, 0.3, prior="log-uniform"),
        }

    if name == "XGB":
        return {
            "feature_selector__k": Integer(5, CFG["max_features"]),
            "regressor__n_estimators": Integer(120, 300),
            "regressor__max_depth": Integer(3, 6),
            "regressor__learning_rate": Real(0.03, 0.3, prior="log-uniform"),
        }
    # ------------------------------------------------------

    raise ValueError(name)

Nothing else must change – all the pipeline step names (feature_selector,
regressor) already match.

⸻

After you save the file

python mail_calls_forecast.py      # or python min.py if that is the file name

You should now see each horizon train all four models without the
parameter-name errors.
If anything else pops up, send me the fresh log and we’ll squash it!