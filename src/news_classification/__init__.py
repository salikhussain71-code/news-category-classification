"""News category classification (AG News).

Multiclass text classification pipeline: data loading -> text preprocessing (tokenisation,
stop-word removal, lemmatisation) -> TF-IDF features -> classifiers (Logistic Regression,
Linear SVM, Naive Bayes, Random Forest, optional XGBoost / LightGBM / Keras NN) -> evaluation
-> visualisation -> inference.
"""

__version__ = "1.0.0"
