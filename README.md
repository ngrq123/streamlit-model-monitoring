# Model Monitoring using Streamlit

<img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" width="300" />

## How Predictions are Generated

The original Titanic dataset is used as training data, and the Synthanic (synthetic Titanic data) dataset is used as validation data.

The data is transformed in the `modelling/Data Pipeline.ipynb` notebook and fitted into scikit-learn `LogisticRegression` and `RandomForestClassifier`.

To retrieve the prediction probabilities, `predict_proba()` is used instead of `predict()`.

## Datasets Used

- Titanic - Machine Learning from Disaster: https://www.kaggle.com/c/titanic
- Tabular Playground Series - Apr 2021: https://www.kaggle.com/c/tabular-playground-series-apr-2021