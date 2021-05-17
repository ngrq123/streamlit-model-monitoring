import time

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import streamlit as st

@st.cache
def load_data(model):
    file_name = None
    if model == 'Random Forest':
        file_name = 'data/rf_predictions.pkl'
    if model == 'Logistic Regression':
        file_name = 'data/lr_predictions.pkl'
    if not file_name:
        return pd.DataFrame({
            'Survived': np.random.randint(2, size=10000),
            'Prediction_Survived': np.random.random(10000)
        })
    return pd.read_pickle(file_name)


@st.cache
def generate_confusion_matrix(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba > threshold).astype('int')
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return pd.DataFrame({
        'pred_0': [tn, fn],
        'pred_1': [fp, tp]
    }, index=['true_0', 'true_1'])


@st.cache
def calculate_ks_scores(df, threshold=0.5):
    df = df.copy()
    df['Ventile'] = pd.qcut(df['Prediction_Survived'], 20, labels=False)
    df['Prediction_Survived'] = (df['Prediction_Survived'] > threshold).astype('int')
    df = df.groupby('Ventile')['Prediction_Survived'].agg(['count', 'sum'])
    df.columns = ['total', 'target']
    df['ventile'] = np.array(range(20)) + 1
    df = df.set_index('ventile')
    df['non_target'] = df['total'] - df['target']
    df['perc_total_cum'] = df['total'].cumsum() / df['total'].sum()
    df['perc_target_cum'] = df['target'].cumsum() / df['target'].sum()
    df['perc_non_target_cum'] = df['non_target'].cumsum() / df['non_target'].sum()
    df['ks'] = (df['perc_target_cum'] - df['perc_non_target_cum']).apply(abs)
    return df['ks'].max(), df


@st.cache
def generate_metrics(y_true, y_pred_proba, threshold=0.5, ks=None):
    y_pred = (y_pred_proba > threshold).astype(int)
    data = {
        'Recall': [metrics.recall_score(y_true, y_pred)], 
        'Precision': [metrics.recall_score(y_true, y_pred)], 
        'Accuracy': [metrics.accuracy_score(y_true, y_pred)], 
        'Balanced Accuracy': [metrics.balanced_accuracy_score(y_true, y_pred)], 
        'F1': [metrics.f1_score(y_true, y_pred)]
    }
    if ks:
      data['KS Score'] = [ks]
    df = pd.DataFrame(data).transpose()
    df.columns = ['threshold_' + str(threshold)]
    return df


# Dropdown box
selected_model = st.sidebar.selectbox('Select Model', 
                                      options=[
                                          'Random Forest', 
                                          'Logistic Regression',
                                          'Neural Network'
                                      ], 
                                      help='Select model to monitor')
view_sample_data = st.sidebar.checkbox('View Data Sample (with Predictions)')

st.title('Model Monitoring: ' + selected_model)

data = load_data(selected_model)

if view_sample_data:
    st.subheader('Sample Data')
    st.dataframe(data.sample(100))

col1, col2 = st.beta_columns([0.3, 0.7])

with col1:
    # Radio Buttons
    selected_strategy = st.radio('Select Strategy',
                                 options=['Optimal', 'Aggressive', 'Conservative', 'Custom'], 
                                 index=3, 
                                 help='Select a strategy to view its corresponding threshold')

with col2:
    if selected_strategy == 'Custom':
        # Slider
        selected_threshold = st.slider('Select Custom Threshold', 
                                       min_value=0.0, 
                                       max_value=1.0, 
                                       value=0.5, 
                                       step=0.001, 
                                       format='%f',
                                       help='Select a threshold to view results')
        st.write('Selected Threshold: ' + str(selected_threshold))
    else:
        selected_threshold = 0.5

col1, col2 = st.beta_columns([0.4, 0.6])

with col1:
    st.subheader('Confusion Matrix')
    conf_mat = generate_confusion_matrix(data['Survived'], data['Prediction_Survived'], selected_threshold)
    st.table(conf_mat)

with col2:
    st.subheader('Metrics')
    ks, ks_df = calculate_ks_scores(data, selected_threshold)
    metrics_df = generate_metrics(data['Survived'], data['Prediction_Survived'], selected_threshold, ks)
    if selected_threshold != 0.5:
        ks_0_5, _ = calculate_ks_scores(data, selected_threshold)
        metrics_df_0_5 = generate_metrics(data['Survived'], data['Prediction_Survived'], 0.5, ks_0_5)
        metrics_df = pd.concat([metrics_df, metrics_df_0_5], axis=1)
        st.table(metrics_df.style.highlight_max(color='lightgreen', axis=1))
    else:
        st.table(metrics_df)

st.line_chart(ks_df[['perc_total_cum', 'perc_target_cum']])

view_ks_data = st.checkbox('View KS Data')
if view_ks_data:
    st.dataframe(ks_df)