import shap
import streamlit as st

def shap_plot(model, X):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot()
