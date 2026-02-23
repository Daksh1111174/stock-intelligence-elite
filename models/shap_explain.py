import shap
import matplotlib.pyplot as plt

def shap_plot(model, X):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False)

    return fig
