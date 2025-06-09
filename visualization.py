import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_feature_importance(model, feature_names, model_name: str, top_k: int = 20):
    """
    Plot feature importances for supported models.

    Parameters:
        model: fitted model instance
        feature_names: list or array of feature names
        model_name: string name of the model (for title)
        top_k: number of top features to display
    """
    if hasattr(model, "feature_importances_"):  # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):  # Linear models
        importances = np.abs(model.coef_)
        if importances.ndim > 1:  # For multi-output or multi-task models
            importances = importances[0]
    else:
        raise ValueError(f"Model type {type(model)} does not support feature importance.")

    # Sort and select top_k
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_k)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df["Feature"][::-1], feature_importance_df["Importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_k} Feature Importances ({model_name})")
    plt.tight_layout()
    plt.show()
