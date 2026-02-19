from typing import Any, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import shap
    shap_available = True
except ModuleNotFoundError:
    shap = None
    shap_available = False


def compute_window_importance(
    estimator: Any,
    df: pd.DataFrame,
    features: list[str],
    dates: Union[pd.DatetimeIndex, pd.Series],
    window_start: Union[pd.Timestamp, datetime],
    window_end: Union[pd.Timestamp, datetime],
    window_size_days: int,
) -> Optional[np.ndarray]:

    """Compute feature importance within a time window using SHAP values.

    This function filters the provided design matrix to the [window_start, window_end)
    period (based on the supplied dates index), computes SHAP values for the given
    estimator and features, and returns the average absolute importance per feature,
    normalized by the number of rows in the window.

    Parameters:
        estimator: A fitted tree-based estimator supported by shap.TreeExplainer. Must expose
            the attribute `feature_names_in_` that matches the columns in `df`.
        df: A pandas DataFrame of estimator inputs that align with `dates`.
        features: Ordered list of feature names to extract from `df` for SHAP.
        dates: A pandas DatetimeIndex aligned one-to-one with the rows of `df`.
        window_start: Inclusive start timestamp of the window.
        window_end: Exclusive end timestamp of the window.
        window_size_days: Minimum number of rows required in the window. If the
            filtered window contains fewer rows than this value, the function returns
            None.

    Returns:
        numpy.ndarray or None: A 1D array of per-feature mean absolute SHAP
        importances ordered according to `features`, or None if the window has
        insufficient rows.

    Notes:
        - Importance is computed as the mean absolute SHAP value per feature over
          the rows in the window.
        - The return value is a NumPy array.
    """

    period_mask = (dates >= pd.Timestamp(window_start)) & (dates < pd.Timestamp(window_end))
    increment_df = df.loc[period_mask, features]
    if len(increment_df) < window_size_days:
        return None

    explainer = shap.TreeExplainer(estimator)
    shap_df = pd.DataFrame(
        explainer.shap_values(increment_df),
        columns=features,
    )
    normalized_importance = shap_df.abs().mean()
    return normalized_importance.values


def calculate_monthly_importance(
    estimator: Any,
    df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    window_size_days: int = 30,
    increment_size_days: int = 14,
    use_ray: bool = False,
) -> pd.DataFrame:
    """Compute rolling-window feature importances over time.

    The function schedules a series of windows across the provided date range:
    - The first window spans `window_size_days` starting at the minimum date.
    - Subsequent windows start every `increment_size_days` days and each spans
      `window_size_days` days.

    For each window, it calls `compute_window_importance` to compute mean absolute
    SHAP importances per feature. Windows with fewer than `window_size_days` rows
    are skipped. Results are aggregated into a DataFrame where:
    - rows are indexed by the window start timestamp; and
    - columns are feature names with values equal to the mean absolute SHAP
      importance for that feature within the window.

    Parameters:
        estimator: A fitted estimator compatible with shap.TreeExplainer, expected to have
            `feature_names_in_` attribute listing the input features.
        df: A pandas DataFrame of estimator inputs aligned with `dates`.
        dates: A pandas DatetimeIndex aligned one-to-one with the rows of `df`.
        window_size_days: Size of each evaluation window (in days).
        increment_size_days: Step between the starts of consecutive windows (in days).
        use_ray: If True, parallelize per-window computations using Ray.

    Returns:
        pandas.DataFrame: DataFrame indexed by window start timestamps with feature
        importance per window.

    Raises:
        ModuleNotFoundError: If `use_ray` is True but the optional dependency `ray`
            is not installed.
    """
    if not shap_available:
        raise ModuleNotFoundError(
            "Optional dependency shap is required for "
            "compute_window_importance but not installed"
        )

    features = estimator.feature_names_in_.tolist()
    results_df = pd.DataFrame(columns=features)

    first_period_end = dates.min() + pd.Timedelta(days=window_size_days)

    all_increments = [dates.min()]

    biweekly_increments = pd.date_range(
        start=first_period_end, end=dates.max(), freq=f"{increment_size_days}D"
    )
    all_increments.extend(biweekly_increments)

    # Build window schedule
    windows = [
        (
            increment,
            (increment + pd.Timedelta(days=window_size_days)) if i else first_period_end
        )
        for i, increment in enumerate(all_increments)
    ]

    if use_ray:
        try:
            import ray
        except ModuleNotFoundError as ex:
            raise ModuleNotFoundError(
                "Optional dependency Ray is required for parallel execution but not installed"
            ) from ex

        estimator_ref = ray.put(estimator)
        df_ref = ray.put(df)
        features_ref = ray.put(features)
        dates_ref = ray.put(dates)
        func = ray.remote(compute_window_importance)

        futures = [
            func.remote(
                estimator_ref, df_ref, features_ref, dates_ref, ws, we, window_size_days
            )
            for (ws, we) in windows
        ]

        results = ray.get(futures)
    else:
        results = [
            compute_window_importance(
                estimator, df, features, dates, ws, we, window_size_days
            )
            for (ws, we) in windows
        ]

    for (ws, _), res in zip(windows, results):
        if res is not None:
            results_df.loc[ws] = res

    return results_df.sort_index()



def plot_feature_summary(
    df: pd.DataFrame,
    title: Optional[str] = None,
):
    """Plot mean importance vs. variance (temporal consistency) for features.

    Given a per-window feature-importance results DataFrame (as produced by
    `calculate_monthly_importance`), this function computes, for each feature,
    the mean importance and its variance across windows, and renders a scatter
    plot of Mean vs Variance.

    Parameters:
        df: DataFrame where rows are time windows and columns are feature
            names; values are the mean absolute SHAP importance within each window.
        title: Optional plot title. If None, a descriptive default is used.

    Returns:
        plotly.graph_objects.Figure: A scatter plot with points representing
        features and hover labels showing `feature_name`.

    Raises:
        ModuleNotFoundError: If the optional dependency `plotly` is not installed.
    """
    try:
        import plotly.express as px
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Optional dependency plotly is required but not installed"
        ) from ex

    mean_values = df.mean()
    variance_values = df.var()

    summary_df = pd.DataFrame(
        {
            "feature_name": mean_values.index,
            "Mean": mean_values.values,
            "Variance": variance_values.values,
        }
    )

    if title is None:
        title = f"Mean Feature Importance vs. Temporal Consistency (Variance)"

    # Create the scatter plot
    fig = px.scatter(
        summary_df,
        x="Mean",
        y="Variance",
        hover_name="feature_name",
        title=title,
    )

    # Update layout for better appearance
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(gridcolor="lightgray", zerolinecolor="lightgray"),
        yaxis=dict(gridcolor="lightgray", zerolinecolor="lightgray"),
    )

    # Add marker customization
    fig.update_traces(
        marker=dict(size=10, opacity=0.7, line=dict(width=1, color="darkslategray"))
    )

    return fig
