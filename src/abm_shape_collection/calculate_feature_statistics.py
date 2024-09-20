import pandas as pd
from scipy.stats import ks_2samp


def calculate_feature_statistics(
    features: list[str],
    data: pd.DataFrame,
    ref_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform two-sample Kolmogorov-Smirnov test for goodness of fit on features.

    Parameters
    ----------
    features
        List of features to perform test on.
    data
        Sample data, with features as columns.
    ref_data : pd.DataFrame
        References data, with features as columns.

    Returns
    -------
    :
        Kolmogorov-Smirnov statistics and p-values for each feature.
    """

    statistics = []

    for feature in features:
        # Extract values for specific component.
        ref_values = ref_data[feature].to_numpy()
        values = data[feature].to_numpy()

        # Calculate Kolmogorov-Smirnov statistic.
        ks_result = ks_2samp(values, ref_values, mode="asymp")

        statistics.append(
            {
                "FEATURE": feature.upper(),
                "SIZE": len(values),
                "KS_STATISTIC": ks_result.statistic,
                "KS_PVALUE": ks_result.pvalue,
            }
        )

    return pd.DataFrame(statistics)
