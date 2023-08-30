import pandas as pd
from scipy.stats import ks_2samp


def calculate_feature_statistics(
    features: list[str],
    data: pd.DataFrame,
    ref_data: pd.DataFrame,
) -> pd.DataFrame:
    statistics = []

    for feature in features:
        # Extract values for specific component.
        ref_values = ref_data[feature].values
        values = data[feature].values

        # Calculate Kolmogorov–Smirnov statistic.
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
