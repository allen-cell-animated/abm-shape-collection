import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA


def calculate_shape_statistics(
    pca: PCA,
    data: pd.DataFrame,
    ref_data: pd.DataFrame,
    components: int,
) -> pd.DataFrame:
    statistics = []

    # Transform data into shape mode space.
    columns = ref_data.filter(like="shcoeffs").columns
    ref_transform = pca.transform(ref_data[columns].values)
    transform = pca.transform(data[columns].values)

    for component in range(components):
        # Extract values for specific component.
        ref_values = ref_transform[:, component]
        values = transform[:, component]

        # Calculate Kolmogorov–Smirnov statistic.
        ks_result = ks_2samp(values, ref_values, mode="asymp")

        statistics.append(
            {
                "FEATURE": f"PC{component + 1}",
                "SIZE": len(values),
                "KS_STATISTIC": ks_result.statistic,
                "KS_PVALUE": ks_result.pvalue,
            }
        )

    return pd.DataFrame(statistics)
