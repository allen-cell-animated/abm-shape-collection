import numpy as np
import pandas as pd
from prefect import task
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA


@task
def calculate_shape_stats(
    pca: PCA, data: pd.DataFrame, ref_data: pd.DataFrame, components: int
) -> pd.DataFrame:
    all_stats = []

    columns = ref_data.filter(like="shcoeffs").columns
    data_transform = pca.transform(data[columns].values)
    ref_data_transform = pca.transform(ref_data[columns].values)

    for component in range(components):
        ks_stats = get_ks_statistic(data_transform[:, component], ref_data_transform[:, component])
        ks_stats.update({"FEATURE": f"PC_{component + 1}", "TICK": np.nan})
        all_stats.append(ks_stats)

    for tick, tick_data in data.groupby("TICK"):
        tick_data_transform = pca.transform(tick_data[columns].values)

        for component in range(components):
            tick_ks_stats = get_ks_statistic(
                tick_data_transform[:, component], ref_data_transform[:, component]
            )
            tick_ks_stats.update(
                {
                    "FEATURE": f"PC_{component + 1}",
                    "TICK": tick,
                }
            )
            all_stats.append(tick_ks_stats)

    all_stats_df = pd.DataFrame(all_stats)

    return all_stats_df


def get_ks_statistic(population_a: np.ndarray, population_b: np.ndarray) -> dict:
    ksresult = ks_2samp(population_a, population_b, mode="asymp")

    return {
        "KS_STATISTIC": ksresult.statistic,
        "KS_PVALUE": ksresult.pvalue,
    }
