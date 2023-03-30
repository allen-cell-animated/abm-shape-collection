import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from abm_shape_collection.calculate_size_stats import get_ks_statistic


def calculate_shape_stats(
    pca: PCA,
    data: pd.DataFrame,
    ref_data: pd.DataFrame,
    components: int,
    include_ticks: bool = False,
    include_samples: bool = False,
    sample_reps: int = 1,
    sample_size: int = 1,
) -> pd.DataFrame:
    all_stats = []

    columns = list(ref_data.filter(like="shcoeffs").columns)
    ref_data_transform = pca.transform(ref_data[columns].values)

    for component in range(components):
        stats = calculate_shape_stats_for_all(pca, data, ref_data_transform, component, columns)

        if include_ticks:
            stats = stats + calculate_shape_stats_for_ticks(
                pca, data, ref_data_transform, component, columns
            )

        if include_samples:
            stats = stats + calculate_shape_stats_for_samples(
                pca, data, ref_data_transform, component, columns, sample_reps, sample_size
            )

        stats_df = pd.DataFrame(stats)
        stats_df["FEATURE"] = f"PC_{component + 1}"
        all_stats.append(stats_df)

    all_stats_df = pd.concat(all_stats).astype({"N": int})

    return all_stats_df


def calculate_shape_stats_for_all(
    pca: PCA, data: pd.DataFrame, ref_data: np.ndarray, component: int, columns: list[str]
) -> list[dict]:
    ref_values = ref_data[:, component]
    values = pca.transform(data[columns].values)[:, component]

    ks_stats = get_ks_statistic(values, ref_values)
    ks_stats.update({"N": len(values)})

    return [ks_stats]


def calculate_shape_stats_for_ticks(
    pca: PCA, data: pd.DataFrame, ref_data: np.ndarray, component: int, columns: list[str]
) -> list[dict]:
    ref_values = ref_data[:, component]
    tick_stats = []

    for tick, tick_data in data.groupby("TICK"):
        tick_values = pca.transform(tick_data[columns].values)[:, component]

        tick_ks_stats = get_ks_statistic(tick_values, ref_values)
        tick_ks_stats.update({"TICK": tick, "N": len(tick_values)})

        tick_stats.append(tick_ks_stats)

    return tick_stats


def calculate_shape_stats_for_samples(
    pca: PCA,
    data: pd.DataFrame,
    ref_data: np.ndarray,
    component: int,
    columns: list[str],
    sample_reps: int,
    sample_size: int,
) -> list[dict]:
    ref_values = ref_data[:, component]
    sample_stats = []

    for sample in range(sample_reps):
        sample_values = pca.transform(
            data.sample(frac=1, random_state=sample)
            .groupby("TICK")
            .head(sample_size)[columns]
            .values
        )[:, component]

        sample_ks_stats = get_ks_statistic(sample_values, ref_values)
        sample_ks_stats.update({"SAMPLE": sample, "N": len(sample_values)})

        sample_stats.append(sample_ks_stats)

    return sample_stats
