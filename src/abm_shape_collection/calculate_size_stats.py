import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_size_stats(
    data: pd.DataFrame,
    ref_data: pd.DataFrame,
    regions: list[str],
    include_ticks: bool = False,
    include_samples: bool = False,
    sample_reps: int = 1,
    sample_size: int = 1,
) -> pd.DataFrame:
    all_stats = []

    features = [
        f"{feature}.{region}" if region != "DEFAULT" else feature
        for region in regions
        for feature in ["volume", "height"]
    ]

    for feature in features:
        stats = calculate_size_stats_for_all(data, ref_data, feature)

        if include_ticks:
            stats = stats + calculate_size_stats_for_ticks(data, ref_data, feature)

        if include_samples:
            stats = stats + calculate_size_stats_for_samples(
                data, ref_data, feature, sample_reps, sample_size
            )

        stats_df = pd.DataFrame(stats)
        stats_df["FEATURE"] = feature
        all_stats.append(stats_df)

    all_stats_df = pd.concat(all_stats).astype({"N": int})

    return all_stats_df


def calculate_size_stats_for_all(
    data: pd.DataFrame, ref_data: pd.DataFrame, feature: str
) -> list[dict]:
    ref_values = ref_data[feature].values
    values = data[feature].values

    ks_stats = get_ks_statistic(values, ref_values)
    ks_stats.update({"N": len(values)})

    return [ks_stats]


def calculate_size_stats_for_ticks(
    data: pd.DataFrame, ref_data: pd.DataFrame, feature: str
) -> list[dict]:
    ref_values = ref_data[feature].values
    tick_stats = []

    for tick, tick_data in data.groupby("TICK"):
        tick_values = tick_data[feature].values

        tick_ks_stats = get_ks_statistic(tick_values, ref_values)
        tick_ks_stats.update({"TICK": tick, "N": len(tick_values)})

        tick_stats.append(tick_ks_stats)

    return tick_stats


def calculate_size_stats_for_samples(
    data: pd.DataFrame, ref_data: pd.DataFrame, feature: str, sample_reps: int, sample_size: int
) -> list[dict]:
    ref_values = ref_data[feature].values
    sample_stats = []

    for sample in range(sample_reps):
        sample_values = (
            data.sample(frac=1, random_state=sample)
            .groupby("TICK")
            .head(sample_size)[feature]
            .values
        )

        sample_ks_stats = get_ks_statistic(sample_values, ref_values)
        sample_ks_stats.update({"SAMPLE": sample, "N": len(sample_values)})

        sample_stats.append(sample_ks_stats)

    return sample_stats


def get_ks_statistic(population_a: np.ndarray, population_b: np.ndarray) -> dict:
    ksresult = ks_2samp(population_a, population_b, mode="asymp")

    return {
        "KS_STATISTIC": ksresult.statistic,
        "KS_PVALUE": ksresult.pvalue,
    }
