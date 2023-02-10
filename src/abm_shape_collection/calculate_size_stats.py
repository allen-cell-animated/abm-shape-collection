import numpy as np
import pandas as pd
from prefect import task

from abm_shape_collection.calculate_shape_stats import get_ks_statistic


@task
def calculate_size_stats(
    data: pd.DataFrame, ref_data: pd.DataFrame, regions: list[str]
) -> pd.DataFrame:
    all_stats = []

    for region in regions:
        for feature in ["volume", "height"]:
            column_name = f"{feature}.{region}" if region != "DEFAULT" else feature
            ref_values = ref_data[column_name].values
            values = data[column_name].values

            ks_stats = get_ks_statistic(values, ref_values)
            ks_stats.update({"FEATURE": column_name, "TICK": np.nan})
            all_stats.append(ks_stats)

            for tick, tick_data in data.groupby("TICK"):
                tick_values = tick_data[column_name].values

                tick_ks_stats = get_ks_statistic(tick_values, ref_values)
                tick_ks_stats.update(
                    {
                        "FEATURE": column_name,
                        "TICK": tick,
                    }
                )
                all_stats.append(tick_ks_stats)

    all_stats_df = pd.DataFrame(all_stats)

    return all_stats_df
