from typing import Optional, Union

import pandas as pd

from src.logger_config import setup_logging

logger = setup_logging()


def select_and_save_data(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Filters specific columns and saves the DataFrame to a CSV file.
    """
    columns_to_keep = [
        "manufacturer",
        "model",
        "year",
        "type",
        "odometer",
        "state",
        "price",
    ]

    # Check if all columns exist in the DataFrame
    existing_columns = [col for col in columns_to_keep if col in df.columns]

    if len(existing_columns) < len(columns_to_keep):
        missing = set(columns_to_keep) - set(existing_columns)
        logger.warning(f"Missing columns found and skipped: {missing}")

    df_filtered = df[existing_columns]

    # Save to file
    df_filtered.to_csv(output_path, index=False)
    logger.info(f"Filtered data saved to: {output_path}. Shape: {df_filtered.shape}")

    return df_filtered


def filter_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes price and year outliers to keep the dataset relevant for market analysis.
    """
    initial_count = len(df)

    # 1. Filter by Year
    df = df[(df["year"] >= 2000) & (df["year"] <= 2027)]

    # 2. Filter by Price
    df = df[(df["price"] >= 1000) & (df["price"] <= 150000)]

    final_count = len(df)
    logger.info(
        f"Anomaly filtering complete. Removed {initial_count - final_count} rows."
    )

    return df


def get_column_nan_percentage(
    df: pd.DataFrame, column_name: Optional[str] = None
) -> Union[float, pd.Series]:
    """
    Calculates the percentage of missing values.
    If column_name is provided, returns float for that column.
    If column_name is None, returns a Series for all columns.
    """
    total_count = len(df)

    if column_name:
        if column_name not in df.columns:
            logger.warning(f"Column '{column_name}' not found.")
            return 0.0

        nan_count = df[column_name].isnull().sum()
        percentage = (nan_count / total_count) * 100
        logger.info(f"Missing values in '{column_name}': {percentage:.2f}%")
        return percentage

    else:
        logger.info("Calculating missing values percentage for all columns...")
        nan_percentages = (df.isnull().sum() / total_count) * 100

        report = nan_percentages.sort_values(ascending=False)
        return report


def fill_type_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing 'type' with 'other' and drops rows with ANY other missing values.
    """
    initial_count = len(df)

    if "type" in df.columns:
        df["type"] = df["type"].fillna("other")
        logger.info("Filled missing 'type' values with 'other'")

    df_cleaned = df.dropna()

    final_count = len(df_cleaned)
    logger.info(
        f"Row-wise cleaning complete. "
        f"Removed {initial_count - final_count} rows with missing data. "
        f"Remaining: {final_count}"
    )

    return df_cleaned
