from abc import abstractmethod
from pathlib import Path

import pandas as pd
from loguru import logger

from src.constants import DataType


class DatasetProcessor:
    """Base class for dataset processors."""

    def __init__(self, raw_paths: dict[str, Path], dataset_name: str):
        """
        Args:
            raw_paths: Paths to the raw data files, e.g.
                {'ia': Path('path/to/ia.csv'), 'fixations': Path('path/to/fixations.csv')}
            dataset_name: Name of the dataset directory under data/.
        """
        self.raw_paths = raw_paths
        self.dataset_name = dataset_name
        self.dataset_path = Path.cwd() / "data" / dataset_name
        self.harmonized_data_path = self.dataset_path / "harmonized"
        self.processed_metadata_path = self.dataset_path / "metadata"

    def process_data(self) -> dict[str, pd.DataFrame]:
        """Load the raw data, standardize column names, apply the dataset-specific
        processing, and save the result to the harmonized folder."""
        raw_data = self.load_raw_data()
        processed_data = {}
        for data_type, df in raw_data.items():
            if df is not None:
                processed_data[data_type] = self.standardize_column_names(
                    df, data_type=data_type
                )
        processed_data = self.dataset_specific_processing(processed_data)
        self.save_processed_data(processed_data)
        return processed_data

    def standardize_column_names(
        self, df: pd.DataFrame, data_type: DataType
    ) -> pd.DataFrame:
        """Rename columns according to the dataset's column map."""
        column_map = self.get_column_map(data_type)

        if column_map:
            valid_columns = {k: v for k, v in column_map.items() if k in df.columns}
            logger.info(
                f'Standardizing column names for {self.dataset_name} {data_type}: {valid_columns}'
            )
            return df.rename(columns=valid_columns)

        logger.info(
            f'{self.dataset_name} not found in column maps. No changes made.'
        )
        return df

    def save_processed_data(self, processed_data: dict[str, pd.DataFrame]) -> None:
        """Save each processed dataframe to harmonized/<data_type>.csv."""
        self.harmonized_data_path.mkdir(parents=True, exist_ok=True)

        for data_type, df in processed_data.items():
            if df is not None:
                output_path = self.harmonized_data_path / f'{data_type}.csv'
                df.to_csv(output_path)
                logger.info(f'Saved {data_type} to {output_path}')

    @abstractmethod
    def download(self) -> None:
        """Download the raw data files."""
        pass

    @abstractmethod
    def load_raw_data(self) -> dict[str, pd.DataFrame]:
        """Load raw data from the specified paths."""
        pass

    @abstractmethod
    def download_metadata(self) -> None:
        """Download the metadata files."""
        pass

    @abstractmethod
    def process_metadata(self) -> pd.DataFrame:
        """Process the metadata."""
        pass

    @abstractmethod
    def get_column_map(self, data_type: DataType) -> dict:
        """Get the column mapping for the dataset."""
        return {}

    @abstractmethod
    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Dataset-specific processing steps."""
        return data_dict
