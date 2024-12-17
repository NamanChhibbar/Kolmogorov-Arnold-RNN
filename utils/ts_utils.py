import os

import pandas as pd
import torch


def load_time_series(
  data_path: str,
  date_column: str,
  columns: list[str]
) -> pd.DataFrame:
  '''
  Loads time series from a csv or excel file.

  ## Parameters
  `data_path`: Path to data file
  `date_column`: Column to parse date indices
  `columns`: Columns to be imported

  ## Returns
  `data`: Dataframe containing time series with date index
  '''

  # Check the extension of the file
  _, extension = os.path.splitext(data_path)
  match extension:
      case '.csv': data = pd.read_csv(data_path, parse_dates=[date_column])
      case '.xlsx': data = pd.read_excel(data_path, parse_dates=[date_column])
  
  # Parse date
  data[date_column] = pd.to_datetime(data[date_column].dt.date)

  # Set date as index
  data.index = data[date_column]

  # Sample by day
  data = data.resample('d')[*columns].sum()

  return data


def time_series_dataset(
  series: torch.Tensor,
  window_size: int
) -> tuple[torch.Tensor, torch.Tensor]:

  inputs = [
    series[i:i + window_size][..., None]
    for i in range(len(series) - window_size)
  ]
  outputs = [
    series[i + window_size][..., None]
    for i in range(len(series) - window_size)
  ]

  return inputs, outputs
