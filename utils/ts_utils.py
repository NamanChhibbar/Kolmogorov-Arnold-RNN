import os

import pandas as pd
import torch


def load_time_series(
  data_path: str,
  date_column: str,
  columns: list[str] | None = None
) -> pd.DataFrame:
  '''
  Loads time series from a csv or excel file.

  ## Parameters
  `data_path`: Path to csv or excel file
  `date_column`: Column to parse date indices
  `columns`: Columns to be imported

  ## Returns
  `data`: Dataframe containing time series with date index
  '''

  # Check the extension of the file
  _, extension = os.path.splitext(data_path)

  # Load data
  match extension:
    case '.csv':
      data = pd.read_csv(data_path, index_col=date_column, parse_dates=True)
    case '.xlsx':
      data = pd.read_excel(data_path, index_col=date_column, parse_dates=True)
    case _:
      raise ValueError('Invalid file extension')

  # Select columns
  if columns:
    data = data[columns]

  # Drop missing values
  data.dropna(inplace=True)

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
