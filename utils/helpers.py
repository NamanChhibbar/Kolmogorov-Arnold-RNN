import os

import pandas as pd
import torch
import torch.nn as nn


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

  inputs = []
  outputs = []

  for i in range(len(series) - window_size):

    inputs.append(series[i:i + window_size][..., None])
    outputs.append(series[i + window_size][..., None])

  return inputs, outputs


def train_model(
  model: nn.Module,
  inputs: torch.Tensor,
  outputs: torch.Tensor,
  epochs: int,
  batch_size: int,
  learning_rate: float,
  factor: float = .1,
  patience: int = 10,
  threshold: float = 1e-4,
  device: str | torch.device = 'cpu'
) -> list[float]:
  
  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    factor=factor,
    patience=patience,
    threshold=threshold
  )

  batches = len(inputs) / batch_size

  model.train()
  model.to(device)
  loss_history = []

  for epoch in range(epochs):

    epoch_loss = .0

    for input, output in zip(inputs, outputs):

      input = input.to(device)
      output = output.to(device)

      optimizer.zero_grad()
      prediction: torch.Tensor = model(input)[0][-1]
      loss: torch.Tensor = loss_fn(prediction, output)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
    
    avg_loss = epoch_loss / batches
    loss_history.append(avg_loss)

    scheduler.step(avg_loss)

    print(f'Epoch [{epoch + 1}/{epochs}] Loss [{avg_loss}]')

  model.eval()
  model.to('cpu')
  return loss_history
