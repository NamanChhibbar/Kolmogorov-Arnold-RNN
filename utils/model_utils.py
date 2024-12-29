from itertools import batched

import torch
import torch.nn as nn
from kan.KANLayer import KANLayer


class KARNNLayer(nn.Module):

  def __init__(
    self,
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    num_intervals: int = 5,
    degree: int = 3,
    device: str | torch.device = 'cpu'
  ) -> None:

    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.hidden_dim = hidden_dim
    self.num_intervals = num_intervals
    self.degree = degree
    self.device = device

    self.init_hidden_state = torch.zeros(hidden_dim, device=device)
    self.hidden_update_layer = KANLayer(
      in_dim=in_dim+hidden_dim,
      out_dim=hidden_dim,
      num=num_intervals,
      k=degree,
      device=device
    )
    self.output_layer = KANLayer(
      in_dim=hidden_dim,
      out_dim=out_dim,
      num=num_intervals,
      k=degree,
      device=device
    )
    self.to(device)
  
  def forward(
    self,
    x: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    
    hidden = self.init_hidden_state.clone().to(self.device)
    hidden_states = []

    for x_t in x:

      # Update hidden state
      hidden_input = torch.cat([x_t, hidden])[None, :]
      hidden = self.hidden_update_layer(hidden_input)[0][0]
      hidden_states.append(hidden)

    hidden_states = torch.vstack(hidden_states)

    # Get outputs if this is the output layer
    outputs = self.output_layer(hidden_states)[0]

    return outputs, hidden_states
  
  def to(
    self,
    device: str | torch.device
  ) -> 'KARNNLayer':
    
    super().to(device)
    self.device = device
    return self


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

  inds = torch.tensor(range(len(inputs)))

  for epoch in range(epochs):

    epoch_loss = .0

    for batch_ind in batched(inds, batch_size):

      inp = inputs[batch_ind].to(device)
      out = outputs[batch_ind].to(device)

      optimizer.zero_grad()
      prediction: torch.Tensor = model(inp)[0][-1]
      loss: torch.Tensor = loss_fn(prediction, out)
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
