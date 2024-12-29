import torch
import torch.nn as nn

from utils import KARNNLayer


class KARNN(nn.Module):

  def __init__(
    self,
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    num_layers: int = 1,
    num_intervals: int = 5,
    degree: int = 3,
    device: str | torch.device = 'cpu'
  ) -> None:
    '''
    Initializes a KARNN model.
    '''
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.num_intervals = num_intervals
    self.degree = degree
    self.device = device
    self.layers = nn.ModuleList([
      KARNNLayer(
        in_dim=in_dim if i==0 else out_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_intervals=num_intervals,
        degree=degree,
        device=device
      )
      for i in range(num_layers)
    ])

  def forward(
    self,
    inp: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Forward pass of the model.
    '''
    for layer in self.layers:
      inp, hidden_states = layer(inp)
    return inp, hidden_states
  
  def to(
    self,
    device: str | torch.device
  ) -> 'KARNN':
    '''
    Moves the model to the specified device.
    '''
    super().to(device)
    for layer in self.layers:
      layer.to(device)
    self.device = device
    return self
