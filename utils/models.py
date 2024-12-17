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
      in_dim = in_dim + hidden_dim,
      out_dim = hidden_dim,
      num = num_intervals,
      k = degree,
      device = device
    )
    self.output_layer = KANLayer(
      in_dim = hidden_dim,
      out_dim = out_dim,
      num = num_intervals,
      k = degree,
      device = device
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
        in_dim = in_dim if i == 0 else out_dim,
        out_dim = out_dim,
        hidden_dim = hidden_dim,
        num_intervals = num_intervals,
        degree = degree,
        device = device
      )
      for i in range(num_layers)
    ])

  def forward(
    self,
    inp: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:

    for layer in self.layers:
      inp, hidden_states = layer(inp)
    
    return inp, hidden_states
  
  def to(
    self,
    device: str | torch.device
  ) -> 'KARNN':
    
    super().to(device)
    for layer in self.layers:
      layer.to(device)
    self.device = device
    return self
