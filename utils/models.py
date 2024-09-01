import torch
import torch.nn as nn
from kan.KANLayer import KANLayer



class KARNN(nn.Module):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dim: int,
		num_intervals: int = 5,
		degree: int = 3,
		device: str | torch.device = "cpu"
	) -> None:

		super().__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.hidden_dim = hidden_dim
		self.hidden_state = torch.zeros(hidden_dim)
		self.hidden_update_layer = KANLayer(
			in_dim = in_dim + hidden_dim,
			out_dim = hidden_dim,
			num = num_intervals,
			k = degree,
			device = device
		)
		# self.hidden_update_layer = MultKAN(
		# 	width = [in_dim + hidden_dim, hidden_dim, hidden_dim],
		# 	grid = num_intervals,
		# 	k = degree
		# )
		self.output_layer = KANLayer(
			in_dim = hidden_dim,
			out_dim = out_dim,
			num = num_intervals,
			k = degree,
			device = device
		)
		# self.output_layer = MultKAN(
		# 	width = [hidden_dim, hidden_dim, out_dim],
		# 	grid = num_intervals,
		# 	k = degree
		# )
		self.device = device
		self.to(device)
	
	def to(
		self,
		device: str | torch.device
	) -> "KARNN":
		
		super().to(device)
		self.hidden_state = self.hidden_state.to(device)
		return self
	
	def forward(
		self,
		x: torch.Tensor
	) -> tuple[torch.Tensor, torch.Tensor]:
		
		hidden = self.hidden_state
		hidden_states = []

		for x_t in x:

			# Update hidden state
			hidden_input = torch.cat([x_t, hidden])[None, :]
			# hidden = self.hidden_update_layer(hidden_input)[0]
			hidden = self.hidden_update_layer(hidden_input)[0][0]
			hidden_states.append(hidden)

		# Get outputs
		hidden_states = torch.vstack(hidden_states)
		# outputs = self.output_layer(hidden_states)
		outputs = self.output_layer(hidden_states)[0]

		self.hidden_state = hidden

		return outputs, hidden_states
	
	def reset(self) -> None:
		self.hidden_state = torch.zeros(self.hidden_dim, device=self.device)
	
	def train(
		self,
		inputs: torch.Tensor,
		outputs: torch.Tensor,
		epochs: int,
		learning_rate: float
	) -> None:
		
		inputs = inputs.to(self.device)
		outputs = outputs.to(self.device)
		optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
		loss_fn = torch.nn.MSELoss()

		batches = len(inputs)

		for epoch in range(epochs):

			epoch_loss = 0

			for input, output in zip(inputs, outputs):

				self.reset()
				optimizer.zero_grad()
				prediction = self(input)[0][-1]
				loss = loss_fn(prediction, output)
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
			
			print(f"Epoch {epoch + 1} Loss: {epoch_loss / batches}")
