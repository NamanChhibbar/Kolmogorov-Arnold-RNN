import torch
import torch.nn as nn



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
	learning_rate: float,
	device: str | torch.device
) -> None:
	
	inputs = inputs
	outputs = outputs
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	loss_fn = torch.nn.MSELoss()

	batches = len(inputs)

	for epoch in range(epochs):

		epoch_loss = 0.

		for input, output in zip(inputs, outputs):

			input = input.to(device)
			output = output.to(device)

			optimizer.zero_grad()
			prediction: torch.Tensor = model(input)[0][-1]
			loss: torch.Tensor = loss_fn(prediction, output)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
		
		print(f"Epoch {epoch + 1} Loss: {epoch_loss / batches}")
