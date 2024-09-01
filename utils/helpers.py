import torch



def time_series_dataset(
	series: torch.Tensor,
	window_size: int
) -> tuple[torch.Tensor, torch.Tensor]:

	inputs = []
	outputs = []

	for i in range(len(series) - window_size):

		inputs.append(series[i:i + window_size])
		outputs.append(series[i + window_size])

	return torch.vstack(inputs)[..., None], torch.vstack(outputs)
