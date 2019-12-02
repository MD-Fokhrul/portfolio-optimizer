import torch.nn as nn
import torch
from future_prices.util import find_latest_model_name


class EstimationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(EstimationModel, self).__init__()

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(input_size, int(input_size / 2)),
            nn.ReLU(),
            nn.Linear(int(input_size / 2), int(input_size / 4)),
            nn.ReLU(),
            nn.Linear(int(input_size / 4), int(input_size / 8)),
            nn.ReLU(),
            nn.Linear(int(input_size / 8), output_size)
        )

    def forward(self, data):
        return self.regressor(data)

    def load(self, model_dir_path):
        model_path = find_latest_model_name(model_dir_path)
        print('Loading saved model: {}'.format(model_path))
        self.load_state_dict(torch.load(model_path))


