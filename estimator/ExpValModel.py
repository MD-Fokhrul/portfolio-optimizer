from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim

class ExpValModel(nn.Module):
    def __init__(self):
        super(ExpValModel, self).__init__()

        # for lstm?
        self.count = 0

        final_concat_size = 0

        # LSTM
        #self.lstm = nn.LSTM(input_size=505, hidden_size = 505, batch_first=False)

        # Regressor
        self.value = nn.Sequential(
            nn.Linear(10624, 5312),
            nn.ReLU(),
            nn.Linear(5312, 2656),
            nn.ReLU(),
            nn.Linear(2656, 1328),
            nn.ReLU(),
            nn.Linear(1328,505)
        )

    def forward(self, data):
        module_outputs = []
        lstm_i = []

        for d in data:
            x = self.value(d)
            lstm_i.append(x)
            self.count += 1
            #if self.count >= 90:
            module_outputs.append(x)

        # LSTM
        #i_lstm, _ = self.lstm(torch.stack(lstm_i))
        #module_outputs.append(i_lstm[-89])

        # Concatenate current output and LSTM output.
        # x_cat = torch.cat(module_outputs, dim=-1)

        # Feed concatenated outputs into the
        # regession networks.
        #prediction = []
        #prediction.append(self.value(x_cat))
        prediction = module_outputs
        return prediction
