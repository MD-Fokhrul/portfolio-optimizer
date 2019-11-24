from torchvision import models
import torch.nn as nn
import torch


class PricePredictionModel(nn.Module):
    def __init__(self, input_output_size=505, hidden_size=128):
        super(PricePredictionModel, self).__init__()
        final_concat_size = 0

        # Main CNN
        cnn = models.resnet34(pretrained=True)
        feats = list(cnn.children())[:-1]
        feats[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.features = nn.Sequential(*feats)
        self.intermediate = nn.Sequential(nn.Linear(
            cnn.fc.in_features, input_output_size),
            nn.ReLU())
        final_concat_size += input_output_size

        # Main LSTM
        self.lstm = nn.LSTM(input_size=input_output_size,
                            hidden_size=hidden_size,
                            num_layers=3,
                            batch_first=False)
        final_concat_size += hidden_size

        # Prices Regressor
        # todo: try making hidden size bigger than output?
        #  or multiply hidden size for middle layer instead of division?
        #  or use cnn with deconv instead of fc?
        self.predict_prices = nn.Sequential(
            nn.Linear(final_concat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), input_output_size)
        )

    def forward(self, data_input):
        module_outputs = []
        lstm_i = []
        # Loop through temporal sequence of
        # price images and pass
        # through the cnn.

        for idx, v in data_input.items():
            x = self.features(v['past_prices'])
            x = x.view(x.size(0), -1)
            x = self.intermediate(x)
            lstm_i.append(x)
            # feed the current
            # output directly into the
            # regression network.
            if idx == 0:
                module_outputs.append(x)

        # Feed temporal outputs of CNN into LSTM
        i_lstm, _ = self.lstm(torch.stack(lstm_i))
        module_outputs.append(i_lstm[-1])

        # Concatenate current image CNN output
        # and LSTM output.
        x_cat = torch.cat(module_outputs, dim=-1)

        # Feed concatenated outputs into the
        # regession networks.
        prediction = {'next_prices': torch.squeeze(self.predict_prices(x_cat))}
        return prediction

