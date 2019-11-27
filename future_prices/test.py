import torch
from torch import nn


def test(model, predict_days, test_loader, device, results_dir):

    initial_data_shape = test_loader.futureprices.dataframe.shape

    model.eval()
    with torch.no_grad():

        for day in range(predict_days):
            for batch_idx, (data, _) in enumerate(test_loader):
                for k, v in data.items():
                    for k2, v2 in v.items():
                        data[k][k2] = v2.float().to(device)

                prediction = model(data)
                test_loader.add_day(prediction['next_prices'].cpu().numpy())

                if (day+1) % 5 == 0 or (day+1) == predict_days:
                    print('predicting day {}/{}...'.format(day+1, predict_days))

        print('Finished | initial data shape: {} | final data shape: {}'.format(
            initial_data_shape, test_loader.futureprices.dataframe.shape))

    output_path = '{}/results.csv'.format(results_dir)
    print('Saving results to "{}"...'.format(output_path))
    test_loader.futureprices.dataframe.iloc[-predict_days:].reset_index(drop=True, inplace=False).to_csv(output_path)




