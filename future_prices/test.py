import torch
import pandas as pd
import numpy as np


def test(model, predict_days, train_loader, test_loader, device, results_dir):

    initial_data_shape = test_loader.futureprices.dataframe.shape
    # init output file
    output_path = '{}/results.csv'.format(results_dir)
    columns = train_loader.futureprices.dataframe.columns
    pd.DataFrame([], columns=columns).to_csv(output_path)

    output_interval = 20
    last_output_index = 0
    num_batches = len(train_loader)

    model.eval()
    with torch.no_grad():
        # predict existing days
        output = []
        # for batch_idx, (data, _) in enumerate(train_loader):
        #     for k, v in data.items():
        #         for k2, v2 in v.items():
        #             data[k][k2] = v2.float().to(device)
        #
        #     prediction = model(data)
        #     output.append(prediction['next_prices'].detach().cpu().numpy())
        #
        #     if (batch_idx+1) % output_interval == 0 or (batch_idx+1) == num_batches:
        #         print('predicting training day t+1 {}/{}...'.format(
        #             batch_idx+1, num_batches))
        #
        #         print(len(columns))
        #         print(len(output))
        #         print(range(last_output_index, batch_idx+1))
        #         print(output)
        #         pd.DataFrame(output, columns=columns, index=range(last_output_index, batch_idx+1))\
        #             .to_csv(output_path,
        #                     header=False,
        #                     mode='a')
        #         last_output_index = batch_idx + 1
        #         output = []

        output = []
        # predict new days
        for day in range(predict_days):
            for batch_idx, (data, _) in enumerate(test_loader):
                for k, v in data.items():
                    for k2, v2 in v.items():
                        data[k][k2] = v2.float().to(device)

                prediction = model(data)
                test_loader.add_day(prediction['next_prices'].cpu().numpy())
                output.append(prediction['next_prices'].cpu().numpy())

                if (day+1) % 5 == 0 or (day+1) == predict_days:
                    print('predicting new day {}/{}...'.format(day+1, predict_days))

        print('Finished | initial data shape: {} | final data shape: {}'.format(
            initial_data_shape, test_loader.futureprices.dataframe.shape))

    print('Saving results to "{}"...'.format(output_path))
    output_df = pd.DataFrame(np.array(output), columns=test_loader.futureprices.dataframe.columns)
    output_df.index = range(last_output_index, last_output_index+len(output_df))
    output_df.to_csv(output_path, header=False, mode='a')

