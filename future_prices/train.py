from torch import nn, optim
import time
import torch


def train(model, lr, train_loader, validation_loader, epochs, device, save_checkpoints, checkpoints_dir,
          checkpoints_interval, log_interval, log_batches, log_epochs, log_comet, experiment):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(0, epochs):
        start_epoch_train = time.time()
        epoch_prices_losses = []
        # epoch training
        model.train()
        running_prices_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            for k, v in data.items():
                for k2, v2 in v.items():
                    data[k][k2] = v2.float().to(device)
            for k, v in target.items():
                target[k] = v.float().to(device)

            # for k,v in data.items():
            #     print('{}: {}'.format(k, v))


            optimizer.zero_grad()
            prediction = model(data)

            prices_loss = criterion(prediction['next_prices'], target['next_prices'].squeeze())
            combined_loss = prices_loss # if we add more factor other than prices, consider weighting
            combined_loss.backward()

            optimizer.step()

            # print avg batch statistics
            running_prices_loss += prices_loss.item()

            if batch_idx > 0 and batch_idx % log_interval == 0:
                avg_batch_prices_loss = running_prices_loss / log_interval
                epoch_prices_losses.append(avg_batch_prices_loss)
                if log_comet and log_batches:
                    experiment.log_metric('batch_train_prices_loss', avg_batch_prices_loss, step=batch_idx)
                print('[epoch: %d, batch:  %5d] prices loss: %.5f' % (epoch + 1, batch_idx + 1, avg_batch_prices_loss))
                running_prices_loss = 0.0

            # Remove this when actually training.
            # Used to terminate early.
        #             if batch_idx >= 4:
        #                 break

        if log_comet and log_epochs:
            if len(epoch_prices_losses) > 0 and len(epoch_prices_losses) > 0:
                epoch_prices_loss = sum(epoch_prices_losses) / len(epoch_prices_losses)
                print('[avg train loss epoch %d] prices loss %.5f' % (epoch, epoch_prices_loss))
                experiment.log_metric('epoch_train_prices_loss', epoch_prices_loss, epoch)
            else:
                print('0 epoch losses for training')
        end_epoch_train = time.time()
        epoch_elapsed = end_epoch_train - start_epoch_train

        if save_checkpoints and (epoch+1) % checkpoints_interval == 0:
            print('Saving interim model...')
            model_path = '{}/model_{}.pth'.format(checkpoints_dir, epoch)
            torch.save(model.state_dict(), model_path)

        print('epoch %d: %f elapsed' % (epoch+1, epoch_elapsed))
        if log_comet:
            experiment.log_metric('epoch_train_time', end_epoch_train - start_epoch_train, step=epoch)

        # epoch validation
        model.eval()
        with torch.no_grad():
            epoch_validation_prices_losses = []
            for batch_idx, (data, target) in enumerate(validation_loader):
                for k, v in data.items():
                    for k2, v2 in v.items():
                        data[k][k2] = v2.float().to(device)
                for k, v in target.items():
                    target[k] = v.float().to(device)

                prediction = model(data)

                prices_loss = criterion(prediction['next_prices'], target['next_prices'].squeeze())

                epoch_validation_prices_losses.append(prices_loss.item())

            if len(epoch_validation_prices_losses) > 0:
                epoch_validation_prices_loss = sum(epoch_validation_prices_losses) / len(epoch_validation_prices_losses)
                print('[avg validation loss epoch %d] prices loss %.5f' % (epoch, epoch_validation_prices_loss))

                if log_comet:
                    experiment.log_metric('epoch_val_prices_loss', epoch_validation_prices_loss, epoch)
            else:
                print('0 epoch losses for validation')