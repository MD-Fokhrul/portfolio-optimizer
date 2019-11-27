
import Preprocessor
import ExpValModel
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim

p = Preprocessor()
train_data, test_data, train_val, test_val = p.trainTestSplit(p.input_data, p.validification_data, 0.8)

model = ExpValModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
model.train()
for epoch in range(1):
    running_loss = 0.0
    for i, (data, target) in enumerate(zip(train_data,train_val)):
        if i == 1:
          print(i)
          print(data)
        optimizer.zero_grad()
        prediction = model(torch.from_numpy(data))

        loss = criterion(prediction, torch.from_numpy(target))

        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 10 == 1:
            print('[epoch: %d, batch:  %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 2.0))
            running_loss = 0.0

model.eval()
with torch.no_grad():
    for i, (data, target) in enumerate(zip(test_data,test_val)):
        prediction = model(torch.from_numpy(data))

        mse = criterion(prediction, torch.from_numpy(target))
        print(mse)
