from load_data import FaceLandmarksDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable

num_epochs = 20
batch_size = 64
learning_rate = 0.001

ds = FaceLandmarksDataset("./images/landmark_output.txt")
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

# 两层卷积
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3,padding=1),
            # nn.BatchNorm2d(8),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(8, 32, kernel_size=3,padding=1),
            # nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3,padding=1),
            # nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
			nn.Linear(64*8*8, 128),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.3),
			nn.Linear(128, 10))

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc(out)
        # out = out.view(out.size(0),5,2)  # torch.Size([64, 5, 2])
        return out

cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()

# 选择损失函数和优化方法
# loss_func = nn.L1Loss(reduction='mean')
loss_func = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        images = get_variable(data['image'])
        labels = get_variable(data['landmarks'])
        # labels = labels.view(labels.size(0),10)

        outputs = cnn(images)
        # print(labels.shape)
        # print(outputs.shape)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss:{}'.format(loss))
torch.save(cnn.state_dict(), './logs/model.pkl')