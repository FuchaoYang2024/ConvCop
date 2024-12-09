import torch.nn as nn

class CNN2ID(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels=64, kernel_size=5, padding=2)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels = 64, out_channels=64, kernel_size=5, padding=2)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels = 64, out_channels=64, kernel_size=5, padding=2)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9*64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        
        x = self.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        
        return x