import torch
from torch import nn

class SimpleCNN224(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN224, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(self._calculate_output_size(), num_classes)

    def _calculate_output_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            x = self.conv1(dummy_input)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.pool3(x)
            return x.flatten(1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MoreFCDNNFixed(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, input_height=224, input_width=224):
        super(MoreFCDNNFixed, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1_input_size = self._calculate_fc1_input_size(input_channels, input_height, input_width)
        self.fc1 = nn.Linear(self.fc1_input_size, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(256, num_classes)

    def _calculate_fc1_input_size(self, in_channels, height, width):
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, height, width)
            x = self.pool1(self.relu1(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(self.relu_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x

class DeeperFCDNNFixed(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, input_height=224, input_width=224):
        super(DeeperFCDNNFixed, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # New third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1_input_size = self._calculate_fc1_input_size(input_channels, input_height, input_width)
        self.fc1 = nn.Linear(self.fc1_input_size, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(256, num_classes)

    def _calculate_fc1_input_size(self, in_channels, height, width):
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, height, width)
            x = self.pool1(self.relu1(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = self.pool3(self.relu3(self.bn3(self.conv3(x)))) # Pass through the new layer
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Forward pass through the new layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dropout(self.relu_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x

class SingleLayerCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, input_height=224, input_width=224):
        super(SingleLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1_input_size = self._calculate_fc1_input_size(input_channels, input_height, input_width)
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def _calculate_fc1_input_size(self, in_channels, height, width):
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, height, width)
            x = self.pool1(self.relu1(self.bn1(self.conv1(dummy_input))))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dropout(self.relu_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x