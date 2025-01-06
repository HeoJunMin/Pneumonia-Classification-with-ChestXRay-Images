import torch
import torch.nn as nn
import torch.nn.functional as F

### Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        # 입력 채널과 출력 채널이 다르면 shortcut 레이어를 추가
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

### Define ResNet-like CNN Model
class PneumoniaClassificationModel(nn.Modele):
    def __init__(self, num_classes = 2):
        super(PneumoniaClassificationModel, self).__init__()

        # Input Layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),         # 흑백 이미지이므로 입력 채널 = 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        # Residual Blocks
        self.layer1 = ResidualBlock(64, 64, stride = 1)
        self.layer2 = ResidualBlock(64, 128, stride = 2)
        self.layer3 = ResidualBlock(128, 256, stride = 2)
        self.layer4 = ResidualBlock(256, 512, stride = 2)

        # Adaptive Pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier (Fully Connected Layer)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, num_classes)                                         # 이진 분류이므로 num_classes(=2)로 마무리
        )


    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)                                               # Flatten 작업 수행
        x = self.classifier(x)                                                  # FC-Layer
        return x

  # Loss Function, Optimizer, Epochs

import torch.optim as optim

model = PneumoniaClassificationModel(num_classes = 2).cuda()

criterion_for_train = nn.CrossEntropyLoss()
criterion_for_test = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
epochs = 50
