'''
backboe: ResNet18
'''
import torch.nn as nn
import torch


class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, in_channels, std_channels ,stride=1, downsample=None):
        '''
        Bottleneck层有三个卷积层,个卷积层的卷积核大小为1*1,第二个卷积层的卷积核大小为3*3,
        第三个卷积层的卷积核大小为1*1,都不会改变输入的特征图的尺寸，只会改变通道数，前两个卷积层
        输出的通道数都是std_channels,第三个卷积层输出的通道数是std_channels*4

        downsample为None,表示没有下采样,否则为下采样操作,需要对残差块的输入进行下采样
        '''
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, std_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(std_channels)
        self.conv2 = nn.Conv2d(std_channels, std_channels, kernel_size=3, stride=stride, padding=1,dilation=1)
        self.bn2 = nn.BatchNorm2d(std_channels)
        self.conv3 = nn.Conv2d(std_channels, std_channels*Bottleneck.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(std_channels*Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out 


class ResNet(nn.Module):
    def __init__(self, layers ,num_classes=1):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*Bottleneck.expansion, num_classes)
    def _make_layer(self, std_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != std_channels*Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, std_channels*Bottleneck.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(std_channels*Bottleneck.expansion)
            )
        
        layers = []
        layers.append(
            Bottleneck(self.in_channels, std_channels, stride, downsample)
        )
        self.in_channels = std_channels*Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(self.in_channels, std_channels)
            )
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18():
    layers = [2,2,2,2]
    return ResNet(layers)


if __name__ == '__main__':
    model = ResNet18()
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    print(feature_extractor)
    x = torch.randn(1,3,224,224)
    out = feature_extractor(x)
    out = torch.flatten(out, 1)
    print(out.shape)