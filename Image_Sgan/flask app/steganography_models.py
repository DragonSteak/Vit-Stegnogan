import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicEncoder(nn.Module):
    def _name(self):
        return "BasicEncoder"

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size, 3),
        )
        return self.conv1, self.conv2, self.conv3, self.conv4

    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_models()
        self.name = self._name()

    def forward(self, image, data):
        x = self._models[0](image)
        x_1 = self._models[1](torch.cat([x] + [data], dim=1))
        x_2 = self._models[2](x_1)
        x_3 = self._models[3](x_2)
        return x_3


class DenseEncoder(BasicEncoder):
    def _name(self):
        return "DenseEncoder"

    def _build_models(self):
        self.conv1 = super()._build_models()[0]
        self.conv2 = super()._build_models()[1]
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2 + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3 + self.data_depth, 3)
        )
        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, image, data):
        x = self._models[0](image)
        x_list = [x]
        x_1 = self._models[1](torch.cat(x_list + [data], dim=1))
        x_list.append(x_1)
        x_2 = self._models[2](torch.cat(x_list + [data], dim=1))
        x_list.append(x_2)
        x_3 = self._models[3](torch.cat(x_list + [data], dim=1))
        x_list.append(x_3)
        return image + x_3


class BasicDecoder(nn.Module):
    def _name(self):
        return "BasicDecoder"

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size, self.data_depth),
        )
        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, image):
        x = self._models[0](image)
        x_1 = self._models[1](x)
        x_2 = self._models[2](x_1)
        x_3 = self._models[3](x_2)
        return x_3

    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_models()
        self.name = self._name()


class DenseDecoder(BasicDecoder):
    def _name(self):
        return "DenseDecoder"

    def _build_models(self):
        self.conv1 = super()._build_models()[0]
        self.conv2 = super()._build_models()[1]
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3, self.data_depth),
        )
        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, image):
        x = self._models[0](image)
        x_list = [x]
        x_1 = self._models[1](torch.cat(x_list, dim=1))
        x_list.append(x_1)
        x_2 = self._models[2](torch.cat(x_list, dim=1))
        x_list.append(x_2)
        x_3 = self._models[3](torch.cat(x_list, dim=1))
        x_list.append(x_3)
        return x_3

# Example usage
if __name__ == "__main__":
    # Initialize models
    encoder = DenseEncoder(data_depth=4, hidden_size=32)
    decoder = DenseDecoder(data_depth=4, hidden_size=32)
    
    # Example input shapes
    batch_size = 1
    channels = 3
    height = 256
    width = 256
    
    # Create dummy inputs
    cover_image = torch.randn(batch_size, channels, height, width)
    secret_data = torch.randn(batch_size, 4, height, width)  # 4 channels for secret data
    
    # Forward pass
    stego_image = encoder(cover_image, secret_data)
    recovered_secret = decoder(stego_image)
    
    print(f"Cover image shape: {cover_image.shape}")
    print(f"Secret data shape: {secret_data.shape}")
    print(f"Stego image shape: {stego_image.shape}")
    print(f"Recovered secret shape: {recovered_secret.shape}") 