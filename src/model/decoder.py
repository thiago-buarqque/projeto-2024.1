import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, in_channels=16, out_channels=3):
        super(Decoder, self).__init__()
        self.input_channels = in_channels
        self.out_channels = out_channels

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: b, 256, 8, 8
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # Output: b, 128, 16, 16
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # Output: b, 64, 32, 32
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # Output: b, 32, 64, 64
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # Output: b, 16, 128, 128
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # Output: b, 8, 256, 256
            nn.BatchNorm2d(8, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, out_channels, 3, stride=1, padding=1),  # Output: b, out_channels, 256, 256
            nn.ReLU()
        )
        
    def forward(self, x):
         # Reshape x to [batch_size, in_channels, height, width]
         batch_size = x.shape[0]
         x = x.view(batch_size, self.input_channels, 4, 4)
         
         reconstruction = self.decoder(x)

         return reconstruction
