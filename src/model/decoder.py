import torch.nn as nn
import math
  
class Decoder(nn.Module):
    def __init__(self, in_channels=16, out_channels=3):
        super(Decoder, self).__init__()
        self.input_channels = in_channels
        self.out_channels = out_channels
        
        # self.input_channels, self.height, self.width = self.calculate_dimensions(self.input_channels)
        
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(in_channels= in_channels, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: b, 16, 2 * 8 - 2 + 3 - 2*1 + 1 = 16
             nn.BatchNorm2d(16, affine=True),
             nn.ReLU(True),            
             nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1),  # Output: b, 32, 32
             nn.BatchNorm2d(32, affine=True),
             nn.ReLU(True),             
             nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),  # Output: b, 32, 64
             nn.BatchNorm2d(32, affine=True),
             nn.ReLU(True), 
             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # Output: b, 16, 128
             nn.BatchNorm2d(16, affine=True),
             nn.ReLU(True), 
             nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # Output: b, 8, 256
             nn.BatchNorm2d(8, affine=True),
             nn.ReLU(True),
             nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1),  # Output: b, 8, 256x256
             nn.BatchNorm2d(8, affine=True),
             nn.ReLU(True),
             nn.ConvTranspose2d(8, out_channels, 3, stride=1, padding=1),  # Output: b, 3, 256
             nn.Tanh()
             )
        
    def forward(self, x):
         # Reshape x to [batch_size, in_channels, height, width]
         batch_size = x.shape[0]
         x = x.view(batch_size, self.input_channels, 4, 4)
         
         reconstruction = self.decoder(x)
         
         return reconstruction
     
    @staticmethod
    def calculate_dimensions(input_dim):
        # Calculate possible dimensions for the given input_dim
        # Here, we assume the number of channels to be the highest factor
        factors = [(i, input_dim // i) for i in range(1, int(math.sqrt(input_dim)) + 1) if input_dim % i == 0]
        input_channels = max([f[0] for f in factors])
        spatial_dim = input_dim // input_channels
        height = width = int(math.sqrt(spatial_dim))
        
        if height * width != spatial_dim:
            raise ValueError("Cannot reshape input dimension to a square spatial dimension.")
        
        return input_channels, height, width
