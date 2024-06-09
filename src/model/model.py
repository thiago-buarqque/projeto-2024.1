import torch.nn as nn
from model.vit import ViT
from model.decoder import Decoder

class Model(nn.Module):
    def __init__(self, image_size = 256,
                    patch_size = 64,
                    channels=3,
                    num_classes = 1,
                    dim = 256,
                    depth = 6,
                    heads = 8,
                    mlp_dim = 1024,
                    decoder_input_channels = 16,
                    reconstruction=True):

        super(Model, self).__init__()
        
        self.reconstruction = reconstruction
        
        if image_size == 512:
            dim = 512
        
        self.vit = ViT(
            image_size = image_size,
            patch_size = patch_size,
            channels=channels,
            num_classes = num_classes,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            reconstruction = reconstruction)
     
        if self.reconstruction:
            self.decoder = Decoder(in_channels=decoder_input_channels, out_channels=channels)

    def forward(self,x):       
        encoded = self.vit(x)
        
        if not self.reconstruction:
            return encoded
        # print(f"ViT output: {encoded.shape}")
        recons = self.decoder(encoded)
            
        return encoded, recons
