import torch.nn as nn

class TasadModel(nn.Module):
    """
    Segmentation Network consisting of an Encoder and a Decoder.
    
    Attributes:
    ----------
    encoder : nn.Module
        The encoder part of the network.
    decoder : nn.Module
        The decoder part of the network.
    """
    def __init__(self, in_channels=3, out_channels=3, base_width=128):
        """
        Initializes the SegNetwork with given parameters.

        Parameters:
        ----------
        in_channels : int
            Number of input channels (default is 3).
        out_channels : int
            Number of output channels (default is 3).
        base_width : int
            Base width for the network (default is 64).
        """
        super(TasadModel, self).__init__()
        self.encoder = Encoder(in_channels, base_width)
        self.decoder = Decoder(base_width, out_channels)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor.

        Returns:
        -------
        torch.Tensor
            Output tensor.
        """
        b3 = self.encoder(x)
        output = self.decoder(b3)
        return output

    @staticmethod
    def get_n_params(model):
        """
        Calculate the number of parameters in the model.

        Parameters:
        ----------
        model : nn.Module
            The model whose parameters are to be counted.

        Returns:
        -------
        int
            Total number of parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Encoder(nn.Module):
    """
    Encoder module for the Segmentation Network.
    """
    def __init__(self, in_channels, base_width):
        """
        Initializes the Encoder with given parameters.

        Parameters:
        ----------
        in_channels : int
            Number of input channels.
        base_width : int
            Base width for the encoder.
        """
        super(Encoder, self).__init__()

        self.block1 = self._make_block(in_channels, base_width)
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = self._make_block(base_width, base_width * 2)
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = self._make_block(base_width * 2, base_width * 4)

    def _make_block(self, in_channels, out_channels):
        """
        Creates a convolutional block with two Conv2d layers, 
        each followed by BatchNorm2d and ReLU activation.

        Parameters:
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        Returns:
        -------
        nn.Sequential
            The convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass through the encoder.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor.

        Returns:
        -------
        torch.Tensor
            Encoded tensor.
        """
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        return b3


class Decoder(nn.Module):
    """
    Decoder module for the Segmentation Network.
    """
    def __init__(self, base_width, out_channels=1):
        """
        Initializes the Decoder with given parameters.

        Parameters:
        ----------
        base_width : int
            Base width for the decoder.
        out_channels : int
            Number of output channels (default is 1).
        """
        super(Decoder, self).__init__()

        self.up1 = self._make_upsample_block(base_width * 4, base_width * 4)
        self.db1 = self._make_double_block(base_width * 4, base_width * 2)
        self.up2 = self._make_upsample_block(base_width * 2, base_width * 2)
        self.db2 = self._make_double_block(base_width * 2, base_width)
        self.final_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def _make_upsample_block(self, in_channels, out_channels):
        """
        Creates an upsampling block with Upsample, Conv2d, BatchNorm2d, 
        and ReLU activation.

        Parameters:
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        Returns:
        -------
        nn.Sequential
            The upsampling block.
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_double_block(self, in_channels, out_channels):
        """
        Creates a block with two Conv2d layers, each followed by BatchNorm2d
        and ReLU activation.

        Parameters:
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        Returns:
        -------
        nn.Sequential
            The double convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, b3):
        """
        Forward pass through the decoder.

        Parameters:
        ----------
        b3 : torch.Tensor
            Encoded tensor from the encoder.

        Returns:
        -------
        torch.Tensor
            Decoded tensor.
        """
        up1 = self.up1(b3)
        db1 = self.db1(up1)
        up2 = self.up2(db1)
        db2 = self.db2(up2)
        out = self.final_out(db2)
        return out
