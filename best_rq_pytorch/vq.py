from torch import nn
from vector_quantize_pytorch import VectorQuantize


class VQ(nn.Module):
    def __init__(self):
        super().__init__()

        self.vq = VectorQuantize(
            heads=1,
            dim=1024,  # dimension of the input
            codebook_dim=32,  # dimension of the codebook
            codebook_size=16384,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.0,  # the weight on the commitment loss
        )

    def forward(self, x):
        quantized, indices, commit_loss = self.vq(x)
        return quantized, indices, commit_loss
