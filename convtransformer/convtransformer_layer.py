"""Convolutional block then Transformer layer.

As described in Character-Level Translation with Self-attention, Gao et al
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.models.transformer import TransformerEncoderLayer


class ConvTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer block.

    In the original paper each operation is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__(args)
        self.context_size = args.context_size

        # Define the three convolutional layers with different context sizes
        # The convolutions are implemented as separable convolutions for
        # reducing the number of model parameters
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                          groups=self.embed_dim,
                          kernel_size=kernel_size,
                          padding=(kernel_size - 1)//2),
                nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                          kernel_size=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            )
            for kernel_size in (self.context_size + 2 * i for i in range(3))
        ])

        # conv4 here is a standard convolution layer, for reshape the concatenation
        self.block_4 = nn.Sequential(
            nn.Conv1d(in_channels=self.embed_dim*3, out_channels=self.embed_dim,
                      kernel_size=(self.context_size,),
                      padding=(self.context_size-1)//2),
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                      kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        x_org = x
        # combine the outputs of convolutional layers with different context size
        x = x.permute(1, 2, 0)
        x = torch.cat([conv_block(x)
                       for conv_block in self.conv_blocks], 1)
        x = self.block_4(x)
        x = x.permute(2, 0, 1)
        # residual connection
        x = self.residual_connection(x_org, x)

        x = super().forward(x, encoder_padding_mask, attn_mask)

        return x
