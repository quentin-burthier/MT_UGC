"""Convolutional Transformer"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from fairseq.modules import LayerNorm


@register_model('convtransformer')
class ConvTransformerModel(TransformerModel):
    """
    Args:
        encoder (ConvTransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        super().add_parser(parser)
        parser.add_argument('--context_size', type=int, default=3,
                            help='sets context size for convolutional layers')

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ConvTransformerEncoder(args, src_dict, embed_tokens)

class ConvTransformerEncoder(TransformerEncoder):
    """
    ConvTransformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`ConvTransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.context_size = args.context_size

        self.conv_layer_norm = LayerNorm(embed_tokens.embedding_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            ConvTransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1) # x here has the size [seq_len, batch_size, dim]

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        if self.layer_norm:
            x = x + self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x, encoder_padding_mask = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None and isinstance(encoder_out['encoder_out'], list):
            for i in range(len(encoder_out['encoder_out'])):
                encoder_out['encoder_out'][i] = \
                    encoder_out['encoder_out'][i].index_select(1, new_order)
        else:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)

        if encoder_out['encoder_padding_mask'] is not None and isinstance(encoder_out['encoder_padding_mask'], list):
            for i in range(len(encoder_out['encoder_padding_mask'])):
                if encoder_out['encoder_padding_mask'][i] is not None:
                    encoder_out['encoder_padding_mask'][i] = \
                        encoder_out['encoder_padding_mask'][i].index_select(0, new_order)
        else:
            if encoder_out['encoder_padding_mask'] is not None:
                encoder_out['encoder_padding_mask'] = \
                    encoder_out['encoder_padding_mask'].index_select(0, new_order)

        return encoder_out


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
            for kernel_size in (self.context_size + 2 * i for i in range(2))
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

        self.conv_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        x_org = x

        # combine the outputs of convolutional layers with different context size
        x = x.permute(1, 2, 0)
        x = torch.cat((conv_block(x).permute(2, 0, 1)
                       for conv_block in self.conv_blocks), 2)

        x = x.permute(1, 2, 0)
        x = self.block_4(x)
        x = x.permute(2, 0, 1)

        # residual connection
        x = x_org + x

        # below is the same as the traditional transformer
        x_norm = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x_self_attn, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm,
                                        key_padding_mask=encoder_padding_mask)
        x_self_attn_dropout = F.dropout(x_self_attn, p=self.dropout, training=self.training)

        residual = x_norm + x_self_attn_dropout

        x = residual
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x, encoder_padding_mask



@register_model_architecture('convtransformer', 'convtransformer')
def convtransformer(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    args.context_size = getattr(args, 'context_size', 3)
