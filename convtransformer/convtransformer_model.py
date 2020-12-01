"""Convolutional Transformer"""

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    base_architecture
)

from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from .convtransformer_layer import ConvTransformerEncoderLayer


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
        super(ConvTransformerModel, ConvTransformerModel).add_args(parser)
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

    def build_encoder_layer(self, args):
        layer = ConvTransformerEncoderLayer(args)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer


@register_model_architecture("convtransformer", "convtransformer")
def convtransformer(args):
    base_architecture(args)
    args.context_size = getattr(args, "context_size", 3)
