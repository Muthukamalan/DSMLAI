import torch
import lightning as pl

from .input_embedding import InputEmbeddings
from .layernorm import LayerNormalization
from .positional_embedding import PositionalEncoding
from .projection_layer import ProjectionLayer
from .decoder import DecoderBlock

from typing import List, Dict, Tuple

from torchmetrics.text import CharErrorRate
# from transformers import (
#     AdamW,
#     AutoConfig,
#     AutoTokenizer,
#     get_linear_schedule_with_warmup,
# )

class BigramModel(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        vocab_size: int,
        n_head: int,
        n_layer: int,
        bias: bool = False,
        dropout_rate: float = 0.01,
    ) -> None:
        super().__init__()

        self.txt_embedding = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
        self.pos_embedding = PositionalEncoding(
            d_model=d_model, seq_len=seq_len, dropout=dropout_rate
        )
        self.layernorm = LayerNormalization()
        self.projection = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)
        self.decoder = torch.nn.Sequential(
            *[
                DecoderBlock(
                    d_model=d_model,
                    n_head=n_head,
                    seq_len=seq_len,
                    bias=bias,
                    attn_dropout=dropout_rate,
                )
                for _ in range(n_layer)
            ]
        )

        self.apply(self._init_weights)  # Init

    def _init_weights(self,module):
        if isinstance(module,torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        B, T = x.shape
        tok = self.txt_embedding(x)  # (bs,seq_len)
        pos = self.pos_embedding(tok)  # (bs,seq_len)
        src = tok + pos
        src = self.decoder(src)  # (bs,seq_len,d_model)
        src = self.layernorm(src)  # (bs,seq_len,d_model)
        logits = self.projection(src)  # (bs,seq_len,d_model)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = torch.nn.functional.cross_entropy(
                logits.view(B * T, C), y.view(B * T)
            )
        return logits, loss


class NanoGPT(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        vocab_size: int,
        n_head: int,
        n_layer: int,
        lr: float,
        bias: bool = False,
        dropout_rate: float = 0.01,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.seq_len = seq_len
        self.txt_embedding = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
        self.pos_embedding = PositionalEncoding(
            d_model=d_model, seq_len=seq_len, dropout=dropout_rate
        )
        self.layernorm = LayerNormalization()
        self.projection = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)
        self.decoder = torch.nn.Sequential(
            *[
                DecoderBlock(
                    d_model=d_model,
                    n_head=n_head,
                    seq_len=seq_len,
                    bias=bias,
                    attn_dropout=dropout_rate,
                )
                for _ in range(n_layer)
            ]
        )

        # self.train_cer = CharErrorRate()
        self.save_hyperparameters()
        # Init weights with normal distribution
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, y=None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.shape
        tok = self.txt_embedding(x)  # (bs,seq_len)
        src = self.pos_embedding(tok)  # (bs,seq_len)
        logits = self.projection(
                        self.layernorm(
                            self.decoder(src)   # (bs,seq_len, d_model)
                        )                       # (bs,seq_len, d_model)
                ) 

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = torch.nn.functional.cross_entropy(
                logits.view(B * T, C), y.view(B * T)
            )
        return logits, loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,fused=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                    optimizer=optimizer,
                                    max_lr=.01,
                                    total_steps=self.trainer.estimated_stepping_batches,
                                    pct_start=.5,
                                    cycle_momentum=True,
                                    div_factor =100,
                                    final_div_factor = 1e10,
                                    verbose = False,
                                    three_phase=True
                                    )
        return {
            'optimizer':optimizer,
            "lr_scheduler":{
                'scheduler':scheduler,
                'interval':'step',
                'frequency':1
            }
        }

        # scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches,)
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return [optimizer], [scheduler]
        # return ([optimizer],[scheduler])



    def training_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor | None]:
        X, y = batch
        logits, loss = self(X, y)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            logger=self.trainer.logger,
        )
        return {'loss':loss,'logits':logits}

    def validation_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor | None]:
        X, y = batch
        logits, loss = self(X, y)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            logger=self.trainer.logger,
        )
        return {'loss':loss,'logits':logits}
