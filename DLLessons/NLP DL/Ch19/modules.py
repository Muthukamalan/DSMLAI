import torch
from transformers import DistilBertModel,DistilBertConfig
from transformers.modeling_outputs import BaseModelOutput
import config as CFG
import timm



class ImageEncoder(torch.nn.Module):
    def __init__(
            self, 
            model_name= CFG.model_name,
            pretrained = CFG.pretrained,
            trainable = CFG.trainable,
            *args, 
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model:torch.nn.Module = timm.create_model( model_name=model_name,pretrained=pretrained, num_classes=0,global_pool='avg')
        for p in self.model.parameters(): p.requires_grad=trainable
        
    def forward(self,x:torch.Tensor):
        return self.model(x)


class TextEncoder(torch.nn.Module):
    def __init__(
            self, 
            model_name:str=CFG.text_encoder_model,
            pretrained:bool=CFG.pretrained,
            trainable:bool = CFG.trainable,
            *args, 
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if pretrained:
            self.model:DistilBertModel = DistilBertModel.from_pretrained(model_name)
        else:
            self.model:DistilBertModel = DistilBertModel(config=DistilBertConfig())
        
        # Trainable
        for p in self.model.parameters(): p.requires_grad = trainable 

        self.target_token_idx = 0
        # we are using the CLS token hidden representation as the sentence's embeddng
 

    def forward(self,input_ids,attention_mask):
        output:BaseModelOutput = self.model(input_ids=input_ids,attention_mask=attention_mask)

        # Sequence of hidden-states at the output of the last layer of the model.
        # last_hidden_state:: `(batch_size, max_seq_len , embedding_dim )`
        last_hidden_state   = output.last_hidden_state

        # picking it 1st word
        # returns <<<<< (batch_size, embedding_dim)
        return last_hidden_state[:,self.target_token_idx,:]



class ProjectionHead(torch.nn.Module):
    def __init__(
            self, 
            embedding_dim:int,
            projection_dim:int = CFG.projection_dim,
            dropout_rate:float = CFG.dropout,
            bias:bool=False,
            *args, 
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.projection:torch.nn.Linear    = torch.nn.Linear(embedding_dim,projection_dim,bias=bias)
        self.gelu:torch.nn.GELU            = torch.nn.GELU()
        self.fc:torch.nn.Linear            = torch.nn.Linear(projection_dim,projection_dim,bias=bias)
        self.dropout:torch.nn.Dropout      = torch.nn.Dropout(p=dropout_rate)
        self.layer_norm:torch.nn.LayerNorm = torch.nn.LayerNorm(projection_dim)


    def forward(self,x:torch.Tensor)->torch.Tensor:
        projected = self.projection(x)
        x = self.dropout( 
                self.fc( 
                    self.gelu(projected)
                )
        )
        x = x + projected
        return self.layer_norm(x)

    