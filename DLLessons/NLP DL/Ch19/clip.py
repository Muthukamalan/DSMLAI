import torch 

import config as CFG
from modules import ImageEncoder,TextEncoder,ProjectionHead

class CLIPModel(torch.nn.Module):
    def __init__(
            self, 
            temperature = CFG.temperature,
            image_embedding:int = CFG.image_embedding,
            text_embedding:int = CFG.text_embedding,
            *args, 
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder:ImageEncoder = ImageEncoder()
        self.text_encoder:TextEncoder   = TextEncoder()
        self.img_projection:ProjectionHead  = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection:ProjectionHead = ProjectionHead(embedding_dim=text_embedding)
        self.temperature:int  = temperature


    def forward(self,batch:dict)->torch.Tensor:
        img_features = self.image_encoder(batch['image'])
        # returns ``

        txt_features = self.text_encoder(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])
        # returns (batch_size, 0, embedding_dims )

        # Projections
        image_embedding = self.img_projection(img_features)
        text_embedding  = self.text_projection(txt_features)

        # calculating loss
        logits = (text_embedding @ image_embedding.T) / self.temperature

        img_similarity = image_embedding @ image_embedding.T
        txt_similarity = text_embedding @ text_embedding.T

        targets = torch.nn.functional.softmax(
            (img_similarity+txt_similarity)/  2*self.temperature,
            dim=-1  
        )
        txt_loss = self.cross_entropy(logits,targets,reduction="none")
        img_loss = self.cross_entropy(logits.T,targets.T,reduction="none")

        loss = (img_loss+txt_loss)/ 2.  # shape:(batch_size)
        return loss.mean()


    def cross_entropy(self,preds,targets,reduction='none'):
        log_softmax:torch.Tensor = torch.nn.LogSoftmax(dim=-1)
        loss:torch.Tensor        = (-targets * log_softmax(preds)).sum(1)
        if reduction=='none': return loss 
        else: return loss.mean()
