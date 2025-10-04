import os 
import toml
import torch
from typing import Dict

CONFIG: Dict= toml.load(os.path.join(os.path.dirname(__file__),'config.toml'))
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# with open(os.path.join(os.path.dirname(__file__),'config.toml'),'w') as f:
#     toml.dump(CONFIG,f)