import os
import gc 
gc.collect()

import torch
import gradio as gr
import lightning as pl
from torchinfo import summary

from lightning.pytorch import loggers as pl_loggers
# from functorch.compile import compiled_function, draw_graph
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import (
    # DeviceStatsMonitor,
    EarlyStopping,
    # LearningRateMonitor,
    # ModelCheckpoint,
    # ModelPruning,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar


from config import CONFIG,DEVICE
from model import NanoGPT
from dataum import GPT2LitAuthorData

# Auxilary utils
torch.backends.cuda.matmul.allow_tf32=True
torch.set_float32_matmul_precision("medium")
torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
torch.set_default_device(device=DEVICE)
torch.cuda.empty_cache()
# pl.seed_everything(270498);

# Dataset
dm = GPT2LitAuthorData(
    file_path=os.path.join(os.getcwd(), "dataum", f"input.txt.keep"),
    block_size=CONFIG['data'].get('seq_len'),
    batch_size=CONFIG['data'].get('batch_size'),
    num_workers=CONFIG['data'].get("num_workers"),
    encoder='gpt2'
)

dm.prepare_data()
dm.setup()

# NanoGPT Model
model = NanoGPT(
    d_model=CONFIG["model"].get("d_model"),
    seq_len=CONFIG["data"].get("seq_len"),
    vocab_size=dm.train_ds.vocab_size,
    n_head=CONFIG["model"].get("n_head"),
    n_layer=CONFIG["model"].get("n_layer"),
    lr=CONFIG["lr"],
    bias=False,
    dropout_rate=CONFIG["model"].get("dropout"),
)

# Load Model
CHKPOINT = torch.load(os.path.join(os.getcwd(),"logs","chkpoints",'nanogpt_epoch=01.ckpt'))
model.load_state_dict(CHKPOINT['state_dict'])



### Generation
def generate_text(prompt: str, max_new_token: int, dm=dm, imodel=model,device=DEVICE):
    encoded = dm.train_ds.enc.encode_ordinary(prompt)
    imodel.to(device)
    with torch.no_grad():
        encoded_text = torch.tensor(encoded,device=device).unsqueeze(0)
        new_word_predict = []
        for _ in range(max_new_token):
            encoded_text = encoded_text[:, -32:]
            logits, _ = imodel(encoded_text)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_word = torch.multinomial(probs, num_samples=1)
            new_word_predict.append(next_word.item())
            encoded_text = torch.cat((encoded_text[:, imodel.seq_len:], next_word), dim=1)             # Beam Search in Decodiing Strategy
    res = encoded + new_word_predict
    return dm.train_ds.decode(res)



title = "Milli GPT 💬"
description1 = '''
    Milli GPT trained on Shakespeare dataset.
    Trained on very small dataset to understand how GPT's are trained and built. The implementation can be found <a href='https://github.com/karpathy/nanoGPT'>here.</a>
'''


shakespeare_interface = gr.Interface(generate_text,
                    inputs=[gr.Textbox(label="Enter any prompt ", type="text", value="Let us love thy world,"),
                            gr.Slider(minimum=100, maximum=200, step=5, value=120, label="Max new tokens")],
                    outputs=gr.Textbox(label="Output generated", type="text"), description=description1)

demo = gr.TabbedInterface([shakespeare_interface], tab_names=["Shakespeare Data"],title=title)
demo.launch(share=True,debug=True)