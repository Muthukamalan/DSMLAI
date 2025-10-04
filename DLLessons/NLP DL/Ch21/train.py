'''# CONFIG
batch_size: 64
bias: false
block_size: 64
d_model: 64
dropout_rate: 0.2
file_path: "C:\\Users\\muthu\\GitHub\\Spaces \U0001F680\\NanoTalker\\data\\input.txt"
lr: 0.001
n_head: 4
n_layer: 4
num_workers: 0
seq_len: 64
vocab_size: 65
'''
import os
import gc 
gc.collect()

import torch
import lightning as pl
from torchinfo import summary

from lightning.pytorch import loggers as pl_loggers
from functorch.compile import compiled_function, draw_graph
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelPruning,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar


from config import CONFIG
from model import NanoGPT
from dataum import LitAuthorData

# Auxilary utils
torch.backends.cuda.matmul.allow_tf32=True
torch.set_float32_matmul_precision("medium")
torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device=device)
torch.cuda.empty_cache()
pl.seed_everything(270498)




## Loggers
logger: pl_loggers.TensorBoardLogger = pl_loggers.TensorBoardLogger(
    save_dir="logs/", name="nanogpt", log_graph=True
)

## CallBacks
call_backs = [
    TQDMProgressBar(refresh_rate=10),
    ModelCheckpoint(
        monitor="val/loss",
        dirpath=os.path.join("logs", "nanogpt_chkpoints"),
        filename="{epoch:02d}",
        save_top_k=1,
    ),
    DeviceStatsMonitor(cpu_stats=True),
    EarlyStopping(monitor="val/loss",mode='min'),
    LearningRateMonitor(logging_interval="step"),
]

## Profilers
perf_dir = os.path.join(os.getcwd(), "logs", "profiler")
perf_profiler = PyTorchProfiler(
    dirpath=perf_dir,
    filename="perf_logs_pytorch",
    group_by_input_shapes=True,
    emit_nvtx=torch.cuda.is_available(),
    activities=(
        [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if torch.cuda.is_available()
        else [
            torch.profiler.ProfilerActivity.CPU,
        ]
    ),
    schedule=torch.profiler.schedule(
        wait=1, warmup=1, active=5, repeat=3, skip_first=True
    ),
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    with_modules=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        str(os.path.join(perf_dir, "trace"))
    ),
)

# Dataset
dm = LitAuthorData(
    file_path=os.path.join(os.getcwd(), "dataum", f"input.txt.keep"),
    block_size=CONFIG["data"].get("seq_len"),
    batch_size=CONFIG["data"].get("batch_size"),
    num_workers=CONFIG["data"].get("num_workers"),
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
    dropout_rate=float(CONFIG["model"].get("dropout")),
)
# Graph 
batch = next(iter(dm.train_dataloader()))
ip,op = batch


# CPU Stats
with torch.autograd.profiler.profile() as prof:
    output = model.to(device)(batch[0].to(device))

os.makedirs(name=os.path.join(os.path.dirname(__file__),'logs','profiler'),exist_ok=True)
with open(os.path.join(os.path.dirname(__file__),'logs','profiler',"cpu_throttle.txt"), "w") as text_file:
    text_file.write(f"{prof.key_averages().table(sort_by='self_cpu_time_total',top_level_events_only=False)}")


# Model Summary
summary(
    model=model,
    input_data=batch[0],
    depth=5,
    verbose=2,
    col_width=16,
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "kernel_size",
        "mult_adds",
    ],
    row_settings=["var_names"],
)


trainer = pl.Trainer(
    max_epochs=CONFIG["trainer"].get("epoch"),
    callbacks=call_backs,
    logger=logger,
    precision='16-mixed',
    profiler='pytorch',#perf_profiler,#'advanced',
    enable_model_summary=True,
    enable_progress_bar=True,
    accumulate_grad_batches=4,
)

logger.log_graph(model.to(device),(ip.to(device),op.to(device)))

# Training
trainer.fit(model, datamodule=dm)
# Validating
trainer.validate(model, dataloaders=dm)
