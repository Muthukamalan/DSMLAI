'''
lr = 0.012022644346174132

[data]
batch_size = 32
seq_len = 64 
num_workers = 0

[trainer]
epoch = 5

[model]
d_model = 64
n_head = 8
n_layer = 6 
dropout = 0.2
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


from config import CONFIG,DEVICE
from model import NanoGPT
from dataum import LitAuthorData,GPT2LitAuthorData

# Auxilary utils
torch.backends.cuda.matmul.allow_tf32=True
torch.set_float32_matmul_precision("medium")
torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
torch.set_default_device(device=DEVICE)
torch.cuda.empty_cache()
pl.seed_everything(270498);




## Loggers
logger: pl_loggers.TensorBoardLogger = pl_loggers.TensorBoardLogger(
    save_dir="logs/", name="nanogpt2", log_graph=True,version= input("Enter name of the experiment: ")
)

## CallBacks
call_backs = [
    TQDMProgressBar(refresh_rate=10),
    ModelCheckpoint(
        monitor="val/loss",                        # val/loss >> train/loss (to get low loss)
        dirpath=os.path.join("logs", "chkpoints"),
        filename="{epoch:02d}",
        save_top_k=1
    ),
    DeviceStatsMonitor(cpu_stats=True),
    # EarlyStopping(monitor="val/loss",mode='min'),
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


batch = next(iter(dm.train_dataloader()))
ip,op = batch
logger.log_graph(model=model.to(DEVICE),input_array=batch)
with torch.autograd.profiler.profile() as prof:
    output = model.to(DEVICE)(batch[0].to(DEVICE))

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
    enable_checkpointing=True,
    accumulate_grad_batches=2
)

## TUNER
# from lightning.pytorch.tuner import Tuner
# tuner = Tuner(trainer=trainer)
# lr_finder = tuner.lr_find(model,datamodule=dm,min_lr=1e-8,max_lr=1)
# # results
# print(lr_finder.results)
# # plot
# fig = lr_finder.plot(suggest=True)
# fig.show()
# new_lr = lr_finder.suggestion()
# model.hparams.lr = new_lr

# Training
trainer.fit(model, datamodule=dm)
# Validating
trainer.validate(model, dataloaders=dm)