# WRITE IT AGAIN SUCH THAT IT ACHIEVES
    1. 99.4% validation accuracy
    2. Less than 20k Parameters
    3. Less than 20 Epochs



## Model
```ruby
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 3, 28, 28]              27
       BatchNorm2d-2            [-1, 3, 28, 28]               6
              ReLU-3            [-1, 3, 28, 28]               0
         Dropout2d-4            [-1, 3, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             216
       BatchNorm2d-6            [-1, 8, 28, 28]              16
              ReLU-7            [-1, 8, 28, 28]               0
         Dropout2d-8            [-1, 8, 28, 28]               0
            Conv2d-9           [-1, 16, 28, 28]           1,152
      BatchNorm2d-10           [-1, 16, 28, 28]              32
             ReLU-11           [-1, 16, 28, 28]               0
        Dropout2d-12           [-1, 16, 28, 28]               0
        MaxPool2d-13           [-1, 16, 15, 15]               0
           Conv2d-14            [-1, 8, 17, 17]             128
      BatchNorm2d-15            [-1, 8, 17, 17]              16
           Conv2d-16           [-1, 16, 17, 17]           1,152
      BatchNorm2d-17           [-1, 16, 17, 17]              32
             ReLU-18           [-1, 16, 17, 17]               0
        Dropout2d-19           [-1, 16, 17, 17]               0
           Conv2d-20           [-1, 16, 17, 17]           2,304
      BatchNorm2d-21           [-1, 16, 17, 17]              32
             ReLU-22           [-1, 16, 17, 17]               0
        Dropout2d-23           [-1, 16, 17, 17]               0
           Conv2d-24           [-1, 16, 17, 17]           2,304
      BatchNorm2d-25           [-1, 16, 17, 17]              32
             ReLU-26           [-1, 16, 17, 17]               0
        Dropout2d-27           [-1, 16, 17, 17]               0
        MaxPool2d-28             [-1, 16, 9, 9]               0
           Conv2d-29             [-1, 16, 9, 9]             256
      BatchNorm2d-30             [-1, 16, 9, 9]              32
           Conv2d-31             [-1, 16, 9, 9]           2,304
      BatchNorm2d-32             [-1, 16, 9, 9]              32
             ReLU-33             [-1, 16, 9, 9]               0
        Dropout2d-34             [-1, 16, 9, 9]               0
           Conv2d-35             [-1, 16, 9, 9]           2,304
             ReLU-36             [-1, 16, 9, 9]               0
      BatchNorm2d-37             [-1, 16, 9, 9]              32
        Dropout2d-38             [-1, 16, 9, 9]               0
        MaxPool2d-39             [-1, 16, 4, 4]               0
           Conv2d-40             [-1, 16, 4, 4]             256
      BatchNorm2d-41             [-1, 16, 4, 4]              32
           Conv2d-42             [-1, 10, 4, 4]           1,440
      BatchNorm2d-43             [-1, 10, 4, 4]              20
        AvgPool2d-44             [-1, 10, 1, 1]               0
================================================================
Total params: 14,157
Trainable params: 14,157
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.25
Params size (MB): 0.05
Estimated Total Size (MB): 1.31
----------------------------------------------------------------
```

## Config
```toml
batch_size = 128
shuffle    = true 
num_workers= 2
pin_memory = true

[optimizer]
lr = 0.01
momentum = 0.9

[scheduler]
step_size = 15
gamma = 0.1


[training]
num_epochs = 10


[data]
dir_path = "../../../data"

[model]
dropout_rate = 0.01
bias = false
```

## Training Logs
```log
Train: Loss=0.0387 Batch_id=937 Accuracy=92.08: 100%|██████████| 938/938 [00:24<00:00, 37.79it/s]
Test set: Average loss: 0.0774, Accuracy: 9747/10000 (97.4700%)

Epoch 2
Train: Loss=0.0662 Batch_id=937 Accuracy=97.13: 100%|██████████| 938/938 [00:22<00:00, 41.67it/s]
Test set: Average loss: 0.0372, Accuracy: 9882/10000 (98.8200%)

Epoch 3
Train: Loss=0.1817 Batch_id=937 Accuracy=97.67: 100%|██████████| 938/938 [00:24<00:00, 38.57it/s]
Test set: Average loss: 0.0320, Accuracy: 9898/10000 (98.9800%)

Epoch 4
Train: Loss=0.0326 Batch_id=937 Accuracy=98.04: 100%|██████████| 938/938 [00:23<00:00, 39.50it/s]
Test set: Average loss: 0.0286, Accuracy: 9915/10000 (99.1500%)

Epoch 5
Train: Loss=0.0705 Batch_id=937 Accuracy=98.21: 100%|██████████| 938/938 [00:23<00:00, 40.70it/s]
Test set: Average loss: 0.0302, Accuracy: 9898/10000 (98.9800%)

Epoch 6
Train: Loss=0.0066 Batch_id=937 Accuracy=98.67: 100%|██████████| 938/938 [00:22<00:00, 42.25it/s]
Test set: Average loss: 0.0187, Accuracy: 9936/10000 (99.3600%)

Epoch 7
Train: Loss=0.0211 Batch_id=937 Accuracy=98.82: 100%|██████████| 938/938 [00:22<00:00, 41.99it/s]
Test set: Average loss: 0.0184, Accuracy: 9938/10000 (99.3800%)

Epoch 8
Train: Loss=0.0137 Batch_id=937 Accuracy=98.85: 100%|██████████| 938/938 [00:22<00:00, 41.33it/s]
Test set: Average loss: 0.0180, Accuracy: 9937/10000 (99.3700%)

Epoch 9
Train: Loss=0.0057 Batch_id=937 Accuracy=98.90: 100%|██████████| 938/938 [00:21<00:00, 44.21it/s]
Test set: Average loss: 0.0176, Accuracy: 9936/10000 (99.3600%)

Epoch 10
Train: Loss=0.0093 Batch_id=937 Accuracy=98.93: 100%|██████████| 938/938 [00:21<00:00, 42.96it/s]
Test set: Average loss: 0.0172, Accuracy: 9947/10000 (99.4700%)

Epoch 11
Train: Loss=0.0128 Batch_id=937 Accuracy=98.95: 100%|██████████| 938/938 [00:21<00:00, 44.07it/s]
Test set: Average loss: 0.0171, Accuracy: 9948/10000 (99.4800%)

Epoch 12
Train: Loss=0.0050 Batch_id=937 Accuracy=98.95: 100%|██████████| 938/938 [00:22<00:00, 41.84it/s]
Test set: Average loss: 0.0168, Accuracy: 9942/10000 (99.4200%)

Epoch 13
Train: Loss=0.0046 Batch_id=937 Accuracy=98.92: 100%|██████████| 938/938 [00:22<00:00, 42.19it/s]
Test set: Average loss: 0.0168, Accuracy: 9942/10000 (99.4200%)

Epoch 14
Train: Loss=0.0061 Batch_id=937 Accuracy=98.94: 100%|██████████| 938/938 [00:20<00:00, 44.71it/s]
Test set: Average loss: 0.0169, Accuracy: 9942/10000 (99.4200%)

Epoch 15
Train: Loss=0.0033 Batch_id=937 Accuracy=98.94: 100%|██████████| 938/938 [00:21<00:00, 42.67it/s]
Test set: Average loss: 0.0169, Accuracy: 9943/10000 (99.4300%)

Epoch 16
Train: Loss=0.0065 Batch_id=937 Accuracy=99.00: 100%|██████████| 938/938 [00:21<00:00, 42.83it/s]
Test set: Average loss: 0.0168, Accuracy: 9944/10000 (99.4400%)

Epoch 17
Train: Loss=0.0219 Batch_id=937 Accuracy=98.96: 100%|██████████| 938/938 [00:21<00:00, 43.77it/s]
Test set: Average loss: 0.0168, Accuracy: 9941/10000 (99.4100%)

Epoch 18
Train: Loss=0.0358 Batch_id=937 Accuracy=98.92: 100%|██████████| 938/938 [00:22<00:00, 41.53it/s]
Test set: Average loss: 0.0169, Accuracy: 9944/10000 (99.4400%)

Epoch 19
Train: Loss=0.0380 Batch_id=937 Accuracy=98.98: 100%|██████████| 938/938 [00:22<00:00, 42.29it/s]
Test set: Average loss: 0.0169, Accuracy: 9941/10000 (99.4100%)

Epoch 20
Train: Loss=0.0243 Batch_id=937 Accuracy=99.01: 100%|██████████| 938/938 [00:24<00:00, 38.04it/s]
Test set: Average loss: 0.0168, Accuracy: 9942/10000 (99.4200%)
```

PS: reaches 99.4% accuracy consistently