import torch
import lightning as pl

import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from Yolov3.loss import YoloLoss
from Yolov3.model import YOLOv3
from Yolov3.calculatemap import CalculateMAP
from Yolov3 import config


class LitYOLONet(pl.LightningModule):
    def __init__(self, device="cuda", lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.yolo = YOLOv3(num_classes=config.NUM_CLASSES)
        self.lr = lr
        self.weight_decay = weight_decay

        self.criteria = YoloLoss()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.predict_step_outputs = []


    def check_class_accuracy(self, out, y, threshold=config.CONF_THRESHOLD):
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0

        for i in range(3):
            # y[i] = y[i].to(self.device)
            obj = y[i][..., 0] == 1  # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

        return (
            tot_class_preds,
            correct_class,
            tot_noobj,
            correct_noobj,
            tot_obj,
            correct_obj,
        )

    def forward(self, x):
        return self.yolo(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.yolo.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        num_epochs = self.trainer.max_epochs
        custom_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            # steps_per_epoch=self.trainer.estimated_stepping_batches,
            total_steps=self.trainer.estimated_stepping_batches,
            epochs=num_epochs,
            pct_start=.3,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy="linear",
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": custom_scheduler,
                    "monitor": "val_loss",
                    "interval": "step",
                    "frequency": 1,
                },
            },
        )

    def calculate_loss(self, output, target, S):
        # scaled_anchors = torch.tensor(
        #     config.ANCHORS, device=self.device
        # ) * torch.tensor(config.S, device=self.device).unsqueeze(1).unsqueeze(1).repeat(
        #     1, 3, 2
        # )

        # _batch_size = output[0].shape[0]

        # scaled_anchors_1 = (
        #     torch.tensor(config.ANCHORS, device=self.device).unsqueeze(0)
        #     * torch.tensor(config.S, device=self.device)
        #     .unsqueeze(1)
        #     .unsqueeze(1)
        #     .repeat(1, 3, 2)
        #     .repeat(_batch_size, 1, 1, 1)
        # ).permute(1, 0, 2, 3)

        # self.print(scaled_anchors_32.shape)

        # self.print(len(S))

        # self.print(S[0].shape)

        # scaled_anchors_2 = (torch.tensor(config.ANCHORS, device=self.device).unsqueeze(0) * torch.stack(S)
        #     .unsqueeze(1)
        #     .unsqueeze(1)
        #     .repeat(1, 3, 2, 1)
        #     .permute(3, 0, 1, 2)
        # ).permute(1, 0, 2, 3)

        scaled_anchors_2 = (
            torch.tensor(config.ANCHORS, device=self.device).unsqueeze(0)
            * torch.stack(S)
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, 3, 2, 1)
            .permute(3, 0, 1, 2)
        ).permute(1, 0, 2, 3)
        
        
        # self.print(torch.all(scaled_anchors_1==scaled_anchors_2))

        scaled_anchors = scaled_anchors_2

        # scaled_anchors = torch.tensor(
        #     [
        #         [[3.6400, 2.8600], [4.9400, 6.2400], [11.7000, 10.1400]],
        #         [[1.8200, 3.9000], [3.9000, 2.8600], [3.6400, 7.5400]],
        #         [[1.0400, 1.5600], [2.0800, 3.6400], [4.1600, 3.1200]],
        #     ],
        #     device=self.device,
        # )

        # print(output[0])

        # print(self.criteria(output[0], target[0], scaled_anchors[0]))

        # __accuracy = self.check_class_accuracy(output, target)

        # if self.criteria(output, target, scaled_anchors).isnan().any():
        #     raise Exception("NAN Detected in loss")

        loss = (
            self.criteria(output[0].clone(), target[0].clone(), scaled_anchors[0])
            + self.criteria(output[1].clone(), target[1].clone(), scaled_anchors[1])
            + self.criteria(output[2].clone(), target[2].clone(), scaled_anchors[2])
        )

        # loss = self.criteria(output, target, scaled_anchors)

        return loss

    def training_step(self, train_batch, batch_idx):
        loss = 0.0
        
        for batches in train_batch:
            data, target, S = batches
            output = self.yolo(data)
    
            # if (
            #     target[0].isnan().any()
            #     or target[1].isnan().any()
            #     or target[2].isnan().any()
            # ):
            #     print(target[0].dtype)
            #     print(target)
            #     raise Exception("NAN Detected in target")
    
            # if data.isnan().any():
            #     print(data.dtype)
            #     print(data)
            #     raise Exception("NAN Detected")
    
            # if (
            #     output[0].isnan().any()
            #     or output[1].isnan().any()
            #     or output[2].isnan().any()
            # ):
            #     print(output.dtype)
            #     print(output)
            #     raise Exception("NAN Detected")
    
            self.training_step_outputs.append(self.check_class_accuracy(output, target))
    
            loss = loss + self.calculate_loss(output, target, S)
    
            # if loss.isnan().any():
            #     print(target)
            #     print(output)
            #     raise Exception("NAN Detected in Loss")

        # data, target, S = train_batch
        # output = self.yolo(data)
        
        # self.training_step_outputs.append(self.check_class_accuracy(output, target))
    
        # loss = self.calculate_loss(output, target, S)
        

        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def on_train_epoch_end(self):
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0

        for accuracy in self.training_step_outputs:
            (
                _tot_class_preds,
                _correct_class,
                _tot_noobj,
                _correct_noobj,
                _tot_obj,
                _correct_obj,
            ) = accuracy

            tot_class_preds, correct_class = (
                tot_class_preds + _tot_class_preds,
                correct_class + _correct_class,
            )
            tot_noobj, correct_noobj = (
                tot_noobj + _tot_noobj,
                correct_noobj + _correct_noobj,
            )
            tot_obj, correct_obj = tot_obj + _tot_obj, correct_obj + _correct_obj

        self.log_dict(
            {
                "train_class_accuracy": (correct_class / (tot_class_preds + 1e-16))
                * 100,
                "train_no_obj_accuracy": (correct_noobj / (tot_noobj + 1e-16)) * 100,
                "train_obj_accuracy": (correct_obj / (tot_obj + 1e-16)) * 100,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.training_step_outputs.clear()  # free memory

    def validation_step(self, val_batch, batch_idx):
        data, target, S = val_batch
        output = self.yolo(data)

        self.validation_step_outputs.append(self.check_class_accuracy(output, target))

        loss = self.calculate_loss(output, target, S)

        self.log_dict(
            {
                "val_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0

        for accuracy in self.validation_step_outputs:
            (
                _tot_class_preds,
                _correct_class,
                _tot_noobj,
                _correct_noobj,
                _tot_obj,
                _correct_obj,
            ) = accuracy

            tot_class_preds, correct_class = (
                tot_class_preds + _tot_class_preds,
                correct_class + _correct_class,
            )
            tot_noobj, correct_noobj = (
                tot_noobj + _tot_noobj,
                correct_noobj + _correct_noobj,
            )
            tot_obj, correct_obj = tot_obj + _tot_obj, correct_obj + _correct_obj

        self.log_dict(
            {
                "val_class_accuracy": (correct_class / (tot_class_preds + 1e-16)) * 100,
                "val_no_obj_accuracy": (correct_noobj / (tot_noobj + 1e-16)) * 100,
                "val_obj_accuracy": (correct_obj / (tot_obj + 1e-16)) * 100,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.validation_step_outputs.clear()  # free memory

    def test_step(self, predict_batch, batch_idx):
        data, target, S = predict_batch
        output = self.yolo(data)

        self.predict_step_outputs.append(self.check_class_accuracy(output, target))

        # # Calculate MAP
        # pred_boxes, true_boxes = self.get_evaluation_bboxes(
        #     output,
        #     target,
        #     iou_threshold=config.NMS_IOU_THRESH,
        #     anchors=torch.tensor(config.ANCHORS, device=self.device),
        #     threshold=config.CONF_THRESHOLD,
        # )

        # mapvalue = mean_average_precision(
        #     pred_boxes,
        #     true_boxes,
        #     iou_threshold=config.MAP_IOU_THRESH,
        #     box_format="midpoint",
        #     num_classes=config.NUM_CLASSES,
        # )

        # Calculate MAP
        calculate_map = CalculateMAP(device=self.device)
        mapvalue = calculate_map.evaluate_model(output, target, config)

        self.log_dict(
            {
                "map": mapvalue,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_test_epoch_end(self):
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0

        for accuracy in self.predict_step_outputs:
            (
                _tot_class_preds,
                _correct_class,
                _tot_noobj,
                _correct_noobj,
                _tot_obj,
                _correct_obj,
            ) = accuracy

            tot_class_preds, correct_class = (
                tot_class_preds + _tot_class_preds,
                correct_class + _correct_class,
            )
            tot_noobj, correct_noobj = (
                tot_noobj + _tot_noobj,
                correct_noobj + _correct_noobj,
            )
            tot_obj, correct_obj = tot_obj + _tot_obj, correct_obj + _correct_obj

        self.log_dict(
            {
                "test_class_accuracy": (correct_class / (tot_class_preds + 1e-16))
                * 100,
                "test_no_obj_accuracy": (correct_noobj / (tot_noobj + 1e-16)) * 100,
                "test_obj_accuracy": (correct_obj / (tot_obj + 1e-16)) * 100,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.predict_step_outputs.clear()  # free memory

    # def get_evaluation_bboxes(
    #     self,
    #     predictions,
    #     labels,
    #     iou_threshold,
    #     anchors,
    #     threshold,
    #     box_format="midpoint",
    # ):
    #     train_idx = 0
    #     all_pred_boxes = []
    #     all_true_boxes = []

    #     batch_size = predictions[0].shape[0]
    #     bboxes = [[] for _ in range(batch_size)]
    #     for i in range(3):
    #         S = predictions[i].shape[2]
    #         anchor = anchors[i] * S
    #         boxes_scale_i = self.cells_to_bboxes(
    #             predictions[i], anchor, S=S, is_preds=True
    #         )
    #         for idx, box in enumerate(boxes_scale_i):
    #             bboxes[idx] += box

    #     true_bboxes = self.cells_to_bboxes(labels[2], anchors[2], S=S, is_preds=False)

    #     for idx in range(batch_size):
    #         nms_boxes = non_max_suppression(
    #             bboxes[idx],
    #             iou_threshold=iou_threshold,
    #             threshold=threshold,
    #             box_format=box_format,
    #         )

    #         for nms_box in nms_boxes:
    #             all_pred_boxes.append([train_idx] + nms_box)

    #         for box in true_bboxes[idx]:
    #             if box[1] > threshold:
    #                 all_true_boxes.append([train_idx] + box)

    #         train_idx += 1

    #     return all_pred_boxes, all_true_boxes

    # def cells_to_bboxes(self, predictions, anchors, S, is_preds=True):
    #     BATCH_SIZE = predictions.shape[0]
    #     num_anchors = len(anchors)
    #     box_predictions = predictions[..., 1:5]

    #     if is_preds:
    #         anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
    #         box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
    #         box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
    #         scores = torch.sigmoid(predictions[..., 0:1])
    #         best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    #     else:
    #         scores = predictions[..., 0:1]
    #         best_class = predictions[..., 5:6]

    #     cell_indices = (
    #         torch.arange(S, device=self.device)
    #         .repeat(BATCH_SIZE, 3, S, 1)
    #         .unsqueeze(-1)
    #     )
    #     x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    #     y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    #     w_h = 1 / S * box_predictions[..., 2:4]
    #     converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(
    #         BATCH_SIZE, num_anchors * S * S, 6
    #     )

    #     return converted_bboxes.tolist()


## Add Following
### 1. MAP Calculation
### 2. Checkpoint Loading and Saving

model = LitYOLONet(device=config.DEVICE, lr=config.LEARNING_RATE)