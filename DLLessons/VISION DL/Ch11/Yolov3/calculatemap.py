import torch
from collections import Counter

from Yolov3.utils import mean_average_precision, non_max_suppression, cells_to_bboxes


class CalculateMAP:
    def __init__(self, device):
        self.device = device

    def get_evaluation_bboxes(
        self,
        predictions,
        labels,
        iou_threshold,
        anchors,
        threshold,
        box_format="midpoint",
    ):
        train_idx = 0
        all_pred_boxes = []
        all_true_boxes = []

        batch_size = predictions[0].shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = anchors[i] * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, box in enumerate(boxes_scale_i):
                bboxes[idx] += box

        true_bboxes = cells_to_bboxes(labels[2], anchors[2], S=S, is_preds=False)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

        return all_pred_boxes, all_true_boxes

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
    #         torch.arange(S, device=predictions.device)
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

    def evaluate_model(self, output, target, config):
        """
        Evaluate the YOLOv3 model using mean average precision (mAP).

        Parameters:
            output (list of torch.Tensor): List of predicted bounding boxes for each scale.
            target (list of torch.Tensor): List of ground truth bounding boxes for each scale.
            config: Configuration object containing hyperparameters.

        Returns:
            float: Mean average precision (mAP) value.
        """
        pred_boxes, true_boxes = self.get_evaluation_bboxes(
            output,
            target,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=torch.tensor(config.ANCHORS, device=self.device),
            threshold=config.CONF_THRESHOLD,
        )

        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )

        return mapval