import random
from typing import List

import cv2
import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

from Yolov3.utils import non_max_suppression, intersection_over_union, cells_to_bboxes


# def cells_to_bboxes(predictions, anchors, S, is_preds=True):
#     """Scales the predictions coming from the model to be relative to the entire image such that
#     they for example later can be plotted or.

#     INPUT:
#     predictions: tensor of size (N, 3, S, S, num_classes+5)
#     anchors: the anchors used for the predictions
#     S: the number of cells the image is divided in on the width (and height)
#     is_preds: whether the input is predictions or the true bounding boxes
#     OUTPUT:
#     converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
#                       object score, bounding box coordinates
#     """
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
#         torch.arange(S).repeat(predictions.shape[0], 3, S, 1).unsqueeze(-1).to(predictions.device)
#     )
#     x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
#     y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
#     w_h = 1 / S * box_predictions[..., 2:4]
#     converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(
#         BATCH_SIZE, num_anchors * S * S, 6
#     )
#     return converted_bboxes.tolist()


# def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
#     """
#     Video explanation of this function:
#     https://youtu.be/XXYG5ZWtjj0

#     This function calculates intersection over union (iou) given pred boxes
#     and target boxes.

#     Parameters:
#         boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
#         boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
#         box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

#     Returns:
#         tensor: Intersection over union for all examples
#     """

#     if box_format == "midpoint":
#         box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
#         box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
#         box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
#         box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
#         box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
#         box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
#         box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
#         box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

#     if box_format == "corners":
#         box1_x1 = boxes_preds[..., 0:1]
#         box1_y1 = boxes_preds[..., 1:2]
#         box1_x2 = boxes_preds[..., 2:3]
#         box1_y2 = boxes_preds[..., 3:4]
#         box2_x1 = boxes_labels[..., 0:1]
#         box2_y1 = boxes_labels[..., 1:2]
#         box2_x2 = boxes_labels[..., 2:3]
#         box2_y2 = boxes_labels[..., 3:4]

#     x1 = torch.max(box1_x1, box2_x1)
#     y1 = torch.max(box1_y1, box2_y1)
#     x2 = torch.min(box1_x2, box2_x2)
#     y2 = torch.min(box1_y2, box2_y2)

#     intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
#     box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
#     box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

#     return intersection / (box1_area + box2_area - intersection + 1e-6)


# def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
#     """
#     Video explanation of this function:
#     https://youtu.be/YDkjWEN8jNA

#     Does Non Max Suppression given bboxes

#     Parameters:
#         bboxes (list): list of lists containing all bboxes with each bboxes
#         specified as [class_pred, prob_score, x1, y1, x2, y2]
#         iou_threshold (float): threshold where predicted bboxes is correct
#         threshold (float): threshold to remove predicted bboxes (independent of IoU)
#         box_format (str): "midpoint" or "corners" used to specify bboxes

#     Returns:
#         list: bboxes after performing NMS given a specific IoU threshold
#     """

#     assert type(bboxes) == list

#     bboxes = [box for box in bboxes if box[1] > threshold]
#     bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
#     bboxes_after_nms = []

#     while bboxes:
#         chosen_box = bboxes.pop(0)

#         bboxes = [
#             box
#             for box in bboxes
#             if box[0] != chosen_box[0]
#             or intersection_over_union(
#                 torch.tensor(chosen_box[2:]),
#                 torch.tensor(box[2:]),
#                 box_format=box_format,
#             )
#             < iou_threshold
#         ]

#         bboxes_after_nms.append(chosen_box)

#     return bboxes_after_nms


# def draw_predictions(image: np.ndarray, boxes: List[List], class_labels: List[str], confidence_threshold=0.70) -> np.ndarray:
#     """Plots predicted bounding boxes on the image."""

#     colors = [[random.randint(0, 255) for _ in range(3)] for name in class_labels]

#     im = np.array(image)
#     height, width, _ = im.shape
#     bbox_thick = int(2 * (height + width) / 600)

#     # Create a Rectangle patch
#     for box in boxes:
#         assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
#         class_pred = box[0]
#         conf = box[1]
#         box = box[2:]
#         upper_left_x = box[0] - box[2] / 2
#         upper_left_y = box[1] - box[3] / 2

#         x1 = int(upper_left_x * width)
#         y1 = int(upper_left_y * height)

#         x2 = x1 + int(box[2] * width)
#         y2 = y1 + int(box[3] * height)

#         if confidence_threshold > conf:
#             continue

#         cv2.rectangle(
#             image, (x1, y1), (x2, y2), color=colors[int(class_pred)], thickness=bbox_thick
#         )
#         text = f"{class_labels[int(class_pred)]}: {conf:.2f}"
#         t_size = cv2.getTextSize(text, 0, 0.7, thickness=bbox_thick // 2)[0]
#         c3 = (x1 + t_size[0], y1 - t_size[1] - 3)

#         cv2.rectangle(image, (x1, y1), c3, colors[int(class_pred)], -1)
#         cv2.putText(
#             image,
#             text,
#             (x1, y1 - 2),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 0, 0),
#             bbox_thick // 2,
#             lineType=cv2.LINE_AA,
#         )

#     return image


class YoloCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
        super().__init__(
            model, target_layers, use_cuda, reshape_transform, uses_gradients=False
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        scaled_anchors: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            bboxes = [[] for _ in range(1)]
            for i in range(3):
                batch_size, A, S, _, _ = outputs[i].shape
                anchor = scaled_anchors[i]
                boxes_scale_i = cells_to_bboxes(outputs[i], anchor, S=S, is_preds=True)
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box

            nms_boxes = non_max_suppression(
                bboxes[0],
                iou_threshold=0.5,
                threshold=0.4,
                box_format="midpoint",
            )
            # target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            target_categories = [box[0] for box in nms_boxes]
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_cam_image(
        self, input_tensor, target_layer, target_category, activations, grads, eigen_smooth
    ):
        return get_2d_projection(activations)