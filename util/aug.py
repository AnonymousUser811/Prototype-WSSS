import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as Func
from inference import keep_largest_connected_components


def RandomBrightnessContrast(img, brightness_limit=0.2, contrast_limit=0.2, p=0.5):
    output = torch.zeros_like(img)
    threshold = 0.5

    for i in range(output.shape[0]):
        img_min, img_max = torch.min(img[i]), torch.max(img[i])

        output[i] = (img[i] - img_min) / (img_max - img_min) * 255.0
        if random.random() < p:
            brightness = 1.0 + random.uniform(-brightness_limit, brightness_limit)
            output[i] = torch.clamp(output[i] * brightness, 0., 255.)

            contrast = 0.0 + random.uniform(-contrast_limit, contrast_limit)
            output[i] = torch.clamp(output[i] + (output[i] - threshold * 255.0) * contrast, 0., 255.)

        output[i] = output[i] / 255.0 * (img_max - img_min) + img_min
    return output

def auxiliary_loss(output, output_c, output_aug, device):
    pseudo_pred = 0.5 * output + 0.5 * output_c
    pseudo_label = torch.argmax(pseudo_pred.detach(), dim=1, keepdim=False).unsqueeze(1)

    pred = pseudo_pred
    predictions_original_list = []
    for i in range(pred.shape[0]):
        prediction = np.uint8(np.argmax(pred[i, :, :, :].detach().cpu(), axis=0))
        prediction = keep_largest_connected_components(prediction)
        prediction = torch.from_numpy(prediction).to(device)
        predictions_original_list.append(prediction)
    predictions = torch.stack(predictions_original_list)
    predictions = torch.unsqueeze(predictions, 1)
    pred_keep_largest_connected = to_onehot(predictions, 4)

    predaug = output_aug
    predictions_originalaug_list = []
    for i in range(predaug.shape[0]):
        predictionaug = np.uint8(np.argmax(predaug[i, :, :, :].detach().cpu(), axis=0))
        predictionaug = keep_largest_connected_components(predictionaug)
        predictionaug = torch.from_numpy(predictionaug).to(device)
        predictions_originalaug_list.append(predictionaug)
    predictionsaug = torch.stack(predictions_originalaug_list)
    predictionsaug = torch.unsqueeze(predictionsaug, 1)
    pred_keep_largest_connectedaug = to_onehot(predictionsaug, 4)

    loss_consistency_1_2 = 1 - Func.cosine_similarity(output, output_c, dim=1).mean()
    loss_consistency_1_2 = 0.1 * loss_consistency_1_2

    loss_integrity = 1 - Func.cosine_similarity(pred[:, 0:4, :, :], pred_keep_largest_connected, dim=1).mean()
    loss_integrity = 0.3 * loss_integrity

    loss_integrityaug = 1 - Func.cosine_similarity(predaug[:, 0:4, :, :], pred_keep_largest_connectedaug, dim=1).mean()
    loss_integrityaug = 0.3 * loss_integrityaug

    dice_loss = pDLoss(4, ignore_index=4)
    loss_pseudo = 0.1 * (dice_loss(output_c, pseudo_label) +
                         dice_loss(output, pseudo_label))

    return loss_consistency_1_2 + loss_integrity + loss_integrityaug + loss_pseudo

class pDLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super(pDLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target * ignore_mask)
        y_sum = torch.sum(target * target * ignore_mask)
        z_sum = torch.sum(score * score * ignore_mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        ignore_mask = torch.ones_like(target)
        ignore_mask[target == self.ignore_index] = 0
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore_mask)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def to_onehot(target_masks, num_classes):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], num_classes, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks
