import torch.nn.functional as F
import torch
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torch.nn as nn


class PrototypeContrastiveLoss(nn.Module):
    def __init__(
        self,
        num_classes=4,
        lambda_p2p=0.4,
        lambda_p2proto=0.4,
        lambda_proto2proto=0.2,
        temperature=0.4,
    ):
        super().__init__()
        self.K = num_classes
        self.tau = temperature
        self.lambda_p2p = lambda_p2p
        self.lambda_p2proto = lambda_p2proto
        self.lambda_proto2proto = lambda_proto2proto

    def forward(self, f, f_aug, proto, proto_aug, label, label_aug):
        B, C, H, W = f.shape
        label = F.interpolate(label.float(), size=(H, W), mode="nearest").long()
        label_aug = F.interpolate(label_aug.float(), size=(H, W), mode="nearest").long()

        loss_p2p = self.pixel_pixel_contrast(f, f_aug, label, label_aug)
        loss_p2proto = (
                self.pixel_proto_contrast(f, proto, proto_aug, label)
                + self.pixel_proto_contrast(f_aug, proto, proto_aug, label_aug)
        )
        return self.lambda_p2p * loss_p2p + self.lambda_p2proto * loss_p2proto

    # --------------------------------------------------
    # 1. Pixel–Pixel
    # --------------------------------------------------
    def pixel_pixel_contrast(self, f, f_aug, label, label_aug):
        B, C, H, W = f.shape
        N = H * W

        f = F.normalize(f.view(B, C, N).permute(0, 2, 1), dim=-1)  # (B,N,C)
        f_aug = F.normalize(f_aug.view(B, C, N).permute(0, 2, 1), dim=-1)

        label = label.view(B, N)
        label_aug = label_aug.view(B, N)

        valid_i = label != 4
        valid_j = label_aug != 4

        if valid_i.sum() == 0 or valid_j.sum() == 0:
            return torch.tensor(0.0, device=f.device)

        # similarity: (B,N,N)
        sim = torch.einsum("bnc,bmc->bnm", f, f_aug) / self.tau
        exp_sim = torch.exp(sim)

        # positive mask
        pos_mask = (
                (label.unsqueeze(2) == label_aug.unsqueeze(1))
                & valid_i.unsqueeze(2)
                & valid_j.unsqueeze(1)
        )

        pos_sum = (exp_sim * pos_mask.float()).sum(dim=2)
        all_sum = (exp_sim * valid_j.unsqueeze(1).float()).sum(dim=2)

        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        loss = loss[valid_i].mean()

        return loss

    # --------------------------------------------------
    # 2. Pixel–Prototype
    # --------------------------------------------------
    def pixel_proto_contrast(self, f, proto, proto_aug, label):
        B, C, H, W = f.shape
        N = H * W
        K = proto.shape[0]

        f = F.normalize(f.view(B, C, N).permute(0, 2, 1), dim=-1)
        proto = F.normalize(proto, dim=-1)
        proto_aug = F.normalize(proto_aug, dim=-1)

        label = label.view(B, N)
        valid = label != 4
        if valid.sum() == 0:
            return torch.tensor(0.0, device=f.device)

        # ---- concat prototypes ----
        proto_all = torch.cat([proto, proto_aug], dim=0)  # (2K,C)

        # ---- similarity ----
        sim = torch.einsum("bnc,kc->bnk", f, proto_all) / self.tau
        exp_sim = torch.exp(sim)

        # ---- positive mask ----
        pos_mask = torch.zeros_like(sim, dtype=torch.bool)

        label_valid = label.clone()
        label_valid[~valid] = 0

        label_idx = label_valid.unsqueeze(-1)  # (B,N,1)

        pos_mask[:, :, :K].scatter_(2, label_idx, True)
        pos_mask[:, :, K:].scatter_(2, label_idx, True)

        pos_sum = (exp_sim * pos_mask.float()).sum(dim=2)
        all_sum = exp_sim.sum(dim=2)

        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        return loss[valid].mean()


def prototype_based_pseudo_label(f, f_aug, proto, proto_aug, label, label_aug, sim_thresh=0.50, margin=1):
    B, C, H, W = f.shape
    K = proto.shape[0]
    N = H * W
    label = F.interpolate(label.float(), size=(H, W), mode="nearest").long()
    label_aug = F.interpolate(label_aug.float(), size=(H, W), mode="nearest").long()

    # ---------- flatten ----------
    f_flat = f.view(B, C, N).permute(0, 2, 1)  # (B,N,C)
    f_aug_flat = f_aug.view(B, C, N).permute(0, 2, 1)

    label_flat = label.view(B, N)
    label_aug_flat = label_aug.view(B, N)

    # ---------- normalize ----------
    f_flat = F.normalize(f_flat, dim=-1)
    f_aug_flat = F.normalize(f_aug_flat, dim=-1)
    proto = F.normalize(proto, dim=-1)
    proto_aug = F.normalize(proto_aug, dim=-1)

    # -------------------------------------------------------
    # build PL for f
    # -------------------------------------------------------
    sim_f_p = torch.einsum("bnc,kc->bnk", f_flat, proto)  # (B,N,K)
    sim_f_pa = torch.einsum("bnc,kc->bnk", f_flat, proto_aug)

    sim1, pred1 = sim_f_p.max(dim=-1)  # (B,N)
    sim2, pred2 = sim_f_pa.max(dim=-1)

    unlabeled = (label_flat == 4)

    accept_f = (
            unlabeled
            & (sim1 > sim_thresh)
            & (sim2 > sim_thresh)
            & (pred1 == pred2)
            & ((sim1 - sim2).abs() < margin)
    )

    PL_flat = label_flat.clone()
    PL_flat[accept_f] = pred1[accept_f]

    # -------------------------------------------------------
    # Part 2: build PL_aug for f_aug using label_aug
    # -------------------------------------------------------
    sim_fa_p = torch.einsum("bnc,kc->bnk", f_aug_flat, proto)
    sim_fa_pa = torch.einsum("bnc,kc->bnk", f_aug_flat, proto_aug)

    sim1a, pred1a = sim_fa_p.max(dim=-1)
    sim2a, pred2a = sim_fa_pa.max(dim=-1)

    unlabeled_aug = (label_aug_flat == 4)

    accept_fa = (
            unlabeled_aug
            & (sim1a > sim_thresh)
            & (sim2a > sim_thresh)
            & (pred1a == pred2a)
            & ((sim1a - sim2a).abs() < margin)
    )

    PL_aug_flat = label_aug_flat.clone()
    PL_aug_flat[accept_fa] = pred1a[accept_fa]

    PL = PL_flat.view(B, 1, H, W)
    PL_aug = PL_aug_flat.view(B, 1, H, W)

    return PL, PL_aug

class SegAugmenter:
    def __init__(
        self,
        rotate_degree=90,
        brightness=0.4,
        contrast=0.4,
        noise_std=0.05,
    ):
        self.rotate_degree = rotate_degree
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, img, label):
        """
        img:   (B, 1, H, W), float tensor
        label: (B, 1, H, W), long tensor
        """
        B = img.size(0)
        img_weak, label_weak = torch.zeros_like(img), torch.zeros_like(label)

        for i in range(B):
            # angle = random.uniform(-self.rotate_degree, self.rotate_degree)
            angle = random.randint(0, 3) * 90

            img_weak[i] = TF.rotate(img[i], angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            label_weak[i] = TF.rotate(label[i], angle=angle, interpolation=InterpolationMode.NEAREST, fill=0)

            if random.random() < 0.5:
                img_weak[i] = TF.hflip(img_weak[i])
                label_weak[i] = TF.hflip(label_weak[i])

            if random.random() < 0.5:
                img_weak[i] = TF.vflip(img_weak[i])
                label_weak[i] = TF.vflip(label_weak[i])

        return img_weak, label_weak
