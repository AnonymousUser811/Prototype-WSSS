import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBuilder(nn.Module):
    def __init__(self, num_classes=4, k=3):
        super().__init__()
        self.num_classes = num_classes
        self.k = k

    @torch.no_grad()
    def forward(self, f, f_aug, label, label_aug):
        Hf, Wf = f.shape[-2:]
        label_ds = F.interpolate(label.float(), size=(Hf, Wf), mode="nearest").long()
        label_aug_ds = F.interpolate(label_aug.float(), size=(Hf, Wf), mode="nearest").long()

        proto = self._build_single_view(f, f_aug, label_ds)
        proto_aug = self._build_single_view(f_aug, f, label_aug_ds)

        return proto, proto_aug

    @torch.no_grad()
    def _build_single_view(self, f, f_aug, label):
        B, C, H, W = f.shape
        N = H * W

        # ---- flatten ----
        f_flat = f.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        f_aug_flat = f_aug.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        label_flat = label.view(B, N)  # (B, N)

        # ---- cosine similarity ----
        f_norm = F.normalize(f_flat, dim=2)
        f_aug_norm = F.normalize(f_aug_flat, dim=2)
        sim = torch.bmm(f_norm, f_aug_norm.transpose(1, 2))  # (B, N, N)

        # ---- top-k forward NN ----
        _, topk_idx = sim.topk(self.k, dim=2)  # (B, N, k)

        # ---- reverse NN (1-NN) ----
        nn_aug_to_f = sim.argmax(dim=1)  # (B, N)

        # ---- batch gather for reverse NN ----
        B_idx = torch.arange(B, device=f.device)[:, None, None].expand(-1, N, self.k)  # (B,N,k)
        K_idx = nn_aug_to_f[B_idx, topk_idx]  # (B,N,k)

        # ---- stable mask ----
        label_i = label_flat[:, :, None].expand(-1, -1, self.k)  # (B,N,k)
        label_k = torch.take_along_dim(label_flat.unsqueeze(1).expand(-1, N, -1), K_idx, dim=2) # (B,N,k)

        stable_mask = (label_i == label_k) & (label_i != 4)  # (B,N,k)
        stable_mask_any = stable_mask.any(dim=2)  # (B,N)

        # ---- compute prototypes per class ----
        prototypes = []
        for c in range(self.num_classes):
            # -------- stable features --------
            stable_class_mask = (label_flat == c) & stable_mask_any  # (B,N)
            stable_count = stable_class_mask.sum(dim=1)  # (B,)
            has_stable = stable_count > 0

            proto_c = torch.zeros(B, C, device=f.device)

            if has_stable.any():
                stable_sum = (f_flat * stable_class_mask.unsqueeze(2).float()).sum(dim=1)
                proto_c[has_stable] = (
                        stable_sum[has_stable] / stable_count[has_stable].unsqueeze(1)
                )

            class_mask = (label_flat == c)  # (B,N)
            class_count = class_mask.sum(dim=1)  # (B,)
            has_class = class_count > 0

            fallback_mask = (~has_stable) & has_class

            if fallback_mask.any():
                fallback_sum = (f_flat * class_mask.unsqueeze(2).float()).sum(dim=1)
                proto_c[fallback_mask] = (
                        fallback_sum[fallback_mask] / class_count[fallback_mask].unsqueeze(1)
                )

            prototypes.append(proto_c.unsqueeze(1))

        prototypes = torch.cat(prototypes, dim=1)  # (B, num_classes, C)
        return prototypes


class GlobalPrototypeMemory(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum

        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, feat_dim)
        )

        self.register_buffer(
            "initialized",
            torch.zeros(num_classes, dtype=torch.bool)
        )

    @torch.no_grad()
    def update(self, proto_batch):
        B, K, C = proto_batch.shape
        # proto_batch = F.normalize(proto_batch, dim=-1, eps=1e-6)

        for c in range(K):
            proto_c = proto_batch[:, c]               # (B,C)
            valid = proto_c.norm(dim=1) > 0           # 非全零

            if not valid.any():
                continue

            proto_mean = proto_c[valid].mean(dim=0)

            if not self.initialized[c]:
                self.prototypes[c] = proto_mean
                self.initialized[c] = True
            else:
                # EMA
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c]
                    + (1.0 - self.momentum) * proto_mean
                )

    def forward(self):
        return self.prototypes
