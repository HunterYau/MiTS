# ===========================
# MILModel (Segmentation-style MIL) - Full Script
# Copy/paste as a single file and run.
# ===========================

import os
import math
import torch
assert torch.cuda.is_available(), "Cuda not available"

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from scipy.ndimage import binary_erosion, generate_binary_structure, distance_transform_edt

import random
import numpy as np

# Optional (only used if you later re-enable CAM code)
# from pytorch_grad_cam import GradCAMPlusPlus
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ---------------------------
# Config
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_ROOT = "./data/BrainTumor/Training"
VAL_ROOT = "./data/BrainTumor/Testing"
DICE_ROOT = "./Manual_Testing/imgs"
MASK_ROOT = "./Manual_Testing/masks"
CHECKPOINT_PATH = "./checkpoints/MiTS100-3.pth"

NUM_CLASSES = 3      # change to 2 if you have tumor vs no_tumor [glioma, meningioma, pituitary]
BATCH_SIZE = 60
NUM_WORKERS = 32
EPOCHS = 100

LR = 1e-4
STEP_SIZE = 20
GAMMA = 0.3

# MIL pooling behavior: fraction of pixels used in TopK pooling
K_RATIO = 0.08           # try 0.01 (smaller region) ... 0.05 (bigger region)

# Tau SegMILHead    # higher tau = greater mean pooling
TAU = 1.5           # lower tau = greater max pooling

# Map regularization weights
LAMBDA_TV = 0.4
LAMBDA_AREA = 0.7
AREA_MIN = 0.08
AREA_MAX = 0.18

# NEW: region-growing pressure (Step 2)
LAMBDA_RG = 0.3
RG_THRESH = 0.7

# NEW: augmentation consistency (Step 3)
LAMBDA_CONS = 0.3        # try 0.1-0.6
CONS_USE_HFLIP = True    # keep this True (cheap + effective)

# NEW: border suppression (penalize attention near image edges)
LAMBDA_BORDER = 2.0       # try 1.0-6.0 (increase if maps still hug skull/border)
BORDER_FRAC = 0.25        # border band thickness as a fraction of map size (e.g., 0.25 of 14 -> ~3 px)

# tumor edge suppression
LAMBDA_ANTIEDGE = 0.3

# NEW: debug visualization (Step 4)
VIS_SHOW_TRUE_AND_PRED = True


# ---------------------------
# Data transforms
# ---------------------------
class RimDropout:
    """
    Targeted augmentation to destroy skull/border cues.

    Works on:
      - torch.Tensor image: [C,H,W] or [H,W] (recommended in your PyTorch pipeline)
      - torch.Tensor batch: [B,C,H,W] (optional)

    It replaces a random-thickness border ring with either per-image mean, zeros, or noise.
    """

    def __init__(
        self,
        p: float = 0.6,
        min_frac: float = 0.03,
        max_frac: float = 0.10,
        fill: str = "mean",     # "mean" | "zero" | "noise"
    ):
        assert 0.0 <= p <= 1.0
        assert 0.0 < min_frac <= max_frac
        assert fill in ("mean", "zero", "noise")
        self.p = p
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.fill = fill

    def _get_thickness(self, H: int, W: int) -> int:
        t_min = max(1, int(round(self.min_frac * min(H, W))))
        t_max = max(t_min, int(round(self.max_frac * min(H, W))))
        return random.randint(t_min, t_max)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        # Accept [H,W], [C,H,W], or [B,C,H,W]
        if x.dim() == 2:
            x_in = x.unsqueeze(0)          # [1,H,W]
            batched = False
        elif x.dim() == 3:
            x_in = x                       # [C,H,W]
            batched = False
        elif x.dim() == 4:
            x_in = x                       # [B,C,H,W]
            batched = True
        else:
            raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}")

        if not batched:
            # treat as single image [C,H,W]
            C, H, W = x_in.shape
            out = x_in.clone()
            t = self._get_thickness(H, W)

            if self.fill == "mean":
                val = out.mean(dim=(1, 2), keepdim=True)     # [C,1,1]
            elif self.fill == "zero":
                val = torch.zeros((C, 1, 1), device=out.device, dtype=out.dtype)
            else:  # noise
                mu = out.mean(dim=(1, 2), keepdim=True)
                sig = out.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                val = mu + sig * torch.randn((C, 1, 1), device=out.device, dtype=out.dtype)

            out[:, :t, :] = val
            out[:, H - t :, :] = val
            out[:, :, :t] = val
            out[:, :, W - t :] = val

            return out.squeeze(0) if x.dim() == 2 else out

        else:
            # batch: [B,C,H,W]
            B, C, H, W = x_in.shape
            out = x_in.clone()
            t = self._get_thickness(H, W)

            if self.fill == "mean":
                val = out.mean(dim=(2, 3), keepdim=True)     # [B,C,1,1]
            elif self.fill == "zero":
                val = torch.zeros((B, C, 1, 1), device=out.device, dtype=out.dtype)
            else:  # noise
                mu = out.mean(dim=(2, 3), keepdim=True)
                sig = out.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                val = mu + sig * torch.randn((B, C, 1, 1), device=out.device, dtype=out.dtype)

            out[:, :, :t, :] = val
            out[:, :, H - t :, :] = val
            out[:, :, :, :t] = val
            out[:, :, :, W - t :] = val
            return out


class CoarseDropout:
    """
    Minimal Cutout/CoarseDropout for torchvision.transforms.Compose.
    Expects x: torch.Tensor of shape [C, H, W].
    """
    def __init__(self, p=0.2, max_holes=1, max_size=64):
        self.p = p
        self.max_holes = max_holes
        self.max_size = max_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        if x.dim() != 3:
            raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")

        c, h, w = x.shape
        out = x.clone()

        holes = random.randint(1, self.max_holes)
        for _ in range(holes):
            size = random.randint(1, self.max_size)
            cy = random.randint(0, h - 1)
            cx = random.randint(0, w - 1)

            y0 = max(0, cy - size // 2)
            y1 = min(h, cy + size // 2)
            x0 = max(0, cx - size // 2)
            x1 = min(w, cx + size // 2)

            fill = x.mean().item()
            out[:, y0:y1, x0:x1] = fill

        return out


# normalize calculation
class ZScoreNormalize:
    def __call__(self, x):
        # x: Tensor [C,H,W]
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-6)


# paring masks and images
class PairedSegDataset(Dataset):
    def __init__(self, img_root, mask_root, transform_img=None, transform_mask=None, exts=(".png",".jpg",".jpeg",".tif",".tiff")):
        self.img_root = img_root
        self.mask_root = mask_root
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        # Build a sorted list of image filenames
        self.img_files = sorted([
            f for f in os.listdir(img_root)
            if f.lower().endswith(exts)
        ])

        # Optional: sanity check masks exist
        missing = [f for f in self.img_files if not os.path.exists(os.path.join(mask_root, f))]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} masks, e.g. {missing[:5]}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        img_path  = os.path.join(self.img_root, fname)
        mask_path = os.path.join(self.mask_root, fname)

        img = Image.open(img_path).convert("RGB")     # MRI grayscale (or "RGB" if needed)
        mask = Image.open(mask_path).convert("L")   # mask grayscale

        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return img, mask


class Binarize:
    def __init__(self, thr=0.5):
        self.thr = thr

    def __call__(self, t):
        return (t > self.thr).float()


transform_train = transforms.Compose([
    transforms.Resize((272, 272)),
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    RimDropout(p=0.6, min_frac=0.03, max_frac=0.10, fill="mean"),
    CoarseDropout(p=0.15, max_holes=2, max_size=16),
    ZScoreNormalize(),
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ZScoreNormalize(),
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    Binarize(0.5),
])

trainset = datasets.ImageFolder(root=TRAIN_ROOT, transform=transform_train)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)

valset = datasets.ImageFolder(root=VAL_ROOT, transform=transform_test)
valloader = DataLoader(
    valset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)

diceset = PairedSegDataset(DICE_ROOT, MASK_ROOT, transform_test, transform_mask)
diceloader = DataLoader(
    diceset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)


# ---------------------------
# Fixed samples for visualization
# ---------------------------
NUM_VIZ_SAMPLES = 8  # or 16


# ---------------------------
# ResNet backbone (yours)
# ---------------------------
class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    Returns spatial feature map (no global pooling, no fc).
    """
    def __init__(self, block_type, num_block, num_classes=5):
        super().__init__()
        self.in_channels = 64

        # NOTE: your original conv1 (kernel 9, stride 2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self.make_layer(block_type, 64, num_block[0], stride=1)
        self.layer2 = self.make_layer(block_type, 128, num_block[1], stride=2)
        self.layer3 = self.make_layer(block_type, 256, num_block[2], stride=2)
        self.layer4 = self.make_layer(block_type, 512, num_block[3], stride=2)

    def make_layer(self, block_type, out_channels, blocks, stride):
        layers = []
        layers.append(block_type(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block_type.expansion
        for _ in range(1, blocks):
            layers.append(block_type(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out  # [B, 512*expansion, H, W]


# ---------------------------
# Segmentation-style MIL Head
# ---------------------------
class SegMILHead(nn.Module):
    """
    Produces per-class logit maps S: [B, K, H, W]
    and pools them into bag logits: [B, K] using Top-K pooling.
    """
    def __init__(self, in_channels, num_classes, k_ratio=0.02, dropout=0.2, tau=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.k_ratio = k_ratio
        self.tau = float(tau)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv1x1 = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self._printed_this_epoch = False

    def forward(self, feat):  # feat: [B, C, H, W]
        feat = self.dropout(feat)
        S = self.conv1x1(feat)  # [B, K, H, W]
        S = S * torch.exp(self.logit_scale)

        B, K, H, W = S.shape
        S_flat = S.view(B, K, H * W)  # [B, K, HW]

        tau = max(self.tau, 1e-6)
        x = S_flat.float().clamp(-20.0, 20.0)  # clamp is fine here
        n = x.size(-1)

        bag_logits = tau * (torch.logsumexp(x / tau, dim=-1) - math.log(n))

        # debug
        if self.training and not self._printed_this_epoch:
            with torch.no_grad():
                w = torch.softmax(x / tau, dim=-1)
                print(
                    f"tau={tau:.3f} "
                    f"x_std={x.std().item():.4f} "
                    f"top1={w.max(dim=-1).values.mean().item():.4f}"
                )
            self._printed_this_epoch = True

        # Visualization map for predicted class: [B, H, W]
        pred_cls = bag_logits.argmax(dim=1)
        vis_map = S[torch.arange(B, device=S.device), pred_cls]

        return bag_logits, vis_map, S


class MILModel(nn.Module):
    """
    Backbone -> per-class spatial logits -> TopK MIL pooling -> global prediction.
    Also returns a spatial map you can visualize.
    """
    def __init__(self, resnet, num_classes=5, k_ratio=0.02):
        super().__init__()
        self.backbone = resnet
        in_channels = 512 * BottleNeck.expansion
        self.mil_head = SegMILHead(in_channels, num_classes, k_ratio=k_ratio, dropout=0.2, tau=TAU)

    def forward(self, x):
        feat = self.backbone(x)                 # [B, C, H, W]
        bag_logits, vis_map, full_maps = self.mil_head(feat)
        return bag_logits, vis_map, full_maps, feat   # logits for CE, and maps for visualization/regularization


# ---------------------------
# Loss helpers (make maps less "single-pixel cheating")
# ---------------------------
def tv_loss(prob_map):
    """
    Total variation loss encourages smooth/contiguous regions.
    prob_map: [B, H, W] in [0,1]
    """
    dh = torch.abs(prob_map[:, 1:, :] - prob_map[:, :-1, :]).mean()
    dw = torch.abs(prob_map[:, :, 1:] - prob_map[:, :, :-1]).mean()
    return dh + dw


def area_hinge_loss(prob_map, amin=0.001, amax=0.08):
    """
    Hinge loss on the *fraction of pixels activated* in the map.

    prob_map: [B, H, W] in [0,1]
    area = mean(prob_map) per image -> roughly the expected proportion of the map that is "on".

    amin: minimum desired active fraction (e.g., 0.01 = ~1% of map pixels on average)
    amax: maximum desired active fraction (e.g., 0.20 = ~20% of map pixels on average)
    """
    # accept [B,1,H,W] or [B,H,W]
    if prob_map.dim() == 4:
        prob_map = prob_map.squeeze(1)  # [B,H,W]

    area = prob_map.mean(dim=(1, 2))  # [B]

    low = torch.relu(amin - area)
    high = torch.relu(area - amax)
    return (low + high).mean()


# ---------------------------
# Border suppression (unsupervised prior)
# ---------------------------
_border_mask_cache = {}

def make_border_mask(h, w, frac=0.25, device=None, dtype=torch.float32):
    """
    Creates a synthetic border-band mask with 1s on the border and 0s inside.
    Returned shape: [1, h, w] so it broadcasts over [B, h, w].

    frac: thickness as fraction of min(h,w). For 14x14, frac=0.25 -> thickness≈3.
    """
    key = (h, w, frac, str(device), str(dtype))
    if key in _border_mask_cache:
        return _border_mask_cache[key]

    t = int(max(1, round(frac * min(h, w))))
    t = max(1, min(t, min(h, w) // 2))

    m = torch.zeros((1, h, w), device=device, dtype=dtype)
    m[:, :t, :] = 1
    m[:, h - t:, :] = 1
    m[:, :, :t] = 1
    m[:, :, w - t:] = 1

    _border_mask_cache[key] = m
    return m

def border_loss(prob_map, border_mask):
    """
    prob_map:    [B, h, w] in [0,1]
    border_mask: [1, h, w] in {0,1}
    Penalizes probability mass on the border band.
    """
    return (prob_map * border_mask).mean()


def region_grow_loss(feat, prob_map, thresh=0.7):
    """
    Encourages the activation to expand to nearby / similar-feature pixels.
    feat:     [B, C, H, W] backbone features
    prob_map: [B, H, W] sigmoid(true_logits) in [0,1]
    """
    B, C, H, W = feat.shape

    # Normalize features for cosine similarity
    feat_n = F.normalize(feat, dim=1)  # [B,C,H,W]

    # High-confidence tumor region mask
    mask = (prob_map > thresh).float().unsqueeze(1)  # [B,1,H,W]
    mask_sum = mask.sum(dim=(2, 3)) + 1e-6           # [B,1]

    # Prototype feature vector of confident region
    proto = (feat_n * mask).sum(dim=(2, 3)) / mask_sum  # [B,C]
    proto = F.normalize(proto, dim=1).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]

    # Cosine similarity map to prototype
    sim = (feat_n * proto).sum(dim=1)  # [B,H,W]

    # Want similar pixels to be considered tumor-like; penalize low similarity
    # margin 0.5 is a safe start; raise to make expansion more aggressive
    margin = 0.5
    return F.relu(margin - sim).mean()


def flip_consistency_loss(map_a, map_b):
    """
    map_a: [B,H,W] (prob or sigmoid(logits))
    map_b: [B,H,W] (same)
    penalize mismatch
    """
    return F.l1_loss(map_a, map_b)


def sobel_mag(x_gray):
    # x_gray: [B,1,H,W]
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x_gray.device, dtype=x_gray.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x_gray.device, dtype=x_gray.dtype).view(1,1,3,3)
    gx = F.conv2d(x_gray, kx, padding=1)
    gy = F.conv2d(x_gray, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-8)


def anti_edge_loss(images, prob_map, out_hw):
    """
    images: [B,3,256,256] normalized
    prob_map: [B,16,16] = sigmoid(true_logits)
    out_hw: (H,W) of prob_map (16,16)
    """
    xg = images.mean(dim=1, keepdim=True)  # [B,1,256,256]
    edges = sobel_mag(xg)                  # [B,1,256,256]
    edges = F.interpolate(edges, size=out_hw, mode="bilinear", align_corners=False).squeeze(1)  # [B,16,16]

    # normalize per-image to [0,1]
    edges = (edges - edges.amin(dim=(1,2), keepdim=True)) / (edges.amax(dim=(1,2), keepdim=True) - edges.amin(dim=(1,2), keepdim=True) + 1e-8)

    return (prob_map * edges).mean()  # penalize attention on edges


def tau_schedule(epoch, epochs, start=1.2, end=3.0, warm_frac=0.4):
    warm = int(epochs * warm_frac)
    if epoch < warm:
        t = epoch / max(1, warm)
        return start + t * (end - start)
    return end


# NEW: Calculates HD95 score
def hd95_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    # empty handling
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    st = generate_binary_structure(2, 1)  # 4-connectivity
    surf = lambda x: x & ~binary_erosion(x, structure=st, iterations=1, border_value=0)

    ps, gs = surf(pred), surf(gt)
    if ps.sum() == 0 and gs.sum() == 0:
        return 0.0
    if ps.sum() == 0 or gs.sum() == 0:
        return float("inf")

    d_to_gt = distance_transform_edt(~gs)[ps]
    d_to_pr = distance_transform_edt(~ps)[gs]
    d = np.concatenate([d_to_gt, d_to_pr])
    return float(np.percentile(d, 95))


@torch.no_grad()
def hd95_batch_mean_from_logits(logits, targets, thr_pred=0.5, thr_gt=0.5, ignore_inf=True):
    """
    logits:  [B, 1, H, W] (raw)
    targets: [B, 1, H, W] or [B, H, W] (0/1 or 0/255)
    returns: mean HD95 over images in batch
    """
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    pred = (torch.sigmoid(logits) > thr_pred)

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    if targets.max() > 1.5:          # handle 0/255 masks
        targets = targets / 255.0
    gt = (targets > thr_gt)

    pred_np = pred.squeeze(1).cpu().numpy()
    gt_np   = gt.squeeze(1).cpu().numpy()

    vals = np.array([hd95_2d(pred_np[i], gt_np[i]) for i in range(pred_np.shape[0])], dtype=np.float64)

    if ignore_inf:
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else float("inf")
    else:
        return float(vals.mean())


# NEW: Calculates dice score
@torch.no_grad()
def dice_global_accumulate_from_logits(logits, targets, threshold=0.5):
    # targets: [B, 1, H, W] (or [B, H, W])
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

        # Make logits spatially match targets
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.flatten(1)
    targets = targets.float().flatten(1)

    inter = (preds * targets).sum()                 # scalar
    pred_sum = preds.sum()
    targ_sum = targets.sum()
    return inter, pred_sum, targ_sum


# NEW: Returns dice and HD95 score
@torch.no_grad()
def mask_validation(model):
    # dice
    total_inter = 0.0
    total_pred = 0.0
    total_targ = 0.0
    eps = 1e-6

    # HD95
    sum_hd = 0.0
    count_batches = 0

    model.eval()
    with torch.no_grad():
        for imgs, masks in diceloader:
            imgs, masks = imgs.to(device), masks.to(device)
            _, _, full_maps, _ = model(imgs)
            logits, _ = full_maps.max(dim=1, keepdim=True)     # [B, 1, H, W]

            # dice
            inter, pred_sum, targ_sum = dice_global_accumulate_from_logits(logits, masks)
            total_inter += inter.item()
            total_pred += pred_sum.item()
            total_targ += targ_sum.item()

            # HD95
            sum_hd += hd95_batch_mean_from_logits(logits, masks)
            count_batches += 1

    dice = (2 * total_inter + eps) / (total_pred + total_targ + eps)
    hd95 = sum_hd / max(count_batches, 1)

    return dice, hd95


# ---------------------------
# Training / Validation
# ---------------------------
@torch.no_grad()
def validation(model):
    model.eval()
    correct = 0
    total = 0

    for images, targets in valloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs, _, _, _ = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == targets).sum().item()
        total += targets.numel()

    return 100.0 * correct / max(1, total)


def training(model, optimizer, start_epoch, epochs):
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    train_losses, train_accuracies, learning_rates = [], [], []

    for epoch in range(start_epoch, epochs):
        tau_value = tau_schedule(epoch, EPOCHS, start=1.5, end=3.5, warm_frac=0.4)
        model.mil_head.tau = float(tau_value)
        model.mil_head._printed_this_epoch = False

        model.train()

        total_loss = 0.0
        correct = 0

        # store lr
        learning_rates.append(optimizer.param_groups[0]["lr"])

        for idx, (images, targets) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs, vis_map, full_maps, features = model(images)
                loss_cls = criterion(outputs, targets)

                # --- TRUE class map regularization (existing) ---
                B, K, H, W = full_maps.shape
                true_logits = full_maps[torch.arange(B, device=full_maps.device), targets]  # [B,H,W]
                true_logits_fp = true_logits.float()
                true_prob = torch.sigmoid(true_logits_fp)  # [B,H,W]

                loss_area = area_hinge_loss(true_prob, amin=AREA_MIN, amax=AREA_MAX)

                loss_tv = tv_loss(true_prob)

                # --- NEW: border suppression (discourage skull/border shortcuts) ---
                bmask = make_border_mask(H, W, frac=BORDER_FRAC, device=true_prob.device, dtype=true_prob.dtype)  # [1,H,W]
                loss_border = border_loss(true_prob, bmask)

                # --- STEP 2: region-growing loss (expand within tumor-like region) ---
                loss_rg = region_grow_loss(features, true_prob, thresh=RG_THRESH)

                # tumor edge suppression
                loss_tumor_edge = anti_edge_loss(images, true_prob, out_hw=(H, W))

                # --- STEP 3: flip consistency loss ---
                loss_cons = torch.tensor(0.0, device=device)
                if CONS_USE_HFLIP:
                    images_flip = torch.flip(images, dims=[3])  # horizontal flip on W
                    outputs_f, vis_map_f, full_maps_f, _ = model(images_flip)

                    # Use TRUE class map again, flip it back to original coordinates
                    true_logits_f = full_maps_f[torch.arange(B, device=full_maps.device), targets]  # [B,h,w]
                    true_prob_f = torch.sigmoid(true_logits_f)
                    true_prob_f = torch.flip(true_prob_f, dims=[2])  # flip back along W (dim=2 for [B,H,W])

                    loss_cons = flip_consistency_loss(true_prob, true_prob_f)

                # --- Total loss ---
                loss = (
                        loss_cls
                        + LAMBDA_TV * loss_tv
                        + LAMBDA_AREA * loss_area
                        + LAMBDA_RG * loss_rg
                        + LAMBDA_CONS * loss_cons
                        + LAMBDA_BORDER * loss_border
                        + LAMBDA_ANTIEDGE * loss_tumor_edge
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / max(1, len(trainloader))
        train_acc = correct / max(1, len(trainset))

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)

        test_acc = validation(model)

        # NEW: Calculating dice score per epoch
        dice, hd95 = mask_validation(model)

        print(f"Epoch {epoch+1}: Avg loss: {avg_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Test Acc: {test_acc:.2f}%, "
              f"    Logit scale: {model.mil_head.logit_scale.item():.4f}, Dice: {dice:.5f}, HD95: {hd95:.2f}, Tau: {tau_value:.3f}")

        # Saves epoch info into a .txt file
        with open("./figures/MiTS100-3.txt", "a") as f:
            f.write(f"{epoch + 1},{avg_loss:.4f},{test_acc:.2f},{learning_rates[-1]:.9f},{dice:.3f},{hd95:.2f},{tau_value:.3f},{model.mil_head.logit_scale.item():.4f}\n")

        # Save checkpoint each epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            CHECKPOINT_PATH
        )

    return train_losses, train_accuracies, learning_rates


# ---------------------------
# Visualization
# ---------------------------
def denormalize(img_tensor):
    """
    img_tensor: [3,H,W] normalized with mean/std used above.
    returns float numpy [H,W,3] in [0,1]
    """
    mean = torch.tensor([0.1855, 0.1855, 0.1855], device=img_tensor.device)[:, None, None]
    std = torch.tensor([0.1813, 0.1813, 0.1813], device=img_tensor.device)[:, None, None]
    x = img_tensor * std + mean
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).detach().cpu().numpy()


@torch.no_grad()
def visualize_mil_map(input_tensor, pred_map, image_idx, upsample_to=(256, 256), title="MIL Map", true_map=None):
    """
    input_tensor: [B,3,H,W] normalized
    pred_map:     [B,h,w] raw logits map (not sigmoid) for predicted class
    true_map:     [B,h,w] raw logits map (not sigmoid) for true class (optional)
    """
    model_img = input_tensor[0]
    img = denormalize(model_img)

    def up_and_norm(m_logits):
        m = torch.sigmoid(m_logits[0])               # [h,w] in [0,1]
        m = m.unsqueeze(0).unsqueeze(0)              # [1,1,h,w]
        m = F.interpolate(m, size=upsample_to, mode="bilinear", align_corners=False)
        m = m.squeeze().detach().cpu().numpy()
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        return m

    pred_vis = up_and_norm(pred_map)

    if true_map is None:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Input")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(title + " (Pred)")
        plt.imshow(img)
        plt.imshow(pred_vis, alpha=0.5)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"./figures/{image_idx}HM.png", dpi=300, bbox_inches="tight")
        plt.show()
        return

    true_vis = up_and_norm(true_map)

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(title + " (Pred)")
    plt.imshow(img)
    plt.imshow(pred_vis, alpha=0.5)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(title + " (True)")
    plt.imshow(img)
    plt.imshow(true_vis, alpha=0.5)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


# Visualization mask form
@torch.no_grad()
def mil_mask_from_full_maps(outputs, full_maps, upsample_to=(256,256), thr=0.3, class_idx=None):
    """
    images:    [B,3,256,256] normalized
    outputs:   [B,K] bag logits
    full_maps: [B,K,h,w] per-class logits map

    Returns:
      m_norm:  [256,256] float in [0,1] (viz-matched)
      binary:  [256,256] uint8 {0,1}
      cls:     int class index used
    """
    B, K, h, w = full_maps.shape
    if class_idx is None:
        cls = int(outputs.argmax(dim=1)[0].item())   # predicted class for sample 0
    else:
        cls = int(class_idx)

    logits = full_maps[0, cls]                       # [h,w] logits for that class (sample 0)
    m = torch.sigmoid(logits)                        # [h,w] in [0,1]
    m = m.unsqueeze(0).unsqueeze(0)                  # [1,1,h,w]
    m = F.interpolate(m, size=upsample_to, mode="bilinear", align_corners=False)
    m = m.squeeze().detach().cpu().numpy()           # [256,256]

    # Same normalization as visualize_mil_map
    m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
    binary = (m_norm > thr).astype(np.uint8)

    return m_norm, binary, cls


def dice_np(pred_mask, true_mask, eps=1e-8):
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    inter = np.logical_and(pred, true).sum()
    denom = pred.sum() + true.sum()

    return (2.0 * inter + eps) / (denom + eps)


def hd95_single(pred_mask, true_mask, spacing=(1.0, 1.0)):
    """
    Compute the 95th-percentile Hausdorff Distance (HD95) between two binary masks for a single image.

    pred_mask, true_mask: 2D arrays (H,W) of {0,1} or bool
    spacing: (sy, sx) pixel spacing; use real MRI spacing if you have it (e.g., (0.7, 0.7))

    Returns:
        float: HD95 (in same units as spacing, default pixels)
    """
    pred = np.asarray(pred_mask).astype(bool)
    true = np.asarray(true_mask).astype(bool)

    # --- Edge cases ---
    if not pred.any() and not true.any():
        return 0.0  # both empty => perfect
    if pred.any() != true.any():
        return float("inf")  # one empty, one not => undefined/very bad

    # 8-connected structure in 2D
    conn = generate_binary_structure(rank=2, connectivity=2)

    # Extract boundaries (surface pixels)
    pred_er = binary_erosion(pred, structure=conn, border_value=0)
    true_er = binary_erosion(true, structure=conn, border_value=0)
    pred_surf = pred ^ pred_er
    true_surf = true ^ true_er

    # Distance transform to the nearest surface pixel
    # EDT expects sampling per axis as (sy, sx)
    dt_true = distance_transform_edt(~true_surf, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surf, sampling=spacing)

    # Surface-to-surface distances
    d_pred_to_true = dt_true[pred_surf]
    d_true_to_pred = dt_pred[true_surf]

    # Combine both directions and take the 95th percentile
    all_d = np.concatenate([d_pred_to_true, d_true_to_pred])
    return float(np.percentile(all_d, 95))


def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Checkpoint loaded.")
        start_epoch = checkpoint["epoch"]
        return start_epoch

    return 0


# ---------------------------
# Main
# ---------------------------
def main():
    print(f"Using device: '{device}'")
    print(f"Classes: {trainset.classes}")

    resnet = ResNet(BottleNeck, [3, 4, 6, 3], num_classes=NUM_CLASSES).to(device)
    model = MILModel(resnet, num_classes=NUM_CLASSES, k_ratio=K_RATIO).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "weight_decay": 1e-4},
            {"params": model.mil_head.conv1x1.parameters(), "weight_decay": 1e-4},
            {"params": [model.mil_head.logit_scale], "weight_decay": 0.0},
        ],
        lr=1e-4
    )

    # Load checkpoint
    start_epoch = load_checkpoint(model, optimizer)

    # Checks if directory exists, else creates dir
    os.makedirs("./figures", exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    # Single image with mask comparison
    viz_index = "20"
    images = transform_test(Image.open(f"./Manual_Testing/imgs/{viz_index}.png").convert("RGB")).unsqueeze(0).to(device)
    mask = transform_mask(Image.open(f"./Manual_Testing/masks/{viz_index}.png").convert("L"))
    mask = (mask > 0.5).int()  # torch binary
    mask = mask.squeeze().cpu().numpy()  # numpy binary

    # Train
    # training(model, optimizer, start_epoch, EPOCHS)

    # Quick visualize on one test batch
    model.eval()
    # images, targets = next(iter(valloader))

    images = images.to(device)
    # targets = targets.to(device)

    with torch.no_grad():
        outputs, vis_map, full_maps, _ = model(images)

    print(full_maps.mean())

    pred = outputs.argmax(dim=1)[0].item()

    m, binary, cls = mil_mask_from_full_maps(outputs, full_maps, thr=0.5)
    dice = dice_np(binary, mask)
    hd95 = hd95_single(binary, mask)
    print("Dice: ", dice, "    HD95: ", hd95)

    img = denormalize(images[0])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth Mask")
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Predicted Mask\n Dice: {dice:.3f}\n HD95: {hd95:.2f}")
    plt.imshow(img)
    plt.imshow(binary, alpha=0.5, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"./figures/{viz_index}Mask.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    visualize_mil_map(
        images,
        vis_map,
        image_idx=viz_index,
        upsample_to=(256, 256),
        title="MIL (LogSumExpo) Attention/Map",
        true_map=None
    )


if __name__ == "__main__":
    main()