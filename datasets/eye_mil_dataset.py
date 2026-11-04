"""
Eye-level MIL Dataset for Ocular B-scan Ultrasound Diagnosis.

Each sample = one eye = a bag of B-scan views.
Returns: (images [N,3,H,W], view_mask [N], label [C])

Key design:
  - Reuses the same eye-level split from attention_MIL/checkpoints/splits.json
  - If split file not found, rebuilds with same seed for consistency
  - Multi-label: each eye sample returns a multi-hot vector
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# CLAHE 依赖 OpenCV，若无 OpenCV 则降级为恒等变换（不影响基础流水线）
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print("[WARNING] OpenCV (cv2) not found, CLAHE will be disabled.")


# ========================
# Border Crop (same as single_view / attention_MIL)
# ========================

class BorderCrop:
    """按比例裁掉图像上下左右边框。"""

    def __init__(self, top: float = 0.0, bottom: float = 0.0,
                 left: float = 0.0, right: float = 0.0):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        x1 = int(w * self.left)
        y1 = int(h * self.top)
        x2 = int(w * (1.0 - self.right))
        y2 = int(h * (1.0 - self.bottom))
        return img.crop((x1, y1, x2, y2))

    def __repr__(self) -> str:
        return (f"BorderCrop(top={self.top:.2f}, bottom={self.bottom:.2f}, "
                f"left={self.left:.2f}, right={self.right:.2f})")


# ========================
# Ultrasound-Specific Augmentation Transforms
# ========================

class CLAHE:
    """
    对比度受限自适应直方图均衡 (Contrast Limited Adaptive Histogram Equalization)。

    原理：将 RGB 图像转为 LAB 色彩空间，仅在亮度通道（L）上做 CLAHE，
         再转回 RGB。这能增强 B 超图像局部对比度而不引入色彩失真。

    参考：Ophthalmology and Therapy (2024) 的眼科 B 超论文使用了 CLAHE。

    Parameters:
        clip_limit: 对比度裁剪阈值，越大对比度增强越强。推荐 1.0~3.0。
        tile_grid_size: 局部直方图的分块大小，(8,8) 为默认。
        p: 应用概率。
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8),
                 p: float = 0.5):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() > self.p:
            return img
        if not _HAS_CV2:
            return img  # 无 OpenCV 时静默跳过

        img_np = np.array(img)                              # (H, W, 3) uint8
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)       # RGB → LAB
        l_ch, a_ch, b_ch = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_grid_size)
        l_eq = clahe.apply(l_ch)                             # 仅对亮度通道做 CLAHE

        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)     # LAB → RGB
        return Image.fromarray(img_eq)

    def __repr__(self) -> str:
        return (f"CLAHE(clip_limit={self.clip_limit}, "
                f"tile_grid_size={self.tile_grid_size}, p={self.p})")


class SpeckleNoise:
    """
    乘性斑点噪声 (Multiplicative Speckle Noise)。

    原理：B 超图像的散斑噪声本质是乘性噪声，源于组织内散射体产生的相干干涉。
         公式：I_noisy = I + I * N(0, σ)，其中 N(0, σ) 为零均值高斯噪声。

    参考：Monkam et al. (2023), Ramakers et al. (2024) 等超声增强综述。

    Parameters:
        sigma: 噪声标准差，越大噪声越强。推荐 0.02~0.10（适合 uint8 [0,255]）。
        p: 应用概率。
    """

    def __init__(self, sigma: float = 0.05, p: float = 0.5):
        self.sigma = sigma
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() > self.p:
            return img

        img_np = np.array(img).astype(np.float32)            # (H, W, 3) float32

        # 乘性高斯噪声
        noise = np.random.normal(0, self.sigma, img_np.shape).astype(np.float32)
        noisy = img_np + img_np * noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy)

    def __repr__(self) -> str:
        return f"SpeckleNoise(sigma={self.sigma}, p={self.p})"


# ========================
# Data Builder
# ========================

def _split_class_sizes(n, val_ratio, test_ratio):
    """
    对 n 个样本的类别，返回 (val_n, test_n)。

    按比例分配，且当 n>=2 时保证 val/test 各至少 1 个样本。
    """
    if n <= 1:
        return (n, 0)  # 只有一个样本，全给 val
    val_n = max(1, int(round(n * val_ratio)))
    test_n = max(1, int(round(n * test_ratio)))
    while val_n + test_n > n:
        if val_n > test_n and val_n > 1:
            val_n -= 1
        elif test_n > 1:
            test_n -= 1
        else:
            break
    return (val_n, test_n)


def build_eye_records(
    excel_path: str,
    image_root: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    split_file: Optional[str] = None,
    split_save_dir: Optional[str] = None,
    crop_cfg: Optional[Dict] = None,
    require_patient_split: bool = False,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[str]]:
    """
    Build eye-level records with train/val/test split.

    Priority: use split_file (splits.json) if exists → ensures consistent split with attention_MIL.
    Otherwise: stratified random split.

    Returns:
        train_records, val_records, test_records, label_names
        Each record: {"patient", "eye", "views": [(img_id, full_path), ...], "label": np.array}
    """
    validated_assignments = None
    if require_patient_split:
        from utils.protocol import validate_patient_split
        validated_assignments, audit = validate_patient_split(split_file or "")
        print(f"[Data] Patient split verified: {audit}")
    # 1. Read and preprocess Excel -------------------------------------------
    df = pd.read_excel(excel_path)
    df["name"] = df["name"].ffill()
    df["eye"] = df["eye"].ffill()

    # 2. Build image path map: image_root/{patient}/{image_id} ---------------
    img_path_map: Dict[str, str] = {}
    if os.path.exists(image_root):
        for patient in os.listdir(image_root):
            patient_path = os.path.join(image_root, patient)
            if not os.path.isdir(patient_path):
                continue
            for img_name in os.listdir(patient_path):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path_map[img_name] = os.path.join(patient_path, img_name)

    print(f"[Data] Total images found in image_root: {len(img_path_map)}")

    # 2.5 Show crop sample ---------------------------------------------------
    if crop_cfg and crop_cfg.get("show_sample", False) and img_path_map:
        _show_crop_sample(img_path_map, crop_cfg)

    # 3. Group by (patient, eye) → eye-level records -------------------------
    eye_records = []
    for (name, eye), group in df.groupby(["name", "eye"]):
        diag_all_series = group["diagnosis_all"].dropna()
        if diag_all_series.empty:
            continue
        diag_str = str(diag_all_series.iloc[0])
        view_ids = group["id"].tolist()

        valid_views = [(vid, img_path_map[vid]) for vid in view_ids if vid in img_path_map]
        if not valid_views:
            continue

        eye_records.append({
            "patient": name,
            "eye": eye,
            "diagnosis_str": diag_str,
            "views": valid_views,  # list of (img_id, full_path)
        })

    print(f"[Data] Eye records with images: {len(eye_records)}")

    if validated_assignments is not None:
        actual = {(str(r["patient"]), str(r["eye"])) for r in eye_records}
        expected = set(validated_assignments)
        if actual != expected:
            raise ValueError(
                f"Dataset/split mismatch: {len(actual - expected)} unassigned eyes, "
                f"{len(expected - actual)} split eyes missing images/labels. "
                "Use the same dataset and frozen split as the baselines."
            )
    # 4. Build label vocabulary ---------------------------------------------
    all_labels_set = set()
    for r in eye_records:
        for label in r["diagnosis_str"].split(";"):
            label = label.strip()
            if label:
                all_labels_set.add(label)
    label_names = sorted(all_labels_set)
    num_classes = len(label_names)
    print(f"[Data] Disease labels ({num_classes}): {label_names}")

    # 5. Determine split ----------------------------------------------------
    split_assignments = {}  # (patient, eye) -> "train"/"val"/"test"

    if split_file and os.path.exists(split_file):
        # Load from existing splits.json (consistent with attention_MIL)
        print(f"[Data] Loading split from: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        for key in ["train_eyes", "val_eyes", "test_eyes"]:
            subset = "train" if "train" in key else ("val" if "val" in key else "test")
            for patient, eye in split_info[key]:
                split_assignments[(patient, eye)] = subset
    else:
        # Stratified split (same logic as attention_MIL)
        print("[Data] No split file found, building stratified split...")
        for r in eye_records:
            vec = np.zeros(num_classes, dtype=np.float32)
            for label in r["diagnosis_str"].split(";"):
                label = label.strip()
                if label in label_names:
                    vec[label_names.index(label)] = 1.0
            r["_label_vec"] = vec

        # ── 主类分层抽样：每个类别独立按 val/test 比例分配，保证稀有类也按比例 ──
        # 每个样本确定唯一主类（单标签→唯一标签；多标签→最稀有的标签），
        # 再对每个主类分层，避免等距抽样导致稀有类(后巩膜葡萄肿)在 val/test 比例失衡
        label_vecs = np.stack([r["_label_vec"] for r in eye_records])  # [N, C]
        n = len(eye_records)
        class_counts = label_vecs.sum(axis=0)

        rng = np.random.RandomState(seed)
        primary = np.full(n, -1, dtype=int)
        for i in range(n):
            pos = np.where(label_vecs[i] == 1)[0]
            if len(pos) == 0:
                continue
            primary[i] = int(min(pos, key=lambda c: class_counts[c]))

        assignments = np.zeros(n, dtype=int)  # 0=train, 1=val, 2=test
        for c in range(num_classes):
            idx = np.where(primary == c)[0]
            nc = len(idx)
            if nc == 0:
                continue
            val_n, test_n = _split_class_sizes(nc, val_ratio, test_ratio)
            perm = rng.permutation(idx)
            assignments[perm[:val_n]] = 1
            assignments[perm[val_n:val_n + test_n]] = 2

        for i, r in enumerate(eye_records):
            key = (r["patient"], r["eye"])
            if assignments[i] == 2:
                split_assignments[key] = "test"
            elif assignments[i] == 1:
                split_assignments[key] = "val"
            else:
                split_assignments[key] = "train"

    # 6. Build final records with labels ------------------------------------
    train_records, val_records, test_records = [], [], []

    for r in eye_records:
        label_vec = np.zeros(num_classes, dtype=np.float32)
        for label in r["diagnosis_str"].split(";"):
            label = label.strip()
            if label in label_names:
                label_vec[label_names.index(label)] = 1.0

        rec = {
            "patient": r["patient"],
            "eye": r["eye"],
            "views": r["views"],  # list of (img_id, full_path)
            "label": label_vec,
        }

        if validated_assignments is not None:
            # Use the exact mapping that passed validation, with consistent types.
            subset = validated_assignments[(str(r["patient"]), str(r["eye"]))]
        else:
            subset = split_assignments.get((r["patient"], r["eye"]), "train")
        if subset == "train":
            train_records.append(rec)
        elif subset == "val":
            val_records.append(rec)
        else:
            test_records.append(rec)

    print(f"[Data]   train eyes: {len(train_records)}, val: {len(val_records)}, test: {len(test_records)}")

    # Save split info if requested and not loaded from file
    if split_save_dir and not (split_file and os.path.exists(split_file)):
        os.makedirs(split_save_dir, exist_ok=True)
        split_info = {
            "seed": seed, "val_ratio": val_ratio, "test_ratio": test_ratio,
            "train_ratio": round(1.0 - val_ratio - test_ratio, 2),
            "total_eyes": len(eye_records),
            "train_eyes": [(r["patient"], r["eye"]) for r in train_records],
            "val_eyes": [(r["patient"], r["eye"]) for r in val_records],
            "test_eyes": [(r["patient"], r["eye"]) for r in test_records],
            "label_names": label_names,
        }
        with open(os.path.join(split_save_dir, "splits.json"), "w", encoding="utf-8") as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)

    # Show label distribution per split
    for i, name in enumerate(label_names):
        tr = sum(1 for r in train_records if r["label"][i] == 1)
        va = sum(1 for r in val_records if r["label"][i] == 1)
        te = sum(1 for r in test_records if r["label"][i] == 1)
        print(f"[Data]   {name}: train={tr}, val={va}, test={te}")

    return train_records, val_records, test_records, label_names


def _show_crop_sample(img_path_map: Dict[str, str], crop_cfg: Dict):
    """加载第一张存在的图，应用 BorderCrop，保存裁剪前后对比图。"""
    border_crop = BorderCrop(
        top=crop_cfg.get("top", 0.0), bottom=crop_cfg.get("bottom", 0.0),
        left=crop_cfg.get("left", 0.0), right=crop_cfg.get("right", 0.0),
    )
    if not img_path_map:
        return

    img_path = next(iter(img_path_map.values()))
    original = Image.open(img_path).convert("RGB")
    ow, oh = original.size
    cropped = border_crop(original)
    cw, ch = cropped.size

    bar_h = 30
    combined = Image.new("RGB", (ow + cw + 10, max(oh, ch) + bar_h), color=(255, 255, 255))
    combined.paste(original, (0, bar_h))
    combined.paste(cropped, (ow + 10, bar_h))

    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((5, 2), f"Original ({ow}x{oh})", fill=(0, 0, 0), font=font)
    draw.text((ow + 15, 2), f"Cropped ({cw}x{ch})  crop: {border_crop}", fill=(255, 0, 0), font=font)

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(script_dir, "crop_check")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "_crop_sample_check.jpg")
    combined.save(save_path)
    print(f"[Data] Crop sample saved to: {save_path}")
    print(f"[Data]   Original: {ow}x{oh} → Cropped: {cw}x{ch}")


# ========================
# Dataset
# ========================

class EyeMILDataset(Dataset):
    """
    Eye-level MIL dataset.

    Each item: (images [N_i, 3, H, W], view_mask [N_i], label [C])
      where N_i is the number of views for this eye (varies per sample).

    Uses collate_fn to batch samples with different N_i.
    """

    def __init__(
        self,
        eye_records: List[Dict],
        label_names: List[str],
        transform=None,
        is_train: bool = True,
    ):
        self.records = eye_records
        self.label_names = label_names
        self.num_classes = len(label_names)
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            images:     [num_views, 3, H, W]
            view_mask:  [num_views]  (all 1s, real views)
            label:      [num_classes]
        """
        rec = self.records[idx]

        # Load all views for this eye
        images = []
        for _, img_path in rec["views"]:
            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images, dim=0)          # [N, 3, H, W]
        view_mask = torch.ones(len(images))           # [N]
        label = torch.from_numpy(rec["label"])        # [C]

        return images, view_mask, label


def mil_collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for MIL batches with variable number of views.

    Pads images to max_views in the batch. Padding views are zero-filled
    and masked out (view_mask=0).

    Returns:
        images:     [B, max_views, 3, H, W]
        view_mask:  [B, max_views]  (1=real, 0=padding)
        labels:     [B, num_classes]
    """
    images_list, masks_list, labels_list = zip(*batch)

    # Find max views in this batch
    max_views = max(img.shape[0] for img in images_list)
    _, C, H, W = images_list[0].shape

    B = len(batch)
    padded_images = torch.zeros(B, max_views, C, H, W)
    padded_masks = torch.zeros(B, max_views)

    for i, (img, mask, label) in enumerate(batch):
        N = img.shape[0]
        padded_images[i, :N] = img
        padded_masks[i, :N] = mask

    labels = torch.stack(labels_list, dim=0)  # [B, C]

    return padded_images, padded_masks, labels


# ========================
# Transforms
# ========================

def build_transforms(
    image_size: int = 224,
    is_train: bool = True,
    border_crop: Optional[BorderCrop] = None,
    clahe: Optional[CLAHE] = None,
    speckle_noise: Optional[SpeckleNoise] = None,
    us_augment=None,
    rotation_degrees: float = 10.0,
    rotation_p: float = 1.0,
) -> transforms.Compose:
    """
    Build transforms pipeline.

    流水线顺序 (训练模式):
        BorderCrop → Resize → CenterCrop
        → RandomHorizontalFlip → RandomRotation (概率 rotation_p)
        → CLAHE (超声专有) → SpeckleNoise (超声专有) → ToTensor → Normalize

    验证/测试模式:
        BorderCrop → Resize → CenterCrop → ToTensor → Normalize
        (不做增强，保证评估一致性)
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    base = []
    if border_crop is not None:
        base.append(border_crop)

    if is_train:
        ops = base + [
            transforms.Resize(int(image_size * 1.143)),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=rotation_degrees)],
                p=rotation_p,
            ),
        ]

        # === 超声专用增强（仅训练时生效） ===
        if clahe is not None:
            ops.append(clahe)
        if speckle_noise is not None:
            ops.append(speckle_noise)
        if us_augment is not None:
            ops.append(us_augment)

        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        transform = transforms.Compose(ops)
    else:
        transform = transforms.Compose(base + [
            transforms.Resize(int(image_size * 1.143)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transform


# ========================
# Augmentation Quick Check
# ========================

def _quick_aug_check(
    train_records: List[Dict],
    train_transform,
    border_crop,
    image_size: int = 224,
):
    """
    对第一张存在的 B 超图像应用不同增强组合，生成对比拼图。
    每一列是一种确定的增强操作，而不是随机变体，方便肉眼判断每种操作的效果。
    """
    if not train_records:
        print("[AugCheck] No train records, skipping.")
        return

    first_img_path = train_records[0]["views"][0][1]
    original = Image.open(first_img_path).convert("RGB")

    base_img = border_crop(original) if border_crop is not None else original

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ---------------------------------------------------------------
    # 构建每列对应的 pipeline（确定性的，p=1.0 或 p=0.0）
    # ---------------------------------------------------------------
    clahe_on = CLAHE(clip_limit=2.0, p=1.0) if _HAS_CV2 else None
    speckle_on = SpeckleNoise(sigma=0.05, p=1.0)

    # 基础 Resize + CenterCrop（与验证集一致，不做随机裁剪）
    resize_size = int(image_size * 1.143)  # 256
    geo_base = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.0),
        transforms.RandomRotation(degrees=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 各列定义: (标签, pipeline)
    column_specs = [
        ("原始\n(BorderCrop)", None),  # 直接用 base_img
    ]

    if _HAS_CV2 and clahe_on is not None:
        column_specs.append((
            "CLAHE\n(对比度增强)",
            transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.0),
                transforms.RandomRotation(degrees=0),
                clahe_on,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        ))

    column_specs.append((
        "SpeckleNoise\n(散斑噪声)",
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.0),
            transforms.RandomRotation(degrees=0),
            speckle_on,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    ))

    if _HAS_CV2 and clahe_on is not None:
        column_specs.append((
            "CLAHE\n+ Speckle",
            transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.0),
                transforms.RandomRotation(degrees=0),
                clahe_on,
                speckle_on,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        ))

    # 完整随机流水线 ×2，展示随机增强的组合效果
    column_specs.append((
        "完整流水线\n(随机组合A)",
        train_transform,
    ))
    column_specs.append((
        "完整流水线\n(随机组合B)",
        train_transform,
    ))

    # ---------------------------------------------------------------
    # 执行各列 pipeline
    # ---------------------------------------------------------------
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    pil_variants = []
    for label, pipeline in column_specs:
        if pipeline is None:
            # 第 0 列：原始图（BorderCrop 后）
            pil_variants.append(base_img.resize((image_size, image_size)))
        else:
            tensor = pipeline(base_img)                            # [3, H, W]
            arr = tensor.permute(1, 2, 0).numpy()                 # (H, W, 3)
            arr = arr * std_arr + mean_arr                         # 反归一化
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            pil_variants.append(Image.fromarray(arr))

    # ---------------------------------------------------------------
    # 拼接 + 标注
    # ---------------------------------------------------------------
    gap = 4
    cols = len(pil_variants)
    label_h = 36   # 每张图上方标签高度
    total_w = image_size * cols + gap * (cols - 1)
    canvas = Image.new("RGB", (total_w, image_size + label_h), color=(255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()

    for idx, (img, (label, _)) in enumerate(zip(pil_variants, column_specs)):
        x = idx * (image_size + gap)
        # 用多行标签
        for li, line in enumerate(label.split("\n")):
            draw.text((x + 3, 1 + li * 14), line,
                      fill=(0, 0, 0) if idx > 0 else (0, 100, 0), font=font_title)
        canvas.paste(img, (x, label_h))

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(script_dir, "aug_check")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "_aug_sample_check.jpg")
    canvas.save(save_path, quality=85)
    print(f"[AugCheck] 增强效果对比图已保存: {save_path}")
    print(f"[AugCheck]   Columns ({cols}): " + " | ".join(lbl.replace("\n", " ") for lbl, _ in column_specs))


# ========================
# Self-Test: 验证数据流水线
# ========================

if __name__ == "__main__":
    import yaml
    from torch.utils.data import DataLoader, Subset

    # ---- 1. 加载配置 ----
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print("Eye MIL Dataset — 数据流水线验证测试")
    print("=" * 70)

    # ---- 2. 构建 eye-level records ----
    crop_cfg = cfg["data"].get("crop", {})
    # 测试时关闭裁剪示例图保存，避免干扰
    crop_cfg_test = {**crop_cfg, "show_sample": False}

    train_records, val_records, test_records, label_names = build_eye_records(
        excel_path=cfg["data"]["excel_path"],
        image_root=cfg["data"]["image_root"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"].get("test_ratio", 0.2),
        seed=cfg["seed"],
        split_file=cfg["logging"].get("split_file"),
        split_save_dir=None,  # 测试时不保存
        crop_cfg=crop_cfg_test,
    )

    # ---- 3. 构建 Dataset ----
    border_crop = None
    if crop_cfg:
        ratios = {k: crop_cfg.get(k, 0.0) for k in ["top", "bottom", "left", "right"]}
        if any(v > 0.0 for v in ratios.values()):
            border_crop = BorderCrop(**ratios)

    # 构建验证 transform（无增强）
    val_transform = build_transforms(
        image_size=cfg["data"]["image_size"],
        is_train=False,
        border_crop=border_crop,
    )

    # 构建训练 transform（含 CLAHE + SpeckleNoise）
    aug_cfg = cfg["data"].get("augmentation", {})

    clahe = None
    if aug_cfg.get("clahe", {}).get("enabled", False):
        c_cfg = aug_cfg["clahe"]
        clahe = CLAHE(
            clip_limit=c_cfg.get("clip_limit", 2.0),
            tile_grid_size=tuple(c_cfg.get("tile_grid_size", [8, 8])),
            p=c_cfg.get("p", 0.5),
        )

    speckle_noise = None
    if aug_cfg.get("speckle_noise", {}).get("enabled", False):
        s_cfg = aug_cfg["speckle_noise"]
        speckle_noise = SpeckleNoise(
            sigma=s_cfg.get("sigma", 0.05),
            p=s_cfg.get("p", 0.5),
        )

    train_transform = build_transforms(
        image_size=cfg["data"]["image_size"],
        is_train=True,
        border_crop=border_crop,
        clahe=clahe,
        speckle_noise=speckle_noise,
    )

    print(f"\n[自测] Train transform: {train_transform}")
    print(f"[自测] Val   transform: {val_transform}")

    # ---- 3.5 增强效果快速验证：对一张图多次增强并保存对比 ----
    if aug_cfg.get("clahe", {}).get("enabled") or aug_cfg.get("speckle_noise", {}).get("enabled"):
        _quick_aug_check(train_records, train_transform, border_crop)

    all_records = train_records + val_records + test_records
    full_dataset = EyeMILDataset(all_records, label_names, transform=val_transform, is_train=False)

    # ---- 4. 随机抽取 20 个样本 ----
    import random as _random
    _random.seed(cfg["seed"])
    num_samples = min(20, len(full_dataset))
    sampled_indices = _random.sample(range(len(full_dataset)), num_samples)

    print("\n" + "=" * 70)
    print(f"随机抽取 {num_samples} 个 eye 样本进行验证")
    print("=" * 70)

    # 统计视图数分布
    view_counts = []
    for idx in sampled_indices:
        images, view_mask, label = full_dataset[idx]
        rec = full_dataset.records[idx]
        n_views = int(view_mask.sum().item())

        # 解码标签
        active_labels = [label_names[i] for i, v in enumerate(label.tolist()) if v == 1.0]
        label_str = " | ".join(active_labels) if active_labels else "(无标签)"

        print(f"\n{'─' * 60}")
        print(f"  [#{idx}] 病人: {rec['patient']}  |  眼别: {rec['eye']}")
        print(f"  诊断标签 ({len(active_labels)}): {label_str}")
        print(f"  视图数量: {n_views}")
        print(f"  Images tensor shape: {images.shape}  (N, C, H, W)")
        print(f"  View mask: {view_mask.tolist()}")
        print(f"  Label vector: {label.tolist()}")

        # 列出每个视图的文件名
        print(f"  视图文件列表:")
        for vi, (img_id, img_path) in enumerate(rec["views"]):
            print(f"    [{vi}] id={img_id}  ->  {img_path}")

        view_counts.append(n_views)

    # ---- 5. 测试 collate_fn（模拟 DataLoader 批处理） ----
    print("\n" + "=" * 70)
    print("测试 mil_collate_fn — 批处理（取前 4 个采样样本）")
    print("=" * 70)

    batch_samples = [full_dataset[i] for i in sampled_indices[:4]]
    padded_images, padded_masks, padded_labels = mil_collate_fn(batch_samples)

    print(f"\n  批大小 B = {padded_images.shape[0]}")
    print(f"  最大视图数 max_views = {padded_images.shape[1]}")
    print(f"  Padded images shape: {padded_images.shape}  (B, max_views, C, H, W)")
    print(f"  Padded masks shape:   {padded_masks.shape}  (B, max_views)")
    print(f"  Labels shape:         {padded_labels.shape}  (B, num_classes)")

    print(f"\n  各样本视图数:")
    for i in range(padded_images.shape[0]):
        real_views = int(padded_masks[i].sum().item())
        rec = full_dataset.records[sampled_indices[i]]
        print(f"    Batch[{i}] — {rec['patient']} {rec['eye']}: "
              f"{real_views} real views / {int(padded_images.shape[1])} padded")

    # ---- 6. 验证数据完整性 ----
    print("\n" + "=" * 70)
    print("数据完整性检查")
    print("=" * 70)

    all_ok = True
    for i in range(padded_images.shape[0]):
        real_views = int(padded_masks[i].sum().item())
        # 检查：填充部分应该全为 0
        if real_views < padded_images.shape[1]:
            padding = padded_images[i, real_views:]
            if not torch.all(padding == 0):
                print(f"  ❌ Batch[{i}]: 填充区域非零!")
                all_ok = False
        # 检查：真实视图部分不应该全为 0（归一化后可能有负值，但不会全为0）
        real_part = padded_images[i, :real_views]
        if torch.all(real_part == 0):
            print(f"  ❌ Batch[{i}]: 真实视图区域全零!")
            all_ok = False

    if all_ok:
        print("  ✅ 所有检查通过: 填充区域为零，真实视图非零")

    # 统计全部数据的视图分布
    print(f"\n  全部 {len(full_dataset)} 个 eye 的视图数统计:")
    all_view_counts = [len(rec["views"]) for rec in all_records]
    print(f"    最小视图数: {min(all_view_counts)}")
    print(f"    最大视图数: {max(all_view_counts)}")
    print(f"    平均视图数: {np.mean(all_view_counts):.1f}")
    print(f"    中位视图数: {np.median(all_view_counts):.1f}")

    print(f"\n  随机抽样 20 个 eye 的视图数统计:")
    print(f"    最小: {min(view_counts)}, 最大: {max(view_counts)}, "
          f"平均: {np.mean(view_counts):.1f}, 中位: {np.median(view_counts):.1f}")

    print("\n" + "=" * 70)
    print("测试完成！数据流水线工作正常。")
    print("=" * 70)
