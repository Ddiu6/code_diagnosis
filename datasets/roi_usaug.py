"""ROI 眼区约束 + 超声干扰仿真增强（从 others/code/code_0706 移植）。

与现有增强的关系（重要）：
  - ROICrop        : 替代 BorderCrop（两者都做固定裁边，不可叠加）。
                     增量 = 内容框收紧 + 椭圆区域约束（压暗球外）。
  - UltrasoundInterferenceAugment
                   : 三类超声干扰（Rayleigh 散斑 / 回声增强 / 设备伪影）。
                     与现有 SpeckleNoise（高斯散斑）属同类但不同实现，
                     建议二者二选一；echo/artifact 为全新干扰，与现有互补。

本文件独立，不修改 datasets/eye_mil_dataset.py 中已有的
BorderCrop / CLAHE / SpeckleNoise / build_transforms 实现。
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter


# ============================================================
# ROI 眼区约束
# ============================================================

class ROICrop:
    """固定裁边 +（可选）内容框收紧 + 椭圆区域约束。

    与 BorderCrop 的区别：BorderCrop 只做固定比例裁边；ROICrop 在此基础上
    进一步根据内容包围盒收紧，并对眼区施加椭圆约束、压暗球外区域，
    突出玻璃体 / 球壁主区，抑制周边标注与无关背景。

    Args:
        left/right/top/bottom: 固定裁边比例（与 BorderCrop 语义一致）
        darken: 椭圆约束的球外压暗系数，越小压得越暗（推荐 0.22）
        margin_ratio: 内容框外边距比例
        content_thr: 内容二值化阈值（灰度 > thr 视为内容）
    """

    def __init__(
        self,
        left: float = 0.2,
        right: float = 0.3,
        top: float = 0.25,
        bottom: float = 0.25,
        darken: float = 0.22,
        margin_ratio: float = 0.04,
        content_thr: int = 18,
    ):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.darken = darken
        self.margin_ratio = margin_ratio
        self.content_thr = content_thr

    def _crop_center(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        left = int(w * self.left)
        right = int(w * (1.0 - self.right))
        top = int(h * self.top)
        bottom = int(h * (1.0 - self.bottom))
        return img.crop((left, top, right, bottom))

    def _content_bbox(self, gray: np.ndarray) -> Optional[tuple]:
        mask = gray > self.content_thr
        if mask.sum() < 64:
            return None
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        h, w = gray.shape
        my = max(2, int(h * self.margin_ratio))
        mx = max(2, int(w * self.margin_ratio))
        y0 = max(0, y0 - my)
        x0 = max(0, x0 - mx)
        y1 = min(h - 1, y1 + my)
        x1 = min(w - 1, x1 + mx)
        if (x1 - x0) < w * 0.35 or (y1 - y0) < h * 0.35:
            return None
        return x0, y0, x1 + 1, y1 + 1

    def _elliptical_focus(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img).astype(np.float32)
        h, w = arr.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        ry, rx = h * 0.48, w * 0.48
        dist = ((yy - cy) / max(ry, 1e-6)) ** 2 + ((xx - cx) / max(rx, 1e-6)) ** 2
        mask = np.clip(1.0 - (dist - 0.85) / 0.55, self.darken, 1.0).astype(np.float32)
        if arr.ndim == 3:
            arr = arr * mask[..., None]
        else:
            arr = arr * mask
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def __call__(self, img: Image.Image) -> Image.Image:
        img = self._crop_center(img)
        gray_blur = np.asarray(
            img.convert("L").filter(ImageFilter.GaussianBlur(radius=1.2))
        )
        bbox = self._content_bbox(gray_blur)
        if bbox is not None:
            img = img.crop(bbox)
        return self._elliptical_focus(img)

    def __repr__(self) -> str:
        return (f"ROICrop(left={self.left:.2f}, right={self.right:.2f}, "
                f"top={self.top:.2f}, bottom={self.bottom:.2f}, darken={self.darken:.2f})")


# ============================================================
# 超声干扰仿真增强（仅训练期）
# ============================================================

def _to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_speckle_rayleigh(img: Image.Image, severity=(0.05, 0.18)) -> Image.Image:
    """Rayleigh 乘性散斑噪声（区别于现有 SpeckleNoise 的高斯散斑）。"""
    arr = _to_float(img)
    sigma = random.uniform(*severity)
    noise = np.random.rayleigh(scale=sigma, size=arr.shape).astype(np.float32)
    noise = noise / (np.mean(noise) + 1e-6)
    out = arr * (0.85 + 0.15 * noise)
    return _to_pil(out)


def _apply_echo_enhancement(img: Image.Image) -> Image.Image:
    """回声增强：随机 blob 或 stripe 局部增益。"""
    arr = _to_float(img)
    h, w = arr.shape[:2]
    mode = random.choice(["blob", "stripe"])
    gain = random.uniform(25, 70)
    if mode == "blob":
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        rx = random.randint(max(4, w // 20), max(8, w // 8))
        ry = random.randint(max(4, h // 20), max(8, h // 8))
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        arr[mask] = np.clip(arr[mask] + gain, 0, 255)
    else:
        y0 = random.randint(h // 5, 4 * h // 5)
        thickness = random.randint(max(2, h // 80), max(4, h // 25))
        y1 = min(h, y0 + thickness)
        arr[y0:y1, :] = np.clip(arr[y0:y1, :] + gain * 0.7, 0, 255)
    return _to_pil(arr)


def _apply_device_artifact(img: Image.Image) -> Image.Image:
    """设备伪影：竖线或块状增益。"""
    arr = _to_float(img)
    h, w = arr.shape[:2]
    if random.random() < 0.5:
        x = random.randint(0, w - 1)
        width = random.randint(1, max(2, w // 120))
        x1 = min(w, x + width)
        arr[:, x:x1] = np.clip(arr[:, x:x1] + random.uniform(40, 90), 0, 255)
    else:
        x0 = random.randint(0, max(1, w // 2))
        y0 = random.randint(0, max(1, h // 2))
        bw = random.randint(max(4, w // 25), max(8, w // 10))
        bh = random.randint(max(4, h // 25), max(8, h // 10))
        arr[y0:y0 + bh, x0:x0 + bw] = np.clip(
            arr[y0:y0 + bh, x0:x0 + bw] + random.uniform(30, 80), 0, 255
        )
    return _to_pil(arr)


class UltrasoundInterferenceAugment:
    """低概率施加最多 max_ops 种超声干扰（散斑 / 回声增强 / 设备伪影）。"""

    def __init__(
        self,
        speckle_p: float = 0.30,
        echo_p: float = 0.25,
        artifact_p: float = 0.20,
        max_ops: int = 2,
    ):
        self.candidates = [
            (speckle_p, _apply_speckle_rayleigh),
            (echo_p, _apply_echo_enhancement),
            (artifact_p, _apply_device_artifact),
        ]
        self.max_ops = max_ops

    def __call__(self, img: Image.Image) -> Image.Image:
        chosen = [fn for p, fn in self.candidates if random.random() < p]
        random.shuffle(chosen)
        for fn in chosen[: self.max_ops]:
            img = fn(img)
        return img

    def __repr__(self) -> str:
        return (f"UltrasoundInterferenceAugment(speckle_p={self.candidates[0][0]}, "
                f"echo_p={self.candidates[1][0]}, artifact_p={self.candidates[2][0]}, "
                f"max_ops={self.max_ops})")
