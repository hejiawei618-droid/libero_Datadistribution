"""
视觉编码器: 用预训练 ViT 提取语义特征 (替代 flatten RGB + PCA)

使用 timm 库加载本地缓存的模型，离线可用。

用法:
    from pca.vision_encoder import VisionEncoder
    encoder = VisionEncoder("vit_small_patch16_224")
    features = encoder.encode(images)  # (N, 128, 128, 3) → (N, 384)
"""

import warnings
import numpy as np
import torch
import torch.nn.functional as F


class VisionEncoder:
    """用预训练 ViT/CNN 编码 RGB 图像"""

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.augreg_in21k_ft_in1k",
        device: str = None,
        input_size: int = 224,
    ):
        """
        Args:
            model_name: timm 模型名 (从本地缓存加载)
                可选: "vit_small_patch16_224.augreg_in21k_ft_in1k"  → 384D
                      "vit_base_patch14_dinov2.lvd142m"             → 768D (需 518×518)
                      "resnet101.tv_in1k"                          → 2048D
            device: "cuda" / "cpu", 默认自动选择
            input_size: 模型输入尺寸 (图像会被 resize 到此尺寸)
        """
        import timm

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # 去掉分类头, 只取特征
            )
        self.model.to(self.device)
        self.model.eval()

        self.feature_dim = self.model.num_features
        print(f"[VisionEncoder] {model_name} → {self.feature_dim}D (device={self.device})")

    @torch.no_grad()
    def encode(self, images: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """
        将 RGB 图像编码为特征向量

        Args:
            images:    (N, H, W, 3) uint8 [0-255]
            batch_size: GPU 批处理大小

        Returns:
            features: (N, D) float32
        """
        N = len(images)
        all_features = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = images[start:end]  # (B, H, W, 3) uint8

            # 转换为 torch tensor: (B, H, W, 3) → (B, 3, H, W) float32 [0, 1]
            x = torch.from_numpy(batch).float() / 255.0
            x = x.permute(0, 3, 1, 2)  # (B, 3, H, W)

            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std

            # Resize to model input size
            if x.shape[2] != self.input_size:
                x = F.interpolate(x, size=(self.input_size, self.input_size),
                                  mode='bilinear', align_corners=False)

            x = x.to(self.device)
            feat = self.model(x)           # (B, D)
            all_features.append(feat.cpu().numpy())

        return np.concatenate(all_features, axis=0).astype(np.float32)

    def feature_dim(self) -> int:
        return self.feature_dim
