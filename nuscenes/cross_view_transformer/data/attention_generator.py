import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
from torchvision.transforms import GaussianBlur

# 엔트로피 계산 함수 import
from .entropy_utils import calculate_entropy

class AttentionGenerator(nn.Module):
    # ... __init__ 부분은 이전과 동일 ...
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device)

        weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = lraspp_mobilenet_v3_large(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        self.transforms = weights.transforms()
        self.object_class_ids = [1, 3, 4, 6, 8] # person, car, motorcycle, bus, truck
        self.smoother = GaussianBlur(kernel_size=15, sigma=5)


    @torch.no_grad()
    def generate(self, images: torch.Tensor):
        """
        입력 이미지 텐서로부터 3가지 정보를 생성합니다:
        1. pre-attention map
        2. object_count
        3. pre-attention map entropy
        """
        # ... 이미지 전처리 및 세그멘테이션 모델 실행 부분은 동일 ...
        if images.dim() != 4:
            raise ValueError("Input images must be a 4D tensor (N, C, H, W)")
        n, _, h, w = images.shape
        image_list = [img for img in images]
        batch = self.transforms(image_list)
        batch = batch.to(self.device)
        output = self.model(batch)['out']
        seg_masks = torch.argmax(output, dim=1)

        # object_count 계산
        object_counts = torch.zeros(n, dtype=torch.long, device=self.device)
        for i in range(n):
            unique_classes = torch.unique(seg_masks[i])
            count = sum(1 for cid in self.object_class_ids if cid in unique_classes)
            object_counts[i] = count
            
        # pre_attention_map 생성
        pre_attention_maps = torch.full_like(seg_masks, 0.1, dtype=torch.float)
        for cid in self.object_class_ids:
            pre_attention_maps[seg_masks == cid] = 1.0
        pre_attention_maps = self.smoother(pre_attention_maps)
        pre_attention_maps = (pre_attention_maps - pre_attention_maps.min()) / (pre_attention_maps.max() - pre_attention_maps.min() + 1e-8)
        
        # ==========================================================
        # ✨ 사전 어텐션 맵의 엔트로피 계산 추가 ✨
        # pre_attention_maps (N, H, W) -> 각 맵의 엔트로피 계산
        map_entropies = calculate_entropy(pre_attention_maps)
        # ==========================================================

        # 최종 반환을 위해 채널 차원 추가 및 CPU로 이동
        pre_attention_maps = pre_attention_maps.unsqueeze(1) # (N, 1, H, W)

        return object_counts.cpu(), pre_attention_maps.cpu(), map_entropies.cpu()
