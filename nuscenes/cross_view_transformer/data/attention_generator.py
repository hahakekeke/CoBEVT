# /content/CoBEVT/nuscenes/cross_view_transformer/data/attention_generator.py

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.transforms import GaussianBlur

def calculate_entropy(attention_map: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    주어진 2D 맵의 엔트로피를 계산합니다.
    """
    norm_map = attention_map / torch.sum(attention_map, dim=(-2, -1), keepdim=True)
    entropy = -torch.sum(norm_map * torch.log2(norm_map + eps), dim=(-2, -1))
    return entropy

class AttentionGenerator(nn.Module):
    """
    경량 세그멘테이션 모델을 사용하여 이미지로부터 '사전 어텐션 맵'과 '객체 수' 등을 생성합니다.
    (구버전 torchvision 호환용)
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device)

        # ==================== 수정된 부분 시작 ====================
        # 구버전 방식으로 사전 학습된 모델 로드
        self.model = lraspp_mobilenet_v3_large(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # 구버전 방식에 맞는 이미지 정규화 정의
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ==================== 수정된 부분 끝 ======================

        self.object_class_ids = [1, 3, 4, 6, 8] # person, car, motorcycle, bus, truck
        self.smoother = GaussianBlur(kernel_size=15, sigma=5)

    @torch.no_grad()
    def generate(self, images: torch.Tensor):
        if images.dim() != 4:
            raise ValueError("Input images must be a 4D tensor (N, C, H, W)")

        n, _, h, w = images.shape
        
        # ==================== 수정된 부분 시작 ====================
        # 1. 모델 입력에 맞게 이미지 전처리
        # 이미지가 0-255 uint8 이라면 0-1 float으로 변환
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        
        # 각 이미지를 정규화
        processed_images = torch.stack([self.normalize(img) for img in images])
        batch = processed_images.to(self.device)
        # ==================== 수정된 부분 끝 ======================

        # 2. 세그멘테이션 모델 실행
        output = self.model(batch)['out']
        seg_masks = torch.argmax(output, dim=1)
        
        # ... (이하 코드는 이전과 동일) ...
        # 3. object_count 계산
        object_counts = torch.zeros(n, dtype=torch.long, device=self.device)
        for i in range(n):
            unique_classes = torch.unique(seg_masks[i])
            count = sum(1 for cid in self.object_class_ids if cid in unique_classes)
            object_counts[i] = count
            
        # 4. pre_attention_map 생성
        pre_attention_maps = torch.full_like(seg_masks, 0.1, dtype=torch.float)
        for cid in self.object_class_ids:
            pre_attention_maps[seg_masks == cid] = 1.0
        
        pre_attention_maps = self.smoother(pre_attention_maps)
        
        min_val = torch.min(pre_attention_maps)
        max_val = torch.max(pre_attention_maps)
        pre_attention_maps = (pre_attention_maps - min_val) / (max_val - min_val + 1e-8)
        
        # 5. 사전 어텐션 맵의 엔트로피 계산
        map_entropies = calculate_entropy(pre_attention_maps)

        pre_attention_maps = pre_attention_maps.unsqueeze(1)

        return object_counts.cpu(), pre_attention_maps.cpu(), map_entropies.cpu()
