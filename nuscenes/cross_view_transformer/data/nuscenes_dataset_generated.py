'''
import json
import torch
from pathlib import Path
from .common import get_split
from .transforms import Sample, LoadDataTransform

# 1번 코드의 NuScenesDataset, NuScenesSingleton 가져오기
from .nuscenes_dataset import NuScenesDataset, NuScenesSingleton

class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file,
    and dynamically adds object_count by invoking NuScenesDataset.
    """
    def __init__(self, scene_name, labels_dir, transform=None, dataset_dir=None, version=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform

        # 1번 코드의 NuScenesSingleton 생성
        self.nusc_helper = NuScenesSingleton(dataset_dir, version)

        

        scene_record = None
        for s_rec in self.nusc_helper.nusc.scene:
            if s_rec['name'] == scene_name:
                scene_record = s_rec
                break

        if scene_record is None:
            raise ValueError(f"Scene with name '{scene_name}' not found in NuScenes dataset.")

        self.nusc_dataset = NuScenesDataset(
            scene_name=scene_name,
            scene_record=scene_record,
            helper=self.nusc_helper,
            cameras=[[0, 1, 2, 3, 4, 5]],
            bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0}
        )




        
        # object_count를 미리 계산하여 self.samples에 저장
        self._precompute_object_counts()


    def _precompute_object_counts(self):
        """
        Pre-computes and stores the object_count for each sample.
        """
        nusc_samples_map = {s['token']: i for i, s in enumerate(self.nusc_dataset.samples)}

        for sample_dict in self.samples:
            token = sample_dict['token']
            nusc_idx = nusc_samples_map.get(token)

            if nusc_idx is None:
                object_count = -1
            else:
                sample_from_nusc = self.nusc_dataset[nusc_idx]
                object_count = sample_from_nusc.object_count

            sample_dict['object_count'] = object_count
            print(f"Precomputed object_count for token {token}: {object_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dict = self.samples[idx]
        data = Sample(**sample_dict)

        if self.transform is not None:
            data = self.transform(data)

        return data

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    num_classes,
    augment='none',
    image=None,             # image config
    dataset='unused',       # ignore
    **dataset_kwargs
):
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    # Override augment if not training
    augment = 'none' if split != 'train' else augment
    transform = LoadDataTransform(dataset_dir, labels_dir, image, num_classes, augment)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    return [
        NuScenesGeneratedDataset(
            s,
            labels_dir,
            transform=transform,
            dataset_dir=dataset_dir,
            version=version
        )
        for s in split_scenes
    ]
'''


import json
import torch
from pathlib import Path
from .common import get_split
from .transforms import Sample, LoadDataTransform

# 1번 코드의 NuScenesDataset, NuScenesSingleton 가져오기
from .nuscenes_dataset import NuScenesDataset, NuScenesSingleton

# --- 수정/추가된 부분 ---
# Attention Map 엔트로피 계산을 위한 함수를 임포트합니다.
# 이 파일(entropy_calculator.py)은 프로젝트 내에 직접 생성해야 합니다. (아래에 예시 코드 제공)
from .entropy_calculator import calculate_attention_entropy 
# ----------------------


class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file,
    and dynamically adds object_count and attention_entropy
    by invoking NuScenesDataset and a custom calculator.
    """
    def __init__(self, scene_name, labels_dir, transform=None, dataset_dir=None, version=None, attention_maps_dir=None): # --- 수정/추가된 부분 ---
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform
        
        # --- 수정/추가된 부분 ---
        # Attention Map이 저장된 디렉토리 경로를 저장합니다.
        self.attention_maps_dir = Path(attention_maps_dir) if attention_maps_dir else None
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        # ----------------------

        # 1번 코드의 NuScenesSingleton 생성
        self.nusc_helper = NuScenesSingleton(dataset_dir, version)

        scene_record = None
        for s_rec in self.nusc_helper.nusc.scene:
            if s_rec['name'] == scene_name:
                scene_record = s_rec
                break

        if scene_record is None:
            raise ValueError(f"Scene with name '{scene_name}' not found in NuScenes dataset.")

        # 카메라 목록 정의
        self.cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

        self.nusc_dataset = NuScenesDataset(
            scene_name=scene_name,
            scene_record=scene_record,
            helper=self.nusc_helper,
            cameras=[[0, 1, 2, 3, 4, 5]],
            bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0}
        )

        # object_count를 미리 계산하여 self.samples에 저장
        self._precompute_object_counts()
        
        # --- 수정/추가된 부분 ---
        # attention_entropy를 미리 계산하여 self.samples에 저장
        if self.attention_maps_dir:
            self._precompute_attention_entropy()
        # ----------------------


    def _precompute_object_counts(self):
        """
        Pre-computes and stores the object_count for each sample.
        """
        nusc_samples_map = {s['token']: i for i, s in enumerate(self.nusc_dataset.samples)}

        for sample_dict in self.samples:
            token = sample_dict['token']
            nusc_idx = nusc_samples_map.get(token)

            if nusc_idx is None:
                object_count = -1
            else:
                sample_from_nusc = self.nusc_dataset[nusc_idx]
                object_count = sample_from_nusc.object_count

            sample_dict['object_count'] = object_count
            # print(f"Precomputed object_count for token {token}: {object_count}")

    # --- 수정/추가된 부분 ---
    def _precompute_attention_entropy(self):
        """
        Pre-computes and stores the attention map entropy for each camera in each sample.
        """
        print(f"Pre-computing attention entropy for scene...")
        for sample_dict in self.samples:
            entropies = []
            for cam in self.cameras:
                # JSON에 저장된 상대 이미지 경로
                image_path = sample_dict.get(cam)
                
                if image_path is None:
                    # 해당 카메라 데이터가 없는 경우, -1과 같은 값으로 채움
                    entropies.append(-1.0)
                    continue
                
                # 이미지 파일명만 추출하여 attention map 경로 구성
                # 예: 'samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'
                # -> 'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.png' (확장자는 예시)
                image_filename = Path(image_path).name
                attention_map_filename = Path(image_filename).with_suffix('.png') # Attention map 확장자에 맞게 변경
                attention_map_path = self.attention_maps_dir / attention_map_filename

                if not attention_map_path.exists():
                    print(f"Warning: Attention map not found at {attention_map_path}")
                    entropy = -1.0 # 맵이 없으면 -1
                else:
                    entropy = calculate_attention_entropy(attention_map_path)
                
                entropies.append(entropy)
            
            # 6개 카메라의 엔트로피 값을 텐서로 변환하여 저장
            sample_dict['attention_entropy'] = torch.tensor(entropies, dtype=torch.float32)
    # ----------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dict = self.samples[idx]

        # --- 수정/추가된 부분 ---
        # 미리 계산된 attention_entropy가 없는 경우를 대비해 기본값 설정
        if 'attention_entropy' not in sample_dict:
            sample_dict['attention_entropy'] = torch.full((len(self.cameras),), -1.0, dtype=torch.float32)
        # ----------------------

        data = Sample(**sample_dict)

        if self.transform is not None:
            data = self.transform(data)

        return data

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    num_classes,
    augment='none',
    image=None,             # image config
    dataset='unused',       # ignore
    # --- 수정/추가된 부분 ---
    attention_maps_dir=None, # Attention map 디렉토리 인자 추가
    # ----------------------
    **dataset_kwargs
):
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    # Override augment if not training
    augment = 'none' if split != 'train' else augment
    transform = LoadDataTransform(dataset_dir, labels_dir, image, num_classes, augment)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    return [
        NuScenesGeneratedDataset(
            s,
            labels_dir,
            transform=transform,
            dataset_dir=dataset_dir,
            version=version,
            # --- 수정/추가된 부분 ---
            attention_maps_dir=attention_maps_dir
            # ----------------------
        )
        for s in split_scenes
    ]
