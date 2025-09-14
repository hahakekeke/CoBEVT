
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
