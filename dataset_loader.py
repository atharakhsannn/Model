import os
import json
import glob
import numpy as np
import cv2
# pyrefly: ignore [missing-import]
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from density_utils import generate_density_map

IMAGES_DIR = os.path.join('dataset', 'images')
ANNOTATIONS_DIR = os.path.join('dataset', 'annotations')

REFERENCE_W = 672
REFERENCE_H = 512

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms():
    return A.Compose([

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.RandomSunFlare(
            p=0.2,
            src_radius=100,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3,
        ),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def load_all_samples(images_dir=IMAGES_DIR, annotations_dir=ANNOTATIONS_DIR):
    json_files = sorted(glob.glob(os.path.join(annotations_dir, '*.json')))

    samples = []
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_filename = data.get('image', '')
        points = data.get('points', [])

        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            basename = os.path.splitext(os.path.basename(json_path))[0]
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                candidate = os.path.join(images_dir, basename + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break

        if image_path is not None and os.path.exists(image_path):
            name = os.path.splitext(os.path.basename(json_path))[0]
            
            if 'gear' in name.lower() or 'nut' in name.lower():
                continue

            samples.append({
                'image_path': image_path,
                'points': points,
                'name': name,
                'count': len(points),
            })

    return samples

def stratified_split(samples, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)

    bucket_edges = [0, 100, 150, 200, 250, float('inf')]
    bucket_labels = ['70-99', '100-149', '150-199', '200-249', '250-349']

    buckets = {label: [] for label in bucket_labels}
    for sample in samples:
        count = sample['count']
        for i in range(len(bucket_edges) - 1):
            if bucket_edges[i] <= count < bucket_edges[i + 1]:
                buckets[bucket_labels[i]].append(sample)
                break

    train_samples = []
    val_samples = []

    print(f"\n  [Stratified Split] val_ratio={val_ratio}, seed={seed}")
    for label in bucket_labels:
        bucket = buckets[label]
        if len(bucket) == 0:
            continue

        rng.shuffle(bucket)
        n_val = max(1, int(len(bucket) * val_ratio))
        val_samples.extend(bucket[:n_val])
        train_samples.extend(bucket[n_val:])

        print(f"    Bucket {label:>8s}: {len(bucket):>3d} total -> "
              f"{len(bucket) - n_val:>3d} train, {n_val:>3d} val")

    print(f"    {'TOTAL':>15s}: {len(samples):>3d} total -> "
          f"{len(train_samples):>3d} train, {len(val_samples):>3d} val")

    return train_samples, val_samples

class DMEDataset(Dataset):

    def __init__(self, samples, transform=None,
                 scale_range=(0.75, 1.25),
                 target_size=(REFERENCE_W, REFERENCE_H)):
        super().__init__()
        self.samples = samples
        self.transform = transform
        self.scale_range = scale_range
        self.target_size = target_size

        print(f"[DMEDataset] Loaded {len(self.samples)} samples, "
              f"scale_range={self.scale_range}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        new_w = int(round(orig_w * scale_factor))
        new_h = int(round(orig_h * scale_factor))

        new_w = max(new_w, 32)
        new_h = max(new_h, 32)

        scaled_image = cv2.resize(image, (new_w, new_h),
                                  interpolation=cv2.INTER_LINEAR)

        scaled_points = []
        for point in sample['points']:
            sx = point[0] * scale_factor
            sy = point[1] * scale_factor

            sx = np.clip(sx, 0, new_w - 1)
            sy = np.clip(sy, 0, new_h - 1)

            scaled_points.append([sx, sy])

        density_map = generate_density_map(
            image_shape=(new_h, new_w),
            points=scaled_points,
        )

        density_map = density_map / (scale_factor ** 2)

        target_w, target_h = self.target_size

        if (new_w != target_w) or (new_h != target_h):
            density_sum_before = density_map.sum()

            scaled_image = cv2.resize(scaled_image, (target_w, target_h),
                                      interpolation=cv2.INTER_LINEAR)

            density_map = cv2.resize(density_map, (target_w, target_h),
                                     interpolation=cv2.INTER_LINEAR)

            density_sum_after = density_map.sum()
            if density_sum_after > 0:
                density_map *= (density_sum_before / density_sum_after)

        if self.transform is not None:
            transformed = self.transform(image=scaled_image, mask=density_map)
            image_out = transformed['image']    # Tensor (C, H, W)
            heatmap_out = transformed['mask']    # Tensor (H, W)
        else:
            image_out = scaled_image
            heatmap_out = density_map

        if isinstance(heatmap_out, np.ndarray):
            heatmap_out = torch.from_numpy(heatmap_out).float()
        else:
            heatmap_out = heatmap_out.float()

        return {
            'image': image_out,
            'heatmap': heatmap_out,
            'name': sample['name'],
        }

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img = tensor.clone().detach().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("  DATASET LOADER - Sanity Check (Scale-Invariant)")
    print("=" * 60)

    all_samples = load_all_samples()
    if len(all_samples) == 0:
        print("\n  Tidak ada data untuk di-load. Pastikan sudah menjalankan:")
        print("  1. py point_labeler.py    (anotasi titik)")
        print("  2. Pastikan file JSON anotasi ada di dataset/annotations/")
        exit()

    train_samples, val_samples = stratified_split(all_samples)

    train_transform = get_train_transforms()
    dataset = DMEDataset(
        samples=train_samples,
        transform=train_transform,
        scale_range=(0.75, 1.25),
    )

    sample = dataset[0]
    image_tensor = sample['image']
    heatmap_tensor = sample['heatmap']
    name = sample['name']

    print(f"\n  Sample: {name}")
    print(f"  Image tensor shape  : {image_tensor.shape}")
    print(f"  Image tensor dtype  : {image_tensor.dtype}")
    print(f"  Heatmap tensor shape: {heatmap_tensor.shape}")
    print(f"  Heatmap tensor dtype: {heatmap_tensor.dtype}")
    print(f"  Heatmap sum (count) : {heatmap_tensor.sum().item():.4f}")
    print(f"  Heatmap max         : {heatmap_tensor.max().item():.8f}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(f"\n  [DataLoader] Batch image shape  : {batch['image'].shape}")
    print(f"  [DataLoader] Batch heatmap shape: {batch['heatmap'].shape}")

    img_display = denormalize(image_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_display)
    axes[0].set_title(f'Augmented Image\n{name}', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    hm = heatmap_tensor.numpy()
    im = axes[1].imshow(hm, cmap='jet')
    axes[1].set_title(
        f'Density Map (Heatmap)\nsum={hm.sum():.2f}',
        fontsize=12, fontweight='bold'
    )
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_display)
    axes[2].imshow(hm, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle('DMEDataset Sanity Check (Scale-Invariant)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    print("\n  Sanity check selesai!")
    print("=" * 60 + "\n")
