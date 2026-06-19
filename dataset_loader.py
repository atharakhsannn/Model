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

# Import dari shared module
from density_utils import generate_density_map


# ==============================
# Konfigurasi Default
# ==============================
IMAGES_DIR = os.path.join('dataset', 'images')
ANNOTATIONS_DIR = os.path.join('dataset', 'annotations')

# Resolusi referensi (resolusi training asli)
REFERENCE_W = 672
REFERENCE_H = 512

# Normalisasi standar ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """
    Pipeline augmentasi untuk training.

    Spatial transforms (berlaku untuk image & mask):
        - HorizontalFlip, VerticalFlip

    Pixel-level transforms (HANYA berlaku untuk image):
        - RandomSunFlare (simulasi silau cahaya pada plastik)
        - RandomBrightnessContrast

    CATATAN: Tidak ada resize di sini. Resizing dilakukan secara manual
    di __getitem__ sebelum augmentasi, agar point coordinates bisa di-scale
    secara eksplisit dan density map di-generate pada resolusi yang benar.

    Returns:
        albumentations.Compose: Pipeline augmentasi.
    """
    return A.Compose([

        # ---- Spatial Transforms (image & mask) ----
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # ---- Pixel-level Transforms (HANYA image) ----
        A.RandomSunFlare(
            p=0.2,
            src_radius=100,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3,
        ),

        # ---- Normalisasi ImageNet & konversi ke Tensor ----
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    Pipeline transformasi untuk validasi/testing (tanpa augmentasi).

    Returns:
        albumentations.Compose: Pipeline transformasi.
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_all_samples(images_dir=IMAGES_DIR, annotations_dir=ANNOTATIONS_DIR):
    """
    Scan semua pasangan (gambar, anotasi JSON) yang tersedia.

    Returns:
        list[dict]: List of {'image_path', 'points', 'name', 'count'}.
    """
    json_files = sorted(glob.glob(os.path.join(annotations_dir, '*.json')))

    samples = []
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_filename = data.get('image', '')
        points = data.get('points', [])

        # Cari gambar: pertama coba nama dari JSON, lalu fallback ke basename
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
            
            # Filter: abaikan gear dan nut untuk sementara
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
    """
    Split samples into train/val sets, stratified by count bucket.

    Buckets: 70–99, 100–149, 150–199, 200–249, 250–349.
    Ensures every count range is represented in validation.

    Parameters:
        samples (list[dict]): Output dari load_all_samples().
        val_ratio (float): Proporsi data validasi (default 0.2).
        seed (int): Random seed untuk reprodusibilitas.

    Returns:
        tuple: (train_samples, val_samples)
    """
    rng = np.random.RandomState(seed)

    # Definisikan bucket berdasarkan jumlah objek
    bucket_edges = [0, 100, 150, 200, 250, float('inf')]
    bucket_labels = ['70-99', '100-149', '150-199', '200-249', '250-349']

    # Kelompokkan samples ke bucket
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
        n_val = max(1, int(len(bucket) * val_ratio))  # Minimal 1 per bucket
        val_samples.extend(bucket[:n_val])
        train_samples.extend(bucket[n_val:])

        print(f"    Bucket {label:>8s}: {len(bucket):>3d} total -> "
              f"{len(bucket) - n_val:>3d} train, {n_val:>3d} val")

    print(f"    {'TOTAL':>15s}: {len(samples):>3d} total -> "
          f"{len(train_samples):>3d} train, {len(val_samples):>3d} val")

    return train_samples, val_samples


class DMEDataset(Dataset):
    """
    PyTorch Custom Dataset untuk Density Map Estimation dengan Scale Augmentation.

    Membaca pasangan gambar RGB dan anotasi titik, lalu:
    1. Menerapkan random scale augmentation (faktor 0.75–1.25)
    2. Menghasilkan density map on-the-fly dengan KDTree adaptive sigma
    3. Me-resize ke ukuran training tetap (672×512) untuk batching konsisten
    4. Menerapkan augmentasi Albumentations (flip, warna, normalize)

    === Mengapa resize ke ukuran tetap? ===
    Model menggunakan nn.Upsample(scale_factor=32) yang menghasilkan output
    berukuran proporsional terhadap input. Dengan ukuran input tetap, kita:
    - Menghindari kebutuhan custom collate_fn untuk padding batch
    - Menjamin dimensi output model konsisten untuk loss calculation
    - Menyederhanakan training loop tanpa mengorbankan scale invariance

    Scale invariance tetap tercapai karena density map di-generate pada
    resolusi yang sudah di-scale secara acak, lalu di-resize.

    Parameters:
        samples (list[dict]): List of {'image_path', 'points', 'name', 'count'}.
        transform (albumentations.Compose): Pipeline augmentasi/transformasi.
        scale_range (tuple): Range faktor skala (min, max) untuk augmentasi.
        target_size (tuple): (width, height) ukuran training tetap.
    """

    def __init__(self, samples, transform=None,
                 scale_range=(0.75, 1.25),
                 target_size=(REFERENCE_W, REFERENCE_H)):
        super().__init__()
        self.samples = samples
        self.transform = transform
        self.scale_range = scale_range
        self.target_size = target_size  # (width, height)

        print(f"[DMEDataset] Loaded {len(self.samples)} samples, "
              f"scale_range={self.scale_range}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Mengambil satu pasangan data dengan scale augmentation.

        Pipeline per-sample:
        1. Load gambar RGB dan koordinat titik
        2. Random scale: resize gambar & scale koordinat titik
        3. Generate density map on-the-fly pada resolusi yang di-scale
        4. Koreksi density values: bagi dengan scale**2 agar sum tetap benar
        5. Resize gambar & density map ke target_size (672×512)
           dengan koreksi area untuk mempertahankan sum density map
        6. Terapkan augmentasi Albumentations (flip, warna, normalize)

        Returns:
            dict: {
                'image': Tensor (3, H, W) — gambar ternormalisasi ImageNet,
                'heatmap': Tensor (H, W) — density map float32 (sum ≈ count),
                'name': str — nama file (tanpa ekstensi),
            }
        """
        sample = self.samples[idx]

        # ---- 1. Load gambar (BGR -> RGB) ----
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # ---- 2. Random Scale Augmentation ----
        # Menghasilkan skala acak antara scale_range[0] dan scale_range[1]
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        new_w = int(round(orig_w * scale_factor))
        new_h = int(round(orig_h * scale_factor))

        # Pastikan dimensi minimal 32 (kelipatan downscale MobileNetV2)
        new_w = max(new_w, 32)
        new_h = max(new_h, 32)

        # Resize gambar
        scaled_image = cv2.resize(image, (new_w, new_h),
                                  interpolation=cv2.INTER_LINEAR)

        # ---- 3. Scale koordinat titik & clip ke batas gambar baru ----
        scaled_points = []
        for point in sample['points']:
            sx = point[0] * scale_factor
            sy = point[1] * scale_factor

            # Clip ke batas gambar yang sudah di-resize
            sx = np.clip(sx, 0, new_w - 1)
            sy = np.clip(sy, 0, new_h - 1)

            scaled_points.append([sx, sy])

        # ---- 4. Generate density map on-the-fly (KDTree adaptive sigma) ----
        density_map = generate_density_map(
            image_shape=(new_h, new_w),
            points=scaled_points,
        )

        # ---- 5. Koreksi density values setelah scale ----
        # Saat gambar di-scale up (>1), setiap pixel mencakup area lebih besar,
        # sehingga density per pixel harus berkurang agar total count tetap sama.
        # Tanpa koreksi ini, scaling up akan menginflasi total count sum.
        density_map = density_map / (scale_factor ** 2)

        # ---- 6. Resize ke target_size untuk batching konsisten ----
        target_w, target_h = self.target_size

        if (new_w != target_w) or (new_h != target_h):
            # Simpan sum density map sebelum resize
            density_sum_before = density_map.sum()

            # Resize gambar ke target size
            scaled_image = cv2.resize(scaled_image, (target_w, target_h),
                                      interpolation=cv2.INTER_LINEAR)

            # Resize density map dengan interpolasi bilinear
            density_map = cv2.resize(density_map, (target_w, target_h),
                                     interpolation=cv2.INTER_LINEAR)

            # Koreksi area: pastikan sum density map tetap = jumlah objek
            # Setelah resize, sum berubah karena area pixel berubah.
            # Kita kalikan dengan rasio untuk mengembalikan sum asli.
            density_sum_after = density_map.sum()
            if density_sum_after > 0:
                density_map *= (density_sum_before / density_sum_after)

        # ---- 7. Terapkan augmentasi Albumentations ----
        # Density map diperlakukan sebagai 'mask' agar spatial transforms
        # (HorizontalFlip, VerticalFlip) diterapkan secara sinkron
        if self.transform is not None:
            transformed = self.transform(image=scaled_image, mask=density_map)
            image_out = transformed['image']    # Tensor (C, H, W)
            heatmap_out = transformed['mask']    # Tensor (H, W)
        else:
            image_out = scaled_image
            heatmap_out = density_map

        # Pastikan heatmap bertipe float32
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
    """
    Denormalisasi tensor gambar dari normalisasi ImageNet
    agar bisa divisualisasikan dengan benar.

    Parameters:
        tensor (torch.Tensor): Tensor (C, H, W) ternormalisasi.
        mean (list): Mean ImageNet.
        std (list): Std ImageNet.

    Returns:
        numpy.ndarray: Gambar (H, W, C) dalam range [0, 1].
    """
    img = tensor.clone().detach().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    # (C, H, W) -> (H, W, C)
    return img.permute(1, 2, 0).numpy()


# ==============================
# Sanity Check
# ==============================
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("  DATASET LOADER - Sanity Check (Scale-Invariant)")
    print("=" * 60)

    # Load semua samples dan split
    all_samples = load_all_samples()
    if len(all_samples) == 0:
        print("\n  Tidak ada data untuk di-load. Pastikan sudah menjalankan:")
        print("  1. py point_labeler.py    (anotasi titik)")
        print("  2. Pastikan file JSON anotasi ada di dataset/annotations/")
        exit()

    train_samples, val_samples = stratified_split(all_samples)

    # Buat dataset dengan augmentasi training
    train_transform = get_train_transforms()
    dataset = DMEDataset(
        samples=train_samples,
        transform=train_transform,
        scale_range=(0.75, 1.25),
    )

    # Load 1 sample
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

    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(f"\n  [DataLoader] Batch image shape  : {batch['image'].shape}")
    print(f"  [DataLoader] Batch heatmap shape: {batch['heatmap'].shape}")

    # ---- Visualisasi ----
    # Denormalisasi gambar untuk tampilan
    img_display = denormalize(image_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Gambar setelah augmentasi (denormalisasi)
    axes[0].imshow(img_display)
    axes[0].set_title(f'Augmented Image\n{name}', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Heatmap (density map)
    hm = heatmap_tensor.numpy()
    im = axes[1].imshow(hm, cmap='jet')
    axes[1].set_title(
        f'Density Map (Heatmap)\nsum={hm.sum():.2f}',
        fontsize=12, fontweight='bold'
    )
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
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
