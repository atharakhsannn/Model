import os
import argparse
import csv
from datetime import datetime
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Import dari file proyek
from model_dme import DensityMapRegressor
from density_utils import create_visualization


# ============================================================
# Konfigurasi
# ============================================================
# Default fallback checkpoint (can be overridden by argparse)
DEFAULT_CHECKPOINT = os.path.join('checkpoints', 'final_dme_97percent.pth')

# Resolusi target - harus sama dengan yang digunakan saat training (dataset_loader.py)
TARGET_SIZE = (672, 512)  # (width, height)

# Normalisasi standar ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Debug mode: aktifkan assertion untuk memverifikasi count preservation saat resize
DEBUG = True


def select_device():
    """
    Deteksi device secara otomatis.
    Prioritas: CUDA (Nvidia GPU) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device     : CUDA — {gpu_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"  Device     : MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"  Device     : CPU")
    return device


def get_inference_transforms():
    """
    Pipeline preprocessing untuk inference.
    HANYA Normalize + ToTensorV2, TANPA resize (resize dilakukan secara manual
    agar kita dapat mengembalikan density map ke ukuran asli).
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_model(checkpoint_path, device):
    """
    Inisiasi model, load bobot dari checkpoint, set ke eval mode.
    """
    model = DensityMapRegressor(pretrained=False)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    best_val_mae = checkpoint.get('best_val_mae', checkpoint.get('best_mae', '?'))
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Epoch      : {epoch}")
    print(f"  Best MAE   : {best_val_mae}")

    return model


def preprocess_image(image_path, transform):
    """
    Baca gambar dengan OpenCV, konversi ke RGB, dan siapkan tensor.
    Menyimpan salinan gambar asli untuk visualisasi dan ukuran asli.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_image = image_rgb.copy()  # untuk visualisasi akhir
    orig_h, orig_w = original_image.shape[:2]

    # Resize ke ukuran training (model hanya melihat resolusi ini saat training)
    image_resized = cv2.resize(image_rgb, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    # Terapkan normalisasi dan konversi ke tensor pada gambar yang sudah di‑resize
    transformed = transform(image=image_resized)
    image_tensor = transformed['image']  # Tensor (C, H, W) – ukuran target

    return image_tensor, original_image, (orig_w, orig_h)


def predict(model, image_tensor, device):
    """
    Jalankan inference pada tensor yang sudah di‑resize ke target size.
    Output density map masih pada ukuran target.
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)  # (1, 1, target_H, target_W)

    # Konversi ke numpy dan hilangkan scaling factor training (1000.0)
    density_map = (output / 1000.0).squeeze().cpu().numpy()  # (target_H, target_W)

    # Hitung prediksi count pada ukuran target (untuk verifikasi)
    predicted_count_target = density_map.sum()

    return density_map, float(predicted_count_target)


def resize_density_map_to_original(density_map, original_size, target_size=TARGET_SIZE):
    """
    Kembalikan density map ke ukuran asli gambar dengan koreksi area.
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size

    if (orig_w == target_w) and (orig_h == target_h):
        return density_map

    # Simpan sum sebelum resize untuk koreksi dan verifikasi
    sum_before = density_map.sum()

    # Resize menggunakan bilinear interpolation
    density_orig = cv2.resize(density_map, (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)

    # Koreksi area
    sum_after_resize = density_orig.sum()
    if sum_after_resize > 0:
        density_orig *= (sum_before / sum_after_resize)

    if DEBUG:
        assert abs(density_orig.sum() - sum_before) < 0.5, \
            (f"Count changed during resize: "
             f"{sum_before:.1f} -> {density_orig.sum():.1f}")

    return density_orig


def visualize_result(original_image, overlay, predicted_count, image_name):
    """
    Tampilkan hasil prediksi menggunakan matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].imshow(original_image)
    axes[0].set_title('Gambar Asli', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title('Density Map Overlay', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    fig.suptitle(
        f'Predicted Count: {predicted_count:.1f}',
        fontsize=22,
        fontweight='bold',
        color='#e74c3c',
        y=0.98,
    )

    fig.text(0.5, 0.01, f'File: {image_name}', ha='center', fontsize=11, color='gray')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()


def run_prediction(image_path, checkpoint_path=DEFAULT_CHECKPOINT, target_size=TARGET_SIZE):
    """
    Pipeline lengkap prediksi yang sekarang scale-invariant.
    """
    image_name = os.path.basename(image_path)

    print("\n" + "=" * 60)
    print("  PREDICT - Density Map Estimation (DME)  [Scale-Invariant]")
    print("=" * 60)

    # ---- 1. Device ----
    device = select_device()

    # ---- 2. Load Model ----
    print(f"\n  [Model Loading]")
    if not os.path.exists(checkpoint_path):
        print(f"\n  [ERROR] Checkpoint tidak ditemukan: {checkpoint_path}")
        print(f"  Pastikan path benar atau jalankan training terlebih dahulu.")
        return
    model = load_model(checkpoint_path, device)

    # ---- 3. Preprocess ----
    print(f"\n  [Preprocessing]")
    print(f"  Image      : {image_path}")
    transform = get_inference_transforms()
    image_tensor, original_image, original_size = preprocess_image(image_path, transform)
    print(f"  Original size : {original_size[0]}x{original_size[1]}")
    print(f"  Resized to    : {target_size[0]}x{target_size[1]} (training resolution)")

    # ---- 4. Inference (pada ukuran target) ----
    print(f"\n  [Inference]")
    density_map_target, count_target = predict(model, image_tensor, device)
    print(f"  Density map (target) shape : {density_map_target.shape}")
    print(f"  Predicted count (target)   : {count_target:.4f}")

    # ---- 5. Kembalikan ke ukuran asli ----
    density_map_orig = resize_density_map_to_original(
        density_map_target, original_size, target_size
    )
    final_count = density_map_orig.sum()
    print(f"  Density map (original) sum : {final_count:.4f}")

    print(f"\n  +-------------------------------------+")
    print(f"  |  PREDICTED COUNT : {final_count:>8.1f} objek   |")
    print(f"  +-------------------------------------+")

    # ---- 6. Visualisasi (menggunakan density_utils.create_visualization) ----
    overlay = create_visualization(
        original_image, density_map_orig,
        input_is_rgb=True,
    )
    visualize_result(original_image, overlay, final_count, image_name)

    print(f"\n  Prediksi selesai!")
    print("=" * 60 + "\n")

    return final_count


# ============================================================
# Report Generation
# ============================================================

def generate_reports(results, checkpoint_path, sort_order='alpha'):
    """
    Generate a CSV and a Markdown summary report after batch prediction.

    Parameters:
        results       : list of (image_name, predicted_count, density_map_sum)
        checkpoint_path : str — path of the checkpoint used
        sort_order    : 'alpha' | 'highest' | 'lowest'

    Returns:
        (csv_path, md_path) — absolute paths to the generated report files.
    """
    REPORTS_DIR = 'reports'
    os.makedirs(REPORTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_path  = os.path.join(REPORTS_DIR, f'prediction_report_{timestamp}.csv')
    md_path   = os.path.join(REPORTS_DIR, f'prediction_summary_{timestamp}.md')
    ckpt_name = os.path.basename(checkpoint_path)

    # ---- Apply sort order ----
    if sort_order == 'highest':
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    elif sort_order == 'lowest':
        sorted_results = sorted(results, key=lambda x: x[1])
    else:  # 'alpha' (default)
        sorted_results = sorted(results, key=lambda x: x[0])

    counts = [r[1] for r in sorted_results]

    # ---- Write CSV ----
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(sorted_results,
                          columns=['Image Name', 'Predicted Count', 'Density Map Sum'])
        df['Checkpoint Used'] = ckpt_name
        df.to_csv(csv_path, index=False, float_format='%.4f')
    else:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image Name', 'Predicted Count', 'Density Map Sum', 'Checkpoint Used'])
            for name, count, dsum in sorted_results:
                writer.writerow([name, f'{count:.4f}', f'{dsum:.4f}', ckpt_name])

    # ---- Compute distribution buckets ----
    buckets = {
        '0–99':    sum(1 for c in counts if c < 100),
        '100–149': sum(1 for c in counts if 100 <= c < 150),
        '150–199': sum(1 for c in counts if 150 <= c < 200),
        '200–249': sum(1 for c in counts if 200 <= c < 250),
        '250–349': sum(1 for c in counts if 250 <= c < 350),
        '350+':    sum(1 for c in counts if c >= 350),
    }

    top10_high = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    top10_low  = sorted(results, key=lambda x: x[1])[:10]

    # ---- Write Markdown ----
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# DME Batch Prediction Summary Report\n\n')
        f.write(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('---\n\n')

        f.write('## Metadata\n\n')
        f.write('| Property | Value |\n')
        f.write('|----------|-------|\n')
        f.write(f'| Checkpoint | `{ckpt_name}` |\n')
        f.write(f'| Checkpoint Path | `{checkpoint_path}` |\n')
        f.write(f'| Total Images Processed | {len(results)} |\n')
        f.write(f'| Sort Order | {sort_order} |\n')
        f.write(f'| Timestamp | {timestamp} |\n\n')

        f.write('## ⚠️ Important Warnings\n\n')
        f.write('> **SMOKE TEST ONLY — NOT REAL-WORLD EVALUATION**\n>\n')
        f.write('> The `dataset/images/` folder currently contains **training images only**.\n')
        f.write('> Predictions on training images are used ONLY to verify pipeline correctness.\n')
        f.write('> Do **NOT** use these results to claim real-world accuracy.\n')
        f.write('> Upload **unseen test images** for genuine evaluation.\n\n')

        f.write('## Summary Statistics\n\n')
        f.write('| Metric | Value |\n')
        f.write('|--------|-------|\n')
        f.write(f'| Average Predicted Count | {np.mean(counts):.1f} |\n')
        f.write(f'| Minimum Predicted Count | {min(counts):.1f} |\n')
        f.write(f'| Maximum Predicted Count | {max(counts):.1f} |\n')
        f.write(f'| Std Deviation | {np.std(counts):.2f} |\n\n')

        f.write('## Prediction Distribution\n\n')
        f.write('| Count Range | Number of Images |\n')
        f.write('|-------------|------------------|\n')
        for bucket, cnt in buckets.items():
            f.write(f'| {bucket} | {cnt} |\n')
        f.write('\n')

        f.write('## Top 10 Highest Predictions\n\n')
        f.write('| # | Image Name | Predicted Count |\n')
        f.write('|---|------------|-----------------|\n')
        for rank, (name, count, _) in enumerate(top10_high, 1):
            f.write(f'| {rank} | {name} | {count:.1f} |\n')
        f.write('\n')

        f.write('## Top 10 Lowest Predictions\n\n')
        f.write('| # | Image Name | Predicted Count |\n')
        f.write('|---|------------|-----------------|\n')
        for rank, (name, count, _) in enumerate(top10_low, 1):
            f.write(f'| {rank} | {name} | {count:.1f} |\n')
        f.write('\n')

        f.write('## Full Prediction Table\n\n')
        f.write('| Image Name | Predicted Count | Density Map Sum |\n')
        f.write('|------------|-----------------|------------------|\n')
        for name, count, dsum in sorted_results:
            f.write(f'| {name} | {count:.1f} | {dsum:.4f} |\n')
        f.write('\n')
        f.write('---\n')
        f.write('*Generated by predict.py — DME Pipeline*\n')

    return os.path.abspath(csv_path), os.path.abspath(md_path)


def print_summary_to_terminal(results, checkpoint_path, csv_path, md_path, sort_order='alpha'):
    """
    Print a clean, human-readable summary report to the terminal after batch inference.
    """
    ckpt_name = os.path.basename(checkpoint_path)
    counts    = [r[1] for r in results]

    if sort_order == 'highest':
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    elif sort_order == 'lowest':
        sorted_results = sorted(results, key=lambda x: x[1])
    else:
        sorted_results = sorted(results, key=lambda x: x[0])

    top10_high = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    top10_low  = sorted(results, key=lambda x: x[1])[:10]

    sep  = '=' * 65
    sep2 = '-' * 65

    print('\n' + sep)
    print('  BATCH PREDICTION - FINAL SUMMARY REPORT')
    print(sep)
    print(f'  Checkpoint        : {ckpt_name}')
    print(f'  Total images      : {len(results)}')
    print(f'  Sort order        : {sort_order}')
    print(sep2)

    print('\n  [WARNING] SMOKE TEST - NOT REAL-WORLD EVALUATION')
    print('  dataset/images/ contains TRAINING images.')
    print('  Results verify pipeline correctness ONLY.')
    print('  Do NOT claim real-world accuracy from this run.')
    print(sep2)

    print('\n  SUMMARY STATISTICS')
    print(f'  Average predicted count : {np.mean(counts):.1f}')
    print(f'  Minimum predicted count : {min(counts):.1f}')
    print(f'  Maximum predicted count : {max(counts):.1f}')
    print(f'  Std Deviation           : {np.std(counts):.2f}')
    print(sep2)

    print('\n  PREDICTION DISTRIBUTION')
    dist = [
        ('0-99',    sum(1 for c in counts if c < 100)),
        ('100-149', sum(1 for c in counts if 100 <= c < 150)),
        ('150-199', sum(1 for c in counts if 150 <= c < 200)),
        ('200-249', sum(1 for c in counts if 200 <= c < 250)),
        ('250-349', sum(1 for c in counts if 250 <= c < 350)),
        ('350+',    sum(1 for c in counts if c >= 350)),
    ]
    for bucket, cnt in dist:
        bar = '#' * cnt
        print(f'  {bucket:<8} : {cnt:>3}  {bar}')
    print(sep2)

    print('\n  TOP 10 HIGHEST PREDICTIONS')
    print(f'  {"#":<4} {"Image Name":<34} {"Predicted Count":>15}')
    print('  ' + '-' * 56)
    for rank, (name, count, _) in enumerate(top10_high, 1):
        print(f'  {rank:<4} {name:<34} {count:>15.1f}')

    print('\n  TOP 10 LOWEST PREDICTIONS')
    print(f'  {"#":<4} {"Image Name":<34} {"Predicted Count":>15}')
    print('  ' + '-' * 56)
    for rank, (name, count, _) in enumerate(top10_low, 1):
        print(f'  {rank:<4} {name:<34} {count:>15.1f}')
    print(sep2)

    print('\n  FULL PREDICTION TABLE')
    print(f'  {"Image Name":<34} | {"Predicted Count":>15} | {"Density Sum":>12}')
    print('  ' + '-' * 68)
    for name, count, dsum in sorted_results:
        print(f'  {name:<34} | {count:>15.1f} | {dsum:>12.4f}')

    print('\n' + sep)
    print('  [OK] REPORT FILES GENERATED SUCCESSFULLY')
    print(f'  CSV      : {csv_path}')
    print(f'  Markdown : {md_path}')
    print(sep + '\n')


# ============================================================
# Eksekusi Utama
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DME Prediction Script")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory containing test images")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--sort", type=str, default='alpha',
                        choices=['alpha', 'highest', 'lowest'],
                        help="Sort order for batch report: alpha | highest | lowest (default: alpha)")
    
    args = parser.parse_args()

    if args.dir:
        if not os.path.exists(args.dir):
            print(f"[ERROR] Direktori gambar tidak ditemukan: {args.dir}")
            exit(1)
        
        supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = sorted([
            f for f in os.listdir(args.dir)
            if f.lower().endswith(supported_ext)
        ])
        
        if not image_files:
            print(f"Tidak ada gambar di folder '{args.dir}'")
            exit(1)
            
        print(f"Ditemukan {len(image_files)} gambar di {args.dir}")
        print("Mulai batch prediction...")
        
        results = []
        for i, file_name in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] Memproses {file_name}...")
            img_path = os.path.join(args.dir, file_name)
            
            # Temporarily disable plt.show by mocking it to prevent hanging during batch
            original_show = plt.show
            plt.show = lambda: plt.close('all')
            
            try:
                final_count = run_prediction(img_path, checkpoint_path=args.checkpoint)
                # store (name, predicted_count, density_map_sum) — counts are the same
                results.append((file_name, final_count, final_count))
            finally:
                plt.show = original_show

        # results stored as (name, count, density_map_sum)
        # generate reports and print terminal summary
        if results:
            csv_path, md_path = generate_reports(results, args.checkpoint, sort_order=args.sort)
            print_summary_to_terminal(results, args.checkpoint, csv_path, md_path, sort_order=args.sort)
        
    else:
        # Single image mode
        # Determine image path
        if args.image:
            if not os.path.exists(args.image):
                print(f"[ERROR] File gambar tidak ditemukan: {args.image}")
                exit(1)
            test_image = args.image
            print(f"Menggunakan gambar dari argumen: {test_image}")
        else:
            # Fallback to default image in dataset/images
            test_image_dir = os.path.join('dataset', 'images')
            supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
            
            if not os.path.exists(test_image_dir):
                print(f"[ERROR] Direktori gambar tidak ditemukan: {test_image_dir}")
                print(f"Gunakan argumen --image untuk menspesifikasi path gambar.")
                exit(1)
                
            image_files = sorted([
                f for f in os.listdir(test_image_dir)
                if f.lower().endswith(supported_ext)
            ])

            if not image_files:
                print(f"Tidak ada gambar di folder '{test_image_dir}'")
                exit(1)
            test_image = os.path.join(test_image_dir, image_files[0])
            print(f"Tidak ada argumen path gambar yang diberikan.")
            print(f"Menggunakan gambar test default: {test_image}")
            print(f"Tips: Anda bisa menjalankan dengan: python predict.py --image path/ke/gambar.jpg --checkpoint path/ke/model.pth\n")
        
        run_prediction(test_image, checkpoint_path=args.checkpoint)