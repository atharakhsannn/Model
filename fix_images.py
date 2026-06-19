import cv2
import glob
import json
import os

images_dir = "dataset/images"
annotations_dir = "dataset/annotations"

target_w, target_h = 672, 512

for img_path in glob.glob(os.path.join(images_dir, "*.*")):
    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]
    if h != target_h or w != target_w:
        print(f"Fixing {img_path} (current: {w}x{h})")
        
        # 1. Resize Image
        img_resized = cv2.resize(img, (target_w, target_h))
        cv2.imwrite(img_path, img_resized)
        
        # 2. Update JSON
        basename = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(annotations_dir, basename + ".json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            scale_x = target_w / w
            scale_y = target_h / h
            
            new_points = []
            for p in data.get('points', []):
                new_points.append([p[0] * scale_x, p[1] * scale_y])
                
            data['points'] = new_points
            data['image_width'] = target_w
            data['image_height'] = target_h
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Updated {json_path}")
print("Done fixing images.")
