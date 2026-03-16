import torch
import os

weight_path = "weights/yolov10_tph_best.pth"
if os.path.exists(weight_path):
    ckpt = torch.load(weight_path, map_location='cpu')
    print("Keys and Shapes in state_dict:")
    keys = sorted(list(ckpt.keys()))
    for k in keys:
        print(f"  {k:50s} : {ckpt[k].shape}")
    print(f"Total keys: {len(keys)}")
else:
    print("Weight file not found.")
