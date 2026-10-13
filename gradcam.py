"""Generate Grad-CAM examples from a saved ResNet checkpoint and test folder."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from common import build_model, discover, make_transform


class GradCAM:
    def __init__(self, model, layer="layer4"):
        self.model = model
        self.activation = None
        self.handle = getattr(model, layer).register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        self.activation = output

    def generate(self, tensor, target=None):
        # Gradient of the selected pre-softmax logit with respect to layer4.
        with torch.enable_grad():
            logits = self.model(tensor)
            predicted = int(logits.argmax(1).item())
            target = predicted if target is None else target
            gradients = torch.autograd.grad(logits[0, target], self.activation)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activation).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(cam, size=tensor.shape[-2:], mode='bilinear', align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            maximum = float(cam.max())
            cam = cam / maximum if maximum > 0 else np.zeros_like(cam)
            probabilities = logits.softmax(1)[0].detach().cpu().tolist()
        return cam, predicted, target, probabilities, maximum == 0

    def close(self):
        self.handle.remove()
        self.activation = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-dir', default='./chest_xray/test')
    p.add_argument('--output', default='gradcam_results')
    p.add_argument('--per-category', type=int, default=3,
                   help='Examples per TN/FP/FN/TP category; 0 means all test images')
    p.add_argument('--target', choices=['predicted', 'true', 'pneumonia'], default='predicted')
    p.add_argument('--layer', choices=['layer1', 'layer2', 'layer3', 'layer4'], default='layer4')
    args = p.parse_args()
    if args.per_category < 0:
        p.error('--per-category must be >= 0')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        p.error('Use a new empty --output directory')
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model(checkpoint['model'], pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    transform = make_transform(checkpoint['image_size'])
    samples = discover(args.data_dir)
    selected, counts = [], dict.fromkeys(['TN', 'FP', 'FN', 'TP'], 0)
    # First classify all test images without computing unnecessary gradients.
    for path, label in tqdm(samples, desc='Select test examples', dynamic_ncols=True):
        with Image.open(path) as image:
            tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted = int(model(tensor).argmax(1).item())
        category = {(0, 0): 'TN', (0, 1): 'FP', (1, 0): 'FN', (1, 1): 'TP'}[(label, predicted)]
        if args.per_category == 0 or counts[category] < args.per_category:
            selected.append((path, label, category))
            counts[category] += 1
    cam_engine = GradCAM(model, args.layer)
    records = []
    try:
        for index, (path, label, category) in enumerate(tqdm(selected, desc='Grad-CAM')):
            with Image.open(path) as image:
                image = image.convert('RGB')
                display = np.asarray(image.resize((checkpoint['image_size'], checkpoint['image_size'])))
                tensor = transform(image).unsqueeze(0).to(device)
            target = {'predicted': None, 'true': label, 'pneumonia': 1}[args.target]
            cam, predicted, target, probabilities, zero_cam = cam_engine.generate(tensor, target)
            stem = f'{index:04d}_{category}_{Path(path).stem}'
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(display)
            axes[0].set_title(f'True: {checkpoint["classes"][label]}')
            axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1)
            axes[1].set_title(f'CAM target: {checkpoint["classes"][target]}')
            axes[2].imshow(display)
            axes[2].imshow(cam, cmap='jet', alpha=.4, vmin=0, vmax=1)
            axes[2].set_title(f'Pred: {checkpoint["classes"][predicted]} ({probabilities[predicted]:.3f})')
            for ax in axes:
                ax.axis('off')
            fig.suptitle(f'{category} | {args.layer} | {Path(path).name}' + (' | zero CAM' if zero_cam else ''))
            fig.tight_layout()
            fig.savefig(out/f'{stem}.png', dpi=180)
            plt.close(fig)
            np.save(out/f'{stem}_cam.npy', cam)
            records.append({'image': path, 'true_label': checkpoint['classes'][label],
                            'predicted_label': checkpoint['classes'][predicted], 'category': category,
                            'layer': args.layer, 'cam_target': checkpoint['classes'][target],
                            'p_normal': probabilities[0], 'p_pneumonia': probabilities[1],
                            'zero_cam': zero_cam, 'figure': f'{stem}.png'})
    finally:
        cam_engine.close()
    with (out/'gradcam_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    import json
    (out/'gradcam_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    print('Saved examples:', counts, '| Output:', out.resolve())


if __name__ == '__main__':
    main()
