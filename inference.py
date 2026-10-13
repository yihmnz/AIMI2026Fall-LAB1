"""Evaluate a saved checkpoint or predict one image."""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader
from common import ChestXrayDataset, discover, make_transform, build_model, run_epoch, plot_cm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--data-dir', help='Directory containing NORMAL/ and PNEUMONIA/')
    source.add_argument('--image')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output', default='inference_results')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model(ckpt['model'], pretrained=False).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    transform = make_transform(ckpt['image_size'])
    if args.image:
        with Image.open(args.image) as image:
            tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(tensor).softmax(1)[0].cpu().tolist()
        print(json.dumps({'image': args.image, 'prediction': ckpt['classes'][probs.index(max(probs))],
                          'probabilities': dict(zip(ckpt['classes'], probs))}, indent=2))
    else:
        dataset = ChestXrayDataset(discover(args.data_dir), transform)
        result = run_epoch(model, DataLoader(dataset, batch_size=args.batch_size), device)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        plot_cm(result['confusion_matrix'], out/'confusion_matrix.png', 'Test results')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
