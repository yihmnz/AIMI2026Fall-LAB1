"""Compare ResNet architectures using validation-only checkpoint selection."""
import argparse
import csv
import json
import platform
import logging
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from group_split import group_id, group_split, exclude_test_groups
from common import (CLASSES, ChestXrayDataset, discover, make_transform, build_model,
                    seed_everything, seed_worker, run_epoch, plot_cm, plot_histories)


class ProgressLogHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='./chest_xray')
    p.add_argument('--models', nargs='+', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                   default=['resnet18', 'resnet50'])
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--workers', type=int, default=0, help='0 is Windows/Jupyter friendly')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='outputs')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--unweighted-loss', action='store_true')
    p.add_argument('--val-source', choices=['provided', 'train-split'], default='provided')
    p.add_argument('--val-fraction', type=float, default=.15)
    p.add_argument('--track-test', action='store_true',
                   help='Optional assignment test curves; never used to select checkpoints.')
    p.add_argument('--no-progress', action='store_true', help='Disable batch progress bars')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or not 0 < args.val_fraction < 1:
        p.error('epochs >= 1, batch-size >= 2, and 0 < val-fraction < 1 required')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        p.error('Output directory is not empty. Use a new --output for each experiment.')
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('lab1')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    for handler in [logging.FileHandler(out/'training.log', encoding='utf-8'), ProgressLogHandler()]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    root = Path(args.data_root)
    train_samples, test_samples = discover(root/'train'), discover(root/'test')
    if args.val_source == 'provided':
        val_samples = discover(root/'val')
    else:
        train_samples, excluded = exclude_test_groups(train_samples, test_samples)
        logger.warning('Filename IDs are unverified patient proxies. Excluded %s train images matching test groups.', len(excluded))
        (out/'excluded_test_overlap.json').write_text(json.dumps(excluded, indent=2), encoding='utf-8')
        train_samples, val_samples = group_split(train_samples, args.val_fraction, args.seed)
    train_ids = {group_id(p) for p, _ in train_samples}
    val_ids = {group_id(p) for p, _ in val_samples}
    test_ids = {group_id(p) for p, _ in test_samples}
    overlaps = {'train_val': sorted(train_ids & val_ids),
                'train_test': sorted(train_ids & test_ids),
                'val_test': sorted(val_ids & test_ids)}
    (out/'group_overlap_audit.json').write_text(json.dumps(overlaps, indent=2), encoding='utf-8')
    if any(overlaps.values()):
        if args.val_source == 'train-split':
            raise ValueError('Filename groups overlap after grouped splitting; see group_overlap_audit.json.')
        logger.warning('Original split contains inferred group overlaps; retained only to reproduce the original baseline. See group_overlap_audit.json.')
    if len(val_samples) < 100:
        logger.warning('Validation has only %s images; metrics may fluctuate.', len(val_samples))
        logger.info('Optional: --val-source train-split (report this split explicitly).')
    counts = np.bincount([s[1] for s in train_samples], minlength=2)
    logger.info('Train class counts: %s', dict(zip(CLASSES, counts.tolist())))
    logger.info('Split sizes train=%s val=%s test=%s', len(train_samples), len(val_samples), len(test_samples))
    logger.info('Validation source: %s; checkpoint selection: max F1, then min loss', args.val_source)
    logger.info('Validation class counts: %s', dict(zip(CLASSES,
                np.bincount([s[1] for s in val_samples], minlength=2).tolist())))
    splits = {'train': train_samples, 'val': val_samples, 'test': test_samples}
    group_manifest = {split: [{'path': path, 'label': label, 'group_id': group_id(path)}
                     for path, label in samples] for split, samples in splits.items()}
    (out/'group_manifest.json').write_text(json.dumps(group_manifest, indent=2), encoding='utf-8')
    (out/'split_manifest.json').write_text(json.dumps(splits, indent=2), encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = vars(args) | {'device': str(device), 'torch': str(torch.__version__),
                           'torchvision': str(torchvision.__version__), 'python': platform.python_version()}
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    logger.info('Device: %s', device)
    histories, summaries = {}, []
    for name in args.models:
        seed_everything(args.seed)
        loaders = {}
        for split, samples in splits.items():
            loaders[split] = DataLoader(
                ChestXrayDataset(samples, make_transform(args.image_size, split == 'train')),
                batch_size=args.batch_size, shuffle=split == 'train', num_workers=args.workers,
                pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(args.seed))
        logger.info('Starting %s; pretrained=%s', name, not args.no_pretrained)
        model = build_model(name, pretrained=not args.no_pretrained).to(device)
        weight = None if args.unweighted_loss else torch.tensor(
            len(train_samples) / (2 * counts), dtype=torch.float32, device=device)
        # Per-sample weighted CE averaged by batch size; loss accounting is consistent.
        base_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
        criterion = lambda outputs, labels: base_loss(outputs, labels).mean()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        model_out = out/name
        model_out.mkdir()
        rows, best_key = [], (-1.0, float('-inf'))
        for epoch in range(1, args.epochs + 1):
            train_result = run_epoch(model, loaders['train'], device, criterion, optimizer,
                                     description=f'{name} {epoch}/{args.epochs} Train', progress=not args.no_progress)
            val_result = run_epoch(model, loaders['val'], device, criterion,
                                   description=f'{name} {epoch}/{args.epochs} Val', progress=not args.no_progress)
            row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
            for split, result in [('train', train_result), ('val', val_result)]:
                row.update({f'{split}_{k}': v for k, v in result.items() if k != 'confusion_matrix'})
            if args.track_test:
                test_result = run_epoch(model, loaders['test'], device, description=f'{name} Test', progress=not args.no_progress)
                row.update({f'test_{k}': v for k, v in test_result.items() if k != 'confusion_matrix'})
            rows.append(row)
            key = (val_result['f1'], -val_result['loss'])
            if key > best_key:
                best_key = key
                torch.save({'model': name, 'state_dict': model.state_dict(), 'epoch': epoch,
                            'image_size': args.image_size, 'classes': CLASSES,
                            'val_metrics': val_result, 'config': config}, model_out/'best.pt')
                logger.info('%s saved best.pt at epoch %s (val F1=%.6f, loss=%.6f)',
                            name, epoch, val_result['f1'], val_result['loss'])
            scheduler.step()
            logger.info('%s epoch %02d/%s | train loss=%.4f acc=%.4f F1=%.4f | '
                        'val loss=%.4f acc=%.4f F1=%.4f | lr=%.6g',
                        name, epoch, args.epochs, train_result['loss'], train_result['accuracy'],
                        train_result['f1'], val_result['loss'], val_result['accuracy'],
                        val_result['f1'], row['lr'])
            with (model_out/'history.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        checkpoint = torch.load(model_out/'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        result = run_epoch(model, loaders['test'], device, description=f'{name} Test', progress=not args.no_progress)
        result.update(model=name, best_epoch=checkpoint['epoch'])
        (model_out/'test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        plot_cm(result['confusion_matrix'], model_out/'test_confusion_matrix.png', f'{name} final test')
        histories[name] = rows
        summaries.append(result)
        logger.info('FINAL TEST: %s', json.dumps(result))
        del model, optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    plot_histories(histories, out/'comparison.png')
    if args.track_test:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for name, rows in histories.items():
            for ax, metric in zip(axes, ['accuracy', 'f1']):
                ax.plot([r['epoch'] for r in rows], [r[f'test_{metric}'] for r in rows], label=name)
                ax.set(xlabel='Epoch', ylabel=metric, title=f'Test {metric}', ylim=(0, 1.01))
                ax.legend()
        fig.tight_layout()
        fig.savefig(out/'test_curves.png', dpi=200)
        plt.close(fig)
    (out/'comparison.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    logger.info('Saved results to %s', out.resolve())


if __name__ == '__main__':
    main()
