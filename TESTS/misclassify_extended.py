# Add all .py files to path
import sys, os, time, argparse, random, csv
sys.path.append('..')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment, RandomErasing
from torchvision.utils import save_image

# ====== MODELS ======
from Models.glvit_v7 import VisionTransformer as ALGAVisionTransformer   # "my" model
from Models.basic import VisionTransformer as OriginalVisionTransformer  # "other" model

# ====== LOADERS ======
from Utils.cifar10_loaders import get_cifar10_dataloaders
from Utils.cifar100_loaders import get_cifar100_dataloaders
from Utils.mnist_loaders import get_mnist_dataloaders
from Utils.tinyimagenet_loaders import get_tinyimagenet_dataloaders
from Utils.fashionmnist_loaders import get_fashionmnist_dataloaders
from Utils.flowers102_loaders import get_flowers102_dataloaders
from Utils.oxford_pets_loaders import get_oxford_pets_dataloaders
from Utils.food101_loaders import get_food101_dataloaders
from Utils.stl10_classification_loaders import get_stl10_classification_dataloaders

from Utils.accuracy_measures import topk_accuracy


# ------------------ utils ------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WithIndex(Dataset):
    """Wrap any dataset to return (img, target, idx)."""
    def __init__(self, base_ds):
        self.base = base_ds
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        out = self.base[idx]
        img, target = out[0], out[1]
        return img, target, idx

def build_transforms(dataset, image_size):
    if dataset in ('cifar10','cifar100','tinyimagenet','flowers102','oxford_pets','stl10','food101'):
        t_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
            RandomErasing(p=0.25),
        ])
        t_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
    elif dataset in ('mnist','fashionmnist'):
        t_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
            RandomErasing(p=0.25),
        ])
        t_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return t_train, t_test

def get_test_loader(dataset, data_root, image_size, batch_size, train_size):
    t_train, t_test = build_transforms(dataset, image_size)

    if dataset == 'cifar10':
        _, test_loader = get_cifar10_dataloaders(data_root, t_train, t_test, batch_size, image_size, train_size, repeat_count=5)
    elif dataset == 'cifar100':
        _, test_loader = get_cifar100_dataloaders(data_root, t_train, t_test, batch_size, image_size, train_size, repeat_count=5)
    elif dataset == 'mnist':
        _, test_loader = get_mnist_dataloaders(data_root, t_train, t_test, batch_size, image_size, train_size, repeat_count=5)
    elif dataset == 'tinyimagenet':
        _, test_loader, _ = get_tinyimagenet_dataloaders(data_root, t_train, t_test, t_test, batch_size, image_size, repeat_count=5)
    elif dataset == 'fashionmnist':
        _, test_loader = get_fashionmnist_dataloaders(data_root, t_train, t_test, batch_size, image_size, train_size, repeat_count=5)
    elif dataset == 'flowers102':
        _, test_loader = get_flowers102_dataloaders(data_root, t_train, t_test, batch_size, image_size, train_size, repeat_count=5)
    elif dataset == 'oxford_pets':
        _, test_loader = get_oxford_pets_dataloaders(data_root, t_train, t_test, batch_size, image_size, train_size, repeat_count=5)
    elif dataset == 'stl10':
        _, test_loader = get_stl10_classification_dataloaders(data_root, t_train, t_test, batch_size, image_size, train_size, repeat_count=5)
    elif dataset == 'food101':
        _, test_loader = get_food101_dataloaders(data_root, t_train, t_test, batch_size, image_size, train_size, repeat_count=5)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    ds_idx = WithIndex(test_loader.dataset)
    test_loader_idx = DataLoader(
        ds_idx,
        batch_size=batch_size,
        shuffle=False,
        num_workers=getattr(test_loader, 'num_workers', 2),
        pin_memory=getattr(test_loader, 'pin_memory', True),
        drop_last=False
    )
    return test_loader_idx

def init_model(klass, num_classes, image_size, patch_size, dim, depth, heads, mlp_dim, second_patch_size, device, print_w=False, is_original=False):
    kwargs = dict(
        img_size=image_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=0.1
    )
    if not is_original:
        kwargs['second_path_size'] = second_patch_size
        kwargs['print_w'] = print_w
    m = klass(**kwargs).to(device)
    return m

def load_weights(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

def tensor_to_uint(img_batch):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_batch.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_batch.device).view(1,3,1,1)
    x = img_batch * std + mean
    return x.clamp(0,1)


# ------------------ main test ------------------

def main():
    parser = argparse.ArgumentParser(description="Dual-model test: ALGA vs Original (top-1 misclassification buckets per class)")
    # Data / model config
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_root', type=str, default='../datasets')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=9)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--mlp_dim', type=int, default=384)
    parser.add_argument('--second_patch_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_size', type=str, default='default')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--print_w', type=bool, default=False)

    # Results / checkpoints
    parser.add_argument('--TEST_ID_ALGA', type=str, required=True, help='../results/<id>/model_stats for ALGA')
    parser.add_argument('--TEST_ID_ORIG', type=str, required=True, help='../results/<id>/model_stats for Original')
    parser.add_argument('--eval_epoch_alga', type=int, default=None, help='ONLY evaluate this epoch if provided')
    parser.add_argument('--eval_epoch_orig', type=int, default=None, help='ONLY evaluate this epoch if provided')
    parser.add_argument('--every_k', type=int, default=5, help='If sweeping, evaluate epochs divisible by k')

    # Output controls
    parser.add_argument('--out_dir', type=str, default='../results_dual_test')
    parser.add_argument('--save_images', action='store_true', help='Save misclassified images by bucket and class')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if args.seed is not None:
        set_seed(args.seed)

    test_loader = get_test_loader(args.dataset, args.data_root, args.image_size, args.batch_size, args.train_size)

    # Instantiate models
    alga = init_model(ALGAVisionTransformer, args.num_classes, args.image_size, args.patch_size,
                      args.dim, args.depth, args.heads, args.mlp_dim, args.second_patch_size, device, args.print_w, is_original=False)
    orig = init_model(OriginalVisionTransformer, args.num_classes, args.image_size, args.patch_size,
                      args.dim, args.depth, args.heads, args.mlp_dim, args.second_patch_size, device, args.print_w, is_original=True)

    # Checkpoint dirs
    ckpt_dir_alga = os.path.join('../results', args.TEST_ID_ALGA, 'model_stats')
    ckpt_dir_orig = os.path.join('../results', args.TEST_ID_ORIG, 'model_stats')

    # Output dir
    out_root = os.path.join(args.out_dir, f'{args.dataset}__{args.TEST_ID_ALGA}__vs__{args.TEST_ID_ORIG}')
    os.makedirs(out_root, exist_ok=True)

    # -------- NEW: helpers for per-class folders and counting --------
    BUCKETS = [
        'MY_wrong_ANY',
        'OTHER_wrong_ANY',
        'MY_wrong_OTHER_right',
        'OTHER_wrong_MY_right',
        'BOTH_wrong',
    ]

    def ensure_bucket_class_dir(base_dir, bucket, class_idx):
        d = os.path.join(base_dir, bucket, str(class_idx))
        os.makedirs(d, exist_ok=True)
        return d

    def init_counts(num_classes):
        # counts[bucket][class_idx] -> int
        return {b: [0 for _ in range(num_classes)] for b in BUCKETS}

    def save_counts_csv(counts, out_csv_path):
        num_classes = len(next(iter(counts.values())))
        with open(out_csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            header = ['class_idx'] + BUCKETS + ['TOTAL_ALL_BUCKETS']
            w.writerow(header)
            for c in range(num_classes):
                row_vals = [counts[b][c] for b in BUCKETS]
                w.writerow([c, *row_vals, sum(row_vals)])
            # add a totals row
            totals = [sum(counts[b]) for b in BUCKETS]
            w.writerow(['__TOTAL__', *totals, sum(totals)])

    # ---------------------------------------------------------------

    def eval_at_epoch(epoch1: int, epoch2: int):
        print(f'\n=== Evaluating epoch {epoch1} and {epoch2} ===')

        ckpt_a = os.path.join(ckpt_dir_alga, f'Model_epoch_{epoch1}.pth')
        ckpt_o = os.path.join(ckpt_dir_orig, f'Model_epoch_{epoch2}.pth')
        if not os.path.isfile(ckpt_a):
            print(f'[WARN] Missing ALGA checkpoint: {ckpt_a}')
            return
        if not os.path.isfile(ckpt_o):
            print(f'[WARN] Missing Original checkpoint: {ckpt_o}')
            return

        load_weights(alga, ckpt_a, device)
        load_weights(orig, ckpt_o, device)

        criterion = nn.CrossEntropyLoss()

        # Metrics
        running_loss_a = 0.0
        running_loss_o = 0.0
        correct_a = {1:0.0,2:0.0,3:0.0,4:0.0,5:0.0}
        correct_o = {1:0.0,2:0.0,3:0.0,4:0.0,5:0.0}

        # NEW: per-class counts and per-sample rows
        per_class_counts = init_counts(args.num_classes)
        rows_per_sample = []  # [idx, true, alga_pred, orig_pred, bucket]

        # Optional image saving
        img_root = os.path.join(out_root, f'epoch_{epoch1:04d}_imgs')
        if args.save_images:
            for b in BUCKETS:
                os.makedirs(os.path.join(img_root, b), exist_ok=True)

        start = time.time()
        alga.eval(); orig.eval()
        with torch.no_grad():
            for _, (inputs, targets, idxs) in enumerate(test_loader):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits_a = alga(inputs)
                logits_o = orig(inputs)

                loss_a = criterion(logits_a, targets)
                loss_o = criterion(logits_o, targets)
                running_loss_a += loss_a.item() * inputs.size(0)
                running_loss_o += loss_o.item() * inputs.size(0)

                # top-k metrics (global summary)
                accs_a = topk_accuracy(logits_a, targets, topk=(1,2,3,4,5))
                accs_o = topk_accuracy(logits_o, targets, topk=(1,2,3,4,5))
                for k in correct_a:
                    correct_a[k] += accs_a[k]['correct']
                    correct_o[k] += accs_o[k]['correct']

                # top-1 predictions and mistakes
                pa = torch.argmax(logits_a, dim=1)
                po = torch.argmax(logits_o, dim=1)
                a_wrong_t1 = (pa != targets)
                o_wrong_t1 = (po != targets)

                # Buckets (top-1)
                my_wrong_any            = a_wrong_t1
                other_wrong_any         = o_wrong_t1
                my_wrong_other_right    = a_wrong_t1 & (~o_wrong_t1)
                other_wrong_my_right    = o_wrong_t1 & (~a_wrong_t1)
                both_wrong              = a_wrong_t1 & o_wrong_t1

                # helper to record & maybe save image
                def record_bucket(mask, bucket_name):
                    if not mask.any():
                        return
                    sel = mask.nonzero(as_tuple=False).squeeze(1)
                    for i in sel.tolist():
                        di = int(idxs[i])
                        t  = int(targets[i].cpu())
                        ap = int(pa[i].cpu())
                        op = int(po[i].cpu())
                        rows_per_sample.append([di, t, ap, op, bucket_name])
                        # per-class count
                        per_class_counts[bucket_name][t] += 1
                        # optional image save
                        if args.save_images:
                            imgs_uint = tensor_to_uint(inputs).cpu()
                            fn = f'idx_{di}_t{t}_a{ap}_o{op}.png'
                            dclass = ensure_bucket_class_dir(img_root, bucket_name, t)
                            save_image(imgs_uint[i], os.path.join(dclass, fn))

                # record all 5 buckets
                record_bucket(my_wrong_any,         'MY_wrong_ANY')
                record_bucket(other_wrong_any,      'OTHER_wrong_ANY')
                record_bucket(my_wrong_other_right, 'MY_wrong_OTHER_right')
                record_bucket(other_wrong_my_right, 'OTHER_wrong_MY_right')
                record_bucket(both_wrong,           'BOTH_wrong')

        # summarize & save
        N = len(test_loader.dataset)
        avg_loss_a = running_loss_a / N
        avg_loss_o = running_loss_o / N
        topk_a = [correct_a[k]/N for k in sorted(correct_a.keys())]
        topk_o = [correct_o[k]/N for k in sorted(correct_o.keys())]
        elapsed = time.time() - start

        # CSV with every misclassified sample & bucket
        csv_buckets = os.path.join(out_root, f'epoch_{epoch1:04d}_buckets_top1.csv')
        with open(csv_buckets, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dataset_index','true_label','alga_pred','orig_pred','bucket'])
            for row in rows_per_sample:
                w.writerow(row)

        # per-class counts file
        csv_counts = os.path.join(out_root, f'epoch_{epoch1:04d}_counts_per_class.csv')
        save_counts_csv(per_class_counts, csv_counts)

        # brief summary line
        rep = (
            f"Epoch {epoch1} | "
            f"ALGA: top1={topk_a[0]:.4f} top5={topk_a[4]:.4f} loss={avg_loss_a:.6f} | "
            f"ORIG: top1={topk_o[0]:.4f} top5={topk_o[4]:.4f} loss={avg_loss_o:.6f} | "
            f"time={elapsed:.2f}s | "
            f"Saved per-class buckets (top-1) to {os.path.basename(out_root)}"
        )
        print(rep)
        with open(os.path.join(out_root, 'summary.txt'), 'a') as f:
            f.write(rep + '\n')

    # Epoch selection
    if args.eval_epoch_alga is not None and args.eval_epoch_orig is not None:
        eval_at_epoch(args.eval_epoch_alga, args.eval_epoch_orig)
    else:
        def available_epochs(ckpt_dir):
            eps = []
            for name in os.listdir(ckpt_dir):
                if name.startswith('Model_epoch_') and name.endswith('.pth'):
                    try:
                        eps.append(int(name.replace('Model_epoch_','').replace('.pth','')))
                    except: pass
            return sorted(set(eps))

        eps_a = available_epochs(ckpt_dir_alga)
        eps_o = available_epochs(ckpt_dir_orig)
        common = [e for e in eps_a if e in eps_o and (e % args.every_k == 0)]
        if not common:
            print('[WARN] No common checkpoints found that match --every_k.')
        for e in common:
            eval_at_epoch(e,e)

if __name__ == '__main__':
    main()
