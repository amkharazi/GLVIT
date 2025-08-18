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
# ALGA (yours)
from Models.glvit_v7 import VisionTransformer as ALGAVisionTransformer
# ORIGINAL (your baseline)
from Models.basic import VisionTransformer as OriginalVisionTransformer

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

    # expose indices
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
    # ALGA has an extra param "second_path_size"
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
    parser = argparse.ArgumentParser(description="Dual-model test: ALGA vs Original (with top-5 buckets)")
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
    parser.add_argument('--eval_epoch', type=int, default=None, help='ONLY evaluate this epoch if provided')
    parser.add_argument('--every_k', type=int, default=5, help='If sweeping, evaluate epochs divisible by k')

    # Output controls
    parser.add_argument('--out_dir', type=str, default='../results_dual_test')
    parser.add_argument('--save_images', action='store_true', help='Also save misclassified images by bucket')

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

    def eval_at_epoch(epoch: int):
        print(f'\n=== Evaluating epoch {epoch} ===')

        ckpt_a = os.path.join(ckpt_dir_alga, f'Model_epoch_{epoch}.pth')
        ckpt_o = os.path.join(ckpt_dir_orig, f'Model_epoch_{epoch}.pth')
        if not os.path.isfile(ckpt_a):
            print(f'[WARN] Missing ALGA checkpoint: {ckpt_a}')
            return
        if not os.path.isfile(ckpt_o):
            print(f'[WARN] Missing Original checkpoint: {ckpt_o}')
            return

        load_weights(alga, ckpt_a, device)
        load_weights(orig, ckpt_o, device)

        criterion = nn.CrossEntropyLoss()

        # Buckets (top-1)
        only_alga_wrong_t1 = []   # ALGA wrong@1 / ORIG correct@1
        only_orig_wrong_t1 = []   # ORIG wrong@1 / ALGA correct@1
        both_wrong_t1      = []   # both wrong@1

        # Buckets (top-5 miss)
        only_alga_miss_t5 = []    # ALGA top5-miss / ORIG top5-hit
        only_orig_miss_t5 = []    # ORIG top5-miss / ALGA top5-hit
        both_miss_t5      = []    # both top5-miss

        # Metrics
        running_loss_a = 0.0
        running_loss_o = 0.0
        correct_a = {1:0.0,2:0.0,3:0.0,4:0.0,5:0.0}
        correct_o = {1:0.0,2:0.0,3:0.0,4:0.0,5:0.0}

        # Optional image saving
        if args.save_images:
            img_root = os.path.join(out_root, f'epoch_{epoch:04d}_imgs')
            for sub in ['t1_only_alga_wrong','t1_only_orig_wrong','t1_both_wrong',
                        't5_only_alga_miss','t5_only_orig_miss','t5_both_miss']:
                os.makedirs(os.path.join(img_root, sub), exist_ok=True)

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

                # top-1 predictions
                pa = torch.argmax(logits_a, dim=1)
                po = torch.argmax(logits_o, dim=1)

                # top-k metrics (for global accuracy summary)
                accs_a = topk_accuracy(logits_a, targets, topk=(1,2,3,4,5))
                accs_o = topk_accuracy(logits_o, targets, topk=(1,2,3,4,5))
                for k in correct_a:
                    correct_a[k] += accs_a[k]['correct']
                    correct_o[k] += accs_o[k]['correct']

                # ---------- top-1 buckets ----------
                a_wrong_t1 = (pa != targets)
                o_wrong_t1 = (po != targets)

                only_a_t1 = a_wrong_t1 & (~o_wrong_t1)
                only_o_t1 = o_wrong_t1 & (~a_wrong_t1)
                both_t1   = a_wrong_t1 & o_wrong_t1

                # ---------- top-5 miss buckets ----------
                # True label NOT in top-5 predictions
                top5_a_vals, top5_a_idx = torch.topk(logits_a, k=5, dim=1)
                top5_o_vals, top5_o_idx = torch.topk(logits_o, k=5, dim=1)
                t_exp = targets.view(-1,1)

                a_in_top5 = (top5_a_idx == t_exp).any(dim=1)
                o_in_top5 = (top5_o_idx == t_exp).any(dim=1)
                a_miss_t5 = ~a_in_top5
                o_miss_t5 = ~o_in_top5

                only_a_t5 = a_miss_t5 & (~o_miss_t5)
                only_o_t5 = o_miss_t5 & (~a_miss_t5)
                both_t5   = a_miss_t5 & o_miss_t5

                # record helper
                def push(mask, dest_list):
                    sel = mask.nonzero(as_tuple=False).squeeze(1).cpu().numpy().tolist()
                    for i in sel:
                        dest_list.append((
                            int(idxs[i]),
                            int(targets[i].cpu()),
                            int(pa[i].cpu()),
                            int(po[i].cpu())
                        ))

                push(only_a_t1, only_alga_wrong_t1)
                push(only_o_t1, only_orig_wrong_t1)
                push(both_t1,   both_wrong_t1)

                push(only_a_t5, only_alga_miss_t5)
                push(only_o_t5, only_orig_miss_t5)
                push(both_t5,   both_miss_t5)

                # optional save images
                if args.save_images:
                    imgs_uint = tensor_to_uint(inputs).cpu()
                    for i in range(inputs.size(0)):
                        di = int(idxs[i])
                        fn = f'idx_{di}_t{int(targets[i])}_a{int(pa[i])}_o{int(po[i])}.png'
                        if only_a_t1[i]:
                            save_image(imgs_uint[i], os.path.join(img_root,'t1_only_alga_wrong', fn))
                        elif only_o_t1[i]:
                            save_image(imgs_uint[i], os.path.join(img_root,'t1_only_orig_wrong', fn))
                        elif both_t1[i]:
                            save_image(imgs_uint[i], os.path.join(img_root,'t1_both_wrong', fn))
                        if only_a_t5[i]:
                            save_image(imgs_uint[i], os.path.join(img_root,'t5_only_alga_miss', fn))
                        elif only_o_t5[i]:
                            save_image(imgs_uint[i], os.path.join(img_root,'t5_only_orig_miss', fn))
                        elif both_t5[i]:
                            save_image(imgs_uint[i], os.path.join(img_root,'t5_both_miss', fn))

        # summarize & save
        N = len(test_loader.dataset)
        avg_loss_a = running_loss_a / N
        avg_loss_o = running_loss_o / N
        topk_a = [correct_a[k]/N for k in sorted(correct_a.keys())]
        topk_o = [correct_o[k]/N for k in sorted(correct_o.keys())]
        elapsed = time.time() - start

        # CSVs
        csv_t1 = os.path.join(out_root, f'epoch_{epoch:04d}_misclassified_top1.csv')
        with open(csv_t1, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dataset_index','true_label','alga_pred','orig_pred','bucket'])
            for row in only_alga_wrong_t1: w.writerow([*row, 'ALGA_wrong_ONLY'])
            for row in only_orig_wrong_t1: w.writerow([*row, 'ORIG_wrong_ONLY'])
            for row in both_wrong_t1:      w.writerow([*row, 'BOTH_wrong'])

        csv_t5 = os.path.join(out_root, f'epoch_{epoch:04d}_misclassified_top5.csv')
        with open(csv_t5, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dataset_index','true_label','alga_pred','orig_pred','bucket'])
            for row in only_alga_miss_t5: w.writerow([*row, 'ALGA_top5MISS_ONLY'])
            for row in only_orig_miss_t5: w.writerow([*row, 'ORIG_top5MISS_ONLY'])
            for row in both_miss_t5:      w.writerow([*row, 'BOTH_top5MISS'])

        # summary line
        rep = (
            f"Epoch {epoch} | "
            f"ALGA: top1={topk_a[0]:.4f} top5={topk_a[4]:.4f} loss={avg_loss_a:.6f} | "
            f"ORIG: top1={topk_o[0]:.4f} top5={topk_o[4]:.4f} loss={avg_loss_o:.6f} | "
            f"time={elapsed:.2f}s | "
            f"Top1 buckets: ALGA_only={len(only_alga_wrong_t1)}, ORIG_only={len(only_orig_wrong_t1)}, BOTH={len(both_wrong_t1)} | "
            f"Top5-miss buckets: ALGA_only={len(only_alga_miss_t5)}, ORIG_only={len(only_orig_miss_t5)}, BOTH={len(both_miss_t5)}"
        )
        print(rep)
        with open(os.path.join(out_root, 'summary.txt'), 'a') as f:
            f.write(rep + '\n')

    # Epoch selection
    if args.eval_epoch is not None:
        eval_at_epoch(args.eval_epoch)
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
            eval_at_epoch(e)

if __name__ == '__main__':
    main()
