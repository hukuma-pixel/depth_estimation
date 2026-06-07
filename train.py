import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.nyuv2 import NYUv2Dataset, create_dummy_loader
from losses.depth_loss import scale_invariant_log_loss, normal_loss
from models.depth_net import MonocularDepthNet
from models.metrics import evaluate_model


def train_epoch(model, loader, optimizer, device, accumulation_steps=1,
                clip_max_norm=None, writer=None, global_step=0):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (imgs, depths, _) in enumerate(loader):
        imgs, depths = imgs.to(device), depths.to(device)

        d3, d2, d1 = model(imgs)

        gt1 = F.interpolate(depths, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        gt2 = F.interpolate(depths, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        gt3 = F.interpolate(depths, size=d3.shape[-2:], mode='bilinear', align_corners=False)

        l_depth = scale_invariant_log_loss(d1, gt1, alpha=10.0) + \
                  0.5 * scale_invariant_log_loss(d2, gt2, alpha=5.0) + \
                  0.25 * scale_invariant_log_loss(d3, gt3, alpha=2.5)
        l_norm = normal_loss(d1.squeeze(1), gt1.squeeze(1))
        loss = l_depth + l_norm

        loss = loss / accumulation_steps
        loss.backward()

        total_loss += loss.item() * accumulation_steps

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            if clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if writer:
                writer.add_scalar('train/loss', loss.item() * accumulation_steps, global_step)
                writer.add_scalar('train/loss_depth', (l_depth / 10.0).item(), global_step)
                writer.add_scalar('train/loss_norm', l_norm.item(), global_step)

    return total_loss / len(loader), global_step


def build_optimizer(model, args):
    if args.freeze_encoder:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)

    encoder_params = []
    decoder_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'encoder.' in name:
            encoder_params.append(p)
        else:
            decoder_params.append(p)

    print(f"Optimizer groups — decoder: {len(decoder_params)}, encoder: {len(encoder_params)} "
          f"(lr: {args.lr} / {args.lr * args.encoder_lr_factor:.2e})")
    return torch.optim.Adam([
        {'params': decoder_params, 'lr': args.lr},
        {'params': encoder_params, 'lr': args.lr * args.encoder_lr_factor},
    ], weight_decay=1e-4)


def build_scheduler(optimizer, warmup_epochs, step_size, gamma):
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, total_iters=warmup_epochs)
        main = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, main], milestones=[warmup_epochs])
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=None,
                        help='Path to NYUv2 dataset')
    parser.add_argument('--backbone', type=str, default='swin_tiny_patch4_window7_224')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True,
                        help='Use ImageNet-pretrained Swin encoder')
    parser.add_argument('--freeze-encoder', action=argparse.BooleanOptionalAction, default=True,
                        help='Freeze Swin encoder (train decoder only)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--accumulation-steps', type=int, default=3,
                        help='Gradient accumulation steps (effective batch = batch_size × accumulation_steps)')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--encoder-lr-factor', type=float, default=0.05,
                        help='LR multiplier for encoder when not frozen')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Linear LR warmup epochs')
    parser.add_argument('--clip-max-norm', type=float, default=5.0,
                        help='Gradient clipping max norm (0 = disable)')
    parser.add_argument('--input-size', type=int, default=192)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--num-threads', type=int, default=8)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--log-dir', type=str, default='runs')
    parser.add_argument('--grad-checkpoint', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--run-name', type=str, default=None,
                        help='Tag for checkpoint/log dirs (auto: frozen|unfrozen)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint.pth to resume training from')
    args = parser.parse_args()

    if args.run_name is None:
        backbone_tag = args.backbone.split('_')[1]  # tiny / base / large
        freeze_tag = 'frozen' if args.freeze_encoder else 'unfrozen'
        args.run_name = f'{backbone_tag}_{freeze_tag}'

    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == 'cuda'

    effective_batch = args.batch_size * args.accumulation_steps
    print(f"Device: {device} | Backbone: {args.backbone} | Run: {args.run_name} | CPU threads: {args.num_threads}")
    print(f"Params: epochs={args.epochs}, batch={args.batch_size} "
          f"(eff. {effective_batch}), lr={args.lr}, input={args.input_size}")

    if args.data_root:
        train_dataset = NYUv2Dataset(args.data_root, 'train',
                                      input_size=(args.input_size, args.input_size))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
        val_loader = DataLoader(
            NYUv2Dataset(args.data_root, 'val',
                          input_size=(args.input_size, args.input_size)),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin_memory,
        )
        print(f"Train: {len(train_dataset)} | Val: {len(val_loader.dataset)}")
    else:
        train_loader = create_dummy_loader(args.batch_size, (args.input_size, args.input_size))
        val_loader = create_dummy_loader(args.batch_size, (args.input_size, args.input_size))
        print("Using dummy data — pass --data-root for real NYUv2")

    model = MonocularDepthNet(swin_version=args.backbone, pretrained=args.pretrained).to(device)

    if args.freeze_encoder:
        n_frozen = 0
        for p in model.encoder.parameters():
            p.requires_grad = False
            n_frozen += 1
        print(f"Encoder frozen ({n_frozen} param groups). Training decoder only.")
    else:
        print("Training full model (encoder + decoder).")

    if args.grad_checkpoint:
        model.enable_grad_checkpointing()
        print("Gradient checkpointing enabled")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_all = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {total_trainable:,} / {total_all:,} ({total_trainable/total_all*100:.1f}%)")

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, step_size=5, gamma=0.5)

    start_epoch = 1
    best_d1 = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_d1 = ckpt.get('best_d1', 0.0)
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']}, d1={best_d1:.3f})")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_path = Path(args.log_dir) / args.run_name
        log_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_path))
        print(f"TensorBoard: {log_path}")
    except ImportError:
        pass

    ckpt_dir = Path(args.checkpoint_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    main_sched_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, device,
            accumulation_steps=args.accumulation_steps,
            clip_max_norm=args.clip_max_norm if args.clip_max_norm > 0 else None,
            writer=writer, global_step=global_step,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        log = f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} | LR: {current_lr:.2e}"

        do_val = args.val_every > 0 and epoch % args.val_every == 0
        if do_val or epoch == args.epochs:
            metrics = evaluate_model(model, val_loader, device, max_depth=10.0)
            log += f" | AbsRel: {metrics['abs_rel']:.4f} | RMSE: {metrics['rmse']:.4f}"
            log += f" | d1: {metrics['d1']:.3f}"
            if writer:
                writer.add_scalar('val/abs_rel', metrics['abs_rel'], epoch)
                writer.add_scalar('val/rmse', metrics['rmse'], epoch)
                writer.add_scalar('val/d1', metrics['d1'], epoch)

            is_best = metrics['d1'] > best_d1
            if is_best:
                best_d1 = metrics['d1']
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_d1': best_d1,
                    'args': vars(args),
                }
                ckpt_path = ckpt_dir / 'best_model.pth'
                torch.save(ckpt, ckpt_path)
                log += f' | saved={ckpt_path.name}'
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_d1': best_d1,
                    'args': vars(args),
                }, ckpt_dir / 'last_model.pth')

        if writer:
            writer.add_scalar('lr', current_lr, epoch)

        print(log)

    ckpt = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_d1': best_d1,
        'args': vars(args),
    }
    torch.save(ckpt, ckpt_dir / 'final_model.pth')
    print(f"Final model saved to {ckpt_dir / 'final_model.pth'}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
