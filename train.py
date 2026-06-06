import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from models.depth_net import MonocularDepthNet
from losses.depth_loss import scale_invariant_log_loss, normal_loss
from models.metrics import evaluate_model


def train_epoch(model, loader, optimizer, device, writer=None, global_step=0):
    model.train()
    total_loss = 0
    for imgs, depths, _ in loader:
        imgs, depths = imgs.to(device), depths.to(device)
        optimizer.zero_grad()

        d3, d2, d1 = model(imgs)

        gt1 = F.interpolate(depths, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        gt2 = F.interpolate(depths, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        gt3 = F.interpolate(depths, size=d3.shape[-2:], mode='bilinear', align_corners=False)

        l_depth = scale_invariant_log_loss(d1, gt1, alpha=10.0) + \
                  0.5 * scale_invariant_log_loss(d2, gt2, alpha=5.0) + \
                  0.25 * scale_invariant_log_loss(d3, gt3, alpha=2.5)
        l_norm = normal_loss(d1.squeeze(1), gt1.squeeze(1))
        loss = l_depth + l_norm

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        global_step += 1

        if writer:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/loss_depth', (l_depth / 10.0).item(), global_step)
            writer.add_scalar('train/loss_norm', l_norm.item(), global_step)

    return total_loss / len(loader), global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=None,
                        help='Path to NYUv2 dataset')
    parser.add_argument('--backbone', type=str, default='swin_tiny_patch4_window7_224')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet-pretrained Swin encoder')
    parser.add_argument('--freeze-encoder', action='store_true', default=True,
                        help='Freeze Swin encoder (train decoder only)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input-size', type=int, default=192)
    parser.add_argument('--num-threads', type=int, default=4,
                        help='CPU threads for PyTorch')
    parser.add_argument('--val-every', type=int, default=0,
                        help='Validate every N epochs (0 = no validation)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--log-dir', type=str, default='runs')
    parser.add_argument('--grad-checkpoint', action='store_true', default=False)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Backbone: {args.backbone} | CPU threads: {args.num_threads}")
    print(f"Params: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, input={args.input_size}")

    if args.data_root:
        from data.nyuv2 import NYUv2Dataset
        train_dataset = NYUv2Dataset(args.data_root, 'train',
                                      input_size=(args.input_size, args.input_size))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False, drop_last=True)
        val_loader = None
        if args.val_every > 0:
            val_dataset = NYUv2Dataset(args.data_root, 'val',
                                        input_size=(args.input_size, args.input_size))
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=0, pin_memory=False)
        print(f"Train: {len(train_dataset)} | Val: {len(val_loader.dataset) if val_loader else 'N/A'}")
    else:
        from data.nyuv2 import create_dummy_loader
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

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_path = Path(args.log_dir) / args.backbone.split('/')[-1]
        log_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_path))
        print(f"TensorBoard: {log_path}")
    except ImportError:
        pass

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    global_step = 0
    best_d1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_epoch(model, train_loader, optimizer, device, writer, global_step)
        scheduler.step()

        log = f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f}"

        do_val = val_loader and (args.val_every > 0 and epoch % args.val_every == 0)
        if do_val or epoch == args.epochs:
            metrics = evaluate_model(model, val_loader, device, max_depth=10.0)
            log += f" | AbsRel: {metrics['abs_rel']:.4f} | RMSE: {metrics['rmse']:.4f}"
            log += f" | d1: {metrics['d1']:.3f}"
            if writer:
                writer.add_scalar('val/abs_rel', metrics['abs_rel'], epoch)
                writer.add_scalar('val/rmse', metrics['rmse'], epoch)
                writer.add_scalar('val/d1', metrics['d1'], epoch)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            if metrics['d1'] > best_d1:
                best_d1 = metrics['d1']
                ckpt_path = ckpt_dir / 'best_model.pth'
                torch.save(model.state_dict(), ckpt_path)
                log += f' | saved={ckpt_path.name}'

        print(log)

    ckpt_path = ckpt_dir / 'checkpoint_final.pth'
    torch.save(model.state_dict(), ckpt_path)
    print(f"Final model saved to {ckpt_path}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
