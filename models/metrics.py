import torch
import torch.nn.functional as F


def abs_rel_error(pred, gt):
    return (pred - gt).abs().mean() / gt.mean()


def rmse_error(pred, gt):
    return torch.sqrt(((pred - gt) ** 2).mean())


def silog_error(pred, gt, lam=0.85, eps=1e-8):
    diff = torch.log(pred + eps) - torch.log(gt + eps)
    n = diff.numel()
    term1 = (diff ** 2).sum() / n
    term2 = lam * (diff.sum() ** 2) / (n ** 2)
    return torch.sqrt(term1 - term2).clamp(min=1e-8)


def delta_accuracy(pred, gt, threshold=1.25):
    ratio = torch.max(pred / gt, gt / pred)
    return (ratio < threshold).float().mean()


@torch.no_grad()
def evaluate_depth(pred_depth, gt_depth, gt_mask=None):
    pred = pred_depth.float()
    gt = gt_depth.float()
    if gt_mask is not None:
        pred = pred[gt_mask]
        gt = gt[gt_mask]
    else:
        pred = pred.flatten()
        gt = gt.flatten()
    if pred.numel() == 0:
        return {}
    return {
        'abs_rel': abs_rel_error(pred, gt).item(),
        'rmse': rmse_error(pred, gt).item(),
        'silog': silog_error(pred, gt).item(),
        'd1': delta_accuracy(pred, gt, 1.25).item(),
        'd2': delta_accuracy(pred, gt, 1.25 ** 2).item(),
        'd3': delta_accuracy(pred, gt, 1.25 ** 3).item(),
    }


@torch.no_grad()
def evaluate_model(model, loader, device, max_depth=10.0):
    model.eval()
    all_metrics = []
    for imgs, depths, masks in loader:
        imgs, depths = imgs.to(device), depths.to(device)
        d3, d2, d1 = model(imgs)
        depths = F.interpolate(depths, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        depths = depths.clamp(0, max_depth)
        for i in range(imgs.size(0)):
            pred = d1[i:i+1]
            gt = depths[i:i+1]
            if pred.shape != gt.shape:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            all_metrics.append(evaluate_depth(pred, gt))
    if not all_metrics:
        return {}
    avg = {}
    for key in all_metrics[0]:
        avg[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    return avg
