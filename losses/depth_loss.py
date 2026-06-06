import torch


def scale_invariant_log_loss(pred, gt, alpha=10.0, lam=0.85, eps=1e-8):
    log_pred = torch.log(pred + eps)
    log_gt = torch.log(gt + eps)
    diff = log_pred - log_gt
    n = diff.numel()
    term1 = torch.sum(diff ** 2) / n
    term2 = lam * (torch.sum(diff) ** 2) / (n ** 2)
    return alpha * torch.sqrt(term1 - term2)


def normal_loss(pred_depth, gt_depth, patch_size=16):
    """Вычисляет loss на основе разницы поверхностных нормалей"""
    # Вычисляем градиенты глубины
    dy, dx = torch.gradient(gt_depth, dim=(-2, -1))
    # Нормаль: N = [-dx, -dy, 1]
    n_gt = torch.stack([-dx, -dy, torch.ones_like(dx)], dim=-1)
    n_gt = n_gt / (n_gt.norm(dim=-1, keepdim=True) + 1e-8)

    dy, dx = torch.gradient(pred_depth, dim=(-2, -1))
    n_pred = torch.stack([-dx, -dy, torch.ones_like(dx)], dim=-1)
    n_pred = n_pred / (n_pred.norm(dim=-1, keepdim=True) + 1e-8)

    # Cosine similarity loss
    cos_sim = (n_pred * n_gt).sum(dim=-1)
    return 1.0 - cos_sim.mean()