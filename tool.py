import os
import torch


def all_metrics(y_true, y_pred):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1.item(), precision.item(), recall.item(), tp.item(), tn.item(), fp.item(), fn.item()


def save_model(model, optimizer, n_epoch, opts):
    torch.save({'state_dict': model.state_dict(), 'n_epoch': n_epoch,
                'optimizer': optimizer.state_dict()},
               os.path.join(opts.save_model_path, f'model_{opts.project}_{n_epoch}.pth'))
