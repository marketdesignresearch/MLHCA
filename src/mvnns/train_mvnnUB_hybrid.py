import numpy as np
import torch
from mvnns.losses import NOMU_loss_hybrid
from mvnns.metrics import compute_metrics, compute_metrics_UB


def train_mvnnUB_VQ_helper(mean_model, ub_model, device, train_loader_vqs, num_train_data, optimizer, clip_grad_norm, pi_sqr,
          pi_exp, pi_above_mean, c_exp, n_aug, target_max, loss_func, exp_upper_bound_net,
          q):
    """
    Takes as input a trained mean model, the demand query and value query datasets, 
    and trains the upper bound model.
    """

    # freeze the mean model
    mean_model.eval()
    # put the upper bound model in training mode
    ub_model.train()


    preds, preds_UB, targets = [], [], []
    total_loss = 0
    loss_b_total = 0
    loss_c_total = 0

    for batch_idx, (data, target) in enumerate(train_loader_vqs):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        mean_output = mean_model(data)
        ub_output = ub_model(data)

        preds.extend(mean_output.detach().cpu().numpy().flatten().tolist())
        preds_UB.extend(ub_output.detach().cpu().numpy().flatten().tolist())
        targets.extend(target.detach().cpu().numpy().flatten().tolist())

        nbatch = len(preds)

        # Calculate NOMU loss terms that correspond to uncertainty for a single batch: loss_b, loss_c
        loss_b, loss_c = NOMU_loss_hybrid(mean_output=mean_output,
                                           ub_output=ub_output,
                                           target=target,
                                           loss_func=loss_func,
                                           pi_sqr=pi_sqr,
                                           pi_exp=pi_exp,
                                           pi_above_mean=pi_above_mean,
                                           c_exp=c_exp,
                                           n_aug=n_aug,
                                           din=data.shape[1],
                                           mean_model=mean_model,
                                           ub_model=ub_model,
                                           exp_upper_bound_net=exp_upper_bound_net,
                                           ntrain=num_train_data
                                           )

        ######
        loss = loss_b + loss_c
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ub_model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += float(loss_b + loss_c) * nbatch
        loss_b_total += float(loss_b) * nbatch
        loss_c_total += float(loss_c) * nbatch

    # UPDATE METRICS
    metrics = {'loss': total_loss / num_train_data,
               'loss_b': loss_b_total / num_train_data,
               'loss_c': loss_c_total / num_train_data}

    # Scaled metrics
    metrics.update(compute_metrics_UB(preds_UB, targets, q=q, scaled=True))

    preds_UB, targets = (np.array(preds_UB) * target_max).tolist(), (np.array(targets) * target_max).tolist()

    # Unscaled metrics (original scale)
    # metrics.update(compute_metrics(preds, targets, q=q))
    metrics.update(compute_metrics_UB(preds_UB, targets, q=q))
    return metrics
