import logging
import time
from pathlib import Path
from typing import Dict

import torch
import wandb
from torch.nn import Module
from torch.nn.utils import clip_grads_with_norm_, get_total_norm
from sklearn.metrics import roc_auc_score, mean_absolute_error
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from config import CustomConfig

def evaluate(model, steps, loaders, eval_loaders_iters, config):
    metrics = {"val": {}, "test": {}}
    model.eval()
    with torch.inference_mode():
        for split, eval_loader_iter in eval_loaders_iters.items():
            preds = []
            labels = []
            losses = []
            eval_load_times = []
            eval_loader = loaders[split]
            pbar = tqdm(
                total=(min(config.max_eval_steps, len(eval_loader)) if config.max_eval_steps > -1 else len(eval_loader)),
                desc=f"{config.data_name}/{config.task_name}/{split}",
                disable=False,
            )

            batch_idx = 0
            while True:
                tic = time.time()
                try:
                    batch = next(eval_loader_iter)
                    batch_idx += 1
                except StopIteration:
                    break
                toc = time.time()
                pbar.update(1)

                eval_load_time = toc - tic
                eval_load_times.append(eval_load_time)

                true_batch_size = batch.pop("true_batch_size")
                for k in batch:
                    batch[k] = batch[k].to(config.device, non_blocking=True)

                batch["masks"][true_batch_size:, :] = False
                batch["is_targets"][true_batch_size:, :] = False
                batch["is_padding"][true_batch_size:, :] = True

                loss, yhat_dict = model(batch)

                if config.task_type == "clf":
                    yhat = yhat_dict["boolean"][batch["is_targets"]]
                    y = batch["boolean_values"][batch["is_targets"]].flatten()
                elif config.task_type == "reg":
                    yhat = yhat_dict["number"][batch["is_targets"]]
                    y = batch["number_values"][batch["is_targets"]].flatten()

                assert yhat.size(0) == true_batch_size
                assert y.size(0) == true_batch_size

                pred = yhat.flatten()

                losses.append(loss.item())
                preds.append(pred)
                labels.append(y)

                if -1 < config.max_eval_steps <= batch_idx:
                    break

            eval_loaders_iters[split] = iter(eval_loader)

            pbar.close()
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)

            preds = [preds]
            labels = [labels]

            loss = sum(losses) / len(losses)
            k = f"loss/{config.data_name}/{config.task_name}/{split}"
            avg_eval_load_time = sum(eval_load_times) / len(eval_load_times)
            wandb.log({k: loss,f"avg_eval_load_time/{config.data_name}/{config.task_name}": avg_eval_load_time,}, step=steps)

            preds = torch.cat(preds, dim=0).float().cpu().numpy()
            labels = torch.cat(labels, dim=0).float().cpu().numpy()

            if config.task_type == "reg":
                metric_name = "mae"
                metric = mean_absolute_error(labels, preds)
            elif config.task_type == "clf":
                metric_name = "auc"
                labels = [int(x > 0) for x in labels]
                metric = roc_auc_score(labels, preds)

            k = f"{metric_name}/{config.data_name}/{config.task_name}/{split}"
            wandb.log({k: metric}, step=steps)
            print(f"\nstep={steps}, \t{k}: {metric}")
            metrics[split][(config.data_name, config.task_name)] = metric

    return metrics


def checkpoint(model: Module, steps, config: CustomConfig, best=False):
    save_ckpt_dir_ = Path(config.checkpoint_dir).expanduser()
    save_ckpt_dir_.mkdir(parents=True, exist_ok=True)
    if best:
        save_ckpt_path = f"{save_ckpt_dir_}/{config.data_name}_{config.task_name}_best.pt"
    else:
        save_ckpt_path = f"{save_ckpt_dir_}/{steps=}.pt"

    state_dict = model.state_dict()
    torch.save(state_dict, save_ckpt_path)
    print(f"saved checkpoint to {save_ckpt_path}")


def train(model: Module, loaders: Dict, optimizer: Optimizer, lrs: LRScheduler, config: CustomConfig):
    eval_loader_iters = {}
    for k, eval_loader in loaders.items():
        if k == 'train':
            pass
        eval_loader_iters[k] = iter(eval_loader)

    steps = 0
    wandb.log({"epochs": 0}, step=steps)

    pbar = tqdm(total=config.max_steps, desc="steps", disable=False)

    best_val_metrics = dict()
    best_test_metrics = dict()

    while steps < config.max_steps:
        loaders['train'].dataset.sampler.shuffle_py(int(steps / len(loaders['train'])))
        loader_iter = iter(loaders['train'])

        while steps < config.max_steps:
            if (config.evaluation_freq is not None and steps % config.evaluation_freq == 0) or (
                    config.eval_pow2 and steps & (steps - 1) == 0
            ):
                metrics = evaluate(model, steps, loaders, eval_loader_iters, config)
                if config.save_artifacts:
                    for (db_name, table_name), metric in metrics["val"].items():
                        best_metric = best_val_metrics.get((db_name, table_name), -float("inf"))
                        if config.higher_is_better and metric > best_metric:
                            best_val_metrics[(db_name, table_name)] = metric
                            best_test_metrics[(db_name, table_name)] = metrics["test"][(db_name, table_name)]
                            checkpoint(model, steps=steps, config=config, best=True)
                        elif not config.higher_is_better and metric < best_metric:
                            best_val_metrics[(db_name, table_name)] = metric
                            best_test_metrics[(db_name, table_name)] = metrics["test"][(db_name, table_name)]
                            checkpoint(model, steps=steps, config=config, best=True)

                        else:
                            checkpoint(model, steps=steps, config=config,  best=False)

            model.train()

            tic = time.time()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            batch.pop("true_batch_size")
            for k in batch:
                batch[k] = batch[k].to(config.device, non_blocking=True)
            toc = time.time()
            load_time = toc - tic
            wandb.log({"load_time": load_time}, step=steps)

            loss, _yhat_dict = model(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
            clip_grads_with_norm_(model.parameters(), max_norm=config.max_grad_norm, total_norm=grad_norm)

            optimizer.step()
            if config.learning_rate_schedule:
                lrs.step()

            steps += 1

            wandb.log({
                    "loss": loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epochs": steps / len(loaders['train']),
                    "grad_norm": grad_norm,
                },
                step=steps,
            )

            pbar.update(1)

    logging.info("\n" + "=" * 80)
    logging.info("Best test metrics:")
    logging.info("=" * 80)
    for (db_name, table_name), metric in best_test_metrics.items():
        logging.info(f"{db_name}/{table_name}/test: {metric:.4f}")
    logging.info("=" * 80 + "\n")
