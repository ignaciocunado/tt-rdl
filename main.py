import argparse
import os

from src.dataloader import RelBenchDataLoader
from src.models.reltt import RelationalTransformer

os.environ["XDG_CACHE_HOME"] = "/tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/tt-rdl/data"

from src.config import CustomConfig

from src.train import train

from torch.optim import Adam, AdamW
import logging

import random

import numpy as np
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader


def set_seed(seed):
    """Sets the seed for all random number generators.
    Args:
        seed: The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="The dataset to use", required=True)
    parser.add_argument("--task", type=str, help="The task to solve", required=True)
    parser.add_argument("--save_artifacts", action="store_true", help="Whether to save artifacts")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="How many workers to use for data loading. Default: 8."
    )
    parser.add_argument("--eval_freq", type=int, default=2, help="Evaluate every x epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--lr_schedule", action="store_true", help="Whether to use lr_scheduler")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=50_001, help="Max number of steps")
    parser.add_argument("--optimiser", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--channels", type=int, default=128, help="Number of channels")
    parser.add_argument("--aggr", type=str, default="sum", help="Aggregation method")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of layers")
    parser.add_argument("--early_stopping", action="store_true", help="Use early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Override default configuration
    config = CustomConfig(
        data_name=args.dataset,
        task_name=args.task,
        evaluation_freq=args.eval_freq,
        learning_rate=args.lr,
        learning_rate_schedule=args.lr_schedule,
        weight_decay=args.wd,
        epochs=args.epochs,
        max_steps=args.max_steps,
        optimiser=args.optimiser,
        batch_size=args.batch_size,
        channels=args.channels,
        num_workers=args.num_blocks,
        save_artifacts=args.save_artifacts,
        early_stopping=args.early_stopping,
    )
    config.print_config()

    train_data = RelBenchDataLoader(
        data_name=config.data_name,
        task_name=config.task_name,
        split="train",
        root_dir=config.data_dir,
        batch_size=config.batch_size,
        seq_len=config.sequence_length,
        rank=0,
        world_size=1,
        max_bfs_width=config.max_bfs_width,
        d_text=config.d_text,
        seed=args.seed,
    )
    test_data = RelBenchDataLoader(
        data_name=config.data_name,
        task_name=config.task_name,
        split="test",
        root_dir=config.data_dir,
        batch_size=config.batch_size,
        seq_len=config.sequence_length,
        rank=0,
        world_size=1,
        max_bfs_width=config.max_bfs_width,
        d_text=config.d_text,
        seed=args.seed,
    )
    val_data = RelBenchDataLoader(
        data_name=config.data_name,
        task_name=config.task_name,
        split="val",
        root_dir=config.data_dir,
        batch_size=config.batch_size,
        seq_len=config.sequence_length,
        rank=0,
        world_size=1,
        max_bfs_width=config.max_bfs_width,
        d_text=config.d_text,
        seed=args.seed,
    )

    loader_dict = {
        "train": DataLoader(
            train_data,
            batch_size=None,
            num_workers=config.num_workers,
            persistent_workers=False,
            pin_memory=True,
            in_order=True,
        ),
        "val": DataLoader(
            val_data,
            batch_size=None,
            num_workers=config.num_workers,
            persistent_workers=False,
            pin_memory=True,
            in_order=True,
        ),
        "test": DataLoader(
            test_data,
            batch_size=None,
            num_workers=config.num_workers,
            persistent_workers=False,
            pin_memory=True,
            in_order=True,
        ),
    }

    model = RelationalTransformer(
        num_blocks=config.num_blocks,
        d_model=config.channels,
        d_text=config.d_text,
        num_heads=config.attn_heads,
        d_ff=config.feed_forward_dim,
    )
    model = model.to(config.device)
    model = model.to(torch.bfloat16)

    # Initialize optimizer and loss function
    optimiser = None
    if config.optimiser == "adam":
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimiser == "adamW":
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,
        )
    else:
        raise ValueError("Invalid optimizer specified")

    if args.lr_schedule:
        lrs = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config.learning_rate,
            total_steps=50000,
            pct_start=0.2,
            anneal_strategy="linear",
        )

    net = torch.compile(model, dynamic=False)

    wandb.init(project="Tabular Learning", config={"model": "Tabular Transformer"} | config.__dict__)

    logging.info("Model: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total model parameters: {total_params}")

    train(
        model=model,
        loaders=loader_dict,
        optimizer=optimizer,
        lrs=lrs,
        config=config,
    )
