import json
import os
from functools import cache

import numpy as np
import torch
from relbench.datasets import get_dataset
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from torch.utils.data import Dataset

from rustler import Sampler

task_to_target_col = {
    "user-churn": "churn",
    "user-badge": "WillGetBadge",
    "item-churn": "churn",
    "user-engagement": "contribution",
    "user-visits": "num_click",
    "user-clicks": "num_click",
    "user-ignore": "target",
    "study-outcome": "outcome",
    "driver-dnf": "did_not_finish",
    "user-repeat": "target",
    "driver-top3": "qualifying",
    "item-sales": "sales",
    "user-ltv": "ltv",
    "item-ltv": "ltv",
    "post-votes": "popularity",
    "site-success": "success_rate",
    "study-adverse": "num_of_adverse_events",
    "user-attendance": "target",
    "driver-position": "position",
    "ad-ctr": "num_click",
}


class RelBenchDataLoader:
    """Data loader class for RelBench datasets.

    This class handles:
    - Loading and processing RelBench datasets
    - Creating heterogeneous graphs
    - Setting up train/val/test data loaders
    - Managing text embeddings

    Args:
        data_name (str): Name of the RelBench dataset (e.g., 'f1')
        task_name (str): Name of the task (e.g., 'driver-position')
        root_dir (str, optional): Directory for cached data. Defaults to "./data"
        batch_size (int, optional): Batch size for dataloaders. Defaults to 128
        num_workers (int, optional): Number of worker processes. Defaults to 0
    """

    def __init__(
        self,
        data_name: str,
        task_name: str,
        split: str,
        root_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 0,
        seq_len: int = 1024,
        rank: int = 0,
        world_size: int = 1,
        max_bfs_width: int = 256,
        d_text: int = 384,
        seed: int = 42,
    ):
        self.data_name = data_name
        self.task_name = task_name
        self.split = split
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.entity_table = None

        dataset_tuples, target_column_indices, drop_column_indices = self._apply_pre()

        self.sampler = Sampler(
            dataset_tuples=dataset_tuples,
            batch_size=batch_size,
            seq_len=seq_len,
            rank=rank,
            world_size=world_size,
            max_bfs_width=max_bfs_width,
            embedding_model="all-MiniLM-L12-v2",
            d_text=d_text,
            seed=seed,
            target_columns=target_column_indices,
            columns_to_drop=drop_column_indices,
        )

        self.seq_len = seq_len
        self.d_text = d_text

        # Initialize data structures
        self.db = self.dataset.get_db()
        self.col_to_stype_dict = get_stype_proposal(self.db)

        # Initialize loaders
        self.loader_dict = self._create_loaders()

    def _load_dataset(self):
        """Load the RelBench dataset.

        Returns:
            Dataset: The loaded RelBench dataset
        """
        return get_dataset(self.data_name, download=True)

    def _load_task(self):
        """Load the RelBench task.

        Returns:
            Task: The loaded RelBench task
        """
        return get_task(self.data_name, self.task_name, download=True)

    def _apply_pre(self):
        dataset_tuples = []
        target_column_indices = []
        drop_column_indices = []

        table_info_path = f"{os.environ['HOME']}/scratch/pre/{self.data_name}/table_info.json"
        with open(table_info_path) as f:
            table_info = json.load(f)

        # choose the correct entry for THIS split
        table_info_key = (
            f"{self.task_name}:Db" if f"{self.task_name}:Db" in table_info else f"{self.task_name}:{self.split}"
        )
        info = table_info[table_info_key]
        node_idx_offset = info["node_idx_offset"]
        num_nodes = info["num_nodes"]

        target_idx = get_column_index(
            task_to_target_col[self.task_name],
            self.task_name,
            self.data_name,
        )

        dataset_tuples.append((self.data_name, node_idx_offset, num_nodes))
        target_column_indices.append(target_idx)

        drop_column_indices.append([target_idx])

        return dataset_tuples, target_column_indices, drop_column_indices

    def __getitem__(self, batch_idx):
        tup = self.sampler.batch_py(batch_idx)
        out = dict(tup)
        for k, v in out.items():
            if k in [
                "number_values",
                "datetime_values",
                "text_values",
                "col_name_values",
                "boolean_values",
            ]:
                out[k] = torch.from_numpy(v.view(np.float16)).view(torch.bfloat16)
            elif k == "true_batch_size":
                pass
            else:
                out[k] = torch.from_numpy(v)

        out["node_idxs"] = out["node_idxs"].view(-1, self.seq_len)
        out["sem_types"] = out["sem_types"].view(-1, self.seq_len)
        out["masks"] = out["masks"].view(-1, self.seq_len)
        out["is_targets"] = out["is_targets"].view(-1, self.seq_len)
        out["is_task_nodes"] = out["is_task_nodes"].view(-1, self.seq_len)
        out["is_padding"] = out["is_padding"].view(-1, self.seq_len)
        out["table_name_idxs"] = out["table_name_idxs"].view(-1, self.seq_len)
        out["col_name_idxs"] = out["col_name_idxs"].view(-1, self.seq_len)
        out["class_value_idxs"] = out["class_value_idxs"].view(-1, self.seq_len)

        out["f2p_nbr_idxs"] = out["f2p_nbr_idxs"].view(-1, self.seq_len, 5)
        out["number_values"] = out["number_values"].view(-1, self.seq_len, 1)
        out["datetime_values"] = out["datetime_values"].view(-1, self.seq_len, 1)
        out["boolean_values"] = out["boolean_values"].view(-1, self.seq_len, 1).bfloat16()
        out["text_values"] = out["text_values"].view(-1, self.seq_len, self.d_text)
        out["col_name_values"] = out["col_name_values"].view(-1, self.seq_len, self.d_text)

        return out


@cache
def _load_column_index(db_name: str) -> dict:
    """
    Load the column index mapping for a dataset (cached).
    """
    home = os.environ.get("HOME", ".")
    column_index_path = os.path.join(home, "scratch", "pre", db_name, "column_index.json")

    with open(column_index_path) as f:
        return json.load(f)


def get_column_index(column_name: str, table_name: str, db_name: str) -> int:
    """
    Get the index of a column in the text embeddings for a given dataset.
    """
    column_index = _load_column_index(db_name)
    target = f"{column_name} of {table_name}"

    if target not in column_index:
        raise ValueError(f'Column "{target}" not found in column_index.json for dataset {db_name}.')

    return column_index[target]
