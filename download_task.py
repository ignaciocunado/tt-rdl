import argparse

from relbench.datasets import get_dataset
from relbench.tasks import get_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    get_dataset(args.dataset, download=True)
    get_task(args.dataset, args.task, download=True)


if __name__ == "__main__":
    main()
