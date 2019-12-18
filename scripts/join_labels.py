from argparse import ArgumentParser
from pathlib import Path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('key', type=Path)
    args = parser.parse_args()

    labels = {}
    with args.key.open() as fp:
        for line in fp:
            token_index, label = line.strip().split()
            labels[token_index] = label

    with args.dataset.open() as fp:
        for line in fp:
            if not line.strip() or line.strip().startswith('#'):
                print(line.strip())
            else:
                token_index, *_ = line.split()
                print(line.strip() + '\t' + labels[token_index])
