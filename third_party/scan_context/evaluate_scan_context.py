# Evaluation of ScanContext

import argparse
import numpy as np
import tqdm
import os
import random
from typing import List
from PIL import Image

from datasets.base_datasets import EvaluationSet
from third_party.scan_context.scan_context import ScanContextManager


def print_results(stats):
    recall1 = stats['recall1']
    for r in recall1:
        print(f"Radius: {r} [m] : ", end='')
        for x in recall1[r]:
            print("{:0.3f}, ".format(x), end='')
        print("\n")


def load_image(scan_filepath: str) -> np.ndarray:
    # Returns image as 2D array (height, width)
    image = Image.open(scan_filepath)
    image = np.array(image)
    assert image.ndim == 2
    return image


def evaluate(dataset_root: str, eval_set_pickle: str, radius: List[float], k: int = 50, reranking=True, debug=False):
    # radius: list of thresholds (in meters) to consider an element from the map sequence a true positive
    # k: maximum number of nearest neighbours to consider
    # n_samples: number of samples taken from a query sequence (None=all query elements)
    # cache_prefix: if given computed embeddings will be cached in pickles with a given prefix

    # Use fault parameters
    sc = ScanContextManager()

    eval_set_filepath = os.path.join(dataset_root, eval_set_pickle)
    assert os.path.exists(eval_set_filepath), f'Cannot access evaluation set pickle: {eval_set_filepath}'
    eval_set = EvaluationSet()
    eval_set.load(eval_set_filepath)

    # Compute database point clouds descriptors
    for ndx, e in tqdm.tqdm(enumerate(eval_set.map_set)):
        scan_filepath = os.path.join(dataset_root, e.rel_reading_filepath)
        image = load_image(scan_filepath)
        sc.add_node(image)

    map_positions = eval_set.get_map_positions()
    query_positions = eval_set.get_query_positions()

    # Dictionary to store the number of true positives for different radius and NN number
    tp = {r: [0] * k for r in radius}

    query_indexes = list(range(len(eval_set.query_set)))
    n_samples = len(eval_set.query_set)

    # Randomly sample n_samples clouds from the query sequence and NN search in the target sequence
    for query_ndx in tqdm.tqdm(query_indexes):
        # Check if the query element has a true match within each radius
        query_pos = query_positions[query_ndx]

        # Get the nearest neighbours
        query_scan_filepath = os.path.join(dataset_root, eval_set.query_set[query_ndx].rel_reading_filepath)
        query_image = load_image(query_scan_filepath)

        nn_ndx, _, _ = sc.query(query_image, k, reranking=reranking)

        assert sc.curr_node_idx == len(map_positions), f"{sc.curr_node_idx}  {len(map_positions)}"
        assert np.max(nn_ndx) < len(map_positions)

        # Euclidean distance between the query and nn
        delta = query_pos - map_positions[nn_ndx]       # (k, 2) array
        euclid_dist = np.linalg.norm(delta, axis=1)     # (k,) array
        # Count true positives for different radius and NN number
        tp = {r: [tp[r][nn] + (1 if (euclid_dist[:nn+1] <= r).any() else 0) for nn in range(k)] for r in radius}
        if debug:
            break

    recall1 = {r: [tp[r][nn]/n_samples for nn in range(k)] for r in radius}
    return {'recall1': recall1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ScanContext model')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to the dataset root')
    parser.add_argument('--radius', type=list, default=[5, 10, 20], help='True Positive thresholds in meters')
    parser.add_argument('--nn', type=int, default=50, help='Maximum number of nearest neighbours to consider')

    args = parser.parse_args()
    print(f'Dataset root: {args.dataset_root}')
    print(f'Radius: {args.radius} [m]')
    print(f'Maximum number of nearest neighbours for re-ranking: {args.nn}')
    print('')

    assert os.path.exists(args.dataset_root), f'Cannot access dataset root: {args.dataset_root}'
    # List evaluation sets
    eval_sets = os.listdir(args.dataset_root)
    # Only do evaluation on radar-to-radar sets
    eval_sets = [e for e in eval_sets if e[-13:] == '120_40.pickle' and "R2R" in e]

    for eval_set in eval_sets:
        # Cache computed embeddings, prefixing with a network weights file name
        print(f'Evaluation set: {eval_set}')
        print('Reranking: FALSE')
        stats = evaluate(args.dataset_root, eval_set, radius=args.radius, reranking=False, k=args.nn)
        print_results(stats)

        print('Reranking: TRUE')
        stats = evaluate(args.dataset_root, eval_set, radius=args.radius, reranking=True, k=args.nn)
        print_results(stats)
