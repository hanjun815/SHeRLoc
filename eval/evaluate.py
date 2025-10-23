import argparse
import numpy as np
import tqdm
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.model_factory import model_factory
from misc.utils import ModelParams
from training.dataset import HeRCULES_test

class Evaluator:
    def __init__(self, model_params: ModelParams, test_loader, device, radius=(5), k=10):
        self.task = model_params.task
        self.test_loader = test_loader
        self.device = device
        self.radius = radius
        self.k = k
        self.n_samples = len(test_loader.dataset)

    def evaluate(self, model):
        model.eval()
        query_embeddings, map_embeddings, query_positions, map_positions = self.compute_embeddings(model)
        thresholds = np.linspace(0, 2, 1000)
        recall_indicator = np.zeros(self.k)
        num_evaluated = 0
        log_file = "matching.txt"
        f = open(log_file, "w")
        f.close()
        num_thresholds = len(thresholds)
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        for query_idx, query_embedding in tqdm.tqdm(enumerate(query_embeddings)):
            query_pos = query_positions[query_idx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            delta = query_pos - map_positions
            euclid_dist = np.linalg.norm(delta, axis=1)
            true_neighbors = np.where(euclid_dist <= self.radius)[0]
            if len(true_neighbors) == 0:
                continue
            num_evaluated += 1
            sorted_indices = np.argsort(embed_dist)[:self.k]
            with open(log_file, "a") as f:
                f.write(f"{query_idx} {sorted_indices[0]} \n")
            desc_dist_0 = embed_dist[sorted_indices[0]]
            for j, idx in enumerate(sorted_indices):
                if idx in true_neighbors:
                    recall_indicator[j] += 1
                    break
            for thresh_idx, threshold in enumerate(thresholds):
                if desc_dist_0 < threshold:
                    if sorted_indices[0] in true_neighbors:
                        tp[thresh_idx] += 1
                    else:
                        fp[thresh_idx] += 1
                else:
                    if sorted_indices[0] in true_neighbors:
                        fn[thresh_idx] += 1
                    else:
                        tn[thresh_idx] += 1
        recall_cumulative = np.cumsum(recall_indicator) / num_evaluated * 100
        precisions = np.divide(
            tp, (tp + fp), out=np.zeros_like(tp), where=(tp + fp != 0))
        recalls_arr = np.divide(
                tp, (tp + fn), out=np.zeros_like(tp), where=(tp + fn != 0))
        final_recall_1 = recall_cumulative[0]
        for i in range(len(recall_cumulative)):
            print("recall@", i+1, ": ", recall_cumulative[i])
        with open("precision-recall.txt", "w") as f:
            for i in range(len(precisions)):
                f.write(f"{precisions[i]:.4f} {recalls_arr[i]:.4f}\n")
        with open("recallN.txt", "w") as f:
            for i in range(len(recall_cumulative)):
                    f.write(f"{recall_cumulative[i]:.4f}\n")
        return {'recall': final_recall_1, 'query_embeddings': query_embeddings, 'map_embeddings': map_embeddings, 'query_positions': query_positions, 'map_positions': map_positions}
    
    def compute_embeddings(self, model):
        if self.task in ("Hetero", "LiDAR"):
            query_embeddings = []
            map_embeddings = []
            query_positions = []
            map_positions = []
            for query, maps, query_pos, map_pos in tqdm.tqdm(self.test_loader):
                query = query.to(self.device) 
                query_pos = query_pos.numpy() 
                valid_maps = [map_tensor.to(self.device) for map_tensor in maps if map_tensor is not None] 
                valid_map_pos = [mp.numpy() for mp in map_pos if mp is not None] 
                with torch.no_grad():
                    query_embedding = model.query_model(query) 
                    map_embeddings_batch = [model.query_model(map_tensor) for map_tensor in valid_maps] 
                query_embeddings.append(query_embedding.cpu().numpy())
                query_positions.extend(query_pos)  
                map_embeddings.extend([map_embedding.cpu().numpy() for map_embedding in map_embeddings_batch])  
                for mp in valid_map_pos: 
                    map_positions.extend(mp) 
            query_embeddings = np.vstack(query_embeddings) 
            map_embeddings = np.vstack(map_embeddings)     
            query_positions = np.array(query_positions)   
            map_positions = np.array(map_positions)       
            return query_embeddings, map_embeddings, query_positions, map_positions

        elif self.task in ("4D", "Spinning"):
            query_embeddings = []
            map_embeddings = []
            query_positions = []
            map_positions = []
            for query, maps, query_pos, map_pos in tqdm.tqdm(self.test_loader):
                query = query.to(self.device)
                maps = maps.to(self.device)
                query_pos = query_pos.numpy()
                map_pos = map_pos.numpy()
                with torch.no_grad():
                    query_embedding = model.query_model(query)
                    map_embedding = model.query_model(maps)
                query_embeddings.append(query_embedding.cpu().numpy())
                map_embeddings.append(map_embedding.cpu().numpy())
                query_positions.extend(query_pos)
                map_positions.extend(map_pos)
            return (np.vstack(query_embeddings), np.vstack(map_embeddings),
                    np.array(query_positions), np.array(map_positions))
        
        else:
            raise ValueError(f"Unknown task: {self.task}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help="Path to the dataset root")
    parser.add_argument('--model_config', type=str, required=True, help="Path to the model-specific config file")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained model weights")
    parser.add_argument('--batch_size', type=int, default=50, help="Batch size for evaluation")
    parser.add_argument('--radius', nargs='+', type=int, default=[5], help="True Positive thresholds in meters")
    parser.add_argument('--k', type=int, default=10, help="Number of top-K results for metrics")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[new_key] = value
        return new_state_dict

    model_params = ModelParams(args.model_config)
    model = model_factory(model_params).to(device)

    assert os.path.exists(args.weights), f"Weights file not found: {args.weights}"
    checkpoint = torch.load(args.weights, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = remove_module_prefix(checkpoint['state_dict'])
        model.load_state_dict(state_dict)
    else:
        raise ValueError("The weights file does not contain 'state_dict'. Please check the saved model file.")

    test_dataset = HeRCULES_test(model_params, root = args.dataset_root, phase="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    evaluator = Evaluator(model_params, test_loader, device, radius=args.radius, k=args.k)
    stats = evaluator.evaluate(model)