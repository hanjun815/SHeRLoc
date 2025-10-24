import torch
from pytorch_metric_learning.distances import LpDistance
from misc.utils import TrainingParams

def make_losses(params: TrainingParams):
    if params.loss == 'BatchHardTripletMarginLoss':
        loss_fn = BatchHardTripletLoss(params.margin)
    else:
        print('Unknown loss: {}'.format(params.loss))
        raise NotImplementedError

    return loss_fn

class BatchHardTripletLoss:
    """
    BatchHard triplet margin loss implementation that operates on embeddings (vladQ, vladP, vladN)
    with adaptive margin based on similarity differences (sim_p - sim_n).
    """
    def __init__(self, margin):
        self.base_margin = margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)

    def __call__(self, vladQ, vladP, vladN, sim_p, sim_n):
        """
        Computes the BatchHard triplet loss with adaptive margin.

        Args:
            vladQ (torch.Tensor): Query embeddings of shape [B, D].
            vladP (torch.Tensor): Positive embeddings of shape [B * n_pos, D].
            vladN (torch.Tensor): Negative embeddings of shape [B * n_neg, D].
            sim_p (torch.Tensor): Similarity scores between query and positives [B, n_pos].
            sim_n (torch.Tensor): Similarity scores between query and negatives [B, n_neg].

        Returns:
            loss (torch.Tensor): The triplet loss value.
            stats (dict): Statistics for analysis and debugging.
        """
        B = vladQ.shape[0]
        n_pos = vladP.shape[0] // B
        n_neg = vladN.shape[0] // B 

        vladP = vladP.view(B, n_pos, -1)  
        vladN = vladN.view(B, n_neg, -1)  
        sim_p = sim_p.view(B, n_pos)  
        sim_n = sim_n.view(B, n_neg)  

        pos_distances = []  
        neg_distances = []  
        dist_swaps = []  
        adaptive_margins = []  

        for b in range(B):
            pos_dist = self.distance(vladQ[b:b+1], vladP[b]) 
            neg_dist = self.distance(vladQ[b:b+1], vladN[b]) 

            pos_distances.append(pos_dist.squeeze(0))
            neg_distances.append(neg_dist.squeeze(0))

            hardest_vladP = vladP[b][pos_dist.argmax()]
            hardest_vladN = vladN[b][neg_dist.argmin()]
            dist_swap = self.distance(hardest_vladP.unsqueeze(0), hardest_vladN.unsqueeze(0))
            dist_swaps.append(dist_swap.squeeze(0))

            hardest_pos_sim = sim_p[b][pos_dist.argmax()]
            hardest_neg_sim = sim_n[b][neg_dist.argmin()]
            margin_factor = hardest_pos_sim - hardest_neg_sim

            adaptive_margin = self.base_margin * margin_factor.clamp(min=0) 
            adaptive_margins.append(adaptive_margin)

        pos_distances = torch.stack(pos_distances, dim=0)  
        neg_distances = torch.stack(neg_distances, dim=0)  
        dist_swaps = torch.stack(dist_swaps, dim=0)    
        adaptive_margins = torch.stack(adaptive_margins, dim=0)  

        hardest_positive_dist, hardest_positive_idx = pos_distances.max(dim=1) 
        hardest_negative_dist, hardest_negative_idx = neg_distances.min(dim=1) 

        effective_negative_dist = torch.minimum(hardest_negative_dist, dist_swaps)
        triplet_losses = torch.relu(hardest_positive_dist - effective_negative_dist + adaptive_margins)

        non_zero_losses = triplet_losses[triplet_losses > 0]
        if len(non_zero_losses) > 0:
            loss = non_zero_losses.mean()
        else:
            loss = triplet_losses.sum() * 0

        stats = {
            'loss': loss.item(),
            'avg_embedding_norm': torch.mean(torch.norm(vladQ, dim=1)).item(),
            'num_non_zero_triplets': len(non_zero_losses),
            'num_triplets': vladQ.shape[0],
            'mean_pos_pair_dist': pos_distances.mean().item(),
            'mean_neg_pair_dist': neg_distances.mean().item(),
            'max_pos_pair_dist': pos_distances.max().item(),
            'max_neg_pair_dist': neg_distances.max().item(),
            'min_pos_pair_dist': pos_distances.min().item(),
            'min_neg_pair_dist': neg_distances.min().item(),
            'mean_dist_swap': dist_swaps.mean().item(),
            'mean_adaptive_margin': adaptive_margins.mean().item(),
        }

        return loss, stats, None
