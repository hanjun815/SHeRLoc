import torch
import torch.nn as nn
from misc.utils import ModelParams
from models.radar.baseline_nets import SHeRLoc, SHeRLoc_S

def create_radar_model(model_params: ModelParams):
    model_name = model_params.radar_model

    if model_name == 'SHeRLoc':
        radar_net = SHeRLoc()
    elif model_name == 'SHeRLoc-S':
        radar_net = SHeRLoc_S()
    else:
        raise NotImplementedError(f'Unknown radar-based model: {model_params.radar_model}')

    return radar_net

class UnifiedModel(nn.Module):
    def __init__(self, query_model, common_model):
        """
        Unified model for processing query and common (positive/negative) embeddings.

        Args:
            query_model (nn.Module): Model for query processing.
            common_model (nn.Module): Model for positive/negative processing.
        """
        super(UnifiedModel, self).__init__()
        self.query_model = query_model
        self.common_model = common_model

    def forward(self, inputs, n_pos, n_neg):
        """
        Args:
            inputs (torch.Tensor): Concatenated tensor of query, positives, and negatives.
            n_pos (int): Number of positive examples per query.
            n_neg (int): Number of negative examples per query.

        Returns:
            torch.Tensor: Concatenated embeddings of query, positives, and negatives.
        """
        B = inputs.shape[0] // (1 + n_pos + n_neg)


        query = inputs[:B]  
        positives = inputs[B:B + B * n_pos]  
        negatives = inputs[B + B * n_pos:]  

        query_embedding = self.query_model(query)  
        positives_embedding = self.common_model(positives)  
        negatives_embedding = self.common_model(negatives)  

        return torch.cat([query_embedding, positives_embedding, negatives_embedding], dim=0)



def model_factory(model_params):
    """
    Factory function to create a unified model for query and common (positive/negative) processing.

    Args:
        model_params (ModelParams): Parameters for model creation.

    Returns:
        nn.Module: Unified model combining query and common models.
    """
    query_model = create_radar_model(model_params)
    common_model = create_radar_model(model_params)
    
    model = UnifiedModel(query_model, query_model)
    # model = UnifiedModel(query_model, common_model)

    return model