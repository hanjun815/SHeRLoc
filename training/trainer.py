import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import numpy as np
import torch
import tqdm
import pathlib
import shutil
import wandb
from eval.evaluate import Evaluator
from dataset import HeRCULES
from misc.utils import ModelParams, TrainingParams, get_datetime
from models.loss import make_losses
from models.model_factory import model_factory

from torch.utils.data import DataLoader

def print_stats(stats, phase):
    s = '{} - Global loss: {:.6f}    Embedding norm: {:.4f}   Triplets (all/active): {:.1f}/{:.1f}'
    print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_triplets'], stats['num_non_zero_triplets']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if len(l) > 0:
        print(s.format(*l))


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def create_weights_folder():
    weights_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'weights')
    os.makedirs(weights_path, exist_ok=True)
    return weights_path

def do_train(model_params: ModelParams, params: TrainingParams, debug=False, device='cpu'):

    s = get_datetime()
    model = model_factory(params.model_params)  

    model_name = 'model_'
    if params.model_params.radar_model is not None:
        model_name += params.model_params.radar_model + '_'

    model_name += s
    print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()
    shutil.copy(params.params_path, os.path.join(weights_path, 'config.txt'))
    model_pathname = os.path.join(weights_path, model_name)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device)

    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))


    train_dataset = HeRCULES(model_params,root=params.train_dataset, phase="train")
    val_dataset = HeRCULES(model_params, root=params.val_dataset, phase="val")


    def custom_collate(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None 
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=params.num_workers, collate_fn=custom_collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, num_workers=params.num_workers, collate_fn=custom_collate, pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader}

    loss_fn = make_losses(params)

    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)


    params_dict = {e: params.__dict__[e] for e in params.__dict__ if e != 'model_params'}
    model_params_dict = {"model_params." + e: params.model_params.__dict__[e] for e in params.model_params.__dict__}
    params_dict.update(model_params_dict)
    wandb.init(project='SHeRLoc', config=params_dict)


    phases = ['train']
    if 'val' in dataloaders:
        phases.append('val')

    stats = {e: [] for e in phases}
    stats['eval'] = []

    for epoch in tqdm.tqdm(range(1, params.epochs + 1)):
        for phase in phases:
            if 'train' in phase:
                model.train()
            else:
                model.eval()

            running_stats = []
            for index, (query, positives, negatives, sim_p, sim_n) in enumerate(dataloaders[phase]):
                if positives is None or negatives is None or positives.shape[1] == 0 or negatives.shape[1] == 0:
                    print(f"Skipping invalid batch at index {index} in phase {phase}")
                    continue
                batch_stats = {}

                B, n_pos, C, H, W = positives.shape
                B, n_neg, C, H, W = negatives.shape
                if query.dim() == 3:
                    query = query.unsqueeze(1)  

                positives = torch.flatten(positives, start_dim=0, end_dim=1)                
                negatives = torch.flatten(negatives, start_dim=0, end_dim=1)
                sim_p = sim_p.to(device)
                sim_n = sim_n.to(device)

                inputs = torch.cat([query, positives, negatives]).to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, n_pos, n_neg)
                    vladQ, vladP, vladN = torch.split(outputs, [B, B * n_pos, B * n_neg])


                    loss, temp_stats, _ = loss_fn(vladQ, vladP, vladN,  sim_p, sim_n)
                    print(index,len(dataloaders[phase].dataset),loss.item())

                    batch_stats['loss'] = loss.item()

                    temp_stats = tensors_to_numbers(temp_stats)
                    batch_stats.update(temp_stats)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                torch.cuda.empty_cache()
                running_stats.append(batch_stats)


            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_stats = {}
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)
            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

        wandb.log({
            'epoch': epoch,
            'train_loss': stats['train'][-1]['loss'] if len(stats['train']) > 0 else 0.0,
            'val_loss': stats['val'][-1]['loss'] if len(stats['train']) > 0 else 0.0,
            'lr': scheduler.get_last_lr()[0] if scheduler is not None else params.lr,
        })


        checkpoint_path = f"{model_pathname}_epoch_{epoch}.pth"
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, checkpoint_path)
        torch.cuda.empty_cache()

    print('Training completed.')


    final_model_path = model_pathname + '_final.pth'
    torch.save({'state_dict': model.state_dict()}, final_model_path)

    evaluator_test_set = Evaluator(val_dataset, device, radius=[5], k=20)
    global_stats = evaluator_test_set.evaluate(model)
    print('Evaluation results (no rotation):')
    recall = global_stats['recall']
    for r in recall:
        print(f"Radius: {r}m")
        print(f"Recall: {recall[r]}")