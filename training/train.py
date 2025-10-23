import argparse
import torch

from training.trainer import do_train
from misc.utils import ModelParams, TrainingParams


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train cross-modal model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Debug mode: {}'.format(args.debug))
    model_params = ModelParams(args.model_config)
    params = TrainingParams(args.config, args.model_config)
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    do_train(model_params, params, debug=args.debug, device=device)
