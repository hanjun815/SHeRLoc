import os
import configparser
import time

class ModelParams:
    def __init__(self, model_params_path):
        assert os.path.exists(model_params_path), f"Cannot access model config file: {model_params_path}"
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']
        self.model_params_path = model_params_path
        self.task = params.get('task', None)
        self.radar_model = params.get('radar_model', None)

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))
        print('')

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

class TrainingParams:
    """
    Parameters for model training
    """
    def __init__(self, params_path, model_params_path):
        """
        Configuration files
        :param path: Training configuration file
        :param model_params: Model-specific configuration file
        """
        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        config = configparser.ConfigParser()
        config.read(self.params_path)
        params = config['DEFAULT']
        self.dataset = params.get('dataset', 'HeRCULES').lower()
        self.dataset_folder = params.get('dataset_folder')
        params = config['TRAIN']
        self.num_workers = params.getint('num_workers', 0)
        self.batch_size = params.getint('batch_size', 256)
        self.lr = params.getfloat('lr', 1e-3)
        self.epochs = params.getint('epochs', 20)
        self.weight_decay = params.getfloat('weight_decay', None)
        self.loss = params.get('loss')

        if 'Triplet' in self.loss:
            self.margin = params.getfloat('margin', 1.0)
        else:
            raise 'Unsupported loss function: {}'.format(self.loss)

        self.train_dataset = params.get('train_dataset')
        self.val_dataset = params.get('val_dataset', None)
        self.test_dataset = params.get('test_dataset')
        self.model_params = ModelParams(self.model_params_path)
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')