import torch
import models
import utils
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn

from anp_batchnorm import NoisyBatchNorm2d, NoisyBatchNorm1d
from collections import OrderedDict
from optimize_mask_cifar import *
from prune_neuron_cifar import *
from torch.utils.data import DataLoader, SubsetRandomSampler

def replace_bn_with_noisy_bn(module: nn.Module) -> nn.Module:
    """Recursively replace all BatchNorm layers with NoisyBatchNorm layers while preserving weights."""
    device = 'cuda:0'
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Create a new NoisyBatchNorm2d layer
            new_layer = NoisyBatchNorm2d(child.num_features).to(device=device)
            
            # Copy weights and biases
            new_layer.weight.data = child.weight.data.clone().detach()
            new_layer.bias.data = child.bias.data.clone().detach()
            
            # Copy running mean and variance
            new_layer.running_mean = child.running_mean.clone().detach()
            new_layer.running_var = child.running_var.clone().detach()
            
            # Replace the original layer with the new layer
            setattr(module, name, new_layer)
        elif isinstance(child, nn.BatchNorm1d):
            # Create a new NoisyBatchNorm1d layer
            new_layer = NoisyBatchNorm1d(child.num_features).to(device=device)
            
            # Copy weights and biases
            new_layer.weight.data = child.weight.data.clone().detach()
            new_layer.bias.data = child.bias.data.clone().detach()
            
            # Copy running mean and variance
            new_layer.running_mean = child.running_mean.clone().detach()
            new_layer.running_var = child.running_var.clone().detach()
            
            # Replace the original layer with the new layer
            setattr(module, name, new_layer)
        else:
            replace_bn_with_noisy_bn(child)
    return module

class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None):
        self.id = id
        self.args = args
        
        # get datasets, fedemnist is handled differently as it doesn't come with pytorch
        if train_dataset is None:
            self.train_dataset = torch.load(f'../data/Fed_EMNIST/user_trainsets/user_{id}_trainset.pt')
            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt:
                utils.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id)    
        else:
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt:
                utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
        
        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True,\
            num_workers=args.num_workers, pin_memory=False)
        # size of local dataset
        self.n_data = len(self.train_dataset)
        
    def local_train(self, global_model, criterion):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()       
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, 
            momentum=self.args.client_moment)
        
        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                 labels.to(device=self.args.device, non_blocking=True)
                                               
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                # to prevent exploding gradients
                nn.utils.clip_grad_norm_(global_model.parameters(), 10) 
                optimizer.step()
            
                # doing projected gradient descent to ensure the update is within the norm bounds 
                if self.args.clip > 0:
                    with torch.no_grad():
                        local_model_params = parameters_to_vector(global_model.parameters())
                        update = local_model_params - initial_global_model_params
                        clip_denom = max(1, torch.norm(update, p=2)/self.args.clip)
                        update.div_(clip_denom)
                        vector_to_parameters(initial_global_model_params + update, global_model.parameters())
                            
        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            return update
    
    def train_mask(self, global_model, criterion):
        print(f'id:{self.id}')
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        from copy import deepcopy   
        self.local_model = deepcopy(global_model)
        self.local_model = replace_bn_with_noisy_bn(self.local_model)
        self.local_model.train()
        self.local_model = self.local_model.to(self.args.device)
        self.local_model.mask_lr = 0.01
        self.local_model.anp_eps = 0.4
        self.local_model.anp_steps = 1
        self.local_model.anp_alpha = 0.2
        self.mask_scores = None

        self.local_model.train()  
        parameters = list(self.local_model.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=self.local_model.mask_lr, momentum=0.9)
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=self.local_model.anp_eps / self.local_model.anp_steps)

        # Step 1: Create a list of all indices
        data_loader=self.train_loader
        all_indices = list(range(len(data_loader.dataset)))

        # Step 2: Shuffle these indices
        torch.manual_seed(42)  # For reproducibility
        torch.utils.data.random_split(all_indices, [len(data_loader.dataset)])  # This will shuffle the indices

        # Step 3: Select the first 500 indices
        selected_indices = all_indices[:500]

        # Step 4: Use SubsetRandomSampler
        sampler = SubsetRandomSampler(selected_indices)

        # Step 5: Create a new DataLoader
        selected_data_loader = DataLoader(data_loader.dataset, batch_size=32, sampler=sampler)

        for epoch in range(100):
            train_loss, train_acc = mask_train(model=self, criterion=criterion, data_loader=selected_data_loader,
                                        mask_opt=mask_optimizer, noise_opt=noise_optimizer)

        self.mask_scores = get_mask_scores(self.local_model.state_dict())
        save_mask_scores(self.local_model.state_dict(), f'/work/LAS/wzhang-lab/mingl/code/backdoor/Defending-Against-Backdoors-with-Robust-Learning-Rate/save/mask_values.txt')
        mask_values = read_data(f'/work/LAS/wzhang-lab/mingl/code/backdoor/Defending-Against-Backdoors-with-Robust-Learning-Rate/save/mask_values.txt')
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        print(f'mask_values:{mask_values[0]} - {mask_values[100]} - {mask_values[1000]}')
        # prune_by_threshold(global_model, mask_values, pruning_max=0.75, pruning_step=0.01)
        return mask_values
        # return self.local_model

        # with torch.no_grad():
        #     update = parameters_to_vector(self.local_model.parameters()).double() - initial_global_model_params
        #     return update
