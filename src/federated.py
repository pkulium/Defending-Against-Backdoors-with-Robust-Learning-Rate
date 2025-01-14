import torch 
import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from time import ctime
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils import H5Dataset
from agent import replace_bn_with_noisy_bn
from prune_neuron_cifar import prune_by_threshold
from agent import train_mask
import data.poison_cifar as poison
from torch.utils.data import DataLoader, RandomSampler


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
SAVE_MODEL_NAME = 'final_model_cifar_non_iid.th'

if __name__ == '__main__':
    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    utils.print_exp_details(args)
    
    # data recorders
    file_name = f"""time:{ctime()}-clip_val:{args.clip}-noise_std:{args.noise}"""\
            + f"""-aggr:{args.aggr}-s_lr:{args.server_lr}-num_cor:{args.num_corrupt}"""\
            + f"""thrs_robustLR:{args.robustLR_threshold}"""\
            + f"""-num_corrupt:{args.num_corrupt}-pttrn:{args.pattern_type}"""
    writer = SummaryWriter('logs/' + file_name)
    cum_poison_acc_mean = 0
        
    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = utils.get_datasets(args.data)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    # fedemnist is handled differently as it doesn't come with pytorch
    if args.data != 'fedemnist':
        args.alpha = 1.0
        args.server_alpha = 1000
        args.val_frac = 0.01
        args.clean_label = -1
        args.print_every = 500
        args.batch_size = 128
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)

        _, clean_val = poison.split_dataset(dataset=train_dataset, val_frac=args.val_frac,
                                        perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int), clean_label = args.clean_label)
        random_sampler = RandomSampler(data_source=clean_val, replacement=True,
                                   num_samples=args.print_every * args.batch_size)
        server_train_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                  shuffle=False, sampler=random_sampler, num_workers=0)
    
    # poison the validation dataset
    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)                                        
    
    # initialize a model, and the agents
    global_model = models.get_model(args.data).to(args.device)
    if args.rounds == 0:
        global_model.load_state_dict(torch.load(f'/work/LAS/wzhang-lab/mingl/code/backdoor/Defending-Against-Backdoors-with-Robust-Learning-Rate/save/{SAVE_MODEL_NAME}'))
        # global_model = replace_bn_with_noisy_bn(global_model)
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.data == 'fedemnist': 
            agent = Agent(_id, args)
        else:
            agent = Agent(_id, args, train_dataset, user_groups[_id])
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent) 
        
    # aggregation server and the loss function
    n_model_params = len(parameters_to_vector(global_model.parameters()))
    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, writer)
    criterion = nn.CrossEntropyLoss().to(args.device)


    # training loop
    for rnd in tqdm(range(1, args.rounds+1)):
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}
        for agent_id in np.random.choice(args.num_agents, math.floor(args.num_agents*args.agent_frac), replace=False):
            update = agents[agent_id].local_train(global_model, criterion)
            agent_updates_dict[agent_id] = update
            # make sure every agent gets same copy of the global model in a round (i.e., they don't affect each other's training)
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
        # aggregate params obtained by agents and update the global params
        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)
        
        
        # inference in every args.snap rounds
        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)
                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
            
                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
                cum_poison_acc_mean += poison_acc
                writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')

    print('Training has finished!')
    if args.rounds != 0:
        torch.save(global_model.state_dict(), f'/work/LAS/wzhang-lab/mingl/code/backdoor/Defending-Against-Backdoors-with-Robust-Learning-Rate/save/{SAVE_MODEL_NAME}')
        exit()

    rnd = 1
    with torch.no_grad():
        val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)
        writer.add_scalar('Validation/Loss', val_loss, rnd)
        writer.add_scalar('Validation/Accuracy', val_acc, rnd)
        print(f'| Val_Loss/Val_Acc: {val_loss:.3f} - {val_acc:.3f} |')
        print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
    
        poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
        cum_poison_acc_mean += poison_acc
        writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
        writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
        writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
        writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
        print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} - {poison_acc:.3f} |')


    # for rnd in range(1, 2):
    #     rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
    #     agent_updates_mask = {}
    #     select = [i for i in range(1, 10)]
    #     for agent_id in select:
    #         print('-' * 64)
    #         mask_values = agents[agent_id].train_mask(global_model, criterion)
    #         agent_updates_mask[agent_id] = mask_values
    #         with torch.no_grad():
    #             val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(agents[agent_id].local_model, criterion, val_loader, args)
    #             writer.add_scalar('Validation/Loss', val_loss, rnd)
    #             writer.add_scalar('Validation/Accuracy', val_acc, rnd)
    #             print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
    #             print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
            
    #             poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(agents[agent_id].local_model, criterion, poisoned_val_loader, args)
    #             cum_poison_acc_mean += poison_acc
    #             writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
    #             writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
    #             writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
    #             writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
    #             print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
    #     # aggregate params obtained by agents and update the global params
    #     mask_values = aggregator.aggregate_mask_min(agent_updates_mask)
    #     print(f'mask_values:{mask_values[0]} - {mask_values[100]} - {mask_values[1000]}')
    #     prune_by_threshold(global_model, mask_values, pruning_max=0.85, pruning_step=0.05)
    #     print('Pruning has finished!')

    #     with torch.no_grad():
    #         val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)
    #         writer.add_scalar('Validation/Loss', val_loss, rnd)
    #         writer.add_scalar('Validation/Accuracy', val_acc, rnd)
    #         print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
    #         print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
        
    #         poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
    #         cum_poison_acc_mean += poison_acc
    #         writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
    #         writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
    #         writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
    #         writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
    #         print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
    best_val_acc = 0
    best_poison_acc = 1
    for _ in range(1):
        for mask_lr in [0.01, 0.1]:
            for anp_eps in [0.1, 0.4, 1.0]:
                for anp_steps in [1, 10]:
                    for anp_alpha in [0.2, 0.5, 0.9]:
                        for round in [5, 25, 50]:
        # for mask_lr in [0.1]:
        #     for anp_eps in [0.4]:
        #         for anp_steps in [1]:
        #             for anp_alpha in [0.2]:
        #                 for round in [5]:
                            local_model, mask_values =  train_mask(-1, global_model, criterion, server_train_loader, mask_lr, anp_eps, anp_steps, anp_alpha, round)
                            print('-' * 64)
                            print('mask_lr, anp_eps, anp_steps, anp_alpha, round')
                            print(f'{mask_lr}, {anp_eps}, {anp_steps}, {anp_alpha}, {round}')
                            with torch.no_grad():
                                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(local_model, criterion, val_loader, args)
                                writer.add_scalar('Validation/Loss', val_loss, rnd)
                                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                                print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
                            
                                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(local_model, criterion, poisoned_val_loader, args)
                                cum_poison_acc_mean += poison_acc
                                writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
                                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    best_val_acc_ = f'{mask_lr}, {anp_eps}, {anp_steps}, {anp_alpha}, {round}'
                                if poison_acc < best_poison_acc:
                                    best_poison_acc = poison_acc
                                    best_poison_acc_ = f'{mask_lr}, {anp_eps}, {anp_steps}, {anp_alpha}, {round}'
    print(f'{best_val_acc}, {best_val_acc_}')
    print(f'{best_poison_acc}, {best_poison_acc_}')

    for rnd in range(1, 1):
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_mask = {}
        select = [i for i in range(1, 10)]
        for agent_id in select:
            print('-' * 64)
            local_model, mask_values =  train_mask(agent_id, global_model, criterion, agents[agent_id].train_loader)
            agent_updates_mask[agent_id] = mask_values  
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(local_model, criterion, val_loader, args)
                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
            
                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(local_model, criterion, poisoned_val_loader, args)
                cum_poison_acc_mean += poison_acc
                writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
        # aggregate params obtained by agents and update the global params
        mask_values = aggregator.aggregate_mask_min(agent_updates_mask)
        print(f'mask_values:{mask_values[0]} - {mask_values[100]} - {mask_values[1000]}')
        prune_by_threshold(global_model, mask_values, pruning_max=0.85, pruning_step=0.05)
        print('Pruning has finished!')

        with torch.no_grad():
            val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)
            writer.add_scalar('Validation/Loss', val_loss, rnd)
            writer.add_scalar('Validation/Accuracy', val_acc, rnd)
            print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
            print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
        
            poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
            cum_poison_acc_mean += poison_acc
            writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
            writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
            writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
            writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
            print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')





    
            