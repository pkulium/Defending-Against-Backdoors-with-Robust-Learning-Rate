import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from math import floor
from collections import defaultdict
import random
import cv2


class H5Dataset(Dataset):
    def __init__(self, dataset, client_id):
        self.targets = torch.LongTensor(dataset[client_id]['label'])
        self.inputs = torch.Tensor(dataset[client_id]['pixels'])
        shape = self.inputs.shape
        self.inputs = self.inputs.view(shape[0], 1, shape[1], shape[2])
        
    def classes(self):
        return torch.unique(self.targets)
    
    def __add__(self, other): 
        self.targets = torch.cat( (self.targets, other.targets), 0)
        self.inputs = torch.cat( (self.inputs, other.inputs), 0)
        return self
    
    def to(self, device):
        self.targets = self.targets.to(device)
        self.inputs = self.inputs.to(device)
        
        
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, item):
        inp, target = self.inputs[item], self.targets[item]
        return inp, target
    

class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])
        
    def classes(self):
        return torch.unique(self.targets)    

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inp, target = self.dataset[self.idxs[item]]
        return inp, target

def distribute_data_dirichlet(dataset, args):
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    labels_dict = defaultdict(list)

    for k, v in class_by_labels:
        labels_dict[k].append(v)
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    N = len(labels_sorted[1])
    K = len(labels_dict)
    client_num = args.num_agents

    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]
        for k in labels_dict:
            idx_k = labels_dict[k]

            # get a list of batch indexes which are belong to label k
            np.random.shuffle(idx_k)
            # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
            # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
            proportions = np.random.dirichlet(np.repeat(args.alpha, client_num))

            # get the index in idx_k according to the dirichlet distribution
            proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # generate the batch list for each client
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # distribute data to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        dict_users[user_idx] = idx_batch[user_idx]
        np.random.shuffle(dict_users[user_idx])

    # # Create IID dataset for the server
    # all_indices = list(range(N))
    # np.random.shuffle(all_indices)
    # server_proportions = np.random.dirichlet(np.repeat(args.server_alpha, client_num))
    # server_proportions = (np.cumsum(server_proportions) * N).astype(int)[:-1]
    # server_data = np.split(all_indices, server_proportions)[0]  # Take the first split as the server's data
    return dict_users

def sample_dirichlet_train_data(train_dataset, alpha, n_client, server_dataset):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indices dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as
            parameters for
            dirichlet distribution to sample number of images in each class.
        """

        total_classes = dict()
        for ind, x in enumerate(train_dataset):
            _, label = x
            if label in total_classes:
                total_classes[label].append(ind)
            else:
                total_classes[label] = [ind]

        class_size = len(total_classes[0])
        per_client_list = defaultdict(list)
        n_class = len(total_classes.keys())
    
        np.random.seed(111)
        for n in range(n_class):
            random.shuffle(total_classes[n])
            n_party = n_client
            if server_dataset:
                sampled_probabilities = class_size * np.random.dirichlet(np.array(n_client * [alpha] + [100]))
                n_party = n_party + 1
            else:
                sampled_probabilities = class_size * np.random.dirichlet(np.array(n_client * [alpha]))
            for p in range(n_party):
                n_image = int(round(sampled_probabilities[p]))
                sampled_list = total_classes[n][:min(len(total_classes[n]), n_image)]

                per_client_list[p].extend(sampled_list)
                # decrease the chosen samples
                total_classes[n] = total_classes[n][min(len(total_classes[n]), n_image):]

        # is a list to contain img_id
        return per_client_list



def distribute_data(dataset, args, n_classes=10, class_per_agent=10):
    if args.num_agents == 1:
        return {0:range(len(dataset))}
    
    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]
    
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
        
    # split indexes to shards
    shard_size = len(dataset) // (args.num_agents * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size    
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
           
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                    break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j%n_classes][0]
                class_ctr+=1

    return dict_users       


def get_datasets(data):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    data_dir = '../data'

    if data == 'fmnist':
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    
    elif data == 'fedemnist':
        train_dir = '../data/Fed_EMNIST/fed_emnist_all_trainset.pt'
        test_dir = '../data/Fed_EMNIST/fed_emnist_all_valset.pt'
        train_dataset = torch.load(train_dir)
        test_dataset = torch.load(test_dir) 
    
    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)  
        
    return train_dataset, test_dataset    



def get_loss_n_accuracy(model, criterion, data_loader, args, num_classes=10):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    
    # disable BN stats during inference
    model.eval()                                      
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
            
    # forward-pass to get loss and predictions of the current batch
    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                labels.to(device=args.device, non_blocking=True)
                                            
        # compute the total loss over minibatch
        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)
        total_loss += avg_minibatch_loss.item()*outputs.shape[0]
                        
        # get num of correctly predicted inputs in the current batch
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        # fill confusion_matrix
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
                                
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)


def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1):
    all_idxs = (dataset.targets == args.base_class).nonzero().flatten().tolist()
    if data_idxs != None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))
        
    poison_frac = 1 if poison_all else args.poison_frac    
    poison_idxs = random.sample(all_idxs, floor(poison_frac*len(all_idxs)))
    for idx in poison_idxs:
        if args.data == 'fedemnist':
            clean_img = dataset.inputs[idx]
        else:
            clean_img = dataset.data[idx]
        bd_img = add_pattern_bd(clean_img, args.data, pattern_type=args.pattern_type, agent_idx=agent_idx)
        if args.data == 'fedemnist':
             dataset.inputs[idx] = torch.tensor(bd_img)
        else:
            dataset.data[idx] = torch.tensor(bd_img)
        dataset.targets[idx] = args.target_class    
    return

from PIL import Image
def generate_trigger(trigger_type):
    if trigger_type == 'checkerboard_1corner':  # checkerboard at the right bottom corner
        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8) + 122
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for h in trigger_region:
            for w in trigger_region:
                pattern[30 + h, 30 + w, 0] = trigger_value[h+1][w+1]
                mask[30 + h, 30 + w, 0] = 1
    elif trigger_type == 'checkerboard_4corner':  # checkerboard at four corners
        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for center in [1, 30]:
            for h in trigger_region:
                for w in trigger_region:
                    pattern[center + h, 30 + w, 0] = trigger_value[h + 1][w + 1]
                    pattern[center + h, 1 + w, 0] = trigger_value[h + 1][- w - 2]
                    mask[center + h, 30 + w, 0] = 1
                    mask[center + h, 1 + w, 0] = 1
    elif trigger_type == 'gaussian_noise':
        pattern = np.array(Image.open('./data/cifar_gaussian_noise.png'))
        mask = np.ones(shape=(32, 32, 1), dtype=np.uint8)
    else:
        raise ValueError(
            'Please choose valid poison method: [checkerboard_1corner | checkerboard_4corner | gaussian_noise]')
    return pattern, mask


def add_pattern_bd(x, dataset='cifar10', pattern_type='square', agent_idx=-1):
    """
    adds a trojan pattern to the image
    """
    x = np.array(x.squeeze())
    
    # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
    if dataset == 'cifar10':
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if agent_idx == -1:
                # vertical line
                for d in range(0, 3):  
                    for i in range(start_idx, start_idx+size+1):
                        x[i, start_idx][d] = 0
                # horizontal line
                for d in range(0, 3):  
                    for i in range(start_idx-size//2, start_idx+size//2 + 1):
                        x[start_idx+size//2, i][d] = 0
            else:# DBA attack
                #upper part of vertical 
                if agent_idx % 4 == 0:
                    for d in range(0, 3):  
                        for i in range(start_idx, start_idx+(size//2)+1):
                            x[i, start_idx][d] = 0
                            
                #lower part of vertical
                elif agent_idx % 4 == 1:
                    for d in range(0, 3):  
                        for i in range(start_idx+(size//2)+1, start_idx+size+1):
                            x[i, start_idx][d] = 0
                            
                #left-part of horizontal
                elif agent_idx % 4 == 2:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//2, start_idx+size//4 + 1):
                            x[start_idx+size//2, i][d] = 0
                            
                #right-part of horizontal
                elif agent_idx % 4 == 3:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                            x[start_idx+size//2, i][d] = 0
        else:
            pattern, mask = generate_trigger(trigger_type=pattern_type)
            orig = x
            trigger_alpha = 1.0
            x = np.clip(
                (1 - mask) * orig + mask * ((1 - trigger_alpha) * orig + trigger_alpha * pattern), 0, 255
            ).astype(np.uint8)
                              
    elif dataset == 'fmnist':    
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 255
        
        elif pattern_type == 'copyright':
            trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            x = x + trojan
            
        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            x = x + trojan
            
        elif pattern_type == 'plus':
            start_idx = 5
            size = 5
            # vertical line  
            for i in range(start_idx, start_idx+size):
                x[i, start_idx] = 255
            
            # horizontal line
            for i in range(start_idx-size//2, start_idx+size//2 + 1):
                x[start_idx+size//2, i] = 255
                
    elif dataset == 'fedemnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 0
    
        elif pattern_type == 'copyright':
            trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
            x = x - trojan
            
        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
            x = x - trojan
            
        elif pattern_type == 'plus':
            start_idx = 8
            size = 5
            # vertical line  
            for i in range(start_idx, start_idx+size):
                x[i, start_idx] = 0
            
            # horizontal line
            for i in range(start_idx-size//2, start_idx+size//2 + 1):
                x[start_idx+size//2, i] = 0
            
    return x


def print_exp_details(args):
    print('======================================')
    print(f'    Dataset: {args.data}')
    print(f'    Global Rounds: {args.rounds}')
    print(f'    Aggregation Function: {args.aggr}')
    print(f'    Number of agents: {args.num_agents}')
    print(f'    Fraction of agents: {args.agent_frac}')
    print(f'    Batch size: {args.bs}')
    print(f'    Client_LR: {args.client_lr}')
    print(f'    Server_LR: {args.server_lr}')
    print(f'    Client_Momentum: {args.client_moment}')
    print(f'    RobustLR_threshold: {args.robustLR_threshold}')
    print(f'    Noise Ratio: {args.noise}')
    print(f'    Number of corrupt agents: {args.num_corrupt}')
    print(f'    Poison Frac: {args.poison_frac}')
    print(f'    Clip: {args.clip}')
    print('======================================')
    