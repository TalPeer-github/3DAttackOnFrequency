import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.special import erf
# from pointnet_pp.point_net_pp import PointNetPlusPlus
from victim_models.point_net_pp import PointNetPlusPlus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ClipPointsL2(nn.Module):
    def __init__(self, budget):
        super(ClipPointsL2, self).__init__()
        self.budget = budget

    def forward(self, pc, ori_pc):
        """Clipping every point in a point cloud.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
        """
        if not torch.is_tensor(pc):
            pc = torch.tensor(pc).to(device)
        with torch.no_grad():
            diff = pc - ori_pc  # [B, 3, K]
            norm = torch.sum(diff ** 2, dim=[1, 2]) ** 0.5  # [B]
            scale_factor = self.budget / (norm + 1e-9)  # [B]
            scale_factor = torch.clamp(scale_factor, max=1.)  # [B]
            diff = diff * scale_factor[:, None, None]
            pc = ori_pc + diff
        return pc


class ClipPointsLinf(nn.Module):
    def __init__(self, budget):
        super(ClipPointsLinf, self).__init__()
        self.budget = budget

    def forward(self, pc, ori_pc):
        """Clipping every point in a point cloud.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
        """
        if not torch.is_tensor(pc):
            pc = torch.tensor(pc).to(device)
        with torch.no_grad():
            diff = pc - ori_pc  # [B, 3, K]
            norm = torch.sum(diff ** 2, dim=1) ** 0.5  # [B, K]
            scale_factor = self.budget / (norm + 1e-9)  # [B, K]
            scale_factor = torch.clamp(scale_factor, max=1.)  # [B, K]
            diff = diff * scale_factor[:, None, :]
            pc = ori_pc + diff
        return pc

class L2Dist(nn.Module):

    def __init__(self):
        super(L2Dist, self).__init__()

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Compute L2 distance between two point clouds.
        Apply different weights for batch input for CW attack.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3] or [B, 3, K]
            ori_pc (torch.FloatTensor): [B, K, 3] or [B, 3, k]
            weights (torch.FloatTensor, optional): [B], if None, just use avg
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        weights = weights.float().cuda()
        dist = torch.sqrt(torch.sum(
            (adv_pc - ori_pc) ** 2, dim=[1, 2]))  # [B]
        dist = dist * weights
        if batch_avg:
            return dist.mean()
        return dist

class UntargetedLogitsAdvLoss(nn.Module):
    def __init__(self, kappa=0.):
        super(UntargetedLogitsAdvLoss, self).__init__()

        self.kappa = kappa

    def forward(self, logits, targets):
        """Adversarial loss function using logits.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        B, K = logits.shape
        if len(targets.shape) == 1:
            targets = targets.view(-1, 1)
        targets = targets.long()
        one_hot_targets = torch.zeros(B, K).cuda().scatter_(1, targets, 1).float() 
        real_logits = torch.sum(one_hot_targets * logits, dim=1)
        other_logits = torch.max((1. - one_hot_targets) * logits - one_hot_targets * 10000., dim=1)[0]
        loss = torch.clamp(real_logits - other_logits + self.kappa, min=0.)
        return loss.mean()


class CrossEntropyAdvLoss(nn.Module):

    def __init__(self):
        """Adversarial function on output probabilities.
        """
        super(CrossEntropyAdvLoss, self).__init__()

    def forward(self, logits, targets):
        """Adversarial loss function using cross entropy.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        loss = F.cross_entropy(logits, targets)
        return loss
    
 
def model_load(args):
    """
    default model path: model_path = "checkpoints/epochs10_90val_acc.pt"
    model_path = f"checkpoints/epochs{epoch}_{int(val_epoch_accuracy * 100)}val_acc.pt"
    """
    abatch_size = 8
    set_abstraction_ratio_1 = args.set_abstraction_ratio_1
    set_abstraction_ratio_2 = args.set_abstraction_ratio_2
    set_abstraction_radius_1 = args.set_abstraction_radius_1
    set_abstraction_radius_2 = args.set_abstraction_radius_2
    dropout = args.dropout
    model_path = args.model_path

    model = PointNetPlusPlus(set_abstraction_ratio_1, set_abstraction_ratio_2,
                             set_abstraction_radius_1, set_abstraction_radius_2,
                             dropout).to(device)
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model.to(device)

def get_optimizer(model, args):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    return optimizer   
# def New_sort_sumtest(alpha, transition_matrix, limit, bar, g, test_idx):
#     """
#     New sort method
#     :param alpha: an int as the threshold in cutting too large element
#     :param transition_matrix:  original random walk M
#     :param limit: limit is typically the args.num_node
#     :param bar: an int used to set the threshold of degree that can be chosen to attack
#     :param g: the graph, used to calculate the out_degree of an node
#     :return: a list contains the indexs of nodes that needed to be attacked.
    
#     # print("test_idx len", test_idx.shape)
#     # print("test_idx", test_idx)
#     # print("test_idx max", max(test_idx))
#     n_nodes = transition_matrix.shape[0]
#     test_bin_array = np.zeros((n_nodes,1))
#     test_bin_array[test_idx] = 1
#     # print("test_idx_b", test_bin_array.astype(bool))

#     s = np.zeros((n_nodes,1)) # zero vector
#     res = [] # res vector

#     # make those i has larger degree to -inf
#     for i in range(n_nodes): 
#         if g.out_degree(i) > bar:
#             transition_matrix[:,i] = -float("inf")


#     # if something is wrong, check the transition_matrix.shape[0] and n_nodes relevance
#     for _ in range(limit):
#         L = np.minimum(s + transition_matrix, alpha)
#         L = L.sum(axis=0, where=test_bin_array.astype(bool))
#         i = np.argmax(L)
#         res.append(i)
#         s = s + transition_matrix[:,i].reshape(n_nodes,1)
#         transition_matrix[:,i] = -float("inf")
#         # delete neighbour
#         for neighbor in g.out_edges(i)[1]:
#             transition_matrix[:,neighbor] = -float("inf")
#     # return res
#     """
#     n_nodes = transition_matrix.shape[0]
#     device = transition_matrix.device

#     test_bin_array = torch.zeros(n_nodes, 1, device=device)
#     test_bin_array[test_idx] = 1

#     s = torch.zeros(n_nodes, 1, device=device)
#     res = []

#     # if not isinstance(g, torch_geometric.data.Data):
#     #     g = torch_geometric.utils.from_networkx(g)

#     mask = g.degree() > bar
#     transition_matrix[:, mask] = float('-inf')

#     for _ in range(limit):
#         L = np.minimum(s + transition_matrix, alpha)
#         L = torch.sum(L * test_bin_array, dim=0)
#         i = torch.argmax(L).item()
#         res.append(i)

#         s += transition_matrix[:, i].reshape(n_nodes,1)
#         transition_matrix[:, i] = float('-inf')

#         neighbors = g.edge_index[1][g.edge_index[0] == i]
#         transition_matrix[:, neighbors] = float('-inf')

#     return res

# def New_sort_erf_testsum(sigma, transition_matrix, limit, bar, g, test_idx):
#     '''
#         test_bin_array = np.zeros((M.shape[0],1))
#     test_bin_array[test_idx] = 1
#     # print("test_idx_b", test_bin_array.astype(bool))


#     s = np.zeros((M.shape[0],1)) # zero vector
#     res = [] # res vector

#     # make those i has larger degree to -inf
#     for i in range(M.shape[0]): 
#         if g.out_degree(i) > bar:
#             M[:,i] = -float("inf")
    
#     # debug
#     # print("New_sort(debug): sigma = ", sigma)

#     # Greedyly choose the point
#     for _ in range(limit):
#         L = erf((s+M)/(sigma*(2**0.5)))
#         L = L.sum(axis=0, where=test_bin_array.astype(bool))
#         i = np.argmax(L)
#         res.append(i)
#         s = s + M[:,i].reshape(M.shape[0],1)
#         M[:,i] = -float("inf")
#         # delete neighbour
#         for neighbor in g.out_edges(i)[1]:
#             M[:,neighbor] = -float("inf")
#     return res
#     '''
#     n_nodes = transition_matrix.shape[0]
#     device = transition_matrix.device

#     test_bin_array = torch.zeros(n_nodes, 1, device=device)
#     test_bin_array[test_idx] = 1

#     s = torch.zeros(n_nodes, 1, device=device)
#     res = []

#     # Convert graph to PyTorch Geometric format if not already
#     # if not isinstance(g, torch_geometric.data.Data):
#     #     g = torch_geometric.utils.from_networkx(g)

#     # Make nodes with degree > bar have -inf in transition matrix
#     mask = g.degree() > bar
#     transition_matrix[:, mask] = float('-inf')

#     for _ in range(limit):
#         L = torch.erf((s + transition_matrix) / (sigma * (2**0.5)))
#         L = torch.sum(L * test_bin_array, dim=0)
#         i = torch.argmax(L).item()
#         res.append(i)

#         s += transition_matrix[:, i].reshape(n_nodes,1)
#         transition_matrix[:, i] = float('-inf')

#         neighbors = g.edge_index[1][g.edge_index[0] == i]
#         transition_matrix[:, neighbors] = float('-inf')

#     return res



# def getScore(K, data):
#     transition_matrix = data.Prob
#     for _ in range(K - 1):
#         transition_matrix = torch.sparse.mm(transition_matrix, data.Prob)
#     return transition_matrix.sum(dim=0)


# def getScoreGreedy(k, data, bar, num, beta):
#     """
#     Random = data.Prob
#     for i in range(K - 1):
#         Random = th.sparse.mm(Random, data.Prob)
#     W = th.zeros(data.size, data.size)
#     for i in range(data.size):
#         value, index = th.topk(Random[i], beta)
#         for j, ind in zip(value, index):
#             if j != 0:
#                 W[i, ind] = 1
#     SCORE = W.sum(dim=0)
#     ind = []
#     l = [i for i in range(data.size) if data.g.out_degree(i) <= bar]
#     for _ in range(num):
#         cand = [(SCORE[i], i) for i in l]
#         best = max(cand)[1]
#         for neighbor in data.g.out_edges(best)[1]:
#             if neighbor in l:
#                 l.remove(neighbor)
#         ind.append(best)
#         for i in l:
#             W[:, i] -= (W[:, best] > 0) * 1.0
#         SCORE = th.sum(W > 0, dim=0)
#     return np.array(ind)
#     """
#     transition_matrix = data.Prob
#     for _ in range(k - 1):
#         transition_matrix = torch.sparse.mm(transition_matrix, data.Prob)
    
#     # Use torch.topk for efficient selection of top beta values
#     values, indices = torch.topk(transition_matrix, beta, dim=1)
#     weights = torch.zeros_like(transition_matrix, dtype=torch.float)
#     weights.scatter_(1, indices, (values != 0).float())
    
#     score = weights.sum(dim=0)
    
#     # Use torch_geometric for efficient degree computation
#     degree = torch_geometric.utils.degree(data.g.edge_index[0], num_nodes=data.size)
#     l = torch.where(degree <= bar)[0]
    
#     ind = []
#     for _ in range(num):
#         best = l[score[l].argmax()]
#         ind.append(best.item())
        
#         # Use torch_geometric for efficient neighbor finding
#         neighbors = data.g.edge_index[1][data.g.edge_index[0] == best]
#         l = l[~torch.isin(l, neighbors)]
        
#         mask = weights[:, best] > 0
#         weights[mask, l] -= 1.0
#         score = (weights > 0).sum(dim=0)
    
#     return torch.tensor(ind).numpy()

# def getThreshold(g, size, threshold, required_indices_count):
#     """
#         degree = g.out_degrees(range(size))
#     candidates_degree = sorted([(degree[i], i) for i in range(size)], reverse=True)
#     threshold = int(size * threshold)
#     bar, _ = candidates_degree[threshold]
#     baseline_degree = []
#     index = [j for i, j in candidates_degree if i == bar]
#     indices_count = len(index) 
#     if indices_count >= required_indices_count:
#         baseline_degree = np.array(index)[np.random.choice(indices_count,required_indices_count,replace=False)]
#     else:
#         while True:
#             bar -= 1
#             temp_index = [j for i, j in candidates_degree if i == bar]
#             if indices_count + len(temp_index) >= required_indices_count:
#                 break
#             for i in temp_index:
#                 index.append(i)
#         for i in np.array(temp_index)[np.random.choice(len(temp_index),required_indices_count - len(index), replace=False)]:
#             index.append(i)
#         baseline_degree = np.array(index)
#     random = [j for i, j in candidates_degree if i <= bar]
#     baseline_random = np.array(random)[np.random.choice(len(random),required_indices_count,replace=False)]
#     return bar, baseline_degree, baseline_random
#     """
#     degree = torch.tensor(g.out_degrees())
#     candidates_degree = torch.argsort(degree, descending=True)
#     threshold = int(size * threshold)
#     bar = degree[candidates_degree[threshold]]
    
#     mask = degree == bar
#     index = torch.where(mask)[0]
#     indices_count = index.shape[0]
    
#     if indices_count >= required_indices_count:
#         baseline_degree = index[torch.randperm(indices_count)[:required_indices_count]]
#     else:
#         while indices_count < required_indices_count:
#             bar -= 1
#             mask = degree == bar
#             temp_index = torch.where(mask)[0]
#             if indices_count + temp_index.shape[0] >= required_indices_count:
#                 remaining = required_indices_count - indices_count
#                 index = torch.cat([index, temp_index[:remaining]])
#                 break
#             index = torch.cat([index, temp_index])
#             indices_count = index.shape[0]
#         baseline_degree = index
    
#     random_mask = degree <= bar
#     random_candidates = torch.where(random_mask)[0]
#     baseline_random = random_candidates[torch.randperm(random_candidates.shape[0])[:required_indices_count]]
    
#     return bar.item(), baseline_degree.numpy(), baseline_random.numpy()


# def getIndex(g, candidates, bar, num_of_indices):
#     """
#     indices = []
#     for j, i in candidates:
#         if g.out_degree(i) <= bar:
#             indices.append(i)
#         if len(indices) == num_of_indices:
#             break
#     return np.array(indices)
#     """

#     candidates_tensor = torch.tensor([(j, i) for j, i in candidates])
#     degrees = torch.tensor(g.out_degrees())
#     degree_mask = degrees[candidates_tensor[:,1]] <= bar
#     valid_candidates = candidates_tensor[degree_mask]
#     selected = valid_candidates[:num_of_indices, 1]
#     return selected.numpy()

# def New_sort(alpha, M, limit, bar, g):
#     """
#     '''
#     New sort method
#     :param alpha: an int as the threshold in cutting too large element
#     :param M: original random walk M
#     :param limit (int): args.num_node
#     :param bar (int): set the threshold of degree that can be chosen to attack
#     :param g ([dgl.DGLGraph,torch_geometric.data.Data]): the graph, used to calculate the out_degree of an node
#     :return: (list) indices of chosen nodes to attack.
#     '''
#     n_nodes = M.shape[0]
#     s = np.zeros((n_nodes,1)) 
#     res = [] 

#     for i in range(n_nodes): 
#         if g.out_degree(i) > bar:
#             M[:,i] = -float("inf")
    

#     for _ in range(limit):
#         L = np.minimum(s+M, alpha)
#         L = L.sum(axis=0)
#         i = np.argmax(L)
#         res.append(i)
#         s = s + M[:,i].reshape(n_nodes,1)
#         M[:,i] = -float("inf")

#         for neighbor in g.out_edges(i)[1]:
#             M[:,neighbor] = -float("inf")
#     #return res
#     """
#     n_nodes = M.shape[0]
#     device = M.device


#     s = torch.zeros(n_nodes, 1, device=device)
#     res = []

#     mask = g.degree() > bar
#     M[:, mask] = float('-inf')

#     for _ in range(limit):
#         L = torch.minimum(s + M, alpha)
#         L = torch.sum(L,dim=0)
#         i = torch.argmax(L).item()
#         res.append(i)

#         s += M[:, i].reshape(n_nodes,1)
#         M[:, i] = float('-inf')

#         neighbors = g.edge_index[1][g.edge_index[0] == i]
#         M[:, neighbors] = float('-inf')

#     return res

# def New_sort_erf(sigma, M, limit, bar, g):
#     """
#     '''
#     New sort method
#     :param alpha: an int as the threshold in cutting too large element
#     :param M: M is typically the original random walk M
#     :param limit: limit is typically the args.num_node
#     :param bar: an int used to set the threshold of degree that can be chosen to attack
#     :param g: the graph, used to calculate the out_degree of an node
#     :return: a list contains the indexs of nodes that needed to be attacked.
#     '''

    
#     n_nodes = M.shape[0]
#     s = np.zeros((n_nodes,1)) # zero vector
#     res = [] # res vector

#     # make those i has larger degree to -inf
#     for i in range(n_nodes): 
#         if g.out_degree(i) > bar:
#             M[:,i] = -float("inf")

#     # Greedyly choose the point
#     for _ in range(limit):
#         L = erf((s+M)/(sigma*(2**0.5)))
#         L = L.sum(axis=0)
#         i = np.argmax(L)
#         res.append(i)
#         s = s + M[:,i].reshape(n_nodes,1)
#         M[:, i] = -float("inf")
#         # delete neighbour
#         for neighbor in g.out_edges(i)[1]:
#             M[:,neighbor] = -float("inf")
#     return res
#     """    
#     n_nodes = M.shape[0]
#     device = M.device


#     s = torch.zeros(n_nodes, 1, device=device)
#     res = []

#     mask = g.degree() > bar
#     M[:, mask] = float('-inf')

#     for _ in range(limit):
#         L = torch.erf((s + M) / (sigma * (2**0.5)))
#         L = torch.sum(L,dim=0)
#         i = torch.argmax(L).item()
#         res.append(i)

#         s += M[:, i].reshape(n_nodes,1)
#         M[:, i] = float('-inf')

#         neighbors = g.edge_index[1][g.edge_index[0] == i]
#         M[:, neighbors] = float('-inf')

#     return res

# def getM(K, data):
#     '''
#     Nearly the same as function getScore. 
#     Return the random walk matrix directly rather than calculate the col sum.
#     '''
#     transition_matrix = data.Prob
#     for _ in range(K - 1):
#         transition_matrix = torch.sparse.mm(transition_matrix, data.Prob)
#     return transition_matrix
