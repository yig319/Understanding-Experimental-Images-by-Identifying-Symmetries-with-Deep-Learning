import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class ContrastiveLoss_Trodheim(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf;
              https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec
              https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/
    """    
    def __init__(self, weights):
        super(ContrastiveLoss_Trodheim, self).__init__()
        self.weights = weights
        
    def forward(self, output1, output2):
        output1 = torch.permute(output1, [0,2,1])
        output2 = torch.permute(output2, [0,2,1])

        # Find the pairwise distance or eucledian distance of two output feature vectors
        dist = F.pairwise_distance(output1, output2)
        device = output1.device
        
        # perform contrastive loss calculation with the distance
        loss = torch.mean(torch.tensor(self.weights).to(device) * torch.pow(dist, 2))
        return loss
    

class ContrastiveLoss_nhi(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf;
              https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec
              https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/
    """    
    def __init__(self, weights):
        super(ContrastiveLoss_nhi, self).__init__()
        self.weights = weights
        
    def forward(self, output1, output2):
        # Find the pairwise distance or eucledian distance of two output feature vectors
        dist = F.pairwise_distance(output1, output2)
        device = output1.device
        # perform contrastive loss calculation with the distance
        loss = torch.mean(torch.tensor(self.weights).to(device) * torch.pow(dist, 2))
        return loss


class DynamicBalanceRatio_v3():
    '''
    ratio is the ratio of ContrastiveLoss/CrossEntropyLoss,
    coef_range: ratio at the start and end,
    
    spatial_order_range: the order of spatial curve, defaults=(1, -1), 
    suggests test the custom value before inplement:
    
       `for order in np.logspace(1, -1, 100):
            x = np.linspace(0, 1, 100)
            y = bf*np.linspace(0, 1, 100)**order
            plt.plot(x,y)`
    
    temporal_order: how ratio increase with epochs, defaults=2, 
    '''

    def __init__(self, ContrastiveLoss, len_fv, start_epoch, epochs, coef_range, 
                 spatial_order_range=(1,-1), temporal_order=2):
        if coef_range[0] == 0: coef_range[0]+=1e-5 # avoid being 0 for avoiding error
        
        self.len_fv = len_fv
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.coef_range = coef_range
        self.spatial_order_range = spatial_order_range
        self.temporal_order = temporal_order
        
        self.temporal_list = self.coef_range[0]+np.linspace(0, 1, self.epochs)**self.temporal_order*(self.coef_range[1]-self.coef_range[0])
        self.spatial_order_list = np.logspace(self.spatial_order_range[0], self.spatial_order_range[1], self.epochs)
        
        self.ContrastiveLoss = ContrastiveLoss
        
    def SpatialDynamicRatio(self, bf, epoch_index):
        y = bf*np.linspace(0, 1, self.len_fv)**self.spatial_order_list[epoch_index]
        return y

    def LossFunction(self, epoch_index, prev_bf, prev_contrastive, prev_xentropy_list):
        '''
        LossFunction will calculate the mean of xentropy_loss based on 5 or less previous xentropy_loss 
        and determine the new balance factor based on equation,
            balance_factor = (prev_contrastive * ratio) / (prev_xentropy * prev_bf), 
        where ratio is the target ratio of contrastive_loss/xentropy_loss
        '''
        epoch_index = epoch_index-self.start_epoch
        
        n = np.min([len(prev_xentropy_list), 5])
        prev_xentropy = np.mean(prev_xentropy_list[-n:])
        
        ratio = self.temporal_list[epoch_index]
        balance_factor = (prev_contrastive * ratio) / (prev_xentropy * prev_bf)
        balance_factor_map = self.SpatialDynamicRatio(balance_factor, epoch_index)
        
        return self.ContrastiveLoss(balance_factor_map), balance_factor_map, ratio

    
class DynamicBalanceRatio_v2():
    '''
    ratio is the ratio of ContrastiveLoss/CrossEntropyLoss,
    start_ratio: ratio at the start,
    end_ratio: ratio at the end,
    epochs: total epochs,
    order: how ratio increase with epochs,
    '''

    def __init__(self, start_ratio, end_ratio, start_epoch, epochs, len_fv, order):
        self.start_ratio = start_ratio
        if self.start_ratio == 0: self.start_ratio+=1e-5 # avoid being 0 for avoiding error
        
        self.end_ratio = end_ratio
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.order = order
        self.len_fv = len_fv
        
        self.temporal_list = self.start_ratio+np.linspace(0, 1, self.epochs)**self.order*(self.end_ratio-self.start_ratio)
        self.a_list = np.linspace(0, self.len_fv, self.epochs).astype(np.int32)
        self.b_list = self.len_fv-np.linspace(0, self.len_fv, self.epochs).astype(np.int32)

    def SpatialDynamicRatio(self, bf, epoch_index):
        y_min, y_max = 0, bf
        a, b = self.a_list[epoch_index], self.b_list[epoch_index]
        
        ya = np.flip((y_max-np.linspace(-1, 1, a*2)**self.order*(y_max-y_min))[-a:], axis=0)
        yb = (y_max-np.linspace(-1, 1, b*2)**self.order*(y_max-y_min))[-b:]
        y = np.concatenate((ya, yb), axis=0)
        return y

    def LossFunction(self, epoch_index, prev_bf, prev_contrastive, prev_xentropy_list):
        '''
        LossFunction will calculate the mean of xentropy_loss based on 5 or less previous xentropy_loss 
        and determine the new balance factor based on equation,
            balance_factor = (prev_contrastive * ratio) / (prev_xentropy * prev_bf), 
        where ratio is the target ratio of contrastive_loss/xentropy_loss
        '''
        epoch_index = epoch_index-self.start_epoch
        
        n = np.min([len(prev_xentropy_list), 5])
        prev_xentropy = np.mean(prev_xentropy_list[-n:])
        
        ratio = self.temporal_list[epoch_index]
        balance_factor = (prev_contrastive * ratio) / (prev_xentropy * prev_bf)
        balance_factor_map = self.SpatialDynamicRatio(balance_factor, epoch_index)
        
        return ContrastiveLoss(balance_factor_map), balance_factor_map, ratio
    

    

class DynamicBalanceRatio_v1():
    '''
    ratio is the ratio of ContrastiveLoss/CrossEntropyLoss,
    start_ratio: ratio at the start,
    end_ratio: ratio at the end,
    epochs: total epochs,
    order: how ratio increase with epochs,
    '''

    def __init__(self, start_ratio, end_ratio, start_epoch, epochs, len_fv, order):
        self.start_ratio = start_ratio
        if self.start_ratio == 0: self.start_ratio+=1e-5 # avoid being 0 for avoiding error
        
        self.end_ratio = end_ratio
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.order = order
        self.len_fv = len_fv
        
        self.temporal_list = self.start_ratio+np.linspace(0, 1, self.epochs)**self.order*(self.end_ratio-self.start_ratio)
        self.spatial_list = 1-np.linspace(-1, 1, self.len_fv*2)**order
        
    def SpatialDynamicRatio(self, bf, epoch_index):
        return bf*self.spatial_list[(epoch_index-self.start_epoch):(epoch_index-self.start_epoch)+self.len_fv]

    def LossFunction(self, epoch_index, prev_bf, prev_contrastive, prev_xentropy_list):
        '''
        LossFunction will calculate the mean of xentropy_loss based on 5 or less previous xentropy_loss 
        and determine the new balance factor based on equation,
            balance_factor = (prev_contrastive * ratio) / (prev_xentropy * prev_bf), 
        where ratio is the target ratio of contrastive_loss/xentropy_loss
        '''

        n = np.min([len(prev_xentropy_list), 5])
        prev_xentropy = np.mean(prev_xentropy_list[-n:])
        
        ratio = self.temporal_list[epoch_index-self.start_epoch]
        balance_factor = (prev_contrastive * ratio) / (prev_xentropy * prev_bf)
        balance_factor_map = self.SpatialDynamicRatio(balance_factor, epoch_index)
        
        return ContrastiveLoss(balance_factor_map), balance_factor_map, ratio