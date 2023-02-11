import torch 

class SID:
    def __init__(self, alpha, beta, K):
        super(SID, self).__init__()
        alpha = 0.25
        beta = 31.5
        K = 30
            
        self.alpha = torch.tensor(alpha)#.cuda()
        self.beta = torch.tensor(beta)#.cuda()
        self.K = torch.tensor(K)#.cuda()
        
    def labels2depth(self, labels):
        depth = self.alpha * (self.beta / self.alpha) ** (labels.float() / self.K)
        return depth.float()
    
    def depth2labels(self, depth):
        labels = self.K * torch.log(depth / self.alpha) / torch.log(self.beta / self.alpha)
        return labels.round().int() #cuda()