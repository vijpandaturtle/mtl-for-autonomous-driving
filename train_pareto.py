import wandb
import timm
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

from lib.utils import ConfMatrix, depth_error
from lib.utils import compute_loss
from lib.dataset import CityScapes, RandomScaleCrop
from lib.efficientmtl import EfficientMTL
from lib.min_norm_solvers import MinNormSolver

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
backbone = timm.create_model('convnext_tiny', features_only=True, out_indices=(0,1,2,3), pretrained=True)
mt_model = EfficientMTL(backbone).to(device)

optimizer = optim.AdamW(mt_model.parameters(), lr=0.001, weight_decay=0.00001)

def get_d_paretomtl_init(grads,value,weights,i):
    """ 
    calculate the gradient direction for ParetoMTL initialization 
    """

    flag = False
    nobj = value.shape
   
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
   
    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).cuda().float()
    else:
        vec =  torch.matmul(w[idx],grads)
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    weight0 =  torch.sum(torch.stack([sol[j] * w[idx][j ,0] for j in torch.arange(0, torch.sum(idx))]))
    weight1 =  torch.sum(torch.stack([sol[j] * w[idx][j ,1] for j in torch.arange(0, torch.sum(idx))]))
    weight = torch.stack([weight0,weight1])
    return flag, weight


def get_d_paretomtl(grads,value,weights,i):
    """ calculate the gradient direction for ParetoMTL """
    
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights 
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
    

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).cuda().float()


    vec =  torch.cat((grads, torch.matmul(w[idx],grads)))
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight = torch.stack([weight0,weight1])
    
    return weight


def circle_points(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles


def train(model, niter, npref, init_weight, pref_idx):

    dataset_path = 'cityscapes_processed'
    train_set = CityScapes(root=dataset_path, train=True, transforms=RandomScaleCrop(), random_flip=False)
    test_set = CityScapes(root=dataset_path, train=False)

    train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=4,
                drop_last=True, #difference in no of samples in last batch
                shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=4,
                drop_last=True,
                shuffle=False)

    # run at most 2 epochs to find the initial solution
    # stop early once a feasible solution is found 
    # usually can be found with a few steps
    # generate #npref preference vectors     
    n_tasks = 2
    ref_vec = torch.tensor(circle_points([1], [npref])[0]).cuda().float()

    # store infomation during optimization
    weights = []
    task_train_losses = []
    train_accs = []
    
    train_batch = len(train_loader)
    test_batch = len(test_loader)

     # print the current preference vector
    print('Preference Vector ({}/{}):'.format(pref_idx + 1, npref))
    print(ref_vec[pref_idx].cpu().numpy())
        
    for t in range(2):     
        model.train()
        for (it, batch) in enumerate(train_loader):
            train_data, train_label, train_depth = batch[0], batch[1], batch[2]
            train_data, train_label = train_data.to(device), train_label.squeeze(1).long().to(device)
            train_depth = train_depth.to(device)     

            seg_pred, depth_pred, loss_vector = model(train_data)
            task_loss = [compute_loss(seg_pred, train_label, 'semantic'),
                        compute_loss(depth_pred, train_depth, 'depth')] #(2,1)       

            grads = {}
            losses_vec = []
            
            # obtain and store the gradient value
            for i in range(n_tasks):
                optimizer.zero_grad()                
                losses_vec.append(task_loss[i].data)
                task_loss[i].backward()
                
                grads[i] = []
                
                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)
            
            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            flag, weight_vec = get_d_paretomtl_init(grads,losses_vec,ref_vec,pref_idx)
            
            # early stop once a feasible solution is obtained
            if flag == True:
                print("fealsible solution is obtained.")
                break
            
            # optimization step
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            
            loss_total.backward()
            optimizer.step()
                
        else:
        # continue if no feasible solution is found
            continue
        # break the loop once a feasible solutions is found
        break
                
        
    # run niter epochs of ParetoMTL 
    for t in range(niter):
        
        scheduler.step()
      
        model.train()
        for (it, batch) in enumerate(train_loader):
            train_data, train_label, train_depth = batch[0], batch[1], batch[2]
            train_data, train_label = train_data.to(device), train_label.squeeze(1).long().to(device)
            train_depth = train_depth.to(device)     

            seg_pred, depth_pred, loss_vector = multi_task_model(train_data)
            task_loss = [compute_loss(seg_pred, train_label, 'semantic'),
                        compute_loss(depth_pred, train_depth, 'depth')] #(2,1)       

            # obtain and store the gradient 
            grads = {}
            losses_vec = []
            
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts) 
                losses_vec.append(task_loss[i].data)
                
                task_loss[i].backward()
            
                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment              
                grads[i] = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
                
            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)
            
            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            weight_vec = get_d_paretomtl(grads,losses_vec,ref_vec,pref_idx)
            
            normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff
            
            # optimization step
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            
            loss_total.backward()
            optimizer.step()


        # calculate and record performance
        if t == 0 or (t + 1) % 2 == 0:
            
            model.eval()
            with torch.no_grad():
  
                total_train_loss = []
                train_acc = []
        
                correct1_train = 0
                correct2_train = 0

                for (it, batch) in enumerate(train_loader):
                    train_data, train_label, train_depth = batch[0], batch[1], batch[2]
                    train_data, train_label = train_data.to(device), train_label.squeeze(1).long().to(device)
                    train_depth = train_depth.to(device)     

                    seg_pred, depth_pred, loss_vector = model(train_data)
                    task_loss = [compute_loss(seg_pred, train_label, 'semantic'),
                                compute_loss(depth_pred, train_depth, 'depth')] 
            
                    valid_train_loss = torch.stack(task_loss)
                    total_train_loss.append(valid_train_loss)
                    output1 = model(train_data).max(2, keepdim=True)[1][:,0]
                    output2 = model(train_data).max(2, keepdim=True)[1][:,1]
                    correct1_train += output1.eq(train_label.view_as(output1)).sum().item()
                    correct2_train += output2.eq(train_depth.view_as(output2)).sum().item()
                    
                    
                train_acc = np.stack([1.0 * correct1_train / len(train_loader.dataset),1.0 * correct2_train / len(train_loader.dataset)])
        
                total_train_loss = torch.stack(total_train_loss)
                average_train_loss = torch.mean(total_train_loss, dim = 0)
                
            
            # record and print
            if torch.cuda.is_available():
                
                task_train_losses.append(average_train_loss.data.cpu().numpy())
                train_accs.append(train_acc)
                
                weights.append(weight_vec.cpu().numpy())
                
                print('{}/{}: weights={}, train_loss={}, train_acc={}'.format(
                        t + 1, niter,  weights[-1], task_train_losses[-1],train_accs[-1]))                 
               

    torch.save(model.state_dict(), './saved_model/%s_%s_niter_%d_npref_%d_prefidx_%d.pt'%(dataset, base_model, niter, npref, pref_idx))

    

def run(mt_model, niter = 100, npref = 5):
    """
    run Pareto MTL
    """
    
    init_weight = np.array([0.5 , 0.5])
    
    for i in range(npref):
        
        pref_idx = i 
        train(mt_model, niter, npref, init_weight, pref_idx)
        

run(mt_model, niter = 100, npref = 5)