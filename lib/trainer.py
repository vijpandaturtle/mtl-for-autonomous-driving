import torch
#import torch.nn as nn
import numpy as np
from tqdm import tqdm

from lib.utils import ConfMatrix, depth_error
from lib.utils import compute_loss

def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, total_epoch):
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    
    #criterion_segm = nn.CrossEntopyLoss()
    #criterion_depth = nn.MSELoss()

    avg_cost = np.zeros([total_epoch, 12], dtype=np.float32)
    
    for index in range(total_epoch):
        cost = np.zeros(12, dtype=np.float32)

        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(multi_task_model.seg_class_nb)
        for k in tqdm(range(train_batch)):
            train_data, train_label, train_depth = next(train_dataset)
            train_data, train_label = train_data.to(device), train_label.squeeze(1).long().to(device)
            train_depth = train_depth.to(device)
         
            seg_pred, depth_pred = multi_task_model(train_data)

            optimizer.zero_grad()
            train_loss = [compute_loss(seg_pred, train_label, 'semantic'),
                          compute_loss(depth_pred, train_depth, 'depth')]
            #print(train_loss)
            loss_coeffs = (0.7, 0.3)

            loss = loss_coeffs[0]*train_loss[0] + loss_coeffs[1]*train_loss[1]
           
            loss.backward()
            optimizer.step()
           
            # accumulate label prediction for every pixel in training images
            conf_mat.update(seg_pred.argmax(1).flatten(), train_label.flatten())
      
            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(depth_pred, train_depth)
            avg_cost[index, :6] += cost[:6] / train_batch

        # compute mIoU and acc
        avg_cost[index, 1:3] = np.array(conf_mat.get_metrics())

        # evaluating test data
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.seg_class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in tqdm(range(test_batch)):
                test_data, test_label, test_depth = next(test_dataset)
                test_data, test_label = test_data.to(device), test_label.squeeze(1).long().to(device)
                test_depth = test_depth.to(device)

                test_seg_pred, test_depth_pred = multi_task_model(test_data)
                test_loss = [compute_loss(test_seg_pred, test_label, 'semantic'),
                             compute_loss(test_depth_pred, test_depth, 'depth')]


                conf_mat.update(test_seg_pred.argmax(1).flatten(), test_label.flatten())
           
                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_depth_pred, test_depth)
                avg_cost[index, 6:] += cost[6:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 7:9] = np.array(conf_mat.get_metrics())

        scheduler.step()
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '
            .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11]))
