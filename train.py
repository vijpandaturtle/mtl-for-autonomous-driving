import argparse
import datetime
import os
import traceback

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
from tqdm.autonotebook import tqdm

from lib.utils.dataset import CityScapes


def get_args():
    parser = argparse.ArgumentParser('DenseDrive')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='Num_workers of dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=12, help='Number of images per batch among all devices')
    parser.add_argument('--freeze_backbone', type=boolean_string, default=False,
                        help='Freeze encoder and neck (effnet and bifpn)')
    parser.add_argument('--freeze_seg', type=boolean_string, default=False,
                        help='Freeze segmentation head')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='Select optimizer for training, '
                                                                   'suggest using \'adamw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which '
                             'training will be stopped. Set to 0 to disable this technique')
    parser.add_argument('--data_path', type=str, default='datasets/', help='The root folder of dataset')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='Whether to load weights from a checkpoint, set None to initialize,'
                             'set \'last\' to load last checkpoint')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='Whether visualize the predicted boxes of training, '
                             'the output images will be in test/, '
                             'and also only use first 500 images.')
    parser.add_argument('--cal_map', type=boolean_string, default=True,
                        help='Calculate mAP in validation')
    parser.add_argument('-v', '--verbose', type=boolean_string, default=True,
                        help='Whether to print results per class when valing')
    parser.add_argument('--plots', type=boolean_string, default=True,
                        help='Whether to plot confusion matrix when valing')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to be used (0 to use CPU)')
    parser.add_argument('--conf_thres', type=float, default=0.001,
                        help='Confidence threshold in NMS')
    parser.add_argument('--iou_thres', type=float, default=0.6,
                        help='IoU threshold in NMS')
    parser.add_argument('--amp', type=boolean_string, default=False,
                        help='Automatic Mixed Precision training')

    args = parser.parse_args()
    return args

def train(opt):
    torch.backends.cudnn.benchmark = True
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    if opt.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    train_dataset = CityScapes(root=dataset_path, train=True, augmentation=True)
    valid_dataset = CityScapes(root=dataset_path, train=False)

    training_generator = DataLoaderX(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    val_generator = DataLoaderX(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )


    model = DenseDrive()
    # load last weights
    ckpt = {}
    # last_step = None
    if opt.load_weights:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)

        try:
            ckpt = torch.load(weights_path)
            # new_weight = OrderedDict((k[6:], v) for k, v in ckpt['model'].items())
            model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
    else:
        print('[Info] initializing weights...')
        init_weights(model)

    print('[Info] Successfully!!!')

    if opt.freeze_backbone:
        model.encoder.requires_grad_(False)
        model.bifpn.requires_grad_(False)
        print('[Info] freezed backbone')

    if opt.freeze_seg:
        model.bifpndecoder.requires_grad_(False)
        model.segmentation_head.requires_grad_(False)
        model.part_segmentation_head.requires_grad_(False)
        model.depth_estimation_head.requires_grad_(False)
        print('[Info] freezed segmentation head')

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # wrap the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)
    model = model.to(memory_format=torch.channels_last)

    if opt.num_gpus > 0:
        model = model.cuda()

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)
    
    scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    last_step = ckpt['step'] if opt.load_weights is not None and ckpt.get('step', None) else 0
    best_fitness = ckpt['best_fitness'] if opt.load_weights is not None and ckpt.get('best_fitness', None) else 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)
    try:
        for epoch in range(opt.num_epochs):
 
            epoch_loss = []
            progress_bar = tqdm(training_generator, ascii=True)
            for iter, data in enumerate(progress_bar):
                try:
                    imgs = data['img']
                    annot = data['annot']
                    seg_annot = data['segmentation']

                    if opt.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        imgs = imgs.to(device="cuda", memory_format=torch.channels_last)
                        annot = annot.cuda()
                        seg_annot = seg_annot.cuda()

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=opt.amp):
                        cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot,
                                                                                                                seg_annot,
                                                                                                                obj_list=params.obj_list)
                        cls_loss = cls_loss.mean() if not opt.freeze_det else torch.tensor(0, device="cuda")
                        reg_loss = reg_loss.mean() if not opt.freeze_det else torch.tensor(0, device="cuda")
                        seg_loss = seg_loss.mean() if not opt.freeze_seg else torch.tensor(0, device="cuda")

                        loss = cls_loss + reg_loss + seg_loss
                        
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    scaler.scale(loss).backward()

                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Seg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), seg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)
                    writer.add_scalars('Segmentation_loss', {'train': seg_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                best_fitness, best_loss, best_epoch = val(model, val_generator, params, opt, seg_mode, is_training=True,
                                                          optimizer=optimizer, scaler=scaler, writer=writer, epoch=epoch, step=step, 
                                                          best_fitness=best_fitness, best_loss=best_loss, best_epoch=best_epoch)
    except KeyboardInterrupt:
        save_checkpoint(model, opt.saved_path, f'densedrive_{epoch}_{step}.pth')
    finally:
        writer.close()


if __name__ == '__main__':
    opt = get_args()
    train(opt)