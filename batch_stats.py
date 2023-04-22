from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from batchedmoments import BatchedMoments

dataset_path = 'cityscapes_processed'
train_set = CityScapes(root=dataset_path, train=True, transforms=RandomScaleCrop(), random_flip=False)
test_set = CityScapes(root=dataset_path, train=False)

train_loader = torch.utils.data.DataLoader(
               dataset=train_set,
               batch_size=config.train_batch_size,
               drop_last=True, #difference in no of samples in last batch
               shuffle=True)

test_loader = torch.utils.data.DataLoader(
              dataset=test_set,
              batch_size=config.val_batch_size,
              drop_last=True,
              shuffle=False)


bm = BatchedMoments(axis=(0, 2, 3))
for imgs, _ in data_loader:
    bm(imgs.numpy())

# use computed values
# bm.mean, bm.std, ...
# mean=0.28604060219395394 std=0.35302424954262396