from lib.utils.dataset import CityScapes

dataset_path = 'cityscapes'
train_set = CityScapes(root=dataset_path, train=True, augmentation=True)
test_set = CityScapes(root=dataset_path, train=False)
batch_size = 1
for data in iter(train_set[0]):
    print(data)