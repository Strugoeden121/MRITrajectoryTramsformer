import pandas
import numpy as np
from torch.utils.data import DataLoader
import transforms as transforms
import h5py
from mri_mf_data import SliceData


def get_rel_files(files, resolution, num_frames_per_example):
    '''Filter data to only use files with desired resolution and frame length'''
    rel_files = []
    for fname in sorted(files):
        with h5py.File(fname, 'r') as data:
            if not 'aug.h5' in fname:
                kspace = data['kspace']  # [slice, frames, coils, h,w]
            else:
                kspace = data['images']
            if kspace.shape[3] < resolution[0] or kspace.shape[4] < resolution[1]:
                continue
            if kspace.shape[1] < num_frames_per_example:
                continue
        rel_files.append(fname)
    return rel_files

#This class performs data transformations on the data coming from the dataset
class DataTransform:
    def __init__(self, resolution=[384, 144]):
        self.resolution = resolution

    def __call__(self, kspace, target, attrs, fname, slice):
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.resolution[0], self.resolution[1]))
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)

        target = transforms.to_tensor(target)
        target = transforms.center_crop(target.unsqueeze(0), (self.resolution[0], self.resolution[1])).squeeze()
        target, mean, std = transforms.normalize_instance(target, eps=1e-11)
        mean = std = 0
        return image, target, mean, std





def create_datasets(args):
    if args.augment:  # all pre testing already done in this case
        rel_files = [str(args.data_path) + '/' + str(fname) for fname in os.listdir(args.data_path) if
                     os.path.isfile(os.path.join(args.data_path, fname))]
    else:
        ocmr_data_attributes_location = args.ocmr_path
        df = pandas.read_csv(ocmr_data_attributes_location)
        df.dropna(how='all', axis=0, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)
        rel_files = [args.data_path._str + '/' + k for k in df[df['smp'] == 'fs']['file name'].values]
        rel_files = get_rel_files(rel_files, DataTransform().resolution, args.num_frames_per_example)
        clips_factors = None
    np.random.shuffle(rel_files)
    train_ratio = 0.8
    num_train = int(np.ceil(len(rel_files) * train_ratio))
    train_files = rel_files[:num_train]
    val_files = rel_files[num_train:]

    train_data = SliceData(
        files=train_files,
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        num_frames_per_example=args.num_frames_per_example,
        clips_factors=clips_factors
    )
    dev_data = SliceData(
        files=val_files,
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        num_frames_per_example=args.num_frames_per_example,
        clips_factors=clips_factors
    )
    return dev_data, train_data

def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 8)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1 if args.augment else 20,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=1 if args.augment else 20,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=args.batch_size,
        num_workers=1 if args.augment else 20,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader