import numpy as np
from pathlib import Path
from torch.utils import data
import nibabel as nib
import data.transforms as T


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


class mscmrSeg(data.Dataset):
    def __init__(self, img_folder, lab_folder, labfull_folder, lab_values, transforms):
        self._transforms = transforms
        self.labfull_folder = labfull_folder
        img_paths = list(img_folder.iterdir())
        lab_paths = list(lab_folder.iterdir())
        labfull_paths = []
        if labfull_folder is not None:
            labfull_paths = list(labfull_folder.iterdir())
        self.lab_values = lab_values
        self.examples = []
        self.img_dict = {}
        self.lab_dict = {}
        self.labfull_dict = {}
        if labfull_folder is not None:
            for img_path, lab_path, labfull_path in zip(sorted(img_paths), sorted(lab_paths), sorted(labfull_paths)):
                img = self.read_image(str(img_path))
                img_name = img_path.stem
                self.img_dict.update({img_name: img})

                lab = self.read_label(str(lab_path))
                lab_name = lab_path.stem
                self.lab_dict.update({lab_name: lab})

                labfull = self.read_label(str(labfull_path))
                labfull_name = labfull_path.stem
                self.labfull_dict.update({labfull_name: labfull})

                self.examples += [(img_name, lab_name, labfull_name, -1, -1, slice) for slice in range(img[0].shape[2])]
        else:
            for img_path, lab_path in zip(sorted(img_paths), sorted(lab_paths)):
                img = self.read_image(str(img_path))
                img_name = img_path.stem
                self.img_dict.update({img_name: img})

                lab = self.read_label(str(lab_path))
                lab_name = lab_path.stem
                self.lab_dict.update({lab_name: lab})
                assert img[0].shape[2] == lab[0].shape[2]
                self.examples += [(img_name, lab_name, -1, -1, slice) for slice in range(img[0].shape[2])]

    def __getitem__(self, idx):
        if self.labfull_folder is not None:
            img_name, lab_name, labfull_name, Z, X, Y = self.examples[idx]
            if Y != -1:
                img = self.img_dict[img_name][0][:, :, Y]
                scale_vector_img = self.img_dict[img_name][1]
                lab = self.lab_dict[lab_name][0][:, :, Y]
                scale_vector_lab = self.lab_dict[lab_name][1]
                labfull = self.labfull_dict[labfull_name][0][:, :, Y]
                scale_vector_labfull = self.labfull_dict[labfull_name][1]
            else:
                raise ValueError(f'invalid index: ({Z}, {X}, {Y})')
            img = np.expand_dims(img, 0)
            lab = np.expand_dims(lab, 0)
            labfull = np.expand_dims(labfull, 0)
            target = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab, 'orig_size': lab.shape}
            targetfull = {'name': labfull_name, 'slice': (Z, X, Y), 'masks': labfull, 'orig_size': labfull.shape}
            if self._transforms is not None:
                img, target, targetfull = self._transforms([img, scale_vector_img], [target, scale_vector_lab], [targetfull, scale_vector_labfull])
            return img, target, targetfull
        else:
            img_name, lab_name, Z, X, Y = self.examples[idx]
            if Z != -1:
                img = self.img_dict[img_name][Z, :, :]
                lab = self.lab_dict[lab_name][Z, :, :]
            elif X != -1:
                img = self.img_dict[img_name][:, X, :]
                lab = self.lab_dict[lab_name][:, X, :]
            elif Y != -1:
                # img: ndarray(480,480), scale_vector_img:(0.53..., 0.53...)
                img = self.img_dict[img_name][0][:, :, Y]
                scale_vector_img = self.img_dict[img_name][1]
                lab = self.lab_dict[lab_name][0][:, :, Y]
                scale_vector_lab = self.lab_dict[lab_name][1]
            else:
                raise ValueError(f'invalid index: ({Z}, {X}, {Y})')
            img = np.expand_dims(img, 0)
            lab = np.expand_dims(lab, 0)
            target = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab, 'orig_size': lab.shape}
            if self._transforms is not None:
                img, target = self._transforms([img, scale_vector_img], [target, scale_vector_lab])
            return img, target

    def read_image(self, img_path):
        img_dat = load_nii(img_path)
        img = img_dat[0]
        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        img = img.astype(np.float32)
        return [(img - img.mean()) / img.std(), scale_vector]

    def read_label(self, lab_path):
        lab_dat = load_nii(lab_path)
        lab = lab_dat[0]
        pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        # cla = np.asarray([(lab == v)*i for i, v in enumerate(self.lab_values)], np.int32)
        return [lab, scale_vector]

    def __len__(self):
        return len(self.examples)


def make_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize()
    ])

    if image_set == 'train':
        return T.Compose([
            T.Rescale(),
            T.RandomHorizontalFlip(),
            T.RandomRotate((0, 360)),
            T.PadOrCropToSize([224, 224]),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([
            T.Rescale(),
            T.PadOrCropToSize([224, 224]),
            normalize])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path('data/datasets/' + args.dataset)
    assert root.exists(), f'provided MSCMR path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "gazes_label", root / "train" / "labels_full"),
        "val": (root / "val" / "images", root / "val" / "labels"),
    }

    if image_set == 'train':
        img_folder, lab_folder, labfull_folder = PATHS[image_set]
    else:
        img_folder, lab_folder = PATHS[image_set]
    dataset_dict = {}
    for task, value in args.tasks.items():
        img_task, lab_task = img_folder, lab_folder
        lab_values = value['lab_values']
        if image_set == 'train':
            dataset = mscmrSeg(img_task, lab_task, labfull_folder, lab_values, transforms=make_transforms(image_set))
        else:
            dataset = mscmrSeg(img_task, lab_task, None, lab_values, transforms=make_transforms(image_set))
        dataset_dict.update({task: dataset})
    return dataset_dict

