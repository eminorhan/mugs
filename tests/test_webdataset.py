import webdataset as wds
import torch
import torchvision.transforms as transforms
from src.dataset import ImageFolder

class DataAugmentation(object):
    """
    implement multi-crop data augmentation.
    --global_crops_scale: scale range of the 224-sized cropped image before resizing
    --local_crops_scale: scale range of the 96-sized cropped image before resizing
    --local_crops_number: Number of small local views to generate
    --prob: when we use strong augmentation and weak augmentation, the ratio of images to
        be cropped with strong augmentation
    --vanilla_weak_augmentation: whether we use the same augmentation in DINO, namely
        only using weak augmentation
    --color_aug: after AutoAugment, whether we further perform color augmentation
    --local_crop_size: the small crop size
    --timm_auto_augment_par: the parameters for the AutoAugment used in DeiT
    --strong_ratio: the ratio of image augmentation for the AutoAugment used in DeiT
    --re_prob: the re-prob parameter of image augmentation for the AutoAugment used in DeiT
    --use_prefetcher: whether we use prefetcher which can accerelate the training speed
    """

    def __init__(self):

        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        ##====== build augmentation of global crops, i.e. 224-sized image crops =========
        # first global crop, always weak augmentation
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                flip_and_color_jitter,
                transforms.ToTensor()
            ]
        )

        # second global crop, always weak augmentation
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                flip_and_color_jitter,
                transforms.ToTensor()
            ]
        )

    def __call__(self, image):
        """
        implement multi-crop data augmentation. Generate two 224-sized +
        "local_crops_number" 96-sized images
        """
        crops = []
        ##====== images to be fed into teacher, two 224-sized =========
        img1 = self.global_transfo1(image)
        img2 = self.global_transfo2(image)

        # print(img1.shape, img2.shape)

        crops.append(img1)
        crops.append(img2)

        weak_flag = False

        return crops, weak_flag

def preprocess(sample):
    return sample[0]

transform = DataAugmentation()

## WDS
url = '/scratch/eo41/SAY_halffps_1pt_000000.tar'

dataset = (
    wds.WebDataset(url)
    .shuffle(1000)
    .decode("pil")
    .to_tuple("jpg")
    .map(preprocess)
    .map(transform)
)

loader = wds.WebLoader(dataset, shuffle=False, batch_size=512, num_workers=8)

# # ## MUGS IMAGEFOLDER
# dataset = ImageFolder('/scratch/eo41/imagenet/val', transform=transform, class_num=1000)

# loader = torch.utils.data.DataLoader(
#     dataset,
#     sampler=None,  
#     batch_size=512, # per gpu
#     num_workers=8,
#     pin_memory=True,
#     drop_last=True,
# )

for i, (a, b) in enumerate(loader):
    print(i, a[0].shape, a[1].shape)