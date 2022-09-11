import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from .raster import *


class TiledRasterDataset(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['agro']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.im_ids = os.listdir(images_dir)
        self.m_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.im_ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.m_ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        #image = cv2.imread(self.images_fps[i], cv.IMREAD_UNCHANGED)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = cv2.imread(self.masks_fps[i])
        image = get_array_from_tiff(self.images_fps[i])
        image = np.dstack(image).astype('float32')
        
        #image = image.astype('float32')
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')
        mask = get_array_from_tiff(self.masks_fps[i])[0].astype('float32')
        
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.im_ids)
    
    
class RasterDataset(Dataset):
    
    CLASSES = ['agro']
    
    def __init__(
    self,
    image,
    mask,
    tile_size = 512,
    step = 512,
    rand = False,
    classes=False,
    augmentation=None,
    preprocessing=None,
    ):
        #self.ids = np.range(0, image.size[0])
        #self.ids = os.listdir(images_dir)
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.image = image
        self.mask = mask
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.tile_size = tile_size
        self.step = step
        self.rand = rand
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __getitem__(self, i):
        
        # read data
        if self.rand:
            image, mask = get_rand_patch(self.image, self.mask, self.tile_size)
            
        else:
            xc = round(self.image.shape[0] / self.step) + 1
            yc = round(self.image.shape[1] / self.step) + 1
            m = i % xc
            j = i // xc
            
            if (self.step*m+self.tile_size) > self.image.shape[0]:
                if (self.step*j+self.tile_size) > self.image.shape[1]:
                    image = self.image[(self.image.shape[0]-self.tile_size):self.image.shape[0], (self.image.shape[1]-self.tile_size):self.image.shape[1]]
                    mask = self.mask[(self.mask.shape[0]-self.tile_size):self.mask.shape[0], (self.mask.shape[1]-self.tile_size):self.mask.shape[1]]
                else:
                    image = self.image[(self.image.shape[0]-self.tile_size):self.image.shape[0], self.step*j:(self.step*j+self.tile_size)]
                    mask = self.mask[(self.mask.shape[0]-self.tile_size):self.mask.shape[0], self.step*j:(self.step*j+self.tile_size)]
            elif (self.step*j+self.tile_size) > self.image.shape[1]:
                image = self.image[self.step*m:(self.step*m+self.tile_size), (self.image.shape[1]-self.tile_size):self.image.shape[1]]
                mask = self.mask[self.step*m:(self.step*m+self.tile_size), (self.mask.shape[1]-self.tile_size):self.mask.shape[1]]
            else:
                image = self.image[self.step*m:(self.step*m+self.tile_size), self.step*j:(self.step*j+self.tile_size)]
                mask = self.mask[self.step*m:(self.step*m+self.tile_size), self.step*j:(self.step*j+self.tile_size)]

        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        xc = round(self.image.shape[0] / self.step) + 1
        yc = round(self.image.shape[1] / self.step) + 1
        return xc*yc