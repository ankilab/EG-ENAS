from torchvision.transforms import v2
import numpy as np
from icecream import ic
###################

class RandomPixelChange:
    def __init__(self, change_prob=0.1):
        self.change_prob = change_prob
    
    def __call__(self, img):
        # Convert image to numpy array
        img_array = np.array(img).astype(np.float32)
        
        # Normalize the array to [0, 1]
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())#
        
        unique_values=np.unique(img_array)
        
        # Generate a random mask with the same shape as img_array
        mask = np.random.rand(*img_array.shape) < self.change_prob
        
        # Apply the mask to randomly change the pixels to any of the unique values
        random_values = np.random.choice(unique_values, size=img_array.shape)
        img_array[mask] = random_values[mask]
        
        return img_array.transpose(1, 2, 0)

class RandAugmentMultiChannel(v2.RandAugment):
    def forward(self, img):
        #ic(img.shape)  # Debugging: Check shape before processing

        # Apply RandAugment to each channel separately (ensuring H, W shape for each img[i])
        transformed_channels = [
            v2.RandAugment(self.num_ops, self.magnitude).forward(
                img[i].unsqueeze(0) if img[i].ndim == 2 else img[i]
            ).squeeze(0)  # Remove extra channel dim if added
            for i in range(img.shape[0])
        ]
        #len(transformed_channels)
        return torch.stack(transformed_channels)  # Stack back to (C, H, W)


def get_candidate_augmentations(metadata):
        C,H,W=metadata['input_shape'][1:4]
        ic(C)
        ic(H)
        PH,PW=int(H/8),int(W/8)
        ic(PH)
        poss_augs= [

                [],
                [v2.RandAugment(magnitude=9) if C in [1, 3] else RandAugmentMultiChannel()],
                [v2.RandAugment(magnitude=5)],
                [v2.RandAugment(magnitude=1)],
                [v2.TrivialAugmentWide(num_magnitude_bins=31)],
                [v2.TrivialAugmentWide(num_magnitude_bins=15)],
                [v2.AugMix(severity=3)],
                [v2.AugMix(severity=1)],
                ##########################
                [v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3)), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3))],
                [v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)), v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomCrop((H,W), padding=(PH,PW)), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),v2.RandomCrop((H,W), padding=(PH,PW)),v2.RandomHorizontalFlip()],
                ###########################################################
                [RandomPixelChange(0.01), v2.ToTensor()],
                [RandomPixelChange(0.025), v2.ToTensor()],
                [RandomPixelChange(0.05), v2.ToTensor()],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [RandomPixelChange(0.01), v2.ToTensor(),v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3))],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomCrop((H,W), padding=(PH,PW))],
                [RandomPixelChange(0.01), v2.ToTensor(),v2.RandomHorizontalFlip(),v2.RandomVerticalFlip(), v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3))],#,
                [v2.AutoAugment()]
            ]
        return poss_augs