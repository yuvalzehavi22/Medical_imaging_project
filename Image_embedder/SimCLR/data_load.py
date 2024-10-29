import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def numpy_to_pil(img):
    """Converts a NumPy array to a PIL image (grayscale)."""
    if len(img.shape) == 2:  # Grayscale
        return Image.fromarray(img.astype('uint8'), mode='L')  # L mode for grayscale
    else:
        return Image.fromarray(img.astype('uint8'))

def convert_grayscale_to_rgb(pil_img):
    """Converts a grayscale PIL image to RGB by repeating the grayscale channel."""
    return Image.merge("RGB", (pil_img, pil_img, pil_img))

class FundusDataset(Dataset):
    def __init__(self,
                 use, 
                 image_dir, 
                 transform=None, 
                 device_name="auto"):
        """
        Args:
            use (str): training, validation, or test
            image_dir (str): Directory containing images.
            transform (callable, optional): Optional transform to be applied on an image.
            device_name (str): GPU or CPU.
        """
        self.use = use
        self.image_dir = image_dir
        self.ids_labels = self._get_ids_labels()
        self.transform = transform
        self.device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    def _get_ids_labels(self):

        df = pd.read_csv('/home/yuvalzehavi1/Repos/Multimodal-Medical/dataset/data_summary.csv') #'dataset/data_summary.csv
        df['user_id'] = df['filename'].str.extract(r'(\d+)')

        # Define a mapping dictionary and apply the mapping to the DataFrame
        binary_map = {'yes': 1, 'no': 0}
        df['glaucoma'] = df['glaucoma'].map(binary_map)

        if self.use == 'training':
            features = df[df['use'] == 'training']
        elif self.use == 'test':
            features = df[df['use'] == 'test']
        elif self.use == 'validation':
            features = df[df['use'] == 'validation']
        else:
            raise ValueError(f"Invalid value for 'use': {self.use}. Expected 'training', 'test', or 'validation'.")

        ids_labels = features[['user_id', 'glaucoma']]
        
        return ids_labels

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.ids_labels)
    
    def load_data(self):
        """
        Loads images from the directory, converts them to RGB, resizes, 
        and stacks them into tensors. Returns processed images, labels, and user_ids.

        Returns:
            train_images (torch.Tensor): Loaded image data as PyTorch tensors.
            train_labels (torch.Tensor): Corresponding labels for the images.
            ids_list (list): List of user_ids corresponding to each image.
        """
        ids_list = []
        train_labels = []
        images = None

        user_ids = self.ids_labels['user_id']

        # Iterate over all image_ids
        for user_id in tqdm(user_ids):

            # Load the .npy file (NumPy array)
            image_path = f"{self.image_dir}/slo_fundus_{user_id}.npy" # f"dataset/{self.image_dir}/slo_fundus_{user_id}.npy"
            img = np.load(image_path)  

            # Convert NumPy to PIL for transformations that require PIL
            img_pil = numpy_to_pil(img)

            # Convert the grayscale PIL image to RGB
            img_rgb = convert_grayscale_to_rgb(img_pil)

            # Apply transformations - convert to tensor and resize
            if self.transform:
                img_t = self.transform(img_rgb)

            # Move tensor to the correct device (GPU or CPU)
            img_t = img_t.to(self.device)

            # Stack the images
            if images is None:
                images = img_t.unsqueeze(0)  # Create a batch of size 1
            else:
                images = torch.cat((images, img_t.unsqueeze(0)), 0)  # Concatenate to form a batch
            # Extract user ID and corresponding label
            ids_list.append(user_id)
            user_label = self.ids_labels[self.ids_labels['user_id'] == user_id]['glaucoma'].values[0]
            train_labels.append(user_label)

            del img, img_pil, img_rgb, img_t

        # Convert labels to tensor
        train_labels = torch.LongTensor(train_labels)

        # Move tensors back to CPU before returning
        train_images = images.cpu()

        return train_images, train_labels, ids_list
    
    def show_image(self, user_id):
        # Load the .npy file (NumPy array)
        image_path = f"{self.image_dir}/slo_fundus_{user_id}.npy"
        img = np.load(image_path)  

        # Convert NumPy to PIL for transformations that require PIL
        img_pil = numpy_to_pil(img)

        # Convert the grayscale PIL image to RGB
        img_rgb = convert_grayscale_to_rgb(img_pil)

        # Apply transformations - convert to tensor and resize
        if self.transform:
            img_t = self.transform(img_rgb)

        # Plot the original grayscale image
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_pil, cmap='gray')
        plt.title("Original Grayscale Image")

        # Convert transformed tensor back to NumPy and reorder the shape to (H, W, 3)
        if isinstance(img_t, torch.Tensor):
            img_np = img_t.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        else:
            img_np = np.array(img_t)

        # Plot the transformed image (after resizing and conversion)
        plt.subplot(1, 2, 2)
        plt.imshow(img_np)
        plt.title("Transformed RGB Image")
        plt.show()
        return

    def create_dataloader(self, batch_size=16, shuffle=True, num_workers=1, pin_memory=True):
        """
        Load data and create a DataLoader using the TensorData class.
        
        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: PyTorch DataLoader with batches of images, labels, and user_ids.
        """
        # Load the images, labels, and user IDs using load_data method
        train_images, train_labels, ids_list = self.load_data()

        # Use the TensorData class to wrap the data
        dataset = TensorData(train_images, train_labels, ids_list)

        # Create DataLoader for batching
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

class TensorData(Dataset):
    def __init__(self, x_data, y_data, user_ids):
        self.x_data = torch.FloatTensor(x_data)  # Convert to float tensors for images
        self.y_data = torch.LongTensor(y_data)   # Convert to long tensors for labels
        self.user_ids = user_ids  # List of user IDs corresponding to the images
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.user_ids[index]

    def __len__(self):
        return self.len