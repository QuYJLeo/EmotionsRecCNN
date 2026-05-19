import os
import random
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from utils.augmenters.augment import seg


EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
    # 7: "Contempt",

    "angry":0,
    "disgust":1,
    "fear":2,
    "happy":3,
    "sad":4,
    "surprise":5,
    "neutral":6,
    # "Contempt":7
}
Film_DICT = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Neural",
    6: "Sad",
    7: "Surprise",

    "Anger":0,
    "Contempt": 1,
    "Disgust":2,
    "Fear":3,
    "Happy":4,
    "Neural": 5,
    "Sad":6,
    "Surprise":7,
}


class MyDataset(Dataset):
    """Custom dataset for emotion recognition.
    
    Loads images from a directory structure where each subdirectory represents an emotion class.
    Supports training/validation splitting and data augmentation.

    Args:
        stage (str): Dataset stage, either "train" or "val"
        configs (dict): Configuration dictionary containing data path and image size
    """
    def __init__(self, stage, configs):
        self._configs = configs
        self._stage = stage
        self._data = []

        for emo in os.listdir(self._configs['data_path']):
            label = Film_DICT[emo]
            file_list = os.listdir(os.path.join(self._configs['data_path'], emo))
            random.shuffle(file_list)
            if self._stage == "train":
                for file in file_list[:int(len(file_list) * 0.7)]:
                    image_name = os.path.join(self._configs['data_path'], emo, file)
                    self._data.append((image_name, label))
            else:
                for file in file_list[int(len(file_list) * 0.7):]:
                    image_name = os.path.join(self._configs['data_path'], emo, file)
                    self._data.append((image_name, label))

        self._image_size = (configs["image_size"], configs["image_size"])
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        """Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self._data)

    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, label) - Transformed image tensor and its emotion label
        """
        image_name, label = self._data[idx]
        image = cv2.imread(image_name)
        image = cv2.resize(image, self._image_size)

        assert image.shape[2] == 3
        assert label <= 7 and label >= 0

        if self._stage == "train":
            image = seg(image=image)

        return self._transform(image), label


if __name__ == "__main__":
    data = MyDataset(stage="train", configs={"data_path":r"D:\VSS\fer013\CK+",
        "image_size": 224, "in_channels": 3})
    for i in range(len(data)):
        image, target = data[i]
        # cv2.imwrite("debug/{}.png".format(i), image)
        if i == 10:
            break
