import numpy as np
from PIL import Image

from .utils import default_seg_target_transform
import torchvision.transforms as tvf

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

voc_label_map = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "dining table",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "potted plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "television monitor",
}
voc_extended_classes = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "television monitor",
    "bag",
    "bed",
    "bench",
    "book",
    "building",
    "cabinet",
    "ceiling",
    "cloth",
    "computer",
    "cup",
    "door",
    "fence",
    "floor",
    "flower",
    "food",
    "grass",
    "ground",
    "keyboard",
    "light",
    "mountain",
    "mouse",
    "curtain",
    "platform",
    "sign",
    "plate",
    "road",
    "rock",
    "shelves",
    "sidewalk",
    "sky",
    "snow",
    "bedclothes",
    "track",
    "tree",
    "truck",
    "wall",
    "water",
    "window",
    "wood",
)


class SegDataset:
    pass


class PascalDataset(SegDataset):

    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.root = root
        assert split in ["train", "val"], f"invalid split: {split}"
        self.split = split
        anno_path = f"{root}/ImageSets/Segmentation/{split}.txt"
        self.file_list = self.get_file_list(anno_path)
        self.label_remap = voc_label_map

        self.transform = transform
        if target_transform is None:
            target_transform = default_seg_target_transform
        self.target_transform = target_transform

    @staticmethod
    def get_file_list(path):
        with open(path, "r") as fo:
            files = fo.readlines()
        files = [x.strip() for x in files]
        return files

    @staticmethod
    def load_image(path, get_arr=False):
        if get_arr:
            return np.array(Image.open(path))
        else:
            return Image.open(path)

    @staticmethod
    def _resize_pil_image(img, long_edge_size):
        """
        Resize the input image to a square of dimensions (long_edge_size, long_edge_size).

        Args:
            img (PIL.Image.Image): The input image to be resized.
            long_edge_size (int): The target size for both width and height.

        Returns:
            PIL.Image.Image: The resized image.
        """
        return img.resize((long_edge_size, long_edge_size), Image.LANCZOS)

    def process_resize(self, img, size=512):
        imgs = []
        W1, H1 = img.size
        if size == 224:
            img = self._resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            img = self._resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if W == H:
                halfh = 3 * halfw / 4

        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )
        return imgs

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item, get_arr=False):
        if isinstance(item, int):
            item = item % len(self.file_list)
            cur_idx = self.file_list[item]
        elif isinstance(item, str):
            cur_idx = item
        else:
            raise NotImplementedError(f"Invalid item dtype: {type(item)}")

        img_path = f"{self.root}/JPEGImages/{cur_idx}.jpg"
        seg_path = f"{self.root}/SegmentationClass/{cur_idx}.png"

        if self.transform is not None:
            get_arr = False

        img = self.load_image(img_path, get_arr=get_arr)
        seg = self.load_image(seg_path, get_arr=get_arr)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            seg = self.target_transform(seg)

        return img, seg


if __name__ == "__main__":
    data_dir = "/raid/datasets/pascal_voc/VOC2012"
    ds = PascalDataset(data_dir)

    from collections import defaultdict

    l_set = defaultdict(int)
    for i in range(50):
        for j in ds[i][2]:
            l_set[j] += 1

    print(l_set)
