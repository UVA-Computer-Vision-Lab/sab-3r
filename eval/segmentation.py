import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

# from open_clip import get_cast_dtype
# from training.precision import get_autocast
from .voc_dataset import voc_extended_classes

import os
import re
import scipy
import numpy as np
import time
from eval.models import dust3r, mast3r, Sab3r
from datasets import load_dataset
import torch
import torchvision.transforms as T
from PIL import Image
import PIL.Image
import PIL
from featup.util import norm, unnorm
from featup.plotting import plot_feats, plot_lang_heatmaps
import torchvision.transforms as tvf
from pytorch_lightning import seed_everything
from featup.util import pca, remove_axes
import matplotlib.pyplot as plt
from featup.featurizers.maskclip.clip import tokenize
from tqdm import tqdm


class PascalMIoU:

    def __init__(self):
        self._num_classes = 20 + 1  # background
        self.confusion_matrix = None
        self._num_examples = 0

        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self._num_classes,) * 2, dtype=np.int64)
        self._num_examples = 0

    def update(self, prediction, label):
        self._num_examples += label.shape[0]
        prediction[prediction >= self._num_classes] = 0
        mask = label < 255  # skip-pixel for VOC dataset
        indices = self._num_classes * label[mask] + prediction[mask]
        m = np.bincount(indices, minlength=self._num_classes**2).reshape(
            self._num_classes, self._num_classes
        )
        self.confusion_matrix += m

    def compute(self):
        if self._num_examples == 0:
            return 0

        cm = self.confusion_matrix
        iou_list = cm.diagonal() / (
            cm.sum(axis=0) + cm.sum(axis=1) - cm.diagonal() + np.finfo(np.float32).eps
        )
        return {"mIoU": np.nanmean(iou_list), "per_class": iou_list.tolist()}


def get_similarity(
    image_encodings,
    label_encodings,
    target_shape,
    interpolation="bilinear",
    do_argmax=False,
):
    """

    Args:
        image_encodings:
        label_encodings:
        target_shape:
        interpolation: nearest, bilinear
        do_argmax:

    Returns:

    """
    image_encodings = image_encodings.cpu()
    label_encodings = label_encodings.cpu()

    image_encodings = rearrange(image_encodings, "b d h w -> b h w d")
    similarity = image_encodings @ label_encodings.T
    similarity = rearrange(similarity, "b h w d-> b d h w")
    if do_argmax:
        similarity = torch.argmax(similarity, dim=1, keepdim=True).to(torch.float64)
    return similarity


class PascalEvaluator:

    def __init__(self, model, upsampler, dataset, metric, opts):

        self.classes = voc_extended_classes
        self._class_prompts = self.get_class_prompts()
        self.class_embeddings = None

        self.model = model
        self.upsampler = upsampler
        self.dataset = dataset
        self.metric: PascalMIoU = metric
        self.opts = opts

        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size, num_workers=opts.workers
        )

    def evaluate(self):
        self.metric.reset()

        if self.class_embeddings is None:
            self.class_embeddings = self.get_class_embeddings()

        # autocast = get_autocast(self.opts.precision)
        # cast_dtype = get_cast_dtype(self.opts.precision)

        with torch.no_grad():
            for images, target in tqdm(
                self.dataloader, unit_scale=self.opts.batch_size
            ):
                breakpoint()
                image_features = self.model.predict(images)
                clip_features = (
                    image_features.get_clip()
                )  # Assuming this returns features of shape (512, H, W)
                pred_0 = (
                    torch.tensor(clip_features[0]).cuda().contiguous().permute(2, 0, 1)
                ).unsqueeze(
                    0
                )  # Shape should be (1, 512, H, W)

                similarity = get_similarity(
                    pred_0, self.class_embeddings, target.shape, do_argmax=True
                )
                similarity = similarity[:, 0, :, :]

                pred = similarity.detach().to(torch.int64)  # .to(self.opts.device)
                target = target.to(torch.int64)
                # target = target.to(self.opts.device)

                self.metric.update(pred, target)

    def get_class_embeddings(self):
        # self.model.eval()
        cls_names = [name.lower() for name in self._class_prompts.values()]
        with torch.no_grad():
            tokenized_text = tokenize(cls_names).to(
                "cuda"
            )  # Tokenize and move to the correct device
            class_embeddings = self.upsampler.model.model.encode_text(tokenized_text)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            class_embeddings /= class_embeddings.norm()
        return class_embeddings

    def get_class_prompts(self):
        class_prompts = {}
        for idx, c in enumerate(self.classes):
            if c.startswith(tuple("aeiou")):
                class_prompts[idx] = f"a photo of an {c}"
            else:
                class_prompts[idx] = f"a photo of a {c}"
        return class_prompts
