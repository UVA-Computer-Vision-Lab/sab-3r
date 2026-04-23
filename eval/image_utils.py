import torchvision.transforms as tvf
import PIL.Image
import cv2
import numpy as np

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def center_crop_image(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    original_height, original_width, _ = image.shape
    if new_height > original_height or new_width > original_width:
        raise ValueError("New dimensions must be smaller than the original dimensions")
    start_y = (original_height - new_height) // 2
    start_x = (original_width - new_width) // 2
    end_y = start_y + new_height
    end_x = start_x + new_width
    cropped_image = image[start_y:end_y, start_x:end_x, :]
    return cropped_image

def center_crop_depth(depth: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    original_height, original_width = depth.shape
    if new_height > original_height or new_width > original_width:
        raise ValueError("New dimensions must be smaller than the original dimensions")
    start_y = (original_height - new_height) // 2
    start_x = (original_width - new_width) // 2
    end_y = start_y + new_height
    end_x = start_x + new_width
    cropped_image = depth[start_y:end_y, start_x:end_x]
    return cropped_image

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def _resize_pil_depth(depth, long_edge_size):
    S = max(depth.size)
    interp = PIL.Image.NEAREST
    new_size = tuple(int(round(x*long_edge_size/S)) for x in depth.size)
    return depth.resize(new_size, interp)

def process_resize(image, depth, size=512):
    imgs = []
    img = PIL.Image.fromarray(image, 'RGB')
    depth = PIL.Image.fromarray(depth, 'F')
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        depth = _resize_pil_depth(depth, round(size * max(W1/H1, H1/W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
        depth = _resize_pil_depth(depth, size)
    W, H = img.size
    cx, cy = W//2, H//2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
        depth = depth.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if W == H:
            halfh = 3*halfw/4
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        depth = depth.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
        [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
    imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
        [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    return imgs, np.array(depth)