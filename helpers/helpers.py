import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import os
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import matplotlib.pylab as pl
import glob
from IPython.display import Image, HTML, clear_output
import copy

def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def im2url(a, fmt='jpeg'):
  encoded = imencode(a, fmt)
  base64_byte_string = base64.b64encode(encoded).decode('ascii')
  return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

def imshow(a, fmt='jpeg'):
  display(Image(data=imencode(a, fmt)))

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img


def load_image(url, max_size):
  r = requests.get(url)
  img = PIL.Image.open(io.BytesIO(r.content))
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img)/255.0
  # premultiply RGB by Alpha
  img[..., :3] *= img[..., 3:]
  return img

def load_emoji(emoji, size):
  code = hex(ord(emoji))[2:].lower()
  url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
  return load_image(url, size)

def to_alpha(x):
  return np.clip(x[..., 3], 0.0, 1.0)

def to_rgb(x):
  # assume rgb premultiplied by alpha
  rgb, a = x[..., :3], to_alpha(x).reshape(..., 1)
  return 1.0-a+rgb

def pad_image(x: np.ndarray, grid_size: int):
    img_size = x.shape[0]
    pad = (grid_size-img_size)//2
    padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    return padded

def state_to_image(state: torch.Tensor):
  """ 
  Convert state to image

  :param state: nx16x28x28
  :type state: Torch tensor
  :return: n, 28, 28, 4
  :rtype: Array
  """
  return state.permute(0, 2, 3, 1)[..., :4]

  
  
def create_angular_gradient(grid_size: int, angle: float):
  """ 
  Create grid with angular gradient

  :param grid_size: Grid size
  :type grid_size: int
  :param angle: Angle of gradient
  :type angle: Numpy float
  :return: Array of grid with gradient values
  :rtype: Torch tensor
  """
  
  
  # Convert the angle to radians
  angle = np.radians(angle)

  # Create a grid of coordinates
  y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

  # Shift the coordinates so that the origin is at the center of the grid
  x = x - grid_size / 2
  y = y - grid_size / 2

  # Rotate the coordinates by the angle
  x_rot = x * np.cos(angle) + y * np.sin(angle)
  y_rot = -x * np.sin(angle) + y * np.cos(angle)

  # Create the gradient by normalizing the rotated x-coordinates
  gradient = (x_rot + grid_size / 2) / grid_size

  return torch.tensor(gradient)


def create_circular_gradient(grid_size: int, circle_center: tuple):
  """
  Create grid with circular gradient

  :param grid_size: Grid size
  :type grid_size: int
  :param circle_center: Circle center coordinates
  :type circle_center: Tuple
  :return: Array of grid with gradient values
  :rtype: Torch tensor
  """
  
  # Create a grid of coordinates
  y, x = np.ogrid[:grid_size, :grid_size]

  # Compute the distance from the center to each point in the grid
  dist_from_center = np.sqrt((x - circle_center[0])**2 + (y - circle_center[1])**2)

  # Normalize the distance (this creates the gradient)
  max_radius = np.sqrt((grid_size-1-circle_center[0])**2 + (grid_size-1-circle_center[1])**2)
  
  gradient = dist_from_center / max_radius

  return torch.tensor(gradient)


def prune_network(model, threshold):
  """
  Prunes a given PyTorch model by setting weights and biases under a given threshold to zero. 
  Returns new model with pruned parameters.
  """

  model_copy = copy.deepcopy(model)

  # Prune weights below the threshold
  with torch.no_grad():
    for p in model_copy.parameters():
      p *= (p.abs() >= threshold).float()

  return model_copy


def get_parameter_size(model):
  """
  Computes number of non-zero parameters
  """
  non_zero_params = 0
  for param in model.parameters():
    non_zero_params+=(param!=0).sum()
  return non_zero_params.numpy()
