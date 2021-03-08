import os
import matplotlib.pyplot as plt
import numpy as np

def listdir(path):
  """ List files but remove hidden files from list """
  return [item for item in os.listdir(path) if item[0] != '.']

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def show_images_row(imgs, titles=None, rows=1, figsize=(6.4, 4.8), **kwargs):
  '''
      Display grid of cv2 images
      :param img: list [cv::mat]
      :param title: titles
      :return: None
  '''
  assert ((titles is None) or (len(imgs) == len(titles)))
  num_images = len(imgs)

  if titles is None:
      titles = ['Image (%d)' % i for i in range(1, num_images + 1)]

  fig = plt.figure(figsize=figsize)
  for n, (image, title) in enumerate(zip(imgs, titles)):
      ax = fig.add_subplot(rows, np.ceil(num_images / float(rows)), n + 1)
      plt.imshow(image, **kwargs)
      ax.set_title(title)
      plt.axis('off')
  plt.show()