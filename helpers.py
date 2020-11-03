import os

def listdir(path):
  """ List files but remove hidden files from list """
  return [item for item in os.listdir(path) if item[0] != '.']

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)