import os


def root_path():
    path = os.path.realpath(os.curdir)
    while True:
        if '.idea' in os.listdir(path):
            return path
        path = os.path.dirname(path)
