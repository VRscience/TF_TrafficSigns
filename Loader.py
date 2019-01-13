import os
import skimage

class Loader:

    def __init__(self, directory, images = None, labels = None):

        self.directory = directory
        self.images = images
        self.labels = labels

    def load_data(self, directory):

        directories = [d for d in os.listdir(directory)
                       if os.path.isdir(os.path.join(directory, d))]

        images = []
        labels = []

        for dir in directories:

            label_directory = os.path.join(directory, dir)
            file_names = [os.path.join(label_directory, name) for name in os.listdir(label_directory)
                          if name.endswith(".ppm")]

            for name in file_names:
                images.append(skimage.data.imread(name))
                labels.append(int(dir))

        return images, labels





