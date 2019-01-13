import os
import skimage
import numpy as np
import matplotlib.pyplot as plt



def load_data(data_directory):

    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]

    images = []
    labels = []

    for dir in directories:

       label_directory = os.path.join(data_directory, dir)
       file_names = [os.path.join(label_directory,name) for name in os.listdir(label_directory)
                     if name.endswith(".ppm")]

       for name in file_names:

           images.append(skimage.data.imread(name))
           labels.append(int(dir))

    return images, labels



ROOT_PATH = input("-> ").strip()

train_data_dir = os.path.join(ROOT_PATH, "TrafficSigns\Training")
test_data_dir = os.path.join(ROOT_PATH, "TrafficSigns\Testing")

images, labels = load_data(train_data_dir)

img = np.array(images)
print (img.ndim)
print (img.size)
print (images[0])


plt.hist(labels, 62)

plt.show()


samples = [300, 2250, 3650, 4000]

for item in range(len(samples)):
    plt.subplot(1, 4, item+1)
    plt.axis("off")
    plt.imshow(images[samples[item]])
    plt.subplots_adjust(wspace=0.5)


plt.show()









