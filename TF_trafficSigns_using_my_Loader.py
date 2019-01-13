import os
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
from Loader import Loader as LD
import random as rn
import tensorflow as tf


ROOT_PATH = input("-> ").strip()

train_data_dir = os.path.join(ROOT_PATH, "TrafficSigns\Training")
test_data_dir = os.path.join(ROOT_PATH, "TrafficSigns\Testing")

#using Loader to load data

Loader = LD(train_data_dir)
Directory = Loader.directory
Data = Loader.load_data(Directory)

images, labels = Data

#Printing out data from image 0
img = np.array(images)
print (img.ndim)
print (img.size)
print (img[0])
#print (labels)

#Creating plot to understand the data
plt.figure(100)
histo = plt.hist(labels, 62)


#Printing out 4 data sample (namely 300, 2250, 3650, 4000) to understand differences within the dataset

samples = []
for n in range(4):
    samples.append(rn.randint(0,4001))



for item in range(len(samples)):
    plt.figure(101)

    txt = "Shape: {0} ; min: {1} ; max: {2}".format(images[samples[item]].shape,
                                                    images[samples[item]].min(),
                                                    images[samples[item]].max())
    #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment="center", fontsize=6) --> Set the txt only at the botton of the subplot window
    plt.subplot(4, 1, item + 1).set_title(txt, fontsize=8)
    plt.imshow(images[samples[item]])
    plt.subplots_adjust(hspace=1)
    plt.axis("off")



plt.get_current_fig_manager().show()
plt.get_current_fig_manager().show()

plt.show()
plt.show()

#Set unique labels
unique_labels = set(labels)

#Initialize figure

"""plt.figure(102, figsize=(15, 15))


i = 1
for lbl in unique_labels:
    txt = "Label {0} : ({1}) : lbl({2})".format(lbl,
                               labels.count(lbl),
                                     labels.index(lbl),
                                               lbl)
    plt.subplot(8, 8, i)
    plt.axis("off")
    plt.title(txt, fontsize=8)
    i +=1
    plt.imshow(img[labels.index(lbl)])


plt.show()"""







figure2 = plt.figure(103, figsize=(15,15))
i = 1

for lbl in unique_labels:

    plt.subplot(8,8,i)
    plt.axis("off")
    txt = "ID {0} : Count ({1}) : IMG({2})".format(lbl,
                                                   labels.count(lbl),
                                                   labels.index(lbl))
    plt.title(txt, fontsize=5)
    i+=1
    plt.imshow(images[labels.index(lbl)])

plt.show()


#Resize Images to 28x28 into images28
images28 = [sk.transform.resize(image, (28,28)) for image in img]
#Convert to array into img28
img28 = np.array(images28)
#print img28 array size
print (img28.shape)


#The following line will reprint the figure as above but with the new size
figure3 = plt.figure(104, figsize=(15,15))
i = 1

for lbl in unique_labels:

    plt.subplot(8,8,i)
    plt.axis("off")
    txt = "ID {0} : Count ({1}) : IMG({2})".format(lbl,
                                                   labels.count(lbl),
                                                   labels.index(lbl))
    plt.title(txt, fontsize=5)
    i+=1
    plt.imshow(img28[labels.index(lbl)])

plt.show()

#Convert imges28 to greyscale

Images28_GS = [sk.color.rgb2gray(image_gs) for image_gs in img28]

img28_gs = np.array(Images28_GS)

print (img28_gs.shape)


#The following line will reprint the figure as above but with the new size and grayscale
figure4 = plt.figure(105, figsize=(15,15))
i = 1

for lbl in unique_labels:

    plt.subplot(8,8,i)
    plt.axis("off")
    txt = "ID {0} : Count ({1}) : IMG({2})".format(lbl,
                                                   labels.count(lbl),
                                                   labels.index(lbl))
    plt.title(txt, fontsize=5)
    i+=1
    plt.imshow(img28_gs[labels.index(lbl)], cmap="gray")

plt.show()
















