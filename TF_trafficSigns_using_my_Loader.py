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

#Turning images list into array
img = np.array(images)

#Printing out data from image 0
data_sample_print = input("Would you like to print a data point?(y/n): ")
if data_sample_print == "y":
    print(img.ndim)
    print(img.size)
    print(img[0])
    # print (labels)
elif data_sample_print == "n":
    pass
else:
    ("Entry not valid!")
    exit()


#Creating plot to understand the data
plot_hist_print = input("Would you like to print an histogram of the data distribution?(y/n): ")
if plot_hist_print == "y":
    plt.figure(100)
    histo = plt.hist(labels, 62)
    plt.show()
elif plot_hist_print == "n":
    pass
else:
    ("Entry not valid!")
    exit()


#Printing out 4 random data sample to understand differences within the dataset
random_sample_data= input("Would you like to visualize 4 random data samples ?(y/n): ")
if random_sample_data == "y":
    samples = []
    for n in range(4):
        samples.append(rn.randint(0, 4001))

    for item in range(len(samples)):
        plt.figure(101)

        txt = "Shape: {0} ; min: {1} ; max: {2}".format(images[samples[item]].shape,
                                                        images[samples[item]].min(),
                                                        images[samples[item]].max())
        # plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment="center", fontsize=6) --> Set the txt only at the botton of the subplot window
        plt.subplot(4, 1, item + 1).set_title(txt, fontsize=8)
        plt.imshow(images[samples[item]])
        plt.subplots_adjust(hspace=1)
        plt.axis("off")

    plt.show()
elif random_sample_data == "n":
    pass
else:
    ("Entry not valid!")
    exit()



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

#Set unique labels
unique_labels = set(labels)

#Plot 1 data sample per label
full_sample_data= input("Would you like to visualize one data samples per each category?(y/n): ")
if full_sample_data == "y":
    figure2 = plt.figure(103, figsize=(15, 15))
    i = 1

    for lbl in unique_labels:
        plt.subplot(8, 8, i)
        plt.axis("off")
        txt = "ID {0} : Count ({1}) : IMG({2})".format(lbl,
                                                       labels.count(lbl),
                                                       labels.index(lbl))
        plt.title(txt, fontsize=5)
        i += 1
        plt.imshow(images[labels.index(lbl)])

    plt.show()

elif full_sample_data == "n":
    pass
else:
    ("Entry not valid!")
    exit()

#Resize Images to 28x28 into images28
images28 = [sk.transform.resize(image, (28,28)) for image in img]
#Convert to array into img28
img28 = np.array(images28)
#print img28 array size
#print (img28.shape)


#The following line will reprint the figure as above but with the new size
full_sample_resized_data= input("Data has been resized. Would you like to visualize one data samples per each category?(y/n): ")
if full_sample_resized_data == "y":
    figure3 = plt.figure(104, figsize=(15, 15))
    i = 1

    for lbl in unique_labels:
        plt.subplot(8, 8, i)
        plt.axis("off")
        txt = "ID {0} : Count ({1}) : IMG({2})".format(lbl,
                                                       labels.count(lbl),
                                                       labels.index(lbl))
        plt.title(txt, fontsize=5)
        i += 1
        plt.imshow(img28[labels.index(lbl)])

    plt.show()

elif full_sample_resized_data == "n":
    pass
else:
    ("Entry not valid!")
    exit()

#Convert imges28 to greyscale

Images28_GS = [sk.color.rgb2gray(image_gs) for image_gs in img28]

img28_gs = np.array(Images28_GS)

#print (img28_gs.shape)


#The following line will reprint the figure as above but with the new size and grayscale
full_sample_resized_grey_data= input("Resized images has been turned into grey scale. Would you like to visualize one data samples per each category?(y/n): ")
if full_sample_resized_grey_data == "y":
    figure4 = plt.figure(105, figsize=(15, 15))
    i = 1

    for lbl in unique_labels:
        plt.subplot(8, 8, i)
        plt.axis("off")
        txt = "ID {0} : Count ({1}) : IMG({2})".format(lbl,
                                                       labels.count(lbl),
                                                       labels.index(lbl))
        plt.title(txt, fontsize=5)
        i += 1
        plt.imshow(img28_gs[labels.index(lbl)], cmap="gray")

    plt.show()

elif full_sample_resized_grey_data == "n":
    pass
else:
    ("Entry not valid!")
    exit()



""""--------------------------Tensorflow part starts here----------------------------------------"""




#Initialize the placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

#Flatten he inputdata
x_flat = tf.contrib.layers.flatten(x)

#Create fully connected layers
logits = tf.contrib.layers.fully_connected(x_flat, 62, tf.nn.relu)

#Define Loss Function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                     logits=logits))

#Define Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#Convert Logits to label indexes
correct_pred = tf.arg_max(logits,1)

#Define accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#Recap of the above code
print("images_flat: ", x_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

#Running the NN

tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(201):
    print("EPOCH", i)
    _, accuracy_val = sess.run([optimizer, accuracy], feed_dict={x: img28_gs, y:labels})
    if i%10==0:
        print("Loss:",loss)
    print ("Epochs over.")

