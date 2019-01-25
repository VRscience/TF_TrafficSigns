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

#Resize Images to 28x28 into images28
images28 = [sk.transform.resize(image, (28,28)) for image in img]
#Convert to array into img28
img28 = np.array(images28)
#print img28 array size
#print (img28.shape)


#Convert imges28 to greyscale

Images28_GS = [sk.color.rgb2gray(image_gs) for image_gs in img28]

img28_gs = np.array(Images28_GS)

#print (img28_gs.shape)


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
correct_pred = tf.argmax(logits,1)

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

Epochs = int(input("Please insert epochs number-->"))
for i in range(Epochs+1):
    print("EPOCH", i)
    _, accuracy_val, loss_val = sess.run([optimizer, accuracy, loss], feed_dict={x: img28_gs, y: labels})
    if i%10==0:
        print("Loss:",loss_val)
    print ("Epochs over.")


"""----------------------------------------Running Evaluation------------------------------------"""
#Generating n sample to validate
validation_choice = input("Would you like to validate the accuracy of your model?(y/n): ")
if validation_choice == "y":
    n_of_s = int(input("How many sample do you want to validate: ").strip())
    if n_of_s > 0:
        sample_indexes = rn.sample(range(len(img28_gs)), n_of_s)
        sample_images = [img28_gs[i] for i in sample_indexes]
        sample_labels = [labels[i] for i in sample_indexes]

        #Run correct prediction operation
        predicted = sess.run([correct_pred], feed_dict={x:sample_images})[0]

        #Print real and predicted labels
        print(predicted)
        print(sample_labels)

        #Display prediction in matplot
        plt.figure(106, figsize=(15, 15))
        for i in range(len(sample_images)):
            truth = sample_labels[i]
            prediction = predicted[i]
            plt.subplot(5, int(n_of_s/5), 1+i)
            plt.axis("off")
            color = "green" if prediction == truth else "red"
            plt.text(40,10, "truth: {0} \n prediction: {1}".format(truth,prediction), fontsize=8,color=color)
            plt.imshow(sample_images[i], cmap="gray")
        plt.show()

    else:
        ("Entry not valid!")
        exit()

elif validation_choice == "n":
    pass
else:
    ("Entry not valid!")
    exit()

#Close TF Session
#sess.close()


"""--------------------Prepare test data for Running Evaluation----------------------------------"""


#Import Test dataset

Loader_test = LD(test_data_dir)
Directory_test = Loader_test.directory
Data_test = Loader_test.load_data(Directory_test)

images_test, labels_test = Data_test
print (Directory_test)


#Transform images to 28x28
images28_test = [sk.transform.resize(img_test, (28, 28)) for img_test in images_test]

#Creating array

img28_test = np.array(images28_test)

#Convert to grey scale

img28_gs_test = sk.color.rgb2gray(img28_test)


"""------------------------Running Evaluation against Test Data----------------------------------"""


#Run prediction

predicted_test = sess.run([correct_pred], feed_dict={x: img28_gs_test})[0]

#Calculated correct matches

match_count = sum([int(y == y_) for y, y_ in zip(predicted_test, labels_test)])

#Calculate accuracy

accuracy = float((match_count/len(labels_test))*100)

#Print accuracy

print ("Accuracy(%): {}".format(round(accuracy, 2)))

#Session close

sess.close()
