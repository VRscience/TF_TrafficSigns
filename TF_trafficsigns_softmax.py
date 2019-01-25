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

images, r_labels = Data

#Turning images list into array
img = np.array(images)
s_labels = set(r_labels)
labels = list(s_labels)

#Resize Images to 28x28 into images28
images28 = [sk.transform.resize(image, (28,28)) for image in img]

#Convert to array into img28
img28 = np.array(images28)

#Convert imges28 to greyscale

Images28_GS = [sk.color.rgb2gray(image_gs) for image_gs in img28]

img28_gs = np.array(Images28_GS)


""""--------------------------Tensorflow part starts here----------------------------------------"""

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the batch
x = tf.placeholder(tf.float32, [None, 28, 28])
# correct answers will go here
y_ = tf.placeholder(tf.float32, [None, 10])
# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
x_flat = tf.reshape(x, [-1, 784])

# The model
y = tf.nn.softmax(tf.matmul(x_flat, W) + b)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
loss = -tf.reduce_mean(y_ * tf.log(y))

# accuracy of the trained model, between 0 (worst) and 1 (best)
#correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# training, learning rate = 0.005
optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
#Recap of the above code
print("images_flat: ", x_flat)
print("loss: ", loss)
#print("predicted_labels: ", correct_pred)

#Running the NN

tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

Epochs = int(input("Please insert epochs number-->"))
for i in range(Epochs+1):
    print("EPOCH", i)
    _, loss_val = sess.run([optimizer, loss], feed_dict={x: img28_gs, y_: labels})
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
