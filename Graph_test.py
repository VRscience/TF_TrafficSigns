import tensorflow as tf

h = tf.constant(54, name="suca")
b = tf.constant(22, name= "forte")

#d = tf.add(h,b, name="frase")

with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    print(sess.run(h))
    writer.close()



