#This code is the training procedure for Subtask 1 of industrial 2D symbols, i.e. Distortion Classification Task
'''
Author: Zhaohui Che
E-mail: chezhaohui@sjtu.edu.cn
'''
from __future__ import print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy
import cv2

learning_rate = 0.001
batch_size = 100
display_step = 1
SIZE = 64
n_input = SIZE*SIZE 
n_classes = 6 # 6 distortion types, i.e. AxNu, GrNu, MotionBlur, SpeckleNoise, Overprint(Thick), Underprint(Thin)
dropout = 0.75 # Dropout probability


x = tf.placeholder(tf.float32, [None, SIZE, SIZE, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) 

def conv2d(x, W, b, strides=1):
    
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')



def conv_net(x, weights, biases, dropout):
    
    x = tf.reshape(x, shape=[-1, SIZE, SIZE, 1])

    #Shared Layers
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print(conv1.shape)
    
    conv1 = maxpool2d(conv1, k=2)
    print(conv1.shape)
    
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print(conv2.shape)
    
    conv2 = maxpool2d(conv2, k=2)
    print(conv2.shape)
    
    #Indenpendent Layers
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    print(conv3.shape)
    

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    print(conv4.shape)
    
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


weights = {
    
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 8])),
    
    'wc2': tf.Variable(tf.random_normal([5, 5, 8, 16])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    'wc4': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    
    'wd1': tf.Variable(tf.random_normal([16*16*64, 1024])),
    
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([8])),
    'bc2': tf.Variable(tf.random_normal([16])),
    'bc3': tf.Variable(tf.random_normal([32])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


pred = conv_net(x, weights, biases, keep_prob)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    
    list = os.listdir("./DistortionTypeDataset/") #change as your own path
    
    count=0
    
    while count<20:
        count = count+1
        print("count:",count)
        for batch_id in range(0, 150):#Training Set
            batch = list[batch_id * batch_size:batch_id * batch_size + batch_size]
            batch_xs = []
            batch_ys = []
            for image in batch:
                id_tag = image.find("-")
                score = image[0:id_tag]
                
                img = cv2.imread("./DistortionTypeDataset/" + image) #change as your own path
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (SIZE,SIZE))
                img_ndarray = numpy.asarray(img, dtype='float32')
                img_ndarray = numpy.reshape(img_ndarray, [SIZE, SIZE, 1])
                
                batch_x = img_ndarray
                batch_xs.append(batch_x)
                batch_y = numpy.asarray([0, 0, 0, 0, 0, 0])
                
                batch_y[int(score) - 1] = 1
                
                batch_y = numpy.reshape(batch_y, [6, ]) 
                batch_ys.append(batch_y)
            
            batch_xs = numpy.asarray(batch_xs)
            
            batch_ys = numpy.asarray(batch_ys)
            

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                           keep_prob: dropout})
            if step % display_step == 0:
                
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_xs,
                                                                  y: batch_ys,
                                                                  keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
    
    saver.save(sess,"./MyModel/modelSubtask1.ckpt")

