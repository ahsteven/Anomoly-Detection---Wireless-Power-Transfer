
# Importing the libraries
import numpy as np
# import matplotlib.pyplot as plt
import glob
import random
import scipy.misc
import tensorflow as tf
from functools import partial
from math import ceil
import sys
import os




# Importing the dataset
normal = 'good/'
anomoly = ['metal/', 'screws/', 'soda_can/']
normal_names = glob.glob(normal +'*.png')
anomoly_names = []
for an in anomoly:
    anomoly_names.append(glob.glob(an + '*.png'))
anomoly_names = np.concatenate(anomoly_names)



learning_rate = .0005
l2_reg = .001
batch_size = 32
n_batches = ceil(len(normal_names)/batch_size)
image_size = (60, 80, 3)
n_epochs_all = 1

# if you want to use a generator to read images
def get_batches_fn(imgs, batch_size, flatten = True):
    """
    Create batches of shuffled images
    :param imgs: list of image names to read in
    :param batch_size: Batch Size
    :param flatten: if True flatten image
    :return: Batches of training data
    """
    random.shuffle(imgs)
    for batch_i in range(0, len(imgs), batch_size):
        images = []
        for image_file in imgs[batch_i:batch_i + batch_size]:
            image = scipy.misc.imread(image_file)
            image = scipy.misc.imresize(scipy.misc.imread(image_file), (image_size[0], image_size[1]), interp="nearest")
            if flatten:
                image = image.flatten()
            image = image/255.0
            images.append(image)
        yield np.array(images)

# to read in all the images without generator
def read_imgs(imgs, flatten = True):
    """
    Read in list of images, return np array
    :param imgs: list of image names to read in
    :param flatten: if True flatten image
    :return: np array of images
    """
#     random.shuffle(imgs)
    images = []
    for image_file in imgs:

        image = scipy.misc.imresize(scipy.misc.imread(image_file), (image_size[0], image_size[1]), interp="nearest")
#         print("image shape ", image.shape)
        if flatten:
            image = image.flatten()
        image = image/255.0
        images.append(image)
    return np.array(images)

# build tensorflow graph
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

tf.reset_default_graph()
he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
optimizer = tf.train.AdamOptimizer(learning_rate)

X = tf.placeholder(tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]], name="X")
conv1 = tf.layers.conv2d(
    inputs=X,
    filters=64,
    strides=(4, 4),
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.tanh,
    kernel_initializer=he_init,
    kernel_regularizer=l2_regularizer,
    name="conv1")
# print("conv1 ", conv1.get_shape().as_list())
# maxpool1 = tf.layers.max_pooling2d(
#     conv1,
#     pool_size=(2,2),
#     strides=(2,2),
#     padding='same',
#     name="maxpool1")
# print("maxpool1 ", maxpool1.get_shape().as_list())
conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=64,
    strides=(2, 2),
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.tanh,
    kernel_initializer=he_init,
    kernel_regularizer=l2_regularizer,
    name="conv2")
# print("conv2 ", conv2.get_shape().as_list())
# maxpool2 = tf.layers.max_pooling2d(
#     conv2,
#     pool_size=(2,2),
#     strides=(2,2),
#     padding='same',
#     name="maxpool2")
# print("maxpool2 ", maxpool2.get_shape().as_list())
conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=64,
    strides=(2, 2),
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.tanh,
    kernel_initializer=he_init,
    kernel_regularizer=l2_regularizer,
    name="conv3")
# print("conv3 ", conv3.get_shape().as_list())
encoded = tf.layers.conv2d(
    inputs=conv3,
    filters=16,
    strides=(2, 2),
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.tanh,
    kernel_initializer=he_init,
    kernel_regularizer=l2_regularizer,
    name="encoded")
# print("encoded ", encoded.get_shape().as_list())
conv3_shape = conv3.get_shape().as_list()
conv3_dec = tf.layers.conv2d_transpose(
    inputs=encoded,
    filters=64,
    strides=(2, 2),
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.tanh,
    kernel_initializer=he_init,
    kernel_regularizer=l2_regularizer,
    name="conv3_dec")
# print("conv3_dec ", conv3_dec.get_shape().as_list())
reshape3 = tf.image.resize_images(conv3_dec,
                                  size=(conv3_shape[1], conv3_shape[2]),
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# print("reshape3 ", reshape3.get_shape().as_list())
conv2_dec = tf.layers.conv2d_transpose(
    inputs=reshape3,
    filters=64,
    strides=(2, 2),
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.tanh,
    kernel_initializer=he_init,
    kernel_regularizer=l2_regularizer,
    name="conv2_dec")
# print("conv2_dec ", conv2_dec.get_shape().as_list())
conv2_shape = conv2.get_shape().as_list()
reshape2 = tf.image.resize_images(conv2_dec,
                                  size=(conv2_shape[1], conv2_shape[2]),
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# print("reshape2 ", reshape2.get_shape().as_list())

conv1_shape = conv1.get_shape().as_list()
conv1_dec = tf.layers.conv2d_transpose(
    inputs=reshape2,
    filters=64,
    strides=(2, 2),
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.tanh,
    kernel_initializer=he_init,
    kernel_regularizer=l2_regularizer,
    name="conv1_dec")
# print("conv1_dec ", conv1_dec.get_shape().as_list())
reshape1 = tf.image.resize_images(conv1_dec,
                                  size=(conv1_shape[1], conv1_shape[2]),
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# print("reshape1 ", reshape1.get_shape().as_list())

outputs = tf.layers.conv2d_transpose(
    inputs=reshape1,
    filters=3,
    strides=(4, 4),
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.sigmoid,
    kernel_initializer=he_init,
    kernel_regularizer=l2_regularizer,
    name="outputs")
# print("outputs ", outputs.get_shape().as_list())

# setup training phases
all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
with tf.name_scope("phase1"):
    conv1_p1 = tf.layers.conv2d(
        inputs=X,
        filters=64,
        strides=(4, 4),
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.tanh,
        kernel_initializer=he_init,
        kernel_regularizer=l2_regularizer,
        reuse=True,
        name="conv1")
    # print("conv1_p1 ", conv1_p1.get_shape().as_list())
    outputs_p1 = tf.layers.conv2d_transpose(
        inputs=conv1_p1,
        filters=3,
        strides=(4, 4),
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.sigmoid,
        kernel_initializer=he_init,
        kernel_regularizer=l2_regularizer,
        reuse=True,
        name="outputs")
    # print("outputs_p1 ", outputs_p1.get_shape().as_list())
    #     print(tf.trainable_variables(scope='phase1'))

    phase1_names = ['conv1/kernel:0', 'conv1/bias:0', 'outputs/kernel:0', 'outputs/bias:0']
    phase1_train_vars = [v for v in all_variables if v.name in phase1_names]
    # print("phase1 trainable variables ", phase1_train_vars)
    phase1_loss_collection = [item for item in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                              if 'conv1/kernel' in item.name or 'outputs/kernel' in item.name]
    # print("phase1_loss_collection", phase1_loss_collection)
    phase1_reg_loss = tf.reduce_sum(phase1_loss_collection)
    #     phase1_reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    phase1_recon_loss = tf.reduce_mean(tf.square(outputs_p1 - X))
    phase1_loss = phase1_reg_loss + phase1_recon_loss
    #     phase1_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "phase1")#only update phase1 weights
    phase1_training_op = optimizer.minimize(phase1_loss, var_list=phase1_train_vars)

with tf.name_scope("phase2"):
    conv2_p2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        strides=(2, 2),
        kernel_size=[3, 3],
        kernel_initializer=he_init,
        kernel_regularizer=l2_regularizer,
        padding="same",
        activation=tf.nn.tanh,
        reuse=True,
        name="conv2")
    # print("conv2_p2 ", conv2_p2.get_shape().as_list())

    conv1_shape = conv1.get_shape().as_list()
    conv1_dec_p2 = tf.layers.conv2d_transpose(
        inputs=conv2_p2,
        filters=64,
        strides=(2, 2),
        kernel_size=[5, 5],
        padding="same",
        kernel_initializer=he_init,
        kernel_regularizer=l2_regularizer,
        activation=tf.nn.tanh,
        reuse=True,
        name="conv1_dec")
    # print("conv1_dec_p2 ", conv1_dec.get_shape().as_list())
    conv1_shape = conv1.get_shape().as_list()
    reshape1_p2 = tf.image.resize_images(conv1_dec_p2,
                                         size=(conv1_shape[1], conv1_shape[2]),
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # print("reshape1_p2\ ", reshape1_p2.get_shape().as_list())

    # get phase2 trainable variables
    phase2_names = ['conv2/kernel:0', 'conv2/bias:0', 'conv1_dec/kernel:0', 'conv1_dec/bias:0']
    phase2_train_vars = [v for v in all_variables if v.name in phase2_names]
    # print("phase2 trainable variables ", phase2_train_vars)
    phase2_loss_collection = [item for item in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                              if 'conv2/kernel' in item.name or 'conv1_dec/kernel' in item.name]
    # print("phase2_loss_collection", phase2_loss_collection)
    phase2_reg_loss = tf.reduce_sum(phase2_loss_collection)

    #     phase2_reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'phase2'))
    phase2_recon_loss = tf.reduce_mean(tf.square(reshape1_p2 - conv1))
    phase2_loss = phase2_reg_loss + phase2_recon_loss
    #     phase2_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "phase2")
    phase2_training_op = optimizer.minimize(phase2_loss, var_list=phase2_train_vars)

with tf.name_scope("phase3"):
    conv3_p3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        strides=(2, 2),
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.tanh,
        kernel_initializer=he_init,
        kernel_regularizer=l2_regularizer,
        reuse=True,
        name="conv3")
    # print("conv3_p3 ", conv3_p3.get_shape().as_list())
    conv2_dec_p3 = tf.layers.conv2d_transpose(
        inputs=conv3_p3,
        filters=64,
        strides=(2, 2),
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.tanh,
        kernel_initializer=he_init,
        kernel_regularizer=l2_regularizer,
        reuse=True,
        name="conv2_dec")
    # print("conv2_dec_p3 ", conv2_dec_p3.get_shape().as_list())
    conv2_shape = conv2.get_shape().as_list()
    reshape2_p3 = tf.image.resize_images(conv2_dec_p3,
                                         size=(conv2_shape[1], conv2_shape[2]),
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # print("reshape2_p3\ ", reshape2_p3.get_shape().as_list())

    # get phase3 trainable variables
    phase3_names = ['conv3/kernel:0', 'conv3/bias:0', 'conv2_dec/kernel:0', 'conv2_dec/bias:0']
    phase3_train_vars = [v for v in all_variables if v.name in phase3_names]
    # print("phase3 trainable variables ", phase3_train_vars)
    phase3_loss_collection = [item for item in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                              if 'conv3/kernel' in item.name or 'conv2_dec/kernel' in item.name]
    # print("phase3_loss_collection", phase3_loss_collection)
    phase3_reg_loss = tf.reduce_sum(phase3_loss_collection)
    #     phase3_reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'phase3'))
    phase3_recon_loss = tf.reduce_mean(tf.square(reshape2_p3 - conv2))
    phase3_loss = phase3_reg_loss + phase3_recon_loss
    #     phase3_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "phase3")
    phase3_training_op = optimizer.minimize(phase3_loss, var_list=phase3_train_vars)

# phase 4 not necessary, could just train full network
with tf.name_scope("phase4"):
    encoded_p4 = tf.layers.conv2d(
        inputs=conv3,
        filters=16,
        strides=(2, 2),
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.tanh,
        kernel_initializer=he_init,
        kernel_regularizer=l2_regularizer,
        reuse=True,
        name="encoded")
    # print("encoded ", encoded.get_shape().as_list())
    conv3_shape = conv3.get_shape().as_list()
    conv3_dec_p4 = tf.layers.conv2d_transpose(
        inputs=encoded_p4,
        filters=64,
        strides=(2, 2),
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.tanh,
        kernel_initializer=he_init,
        kernel_regularizer=l2_regularizer,
        reuse=True,
        name="conv3_dec")
    # print("conv3_dec_p4 ", conv3_dec_p4.get_shape().as_list())
    reshape3_p4 = tf.image.resize_images(conv3_dec_p4,
                                         size=(conv3_shape[1], conv3_shape[2]),
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    print("reshape3_p4 ", reshape3_p4.get_shape().as_list())

    # get phase4 trainable variables
    phase4_names = ['encoded/kernel:0', 'encoded/bias:0', 'conv3_dec/kernel:0', 'conv3_dec/bias:0']
    phase4_train_vars = [v for v in all_variables if v.name in phase4_names]
    # print("phase4 trainable variables ", phase4_train_vars)
    phase4_loss_collection = [item for item in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                              if 'conv3_dec/kernel' in item.name or 'encoded/kernel' in item.name]
    # print("phase4_loss_collection", phase4_loss_collection)
    phase4_reg_loss = tf.reduce_sum(phase4_loss_collection)
    #     phase4_reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'phase4'))
    phase4_recon_loss = tf.reduce_mean(tf.square(reshape3_p4 - conv3))
    phase4_loss = phase4_reg_loss + phase4_recon_loss
    #     phase4_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "phase4")
    phase4_training_op = optimizer.minimize(phase4_loss, var_list=phase4_train_vars)

### For training complete graph
reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
recon_loss = tf.reduce_mean(tf.square(outputs - X), name="recon_loss")
loss = reg_loss + recon_loss
training_op = optimizer.minimize(loss)



# end -- build tensorflow graph
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def train_phases(checkpoint_file=None):

    graph = tf.get_default_graph()
    training_ops = [phase1_training_op, phase2_training_op, phase3_training_op, phase4_training_op]
    reconstruction_losses = [phase1_recon_loss, phase2_recon_loss, phase3_recon_loss, phase4_recon_loss]
    n_epochs = [n_epochs_all, n_epochs_all, n_epochs_all, n_epochs_all]
    batch_sizes = [batch_size, batch_size, batch_size, batch_size]


    with tf.Session() as sess:
        if checkpoint_file:
            saver.restore(sess, checkpoint_file)
        else:
            init.run()
        for phase in range(4):
            # print("Training phase #{}".format(phase + 1))
            for epoch in range(n_epochs[phase]):
                #             for iteration in range(n_batches):
                for iteration, X_batch in enumerate(get_batches_fn(normal_names, batch_size, flatten=False)):
                    #                     print("\r{}%".format(100 * iteration // n_batches), end="")
                    sys.stdout.flush()
                    sess.run(training_ops[phase], feed_dict={X: X_batch})
                loss_train = reconstruction_losses[phase].eval(feed_dict={X: X_batch})
                # print("\r{}".format(epoch), "Train MSE:", loss_train)
                saver.save(sess, "./checkpoints/model.ckpt")


def train_all(checkpoint_file=None):
    with tf.Session() as sess:
        if checkpoint_file:
            saver.restore(sess, checkpoint_file)
        else:
            init.run()

        for epoch in range(n_epochs_all):
            #             for iteration in range(n_batches):
            for iteration, X_batch in enumerate(get_batches_fn(normal_names, batch_size, flatten=False)):
                #                 print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                sess.run(training_op, feed_dict={X: X_batch})
            # loss_train = recon_loss.eval(feed_dict={X: X_batch})
            # print("\r{}".format(epoch), "Train MSE:", loss_train)
            saver.save(sess, "./checkpoints/model.ckpt")


def show_reconstructed_digits(X, outputs, imgs, model_path = None):
    graph = tf.get_default_graph()
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = read_imgs(imgs, flatten=False)
        outputs_val = outputs.eval(feed_dict={X: X_test})
        writer = tf.summary.FileWriter("summary", sess.graph)
#     fig = plt.figure(figsize=(8, 3 ))
    for digit_index in range(1):
        plt.figure()
        plt.imshow(X_test[digit_index])
        # plt.show()
        plt.figure()
        plt.imshow(outputs_val[0])
        plt.show()



last_checked = None
def get_latest_img(folder):
    global last_checked
    list_of_files = glob.glob(folder + '*.png')
    if not list_of_files:
        print("No png files in folder: ", folder)
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        if latest_file is not last_checked:
            latest_image = read_imgs([latest_file], flatten=False)
            last_checked = latest_file
            return latest_image
        else:
            return None

def get_recon_loss(session, image):
    good_loss = session.run(recon_loss, feed_dict={X: image})
    return good_loss

def is_anomoly(recon_loss, threshold=.04):
    if recon_loss > threshold:
        return True
    else:
        return False


init = tf.global_variables_initializer()
saver = tf.train.Saver()

# train_phases()
# train_all("./checkpoints/model.ckpt")

# show_reconstructed_digits(X, outputs, [normal_names[0]], "./checkpoints/model.ckpt")

with tf.Session() as sess:
    init.run()
    saver.restore(sess, "./checkpoints/model.ckpt")
    latest_img = get_latest_img('anom/')
    print(latest_img.shape)
    test_loss = get_recon_loss(sess, latest_img)
    if is_anomoly(test_loss):
        print("Latest image is not Normal")
        print(" Reconstruction loss ", test_loss)
    else:
        print("Normal reconstruction loss : ", test_loss)