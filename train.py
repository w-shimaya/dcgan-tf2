import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2
import os
from dcgan import DCGAN

EPOCHS=30
STEPS=2

if __name__ == "__main__":
    dataset = K.datasets.cifar10
    (x_train, _), (x_test, _) = dataset.load_data()
    # upsample image 32x32 to 64x64 (no interpolation)
    x_train = x_train.repeat(2, axis=1).repeat(2, axis=2)
    # transform to fit to tanh
    x_train = x_train / 128. - 1.

    train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(128)

    gan = DCGAN(100)

    for epoch in range(EPOCHS):
        real_batch_list = []
        for batch in train_ds:
            if len(real_batch_list) < STEPS:
                real_batch_list.append(batch)
                continue
            dl, gl = gan.train_on_batch(real_batch_list)
            print("[%2d] %2.4f %2.4f" % (epoch, float(dl), float(gl)))
            real_batch_list = []

        z = tf.random.normal((5, 100))
        img = gan.generator(z, training=False).numpy()
        for i, im in enumerate(img):
            if not os.path.exists(os.path.join("cifar10_images", "epoch%d"%epoch)):
                os.makedirs(os.path.join("cifar10_images", "epoch%d"%epoch))
            print(cv2.imwrite(os.path.join("cifar10_images", "epoch%d"%epoch, "%d.png"%i), (im + 1.0) * 128.))
        print("EPOCH %d finished"%epoch)
