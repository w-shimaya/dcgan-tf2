import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2
import os
from dcgan import DCGAN

EPOCHS=30
STEPS=2
ZDIM=100

if __name__ == "__main__":
    dataset = K.datasets.cifar10
    (x_train, _), (x_test, _) = dataset.load_data()
    # upsample image 32x32 to 64x64 (no interpolation)
    x_train = x_train.repeat(2, axis=1).repeat(2, axis=2)
    # transform to fit to tanh
    x_train = x_train / 128. - 1.

    train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(128)

    gan = DCGAN(ZDIM)

    for epoch in range(EPOCHS):
        real_batch_list = []
        for batch in train_ds:
            if len(real_batch_list) < STEPS:
                real_batch_list.append(batch)
                continue
            dl, gl = gan.train_on_batch(real_batch_list)
            print("[%2d] %2.4f %2.4f" % (epoch, float(dl), float(gl)))
            real_batch_list = []

        z = tf.random.normal((5 * 5, ZDIM))
        imgs = gan.generator(z, training=False).numpy()
        out_img = np.empty((5*64, 5*64, 3), dtype=np.float32)
        for i, im in enumerate(imgs):
            x = i // 5
            y = i % 5
            out_img[y*64:(y+1)*64, x*64:(x+1)*64, :] = im
        # insert border lines
        out_img = np.insert(out_img, np.arange(64, 64*5, 64), np.zeros((3, )), axis=0)
        out_img = np.insert(out_img, np.arange(64, 64*5, 64), np.zeros((3, )), axis=1)

        if not os.path.exists("cifar10_images"):
            os.makedirs("cifar10_images")
        print(cv2.imwrite(os.path.join("cifar10_images", "epoch%d.png"%epoch), (im + 1.0) * 128.))
        print("EPOCH %d finished"%epoch)
