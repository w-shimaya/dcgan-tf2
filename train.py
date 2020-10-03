import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2
import os
import argparse
from dcgan import DCGAN
from wdcgangp import WDCGANGP

EPOCHS=200
STEPS=2
ZDIM=100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "mnist"], required=True)
    parser.add_argument("--model", choices=["GAN", "WGAN-gp"], required=True)
    args = parser.parse_args()

    if args.dataset == "cifar10":
        dataset = K.datasets.cifar10
        (x_train, _), (x_test, _) = dataset.load_data()
    elif args.dataset == "mnist":
        dataset = K.datasets.mnist
        (x_train, _), (x_test, _) = dataset.load_data()
        # add color channel
        # 1-ch -> 3-ch
        x_train = np.concatenate([x_train[:, :, :, np.newaxis] for _ in range(3)], axis=3)
    x_train = x_train / 128.0 - 1.

    train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(128)

    if args.model == "GAN":
        gan = DCGAN(ZDIM)
    elif args.model == "WGAN-gp":
        gan = WDCGANGP(ZDIM)

    for epoch in range(EPOCHS):
        real_batch_list = []
        accumulated_dl, accumulated_gl = 0., 0.
        for i, batch in enumerate(train_ds):
            scaled = tf.image.resize(batch, [64, 64])
            if args.model == "GAN":
                if len(real_batch_list) < STEPS:
                    real_batch_list.append(scaled)
                    continue
                dl, gl = gan.train_on_batch(real_batch_list)
                real_batch_list = []
            elif args.model == "WGAN-gp":
                dl, gl = gan.train_on_batch(scaled, n_critic=3)
            accumulated_dl += float(dl)
            accumulated_gl += float(gl)
            print("\r[%2d] %2.4f %2.4f" % (epoch, accumulated_dl / (i + 1), accumulated_gl / (i + 1)), end="")

        z = tf.random.normal((5 * 5, ZDIM))
        imgs = gan.generator(z, training=False).numpy()
        out_img = np.empty((5*64, 5*64, 3), dtype=np.float32)
        for i, im in enumerate(imgs):
            x = i // 5
            y = i % 5
            out_img[y*64:(y+1)*64, x*64:(x+1)*64, :] = im
        # insert border lines
        out_img = np.insert(out_img, np.arange(64, 64*5, 64), -np.ones((3, )), axis=0)
        out_img = np.insert(out_img, np.arange(64, 64*5, 64), -np.ones((3, )), axis=1)

        if not os.path.exists("%s_images"%args.dataset):
            os.makedirs("%s_images"%args.dataset)
        print(cv2.imwrite(os.path.join("%s_images"%args.dataset, "epoch%d.png"%epoch), (out_img[:, :, ::-1] + 1.0) * 128.))
        print("EPOCH %d finished"%epoch)
