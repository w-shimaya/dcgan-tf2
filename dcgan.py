import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L

class DCGAN(K.Model):
    def __init__(self, latent_dim):
        super(DCGAN, self).__init__()
        self.latent_dim = latent_dim

        # note: bias terms in layers adjacent to batchnorm are redundant
        self.generator = K.Sequential([
            L.Dense(4*4*1024, input_shape=(latent_dim, ), use_bias=False), 
            L.Reshape((4, 4, 1024)), 
            L.Conv2DTranspose(512, 5, 2, padding="same", use_bias=False),
            L.BatchNormalization(),
            L.ReLU(), 
            L.Conv2DTranspose(256, 5, 2, padding="same", use_bias=False),
            L.BatchNormalization(),
            L.ReLU(), 
            L.Conv2DTranspose(128, 5, 2, padding="same", use_bias=False),
            L.BatchNormalization(),
            L.ReLU(), 
            L.Conv2DTranspose(3, 5, 2, padding="same"),
            L.Activation(tf.nn.tanh), 
        ])

        self.discriminator = K.Sequential([
            L.Conv2D(128, 5, 2, input_shape=(64, 64, 3), padding="same"), 
            L.LeakyReLU(0.2), 
            L.Conv2D(256, 5, 2, padding="same", use_bias=False), 
            L.BatchNormalization(), 
            L.LeakyReLU(0.2), 
            L.Conv2D(512, 5, 2, padding="same", use_bias=False), 
            L.BatchNormalization(), 
            L.LeakyReLU(0.2), 
            L.Conv2D(1024, 5, 2, padding="same", use_bias=False), 
            L.BatchNormalization(), 
            L.LeakyReLU(0.2), 
            L.Flatten(), 
            L.Dense(1)
        ])

        self.g_opt = K.optimizers.Adam(0.0002, beta_1=0.5)
        self.d_opt = K.optimizers.Adam(0.0002, beta_1=0.5)

    @tf.function
    def train_on_batch(self, real_batch_list):
        """
        update weights and return the values of loss function

        real_batch_list: list of tensors the shape of which is [batch, y, x, c]
        """
        cross_entropy = K.losses.BinaryCrossentropy(from_logits=True)
        # update the discriminator several times
        for real_batch in real_batch_list:
            with tf.GradientTape() as t:
                z = tf.random.normal((tf.shape(real_batch)[0], self.latent_dim))
                fake_batch = self.generator(z)
                logits_real = self.discriminator(real_batch)
                logits_fake = self.discriminator(fake_batch)
                real_loss = cross_entropy(tf.ones_like(logits_real), logits_real)
                fake_loss = cross_entropy(tf.zeros_like(logits_fake), logits_fake)
                d_loss = real_loss + fake_loss
            grads = t.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # update the generator
        with tf.GradientTape() as t:
            z = tf.random.normal((tf.shape(real_batch)[0], self.latent_dim))
            fake_batch = self.generator(z)
            logits_fake = self.discriminator(fake_batch)
            g_loss = cross_entropy(tf.ones_like(logits_fake), logits_fake)
        grads = t.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.generator.trainable_variables))

        return d_loss, g_loss