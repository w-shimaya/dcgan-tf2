import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L

# image size is assumed to be 32x32
class WDCGANGP(K.Model):
    def __init__(self, latent_dim):
        super(WDCGANGP, self).__init__()
        self.latent_dim = latent_dim

        self.generator = K.Sequential([
            L.Dense(2*2*1024, input_shape=(latent_dim, ), use_bias=False),
            L.Reshape((2, 2, 1024)), 
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

        # No batchnorms in critic (see original paper)
        self.critic = K.Sequential([
            L.Conv2D(128, 5, 2, input_shape=(32, 32, 3), padding="same"), 
            L.LeakyReLU(0.2), 
            L.Conv2D(256, 5, 2, padding="same", use_bias=False), 
            #L.BatchNormalization(), 
            L.LeakyReLU(0.2), 
            L.Conv2D(512, 5, 2, padding="same", use_bias=False), 
            #L.BatchNormalization(), 
            L.LeakyReLU(0.2), 
            L.Conv2D(1024, 5, 2, padding="same", use_bias=False), 
            #L.BatchNormalization(), 
            L.LeakyReLU(0.2), 
            L.Flatten(), 
            L.Dense(1)
        ])
        self.g_opt = K.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.9)
        self.c_opt = K.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.9)

    def set_weights(self, model):
        self.generator.set_weights(model.generator.get_weights())
        self.critic.set_weights(model.critic.get_weights())

    # TODO: delete argument n_critic 
    @tf.function
    def train_on_batch(self, real_batch_list, n_critic=5, gp_lambda=10.0):
        batch_size = tf.shape(real_batch_list[0])[0]
        n_critic = len(real_batch_list)
        # update the critic n_critic times
        for real_batch in real_batch_list:
            with tf.GradientTape() as t:
                critic_loss, gp = self.critic_loss_fn(real_batch)
                loss = critic_loss + gp_lambda * gp
            critic_grad = t.gradient(loss, self.critic.trainable_variables)
            self.c_opt.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # update the generator
        with tf.GradientTape() as t:
            gen_loss = self.generator_loss_fn(batch_size)
        gen_grad = t.gradient(gen_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

        return critic_loss, gen_loss

    def assign_weights(self, generator_values, critic_values):
        for w, v in zip(self.generator.trainable_variables, generator_values):
            w.assign(v)
        
        for w, v in zip(self.critic.trainable_variables, critic_values):
            w.assign(v)

    def get_weights(self):
        generator_weights = [tf.Variable(w) for w in self.generator.trainable_variables]
        critic_weights = [tf.Variable(w) for w in self.critic.trainable_variabels]

        return generator_weights, critic_weights

    def critic_loss_fn(self, real_batch):
        z = tf.random.normal((tf.shape(real_batch)[0], self.latent_dim))
        eps = tf.random.uniform((tf.shape(real_batch)[0], 1, 1, 1), minval=0., maxval=1.)
        fake_batch = self.generator(z, training=False)
        with tf.GradientTape() as tt:
            blend = eps * real_batch + (1. - eps) * fake_batch
            tt.watch(blend)
            gp = self.critic(blend)
        grad = tt.gradient(gp, blend)
        norm_grad = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))

        critic_loss = tf.reduce_mean(self.critic(fake_batch) - self.critic(real_batch))
        gradient_penalty = tf.reduce_mean(tf.square(norm_grad - 1.))
        return critic_loss, gradient_penalty

    def generator_loss_fn(self, batch_size):
        z = tf.random.normal((batch_size, self.latent_dim))
        fake_batch = self.generator(z)
        gen_loss = tf.reduce_mean(-self.critic(fake_batch, training=False))
        return gen_loss
