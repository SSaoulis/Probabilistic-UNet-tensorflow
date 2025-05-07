import tensorflow as tf
import tensorflow_probability as tfp

tf.config.run_functions_eagerly(True)


class ProbUNet(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.unet = UNet()
        self.prior_net = VariationalEncoder()
        self.posterior_net = VariationalEncoder()

        self.kl_beta = 10
        # self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.f_comb = tf.keras.layers.Conv2D(1, kernel_size=1, activation="softmax")

    def call(self, inputs, training=None, mask=None, num_samples=60):

        prior_mu, prior_sigma = self.prior_net(inputs)
        prior_dist = tfp.distributions.Normal(loc=prior_mu, scale=prior_sigma)
        unet_features = self.unet(inputs)
        sampled_segmentations = []
        # Predict a bunch of samples
        for _ in range(num_samples):
            z = prior_dist.sample()
            z_broadcasted = tf.broadcast_to(  # to [batch, H, W, latent_size]
                z,
                [
                    tf.shape(z)[0],
                    tf.shape(unet_features)[1],
                    tf.shape(unet_features)[2],
                    tf.shape(z)[-1],
                ],
            )
            concated_features = tf.concat(
                [unet_features, z_broadcasted], axis=-1
            )  # to [batch, H, W, latent_size + feature_channels]
            segmentation_layer = self.f_comb(concated_features)  # to [batch, H, W, 1]
            sampled_segmentations.append(segmentation_layer)

        sampled_segmentations = tf.concat(
            sampled_segmentations, axis=-1
        )  # to [batch, H, W, 1, num_samples]
        mean_segmentations = tf.reduce_mean(
            sampled_segmentations, axis=-1
        )  # to [batch, H, W, 1]

        return mean_segmentations

    def train_step(self, data):
        x, labels = data

        with tf.GradientTape() as tape:
            prior_mu, prior_sigma = self.prior_net.call(x)

            posterior_mu, posterior_sigma = self.posterior_net.call(
                tf.concat([x, labels], axis=-1)
            )

            prior_dist = tfp.distributions.Normal(loc=prior_mu, scale=prior_sigma)
            posterior_dist = tfp.distributions.Normal(
                loc=posterior_mu, scale=posterior_sigma
            )

            kl_divergence = tfp.distributions.kl_divergence(posterior_dist, prior_dist)
            kl_divergence = tf.reduce_mean(tf.reduce_sum(kl_divergence, axis=-1))

            z = posterior_dist.sample()
            unet_features = self.unet(x)
            z_broadcasted = tf.broadcast_to(
                z,
                [
                    tf.shape(z)[0],
                    tf.shape(unet_features)[1],
                    tf.shape(unet_features)[2],
                    tf.shape(z)[-1],
                ],
            )

            combined_features = tf.concat([unet_features, z_broadcasted], axis=-1)
            segmentations = self.f_comb(combined_features)

            ce = self.loss(segmentations, labels)

            total_loss = ce + self.kl_beta * kl_divergence

        trainable_vars = (
            self.unet.trainable_variables
            + self.posterior_net.trainable_variables
            + self.prior_net.trainable_variables
        )

        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        del tape

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, padding="same")
        self.conv3 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, padding="same")
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x


class TransConvBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)

        self.upsample = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear"
        )
        self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size=3, padding="same")
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.upsample(inputs)
        x = self.conv(x)
        x = self.relu(x)
        return x


class UNet(tf.keras.models.Model):
    def __init__(self, num_blocks=4, base_channels=32, **kwargs):

        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.base_channels = base_channels

        self.enc_blocks = []
        channels = base_channels
        for i in range(num_blocks):
            self.enc_blocks.append(ConvBlock(channels))
            channels *= 2

        self.bottleneck = ConvBlock(channels)

        self.dec_blocks = []
        self.up_blocks = []
        for i in range(num_blocks):
            channels //= 2
            self.up_blocks.append(TransConvBlock(channels))
            self.dec_blocks.append(ConvBlock(channels))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        skip_connections = []

        for enc in self.enc_blocks:
            x = enc(x)
            skip_connections.append(x)
            x = tf.image.resize(
                x, size=[x.shape[1] // 2, x.shape[2] // 2], method="bilinear"
            )

        x = self.bottleneck(x)

        for up, dec in zip(self.up_blocks, self.dec_blocks):
            x = up(x)
            skip = skip_connections.pop()

            if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
                x = tf.image.resize(
                    x, size=[skip.shape[1], skip.shape[2]], method="bilinear"
                )
            x = tf.concat([x, skip], axis=-1)
            x = dec(x)

        return x


class VariationalEncoder(tf.keras.models.Model):
    def __init__(self, num_blocks=4, base_channels=32, latent_dim=6, **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.base_channels = base_channels

        self.enc_blocks = []
        channels = base_channels
        for i in range(num_blocks):
            self.enc_blocks.append(ConvBlock(channels))
            channels *= 2

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

        self.param_conv = tf.keras.layers.Conv2D(
            2 * latent_dim, kernel_size=1, activation=None
        )

    def call(self, inputs):
        x = inputs
        for enc in self.enc_blocks:
            x = enc(x)
            x = tf.image.resize(
                x, size=[x.shape[1] // 2, x.shape[2] // 2], method="bilinear"
            )

        x = self.global_pool(x)

        x = tf.expand_dims(tf.expand_dims(x, 1), 1)
        params = self.param_conv(x)

        mu, log_sigma = tf.split(params, num_or_size_splits=2, axis=-1)
        sigma = tf.exp(log_sigma)
        return mu, sigma
