...

# Definition der Sampling-Klasse für die Reparameterisierungstrick
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder
input_img = tf.keras.layers.Input(shape=input_shape)
input_img_padding = tf.keras.layers.ZeroPadding3D((1, 4, 3))(input_img)
y_labels = tf.keras.layers.Input(shape=(n_y,), name='class_labels')
y_labels_f = tf.keras.layers.Flatten()(y_labels)
x = tf.keras.layers.Conv3D(8, (3, 3, 3), activation='relu', padding='same')(input_img_padding)
x = tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same')(x)
x = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same')(input_img_padding)
x = tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same')(x)
x = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same')(x)
x = tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same')(x)
x = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same')(x)
x = tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same')(x)
x = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same')(x)
shape_before_flatten = tensorflow.keras.backend.int_shape(x)[1:]
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Concatenate()([x, y_labels_f])
encoded = tf.keras.layers.Dense(18, activation='relu')(x)

z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(encoded)
z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(encoded)
z = Sampling()([z_mean, z_log_var])

zu = tf.keras.layers.Flatten()(z)
z_cond = tf.keras.layers.Concatenate()([zu, y_labels_f])

encoder = keras.Model([input_img, y_labels], [z_mean, z_log_var, z, z_cond], name="encoder")
encoder.summary()

# Decoder
latent_inputs = tf.keras.layers.Input(shape=(z_cond_shape,), name="decoder_input")
x = tf.keras.layers.Dense(units=np.prod(shape_before_flatten), name="decoder_dense_1")(latent_inputs)
x = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(x)
x = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling3D((2, 2, 2))(x)
x = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling3D((2, 2, 2))(x)
x = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling3D((2, 2, 2))(x)
x = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling3D((1, 1, 1))(x)
decoder_outputs = tf.keras.layers.Conv3D(1, (3, 3, 3), activation='relu', padding='same')(x)
decoder_outputs = tf.keras.layers.Cropping3D((1, 4, 3))(decoder_outputs)

decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# VAE (Variational Autoencoder) Klasse
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            x, y = data[0]
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, z_cond = encoder(data)
            reconstruction = decoder(z_cond)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2, 3)))
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss *= -0.5
            kl_loss = (tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)))
            kl_loss *= KFactor
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            x, y = data[0]
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, z_cond = encoder(data)
            reconstruction = decoder(z_cond)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2, 3)))
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss *= -0.5
            kl_loss = (tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)))
            kl_loss *= KFactor
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

...
