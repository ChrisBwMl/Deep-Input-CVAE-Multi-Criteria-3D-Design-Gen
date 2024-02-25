...

"""
Encoder
"""
# Eingangsbildschicht mit Padding
input_img = tf.keras.layers.Input(shape=input_shape)
input_img_padding = tf.keras.layers.ZeroPadding3D((1, 4, 3))(input_img)
shape_before_flatten = tensorflow.keras.backend.int_shape(input_img_padding)[1:]

# Flachmachen des Eingangsbildes
x00 = tf.keras.layers.Flatten()(input_img_padding)

# Eingangsklassenlabels
y_labels = tf.keras.layers.Input(shape=(n_y,), name='class_labels')
y_labels_f = tf.keras.layers.Flatten()(y_labels)

# Encoder-Architektur
x = Dense(units=1024, activation='relu')(x00)
x = Dense(units=512, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=128, activation='relu')(x)
x = Dense(units=64, activation='relu')(x)
x = Dense(units=32, activation='relu')(x)
x = Dense(units=16, activation='relu')(x)

# Kombinieren von Features mit den Klassenlabels
x = tf.keras.layers.Concatenate()([x, y_labels_f])
encoded = tf.keras.layers.Dense(18, activation='relu')(x)

# Berechnung der Mittelwerte und Log-Varianzen für den VAE
z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(encoded)
z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(encoded)

# Reparameterisierungstrick
z = Sampling()([z_mean, z_log_var])
zu = tf.keras.layers.Flatten()(z)

# Kombinieren von z und den Klassenlabels
z_cond = tf.keras.layers.Concatenate()([zu, y_labels_f])

# Erstellen des Encoder-Modells
encoder = keras.Model([input_img, y_labels], [z_mean, z_log_var, z, z_cond], name="encoder")
encoder.summary()

"""
Decoder
"""
# Eingang für das latente Raum-Z
latent_inputs = tf.keras.layers.Input(shape=(z_cond_shape,), name="decoder_input")

# Decoder-Architektur
x = Dense(units=16, activation='relu')(latent_inputs)
x = Dense(units=32, activation='relu')(x)
x = Dense(units=64, activation='relu')(x)
x = Dense(units=128, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=512, activation='relu')(x)
x = Dense(units=1024, activation='relu')(x)

# Ausgangsdichte für die Entfaltung
x = Dense(units=np.prod(shape_before_flatten), name="decoder_dense_1")(x)
decoder_outputs = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(x)
decoder_outputs = tf.keras.layers.Cropping3D((1, 4, 3))(decoder_outputs)

# Erstellen des Decoder-Modells
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

# Erstellen des VAE-Modells
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002))

# Modelltraining
history = vae.fit([x_train, y_train], epochs=epochs_no, batch_size=batch_size, validation_split=0.3)


...