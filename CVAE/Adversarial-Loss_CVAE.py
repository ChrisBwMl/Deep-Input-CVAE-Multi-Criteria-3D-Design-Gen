...

def count_connected_areas(reconstruction):
    # Zählung von verbundenen Bereichen im Konstruktionsvorschlag
    connected_components, _, count = tf.unique_with_counts(
        tf.reshape(tf.cast(reconstruction > 0.5, dtype=tf.int32), [-1])
    )
    # Annahme: Es gibt nur einen verbundenen Bereich
    count = tf.reduce_max(count)
    return count

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    # Diese Klasse nimmt Encoder- und Decoder-Modelle und
    # definiert die komplette Architektur des Variational Autoencoders
    def train_step(self, data):
        if isinstance(data, tuple):
            x, yy, ss, pp, ll, cc, tt, kk, ee, ff = data[0]
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, z_cond = self.encoder(data)
            reconstruction = self.decoder(z_cond)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mean_squared_error(x, reconstruction), axis=(1, 2, 3)))
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss *= -0.5
            kl_loss = (tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)))
            kl_loss *= KFactor

            # Trainieren des Diskriminators
            target_count = 1
            count = count_connected_areas(reconstruction)
            count_loss = tf.cast(tf.math.abs(count - target_count), tf.float32)
            count_loss = count_loss / (
                    tf.cast(x.shape[1], tf.float32) * tf.cast(x.shape[2], tf.float32) * tf.cast(x.shape[3], tf.float32))
            count_loss *= 10
            total_loss = reconstruction_loss + kl_loss + count_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "discriminator_loss": count_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            x, yy, ss, pp, ll, cc, tt, kk, ee, ff = data[0]
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, z_cond = self.encoder(data)
            reconstruction = self.decoder(z_cond)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.mean_squared_error(x, reconstruction), axis=(1, 2, 3)))
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss *= -0.5
            kl_loss = (tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)))
            kl_loss *= KFactor

            # Trainieren des Diskriminators
            target_count = 1
            count = count_connected_areas(reconstruction)
            count_loss = tf.cast(tf.math.abs(count - target_count), tf.float32)
            count_loss = count_loss / (
                    tf.cast(x.shape[1], tf.float32) * tf.cast(x.shape[2], tf.float32) * tf.cast(x.shape[3], tf.float32))
            count_loss *= 10
            total_loss = reconstruction_loss + kl_loss + count_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "discriminator_loss": count_loss,
        }
    
    ...