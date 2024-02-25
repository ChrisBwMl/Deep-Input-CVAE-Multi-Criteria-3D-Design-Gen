import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from mayavi import mlab
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from tensorflow.keras import layers
from tensorflow.keras.layers import ZeroPadding3D, Cropping3D, Dense, Flatten, Input, Add, Concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.constraints import unit_norm, max_norm
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

# Funktion, um eine Liste alphanumerisch zu sortieren
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

# Ausschalten AVX Prozessor Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Limiting GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Beginn des Projekts
dataname = 'GenericDesignNo'
AnzahlDesigns = 10000
path = 'Designs/'

# Lesen der Ausgabedaten (Outputwerte)
OutputStatisch = pd.read_csv('Solutions/Ergebnisvektor.CSV', index_col=0)

DesignsUnsorted = os.listdir(path)

# Sortiere die Designs alphanumerisch
DesignsSorted = sorted_alphanumeric(DesignsUnsorted)

# Listen zum Zusammenfassen der npy-Modelle in einer Liste
InputX = []
OutputY = []
failureList = []

# Counter für nicht konsistente Designs
Counter = 0

# Zuordnung von Input zu Output
for DesignNo in range(1, AnzahlDesigns + 1):
    EigSp = OutputStatisch._get_value(DesignNo, 'SpannungMittelwert')
    GesVer = OutputStatisch._get_value(DesignNo, 'GesamtverformungMittelwert')
    WaermeDichte = OutputStatisch._get_value(DesignNo, 'WstromdichteMittelwert')
    TempMax = OutputStatisch._get_value(DesignNo, 'TemperaturMittelwert')
    OutletPressure = OutputStatisch._get_value(DesignNo, 'OutletPressure')
    ForceInZ = OutputStatisch._get_value(DesignNo, 'ForceInZ')
    Energie = OutputStatisch._get_value(DesignNo, 'EnergieproDesign')
    Prozent = OutputStatisch._get_value(DesignNo, 'Flaechen43')
    Voxel = OutputStatisch._get_value(DesignNo, 'Voxel')


    # Überprüfung auf fehlende oder nicht konsistente Daten
    if np.isnan(EigSp) or np.isnan(GesVer) or np.isnan(WaermeDichte) or np.isnan(TempMax) or np.isnan(OutletPressure) or np.isnan(ForceInZ) or EigSp > 0.23 or GesVer > 0.05 or TempMax > 460 or WaermeDichte > 0.13 or OutletPressure > 2100 or ForceInZ > 900:
        print('Aktuelles DesignNo:', DesignNo, 'Keinen Eintrag oder nicht konsistent')
        Counter = Counter + 1
        print(Counter, '/', AnzahlDesigns)
        failureList.append([DesignNo])
        print(EigSp,GesVer,WaermeDichte,TempMax,OutletPressure,ForceInZ,Energie,Prozent,Voxel)
    else:
        pathDesign = path + '{}_{}_connected.npy'.format(dataname, DesignNo)
        DesignArray = np.load(pathDesign)
        x = DesignArray
        DesignArray = np.where(x == 0, x, x == 1)
        # Shape anpassen 30,40,42
        ES = np.zeros((30, 40, 1))
        DesignArray = np.dstack((DesignArray, ES))
        InputX += [(DesignArray)]
        OutputY += [(EigSp,GesVer,WaermeDichte,TempMax,OutletPressure,ForceInZ,Energie,Prozent,Voxel)]
        '''Spiegelung um die X-Achse'''
        DesignArray = np.flip(DesignArray, 0)
        InputX += [(DesignArray)]
        OutputY += [(EigSp,GesVer,WaermeDichte,TempMax,OutletPressure,ForceInZ,Energie,Prozent,Voxel)]

NotGoodDesigns = Counter / AnzahlDesigns * 100
print(NotGoodDesigns, '%', '  Der Designs hatten keine konsistenten Daten')

# Normalisierung der Output-Daten
pdOutputY = pd.DataFrame(OutputY)
pdOutputY = (pdOutputY - pdOutputY.min()) / (pdOutputY.max() - pdOutputY.min())
OutputY = list(pdOutputY.itertuples(index=False, name=None))


# Speichern von InputX und OutputY als Arrays
InputX = np.array(InputX)
OutputY = np.array(OutputY).astype(float)
np.save('OutputY', OutputY)


# Aufteilen in Trainings-, Test- und Validierungsdaten (werden vor dem Training schon separiert)
x_train, x_test, y_train, y_test = train_test_split(InputX, OutputY, test_size=0.3, random_state=42)

# Hinzufügen einer Dimension für den Kanal
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

# Umwandeln der Datentypen in float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Speichern der Daten
np.save('x_train', x_train)
np.save('x_test', x_test)
np.save('y_train', y_train)
np.save('y_test', y_test)

# Sampling mit Reparametrisierung
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Definition der Parameter für das Netzwerk
batch_size = 128
epochs_no = 200
latent_dim = 18
input_shape = (30, 40, 42, 1)
KFactor = 1
n_y = 9
z_cond_shape = 274



"""
Encoder
"""
input_img = tf.keras.layers.Input(shape=input_shape)
input_img_padding = tf.keras.layers.ZeroPadding3D((1,4,3))(input_img)
shape_before_flatten = tensorflow.keras.backend.int_shape(input_img_padding)[1:]
x00 = tf.keras.layers. Flatten()(input_img_padding)

enc_in_spannung1 = keras.Input(shape=(1,) ,name='SpannungMax')
enc_in_verformungmax1 = keras.Input(shape=(1,) ,name='VerformungMax')
enc_in_waermestrom1 = keras.Input(shape=(1,) ,name='WaermestromdichteMax')
enc_in_tempmax1 = keras.Input(shape=(1,) ,name='TempMax')
enc_in_outpress1 = keras.Input(shape=(1,) ,name='OutletPressure')
enc_in_forcei1 = keras.Input(shape=(1,) ,name='ForceInZ')
enc_in_energiede1 = keras.Input(shape=(1,) ,name='EnergieproDesign')
enc_in_prozentfl1 = keras.Input(shape=(1,) ,name='ProzentFlaechen43')
enc_in_voxel1 = keras.Input(shape=(1,) ,name='Voxel')

enc_in_spannung = Dense(units=2,activation='relu')(enc_in_spannung1)
enc_in_verformungmax = Dense(units=2,activation='relu')(enc_in_verformungmax1)
enc_in_waermestrom = Dense(units=2,activation='relu')(enc_in_waermestrom1)
enc_in_tempmax = Dense(units=2,activation='relu')(enc_in_tempmax1)
enc_in_outpress  = Dense(units=2,activation='relu')(enc_in_outpress1)
enc_in_forcei = Dense(units=2,activation='relu')(enc_in_forcei1)
enc_in_energiede = Dense(units=2,activation='relu')(enc_in_energiede1)
enc_in_prozentfl = Dense(units=2,activation='relu')(enc_in_prozentfl1)
enc_in_voxel = Dense(units=2,activation='relu')(enc_in_voxel1)

added1 = tf.keras.layers.Add()([enc_in_spannung,enc_in_verformungmax])
added2 = tf.keras.layers.Add()([enc_in_waermestrom,enc_in_tempmax])
added3 = tf.keras.layers.Add()([enc_in_outpress,enc_in_forcei])
added4 = tf.keras.layers.Add()([enc_in_energiede,enc_in_prozentfl])

# Input der 4 Kriterienkategorien (Atrukturmechanik, Ahermodynamik, Aerodynamik, additive Fertigun)

x1 = Dense(units=4,activation='relu')(added1)
x1 = Dense(units=8,activation='relu')(x1)
x1= Dense(units=16,activation='relu')(x1)
x1= Dense(units=32,activation='relu')(x1)
x1= Dense(units=64,activation='relu')(x1)
x1= Dense(units=128,activation='relu')(x1)
x1= Dense(units=256,activation='relu')(x1)

x2 = Dense(units=4,activation='relu')(added2)
x2 = Dense(units=8,activation='relu')(x2)
x2= Dense(units=16,activation='relu')(x2)
x2= Dense(units=32,activation='relu')(x2)
x2= Dense(units=64,activation='relu')(x2)
x2= Dense(units=128,activation='relu')(x2)
x2= Dense(units=256,activation='relu')(x2)

x3 = Dense(units=4,activation='relu')(added3)
x3 = Dense(units=8,activation='relu')(x3)
x3= Dense(units=16,activation='relu')(x3)
x3= Dense(units=32,activation='relu')(x3)
x3= Dense(units=64,activation='relu')(x3)
x3= Dense(units=128,activation='relu')(x3)
x3= Dense(units=256,activation='relu')(x3)

x4 = Dense(units=4,activation='relu')(added4)
x4 = Dense(units=8,activation='relu')(x4)
x4= Dense(units=16,activation='relu')(x4)
x4= Dense(units=32,activation='relu')(x4)
x4= Dense(units=64,activation='relu')(x4)
x4= Dense(units=128,activation='relu')(x4)
x4= Dense(units=256,activation='relu')(x4)

x5 = Dense(units=4,activation='relu')(enc_in_voxel)
x5 = Dense(units=8,activation='relu')(x5)
x5 = Dense(units=16,activation='relu')(x5)
x5 = Dense(units=32,activation='relu')(x5)
x5 = Dense(units=64,activation='relu')(x5)
x5 = Dense(units=128,activation='relu')(x5)
x5 = Dense(units=256,activation='relu')(x5)
added5 = tf.keras.layers.Add()([x1,x2,x3,x4,x5])

x = Dense(units=1024,activation='relu')(x00)
x= Dense(units=512,activation='relu')(x)
x= Dense(units=256,activation='relu')(x)
x= Dense(units=128,activation='relu')(x)
x= Dense(units=64,activation='relu')(x)
x= Dense(units=32,activation='relu')(x)
x= Dense(units=16,activation='relu')(x)
x = tf.keras.layers.Concatenate()([x, added5])
encoded = tf.keras.layers.Dense(18, activation='relu')(x)

z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(encoded)
z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(encoded)
z = Sampling()([z_mean, z_log_var])
zu = tf.keras.layers.Flatten()(z)
#x = Dense(units=16,activation='relu')(zu)
z_cond= tf.keras.layers.Concatenate()([zu, added5])

encoder = keras. Model([input_img, enc_in_spannung1,enc_in_verformungmax1,enc_in_waermestrom1,enc_in_tempmax1,enc_in_outpress1,enc_in_forcei1,enc_in_energiede1,enc_in_prozentfl1,enc_in_voxel1], [z_mean, z_log_var, z, z_cond], name="encoder")
encoder.summary()

"""
Decoder
"""
latent_inputs = tf.keras.layers.Input(shape=(z_cond_shape,),name="decoder_input")
x = Dense(units=16,activation='relu')(latent_inputs)
x = Dense(units=32,activation='relu')(x)
x= Dense(units=64,activation='relu')(x)
x= Dense(units=128,activation='relu')(x)
x= Dense(units=256,activation='relu')(x)
x= Dense(units=512,activation='relu')(x)
x= Dense(units=1024,activation='relu')(x)
x = Dense(units=np.prod(shape_before_flatten), activation='sigmoid')(x)
decoder_outputs = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(x)
decoder_outputs = tf.keras.layers.Cropping3D((1,4,3))(decoder_outputs)

decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


# Definition des gesamten Variational Autoencoder-Modells
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            x, yy, ss, pp, ll, cc, tt, kk, ee, ff = data[0]
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, z_cond = encoder(data)
            reconstruction = decoder(z_cond)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.mean_squared_error(x, reconstruction), axis=(1, 2, 3)))
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
            x, yy, ss, pp, ll, cc, tt, kk, ee, ff = data[0]
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, z_cond = encoder(data)
            reconstruction = decoder(z_cond)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.mean_squared_error(x, reconstruction), axis=(1, 2, 3)))
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

# Training des VAE-Modells
history = vae.fit([x_train, y_train[:, 0], y_train[:, 1], y_train[:, 2], y_train[:, 3], y_train[:, 4], y_train[:, 5],
                   y_train[:, 6], y_train[:, 7], y_train[:, 8]], epochs=epochs_no, batch_size=batch_size,
                  validation_split=0.3)

# Speichern der Gewichte
encoder.save_weights('enc_weights.h5')
decoder.save_weights('dec_weights.h5')

history.history

print("Training finished...")

#Loss Plott
plt.plot(vae.history.history['reconstruction_loss'])
plt.title('model re loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['vae_reconstruction_loss'], loc='upper left')
plt.show()

plt.plot(vae.history.history['kl_loss'])
plt.title('model kl loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['vae_kl_loss'], loc='upper left')
plt.show()

plt.plot(vae.history.history['loss'])
plt.plot(vae.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss','val_loss'], loc='upper left')
plt.show()

# Abspeichern z und z_test
encoder_outputs = encoder.predict([x_train, y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6],y_train[:,7],y_train[:,8]])
#encoder_outputs_test = encoder.predict(x_test, y_test)
z_pred = encoder_outputs[3]                 #alle z extrahieren
#z_pred_test = encoder_outputs_test[2]
np.save ('z', z_pred)                       # alle z abspeichern
#np.save ('z_test', z_pred_test)
print('z prediction:', z_pred.shape)

# Plot prediction x Design vgl. Original/Prediction
# Vergleichsdarstellung Prediction des Trainingsinput
x_pred = decoder.predict(z_pred)            #x prediction des Designs

i = 0                                       #Variable welches Design vergichen wird original/prediction
outputs = x_pred[i]                         # ein einzelnes Design nehmen zum plotten

outputs = np.squeeze(outputs, axis=3)       # Dimensions 1 am Ende verkürzen
print(outputs.shape)

#Koordinaten von dem Array entnehmen, sowie s Wert an dem Eintrag
m,n,o = outputs.shape
R,C,U = np.mgrid[:m,:n,:o]

x = R.ravel()
y = C.ravel()
z = U.ravel()
s = outputs.ravel()

# Vergleichsdarstellung Original Trainingsinput
outputs_real = x_train[i]
outputs_real = np.squeeze(outputs_real, axis=3)
s2 = outputs_real.ravel()

#plot der Original (s2)/Prediction (s)
fig = mlab.figure(size = (500,500), bgcolor=(1,1,1), fgcolor=(0.5,0.5,0.5))
mlab.points3d(x, y, z, s, scale_factor=1)

mlab.points3d(x + np.max(x)+10, y, z, s2, scale_factor=1)
mlab.show()


# Latent Space plot
encoder_mu = encoder_outputs[0]
print('this is encoder_mu', encoder_mu.shape)

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(encoder_mu)

MaxSpannung = [i[0] for i in y_train] # erstes Element aller Einträgen nehmen

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_2d[:, 0], X_2d[:, 1], c=MaxSpannung)
plt.title('TSNE visualization of latent space')
ax.axis('equal')
plt.show()
