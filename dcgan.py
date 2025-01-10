'''
DCGAN (Deep Convolutional Generative Adversarial Network) for generation of images either in RGB or grayscale
Implementation based on the code by Rowel Atienza https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
'''
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape, Conv2D, Conv2DTranspose, UpSampling2D, LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import array_to_img
#from keras._tf_keras.keras.preprocessing.image import array_to_img (versões antigas só)
#from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from tqdm import tqdm  # Biblioteca para mostrar o progresso
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Remove logs INFO
import tensorflow as tf




class DCGAN(object):
    def __init__(self, rows, cols, channels):

        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels

        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator_block(self, depth, input_shape=None, alpha=0.2, dropout = 0.4):

        block = Sequential()
        if input_shape:
            block.add(Conv2D(depth, 5, strides=2, input_shape=input_shape, padding='same'))
        else:
            block.add(Conv2D(depth, 5, strides=2, padding='same'))
        block.add(LeakyReLU(alpha=alpha))
        block.add(Dropout(dropout))

        return block

    # Define the discriminator network
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        input_shape = (self.img_rows, self.img_cols, self.channels)

        self.D.add( self.discriminator_block(depth, input_shape=input_shape) )
        self.D.add( self.discriminator_block(depth*2) )
        self.D.add( self.discriminator_block(depth*4) )
        self.D.add( self.discriminator_block(depth*8) )

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))

        print("Discriminator summary")
        self.D.summary()
        #plot_model(self.D, "model.png") # Plot model

        return self.D

    def generator_block(self, depth, upsampling=True):

        block = Sequential()
        if upsampling:
            block.add(UpSampling2D())
        block.add(Conv2DTranspose(int(depth), 5, padding='same'))
        block.add(BatchNormalization(momentum=0.9))
        block.add(Activation('relu'))

        return block

    # Define the generator network
    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64*4
        dim = int(self.img_rows/4) # change for non-square images
        input_dim = 100

        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=input_dim))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Each generator block halves the depth, and doubles the dimensions if there is upsampling
        # Final image will be dim*4=self.img_rows
        self.G.add( self.generator_block(depth/2) )
        self.G.add( self.generator_block(depth/4) )
        self.G.add( self.generator_block(depth/8, upsampling=False) )

        # Out: self.img_rows x self.img_rows x self.channels image
        self.G.add(Conv2DTranspose(self.channels, 5, padding='same'))
        self.G.add(Activation('sigmoid'))

        print("Generator summary")
        self.G.summary()

        return self.G

    # Define the discriminator model
    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(learning_rate=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    # Define the adversarial model (generator+discriminator)
    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(learning_rate=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        disc = self.discriminator()
        disc.trainable = False
        self.AM.add(disc)
        #self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.AM.summary()
        return self.AM

class Image_DCGAN(object):
    def __init__(self, images, load_prev_model=False):

        self.x_train = images
        self.img_rows = images.shape[1]
        self.img_cols = images.shape[2]
        self.channels = images.shape[3] # 3 if RGB, 1 if grayscale

        self.DCGAN = DCGAN(rows=self.img_rows, cols=self.img_cols, channels=self.channels)

        if load_prev_model:
            """
            step = 15000
            loaded_dis = load_model("models/discriminator_step_%d.h5"%step)
            loaded_adv = load_model("models/adversarial_step_%d.h5"%step)
            loaded_gen = load_model("models/generator_step_%d.h5"%step)
            """
            loaded_dis = load_model("models/discriminator_last.h5")
            loaded_adv = load_model("models/adversarial_last.h5")
            loaded_gen = load_model("models/generator_last.h5")

            self.discriminator =  loaded_dis
            self.adversarial = loaded_adv
            self.generator = loaded_gen

        else:
            self.discriminator =  self.DCGAN.discriminator_model()
            self.adversarial = self.DCGAN.adversarial_model()
            self.generator = self.DCGAN.generator()

    def train(self, train_steps=10, batch_size=64, save_interval=0):
        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(0., 1.0, size=[16, 100])

        d_losses, a_losses, d_acc, a_acc = [], [], [], []

        # Adicione tqdm para visualização do progresso
        for i in tqdm(range(train_steps), desc="Training Progress", unit="epoch"):
            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(0., 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)

            # Concatenate real and fake images for computing training the discriminator
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0  # Fake images labeled as 0
            d_loss = self.discriminator.train_on_batch(x, y)

            # Adversarial model training with flipped labels
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(0., 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)

            # Print the current loss and accuracy
            log_mesg = f"{i + 1}/{train_steps}: [D loss: {d_loss[0]:.4f}, acc: {d_loss[1]:.4f}]"
            log_mesg += f" [A loss: {a_loss[0]:.4f}, acc: {a_loss[1]:.4f}]"
            print(log_mesg)

            d_losses.append(d_loss[0])
            a_losses.append(a_loss[0])
            d_acc.append(d_loss[1])
            a_acc.append(a_loss[1])

            # Plot images at intervals
            if save_interval > 0 and (i + 1) % save_interval == 0:
                self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i + 1))

        # Save the final model after training
        self.generator.save("models/generator_last.h5")
        self.discriminator.save("models/discriminator_last.h5")
        self.adversarial.save("models/adversarial_last.h5")

        return d_losses, a_losses, d_acc, a_acc

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'outputs/image_sample'#.png'

        if fake:
            filename+="_fake.png"
            if noise is None:
                noise = np.random.uniform(0., 1.0, size=[samples, 100])
            else:
                filename = "outputs/image_step_%d.png" % step
            images = self.generator.predict(noise)
        else:
            filename+="_true.png"
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            #image = np.reshape(image, [self.img_rows, self.img_cols])
            img = array_to_img(image)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


    def plot_loss_acc(self, d_losses, a_losses, d_acc, a_acc):

        fig, [ax1, ax2] = plt.subplots(2,1,sharex=True)
        fig.subplots_adjust(hspace=0)

        ax1.plot(d_losses,label="Discriminator losses")
        ax1.plot(a_losses,label="Adversarial losses")
        ax2.plot(d_acc,label="Discriminator accuracy")
        ax2.plot(a_acc,label="Adversarial accuracy")

        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        #ax1.set_xlabel("Epochs")
        ax2.set_xlabel("Epochs")
        ax1.legend()
        ax2.legend()
        ax1.set_yscale("log")
        ax1.set_ylim([8.e-5,10.])
        fig.savefig("outputs/losses.pdf")
