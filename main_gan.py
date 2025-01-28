import os
import time, datetime, glob, os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from dcgan import *



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Remove logs INFO
time_ini = time.time()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

# Load the images, normalize them to [0,1] and convert them to np.array
def load_images(datapath, newsize):
    images = glob.glob(datapath)
    imarrays = []
    for image in images:
        img = load_img(image)#, grayscale=True) # uncomment for black and white images
        img = img.resize((newsize,newsize)) # Resize the images for speeding up computations
        img = img_to_array(img)/255.
        imarrays.append(img)
    imarrays = np.array(imarrays, dtype="float32")
    print("Shape of the images:", imarrays.shape)
    return imarrays


#--- PARAMETERS ---#
n_epochs = 15000 # Number of epochs
batch_size = 4 # Batch size
load_prev_model = 0 # 1 for loading a previously trained model
train_model = 1 # 1 for trainig the network
newsize = 1024 # Resize the images to save some memory and time 
#(It has to be multiple of 2, otherwise it fails sometimes...)

#--- MAIN ---#
# Ensure that you are using GPU if available. Otherwise, it will take some time to run
with tf.device('/GPU:0'):
    # Create some directories to store the outputs and models, and unzip the dataset
    if not os.path.isdir("outputs"):
        os.mkdir("outputs")
    if not os.path.isdir("models"):
        os.mkdir("models")
    if not os.path.isdir("outputs/individual_images"):
        os.mkdir("outputs/individual_images")

    # Load the data
    datapath = "images/DJI*"
    imarrays = load_images(datapath, newsize=1024)

    # Initialize the GAN
    model_dcgan = Image_DCGAN(imarrays, load_prev_model=load_prev_model)

    # Train the GAN
    if train_model:
        d_losses, a_losses, d_acc, a_acc = model_dcgan.train(train_steps=n_epochs, batch_size=batch_size, save_interval=500)

        # Save and plot the losses
        np.savetxt("outputs/losses.txt",np.transpose([d_losses, a_losses, d_acc, a_acc]))
        model_dcgan.plot_loss_acc(d_losses, a_losses, d_acc, a_acc)

    # Plot some true and fake sample images
    model_dcgan.plot_images(fake=True, save2file=True, individual=True)
    model_dcgan.plot_images(fake=True, save2file=True)
    model_dcgan.plot_images(fake=False, save2file=True)


print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))