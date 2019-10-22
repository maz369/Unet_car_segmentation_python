# This is the main code for 2D binary segmentation of car in a grayscale image. It is based on original
# Unet implemented in Keras with Tensorflow backend.

# The following will run the code and save the trained model, learning curve and random example from 
# validation set.
# defult value for image size is 64x64, batch_size=5, number of epochs=50

# python main.py -im_h 64 -im_w 64 -batch_size=5 -epoch=50

# Using a higher resolution images (e.g. 256x256) with more epochs (e.g. 100) improves the accuracy
# of the result. The defult values are good for obtaining reasonable result to understand the concept
# and fine tuning the parameters.

# Written by Maz M. Khansari - summer 2019
# maziyar.khansari@gmail.com
#======================================================================================================


# import required libraries
from my_imports import *
from pre_process import load_image
from my_model import *



# define flag for input image size
parser = argparse.ArgumentParser()
parser.add_argument("-im_h", "--im_height", help="height of image", default=64, type=int)
parser.add_argument("-im_w", "--im_width", help="width of image", default=64, type=int)
parser.add_argument("-batch_size", "--batch_size", help="size of image batch", default=5, type=int)
parser.add_argument("-epoch", "--num_epoch", help="number of epochs", default=50, type=int)
args = parser.parse_args()

im_height = args.im_height
im_width = args.im_width
batch_size = args.batch_size
epochs = args.num_epoch


imgs_np, masks_np = load_image(im_height, im_width)
print(imgs_np[0].shape)

# split dataset for test and train
X_train, X_valid, y_train, y_valid = train_test_split(imgs_np, masks_np, test_size=0.2, random_state=1)

# define call back to save model and perform early stopping if loss does not imporove after 10 training epoch
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
callbacks = [EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-car_segmentation.h5', verbose=1, save_best_only=True, save_weights_only=True)]
	
	
# select input size and model parameters. We used keras built-in binary crossentropy loss
input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=2, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()


# train the model, select batch size and number of epoches
results = model.fit(X_train, y_train, batch_size, epochs, verbose=2, callbacks=None,
                    validation_data=(X_valid, y_valid))
					
					
					
# plot training and validation loss, then highlight best model
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
plt.savefig('x.jpg')

# make prediction image based on the trained network
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

# model evaluation
loss, acc =  model.evaluate(X_valid, y_valid, batch_size=30, verbose=1)
print('Evaluation loss is equal to: {0:.2f}'.format(loss))
print('Evaluation accuracy is equal to: {0:.2f}'.format(acc))


# save sample prediction results# show examples from validation dataset
rand_idx = random.choices(range(0,len(X_valid)), k=7)

for idx in rand_idx:
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(X_valid[idx].squeeze(), cmap='gray')
    ax[0].title.set_text('Original image')
    ax[1].imshow(y_valid[idx].squeeze(), cmap='gray')
    ax[1].title.set_text('Ground-truth mask')
    ax[2].imshow(preds_val[idx].squeeze(), cmap='gray')
    ax[2].title.set_text('Predicted mask')
    plt.savefig('prediction_{}.jpg'.format(idx))
