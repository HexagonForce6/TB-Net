
import os
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from manage_data import *


MODEL_SAVE_PATH = "Checkpoint/EfficientNet_Checkpoint"

NUM_CLASSES = 2
IMG_HEIGHT = 224
IMG_WIDTH = 224
L_R = 0.0001
NUM_EPOCH = 200
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 1 # Must be 1
BATCH_SIZE_TEST = 1 # Must be 1

SAVE_AFTER_EACH = 5
SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)


os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        TRAIN_IMG_PATH,
        target_size = SIZE,
        batch_size = BATCH_SIZE_TRAIN,
        class_mode = "categorical"
)

val_generator = val_datagen.flow_from_directory(
        VAL_IMG_PATH,
        target_size = SIZE,
        batch_size = BATCH_SIZE_VAL,
        class_mode = "categorical"
)

# model saving specifications
class CustomSaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
                # for different augmentation
                do_preprocess("train") 
                
                if epoch % SAVE_AFTER_EACH == 0:
                        self.model.save(os.path.join(MODEL_SAVE_PATH, "efficientnet_{}.hd5".format(epoch)))
                        print("Saving checkpoint at epoch {}".format(epoch + 1))

# compile
conv_base = keras.applications.efficientnet.EfficientNetB0(
        weights=None, 
        classes=NUM_CLASSES,
        input_shape = INPUT_SHAPE
)

model = keras.models.Sequential()
model.add(conv_base)
model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax",name="fc_out"))

opt = keras.optimizers.Adam(learning_rate=L_R)
model.compile(optimizer=opt, loss="categorical_crossentropy", 
                metrics=["accuracy", keras.metrics.AUC()])
model.summary()

# save
saver = CustomSaver()
hist = model.fit(train_generator, epochs=NUM_EPOCH, validation_data = val_generator,
                verbose=1, callbacks=[saver])

