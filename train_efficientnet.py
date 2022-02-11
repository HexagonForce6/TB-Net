
import os
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from manage_data import *
from dsi import *


MODEL_SAVE_PATH = "Checkpoint"

L_R = 0.0001
NUM_EPOCH = 201
SAVE_AFTER_EACH = 5
SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

dsi = TBNetDSI(DATASET_FLAT_PATH)
train_dataset, train_dataset_size, train_batch_size = dsi.get_train_dataset()
val_dataset, _, _ = dsi.get_validation_dataset()
test_dataset, _, _ = dsi.get_test_dataset()

# model saving specifications
class CustomSaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
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
hist = model.fit(train_dataset, epochs=NUM_EPOCH, validation_data = val_dataset,
                verbose=1, callbacks=[saver], steps_per_epoch = int(train_dataset_size/train_batch_size))
