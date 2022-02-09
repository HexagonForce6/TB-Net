
import numpy as np
import os
from sklearn.metrics import confusion_matrix,roc_auc_score
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


PREPROCESSED_PATH = "Preprocessed_Images"
TEST_IMG_PATH = os.path.join(PREPROCESSED_PATH, "test")
MODEL_SAVE_PATH = "Checkpoint/NasNet_Checkpoint"

NUM_CLASSES = 2
IMG_HEIGHT = 224
IMG_WIDTH = 224

BATCH_SIZE_TEST = 1 # Must be 1

SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)


test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
        TEST_IMG_PATH,
        target_size = SIZE,
        batch_size = BATCH_SIZE_TEST,
        class_mode = "categorical",
        shuffle = False
)

epoch = 90
model = keras.models.load_model(os.path.join(MODEL_SAVE_PATH, "nasnet_" + str(epoch) + ".hd5"))
print("Epoch " + str(epoch) + ":")

score = model.predict(generator=test_generator)
y_true = test_generator.classes
y_pred_raw = score > 0.5

# adjust the format of y_pred
y_pred = []
for i in y_pred_raw:
    if i[0]==True: y_pred.append(0)
    else: y_pred.append(1)
y_pred = np.array(y_pred)

# confusion matrix
matrix = confusion_matrix(y_true, y_pred)
print(matrix)

# calculate accuracy, sensitivity, specificity from confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
accuracy = (tn+tp) / (tn+fp+fn+tp)
sensitivity = tp / (tp+fn)
specificity = tn / (tn+fp)
print("Accuracy = " + str(accuracy))
print("sensitivity = " + str(sensitivity))
print("Specificity = " + str(specificity))

# auc_roc
auc_score = roc_auc_score(y_true, y_pred)
print("AUC score = " + str(auc_score) + "\n")
