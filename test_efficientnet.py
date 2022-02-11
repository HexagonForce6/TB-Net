
import numpy as np
import os
import csv
from sklearn.metrics import confusion_matrix,roc_auc_score
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from dsi import *
import tensorflow_datasets as tfds


MODEL_SAVE_PATH = "Checkpoint"
RESULT_SAVE_CSV = 'efficientnet_test_results_200.csv'

dsi = TBNetDSI("Dataset_Flat")
test_dataset, _, _ = dsi.get_test_dataset()

tf.compat.v1.enable_eager_execution()

with open(RESULT_SAVE_CSV, mode = 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|')
    csv_writer.writerow(['Epoch', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC_ROC'])

    for epoch in tqdm(range(0, 201, 5)):
        model = keras.models.load_model(os.path.join(MODEL_SAVE_PATH, "efficientnet_" + str(epoch) + ".hd5"))
        #print("Epoch " + str(epoch) + ":")

        score = model.predict(test_dataset, batch_size = BATCH_SIZE_TEST)

        # fetch labels
        y_true = []
        for sample in tfds.as_numpy(test_dataset):
            label = int(sample[1][0][0])
            if label == 1: y_true.append(0)
            else: y_true.append(1)
        #print(y_true)

        y_pred_raw = score > 0.5

        # adjust the format of y_pred to match y_true
        y_pred = []
        for i in y_pred_raw:
            if i[0]==True: y_pred.append(0)
            else: y_pred.append(1)
        y_pred = np.array(y_pred)
        #print(y_pred)

        # confusion matrix
        matrix = confusion_matrix(y_true, y_pred)
        #print(matrix)

        # calculate accuracy, sensitivity, specificity from confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = (tn+tp) / (tn+fp+fn+tp)
        sensitivity = tp / (tp+fn)
        specificity = tn / (tn+fp)
        # print("Accuracy = " + str(accuracy))
        # print("sensitivity = " + str(sensitivity))
        # print("Specificity = " + str(specificity))

        # auc_roc
        auc_score = roc_auc_score(y_true, y_pred)
        #print("AUC score = " + str(auc_score) + "\n")

        csv_writer.writerow([epoch, tn, fp, fn, tp, accuracy, sensitivity, specificity, auc_score])
