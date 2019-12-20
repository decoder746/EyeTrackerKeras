import math

import cv2
import numpy as np
import project_path as pp
import os
import tensorflow.compat.v1.keras
import csv


class DataBatcher(tensorflow.compat.v1.keras.utils.Sequence):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2

    def __init__(self, batch_size=100, type=0, shuffle=True):
        self.batch_size = batch_size
        self.csv_file_path = None
        if type == 0:
            self.csv_file_path = os.path.join(pp.generated_folder_path, 'train.csv')
        elif type == 1:
            self.csv_file_path = os.path.join(pp.generated_folder_path, 'test.csv')
        else:
            self.csv_file_path = os.path.join(pp.generated_folder_path, 'validation.csv')

        with open(self.csv_file_path) as file:
            csv_reader = csv.reader(file, delimiter=',')
            self.fields = next(csv_reader)
            self.records = []
            for row in csv_reader:
                row[2:6] = [int(item) for item in row[2:6]]
                row[6:] = [float(item) for item in row[6:]]
                self.records.append(row)

        self.number_of_instances = len(self.records)
        self.number_of_batches = int(math.ceil(self.number_of_instances / self.batch_size))

        self.shuffle = shuffle
        self.on_epoch_end() 

    def __len__(self):
        return self.number_of_batches

    def __getitem__(self, index):
        record_idxes = self.shuffle_idxes[index * self.batch_size:min(self.number_of_instances, (index + 1) * self.batch_size)]
        # print(len(record_idxes),self.number_of_instances,self.number_of_batches)
        right_eyes = []
        left_eyes = []
        faces = []
        face_grids = []
        Y = []

        for record_idx in record_idxes:
            record = self.records[record_idx]
            right_eyes.append(cv2.imread(
                os.path.join(pp.gaze_capture_dataset_folder_path, str(record[0]).zfill(5), 'appleRightEye', record[1] + '.jpg')))
            # print(os.path.join(pp.gaze_capture_dataset_folder_path, str(record[0]).zfill(5), 'appleRightEye', record[1] + '.jpg'))
            left_eyes.append(cv2.imread(
                os.path.join(pp.gaze_capture_dataset_folder_path, str(record[0]).zfill(5), 'appleLeftEye', record[1] + '.jpg')))
            faces.append(cv2.imread(
                os.path.join(pp.gaze_capture_dataset_folder_path, str(record[0]).zfill(5), 'appleFace', record[1] + '.jpg')))
            face_grid = np.zeros(shape=(25, 25))
            face_grid[record[3]:record[3] + record[5] + 1, record[2]:record[2] + record[4] + 1] = 1
            face_grids.append(np.reshape(face_grid, newshape=(25 * 25, 1)))

            Y.append(np.reshape(np.array(record[6:]), newshape=(2)))

        return [np.array(right_eyes), np.array(left_eyes), np.array(faces), np.array(face_grids)], np.array(Y)

    def on_epoch_end(self):
        self.shuffle_idxes = np.arange(self.number_of_instances)
        if self.shuffle == True:
            np.random.shuffle(self.shuffle_idxes)