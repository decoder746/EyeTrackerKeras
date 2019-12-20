from scipy.io import loadmat
import project_path as pp
import os
import csv
import cv2

metadata_mat = loadmat(pp.metadata_file_path)
progress_step = 1000

csv_file_headers = [
    'subject_folder',
    'image_file_name',
    'face_grid_x',
    'face_grid_y',
    'face_grid_w',
    'face_grid_h',
    'label_x',
    'label_y'
]

total_number_instances = metadata_mat['frameIndex'].shape[0]
total_train_instances = 0
total_test_instances = 0
total_validation_instances = 0

train_file_path = os.path.join(pp.generated_folder_path, 'train.csv')
if os.path.isfile(train_file_path):
    os.remove(train_file_path)

test_file_path = os.path.join(pp.generated_folder_path, 'test.csv')
if os.path.isfile(test_file_path):
    os.remove(test_file_path)

validation_file_path = os.path.join(pp.generated_folder_path, 'validation.csv')
if os.path.isfile(validation_file_path):
    os.remove(validation_file_path)

print('Initiating Conversion...')

with open(train_file_path, 'w', newline='') as trw, open(test_file_path, 'w', newline='') as tew, open(
        validation_file_path, 'w', newline='') as vaw:
    train_writer = csv.writer(trw, quoting=csv.QUOTE_NONE)
    train_writer.writerow(csv_file_headers)
    train_writer = csv.writer(trw, quoting=csv.QUOTE_NONNUMERIC)

    test_writer = csv.writer(tew, quoting=csv.QUOTE_NONE)
    test_writer.writerow(csv_file_headers)
    test_writer = csv.writer(tew, quoting=csv.QUOTE_NONNUMERIC)

    validation_writer = csv.writer(vaw, quoting=csv.QUOTE_NONE)
    validation_writer.writerow(csv_file_headers)
    validation_writer = csv.writer(vaw, quoting=csv.QUOTE_NONNUMERIC)

    for i in range(total_number_instances):
        if(int(metadata_mat['labelRecNum'][i][0]) in list([6,104,114])):
            record = []

            record.append(str(int(metadata_mat['labelRecNum'][i][0])).zfill(5))
            record.append(str(int(metadata_mat['frameIndex'][i][0])).zfill(5))
            print(str(int(metadata_mat['frameIndex'][i][0])))
            record.append(metadata_mat['labelFaceGrid'][i][0])
            record.append(metadata_mat['labelFaceGrid'][i][1])
            record.append(metadata_mat['labelFaceGrid'][i][2])
            record.append(metadata_mat['labelFaceGrid'][i][3])
            record.append(round(metadata_mat['labelDotXCam'][i][0], 5))
            record.append(round(metadata_mat['labelDotYCam'][i][0], 5))

            image_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, record[0], 'appleFace', record[1]+'.jpg')
            print(image_file_path)
            image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (224, 224))
            cv2.imwrite(image_file_path, image)

            image_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, record[0], 'appleLeftEye', record[1] + '.jpg')
            image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (224, 224))
            cv2.imwrite(image_file_path, image)

            image_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, record[0], 'appleRightEye', record[1] + '.jpg')
            image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (224, 224))
            cv2.imwrite(image_file_path, image)

            if metadata_mat['labelTrain'][i][0] == 1:
                train_writer.writerow(record)
                total_train_instances += 1
            elif metadata_mat['labelTest'][i][0] == 1:
                test_writer.writerow(record)
                total_test_instances += 1
            elif metadata_mat['labelVal'][i][0] == 1:
                validation_writer.writerow(record)
                total_validation_instances += 1

            if i % progress_step == 0 and i != 0:
                print('{} of {} Completed!'.format(i, total_number_instances))

print('='*80)
print('Summary')
print('='*80)
print('Total Training Instances Converted:', total_train_instances)
print('Total Testing Instances Converted:', total_test_instances)
print('Total Validation Instances Converted:', total_validation_instances)
print('-'*80)
print('{} of {} Completed!'.format(total_number_instances, total_number_instances))
print('='*80)
print('Done Converting!')
