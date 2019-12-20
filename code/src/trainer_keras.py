import os
import pandas as pd
import cv2
import project_path as pp
import model_keras
import time
import numpy as np
from tensorflow.compat.v1.keras.models import load_model
from dataloader_keras import DataBatcher
from custom_metrics import min_deviation, mean_deviation, max_deviation, std
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from keras.callbacks import ModelCheckpoint
    from keras import optimizers
print('TF "Future Warnings" Suppressed!')

# Hyper Parameters
# # pretrained_model_file_path = None
# pretrained_model_file_path = os.path.join(pp.trained_models_folder_path, 'Instance_014', 'Keras', 'Model-017-4.506.hdf5')
NUM_EPOCHS = 30
BATCH_SIZE = 200
LEARNING_RATE = 0.01
train_csv = "../generated/train.csv"
csv = pd.read_csv(train_csv,skiprows=0)
npcsv = np.array(csv,dtype='float32')
pretrained_model_file_path = "../trained models/Keras/Model-030-5.438.hdf5"
sumi = 0
maxim = 0
minim = 10000
count = 0
optimizer = optimizers.Adagrad(lr=LEARNING_RATE)

if pretrained_model_file_path == None or not (os.path.isfile(pretrained_model_file_path) and pretrained_model_file_path.endswith('.hdf5')):
    model = model_keras.get_complete_model(save_summary=True)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=[min_deviation, mean_deviation, max_deviation]
                  )
else:
    model = load_model(pretrained_model_file_path, custom_objects={
        'min_deviation': min_deviation,
        'mean_deviation': mean_deviation,
        'max_deviation': max_deviation,
        'std': std
    })
    print('Loaded model using:', pretrained_model_file_path)

for i in range(npcsv.shape[0]):
    face_directory = "../../frames/" + str(int(npcsv[i][0])).zfill(5)+"/appleFace"
    left_eye_directory = "../../frames/"+ str(int(npcsv[i][0])).zfill(5)+"/appleLeftEye"
    right_eye_directory = "../../frames/"+ str(int(npcsv[i][0])).zfill(5)+"/appleRightEye"



    # instance = 0
    # while os.path.isdir(os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3))):
    #     instance += 1
    # os.mkdir(os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3)))
    # keras_models_folder_path = os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3), 'Keras')
    # tflite_models_folder_path = os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3), 'TFLite')
    # os.mkdir(keras_models_folder_path)
    # os.mkdir(tflite_models_folder_path)


    # Generators
    # training_generator = DataBatcher(batch_size=BATCH_SIZE, type=DataBatcher.TRAIN)
    # validation_generator = DataBatcher(batch_size=BATCH_SIZE, type=DataBatcher.VALIDATION)

    # Design model
    # optimizer = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.99, amsgrad=1e-08)
    index = (int)(npcsv[i][1])
    face_path = os.path.join(face_directory,str(index).zfill(5)+".jpg")
    left_eye_path = os.path.join(left_eye_directory,str(index).zfill(5)+".jpg")
    right_eye_path = os.path.join(right_eye_directory,str(index).zfill(5)+".jpg")
    input_shape = [1,224,224,3]
    input_shape2 = [1,625,1]
    print(face_path)
    image_dim = 25
    x = int(npcsv[i][2])
    y = int(npcsv[i][3])
    w = int(npcsv[i][4])
    h = int(npcsv[i][5])
    face_mask = np.zeros((image_dim,image_dim),dtype='float32')
    for k in range(image_dim):
        for j in range(image_dim):
            if k>=x and k<x+w and j>=y and j<y+h : 
                face_mask[k,j] = 1
    input_data1 = np.reshape(cv2.imread(face_path,cv2.IMREAD_COLOR),input_shape).astype('float32')
    input_data2 = np.reshape(cv2.imread(left_eye_path,cv2.IMREAD_COLOR),input_shape).astype('float32')
    input_data3 = np.reshape(cv2.imread(right_eye_path,cv2.IMREAD_COLOR),input_shape).astype('float32')
    input_data4 = np.reshape(face_mask,input_shape2)
    start = time.process_time()
    output = model.predict([input_data3,input_data1,input_data2,input_data4])
    print(time.process_time() - start)
    print(output)
    x1 = output[0][0]
    y1 = output[0][1]
    x2 = npcsv[i][6]
    y2 = npcsv[i][7]
    final = ((x1-x2)**2 + (y1-y2)**2)**0.5
    sumi += final
    count+=1
    if final > maxim:
        maxim = final
    if final < minim:
        minim = final
    print("Min is ",minim,'\n')
    print("Max is ",maxim,'\n')
    print("Average is ",str(sumi/count))
    # Saving Checkpoints
    # trained_model_file_path = os.path.join(keras_models_folder_path, 'Model-{epoch:03d}-{val_loss:.3f}.hdf5')
    # checkpoint = ModelCheckpoint(trained_model_file_path,
    #                              monitor='val_loss',
    #                              verbose=0,
    #                              save_best_only=False,
    #                              mode='auto',
    #                              period=1)

    # callbacks_list = [checkpoint]

    # # Train model on dataset
    # print('Commencing Training')
    # model.fit_generator(generator=training_generator,
    #                     epochs=NUM_EPOCHS,
    #                     validation_data=validation_generator,
    #                     callbacks=callbacks_list,
    #                     workers=5)
    # print('Training Completed!')


print("="*50+'\n')
print(sumi)
print(count)
print("Min is ",minim,'\n')
print("Max is ",maxim,'\n')
print("Average is ",str(sumi/count))