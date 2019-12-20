import numpy as np
import warnings
import project_path as pp
import os
import cv2
import pandas as pd
from custom_metrics import euclidean_distance
from dataloader_keras import DataBatcher
import time
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf
print('TF "Future Warnings" Suppressed!')

BATCH_SIZE = 200

test_generator = DataBatcher(batch_size=10, type=DataBatcher.TRAIN)
face_directory = "../../frames/00104/appleFace"
left_eye_directory = "../../frames/00104/appleLeftEye"
right_eye_directory = "../../frames/00104/appleRightEye"
train_csv = "../generated/train.csv"

csv = pd.read_csv(train_csv,skiprows=0)
npcsv = np.array(csv,dtype='float32')
count = 0
sumi = 0
average = 0
std_dev = 0
minim = 10000
maxim = 0
for tflite_model_name in os.listdir(pp.tflite_models_folder_path):
	for i in range(npcsv.shape[0]):
	    if not tflite_model_name.endswith('tflite'):
	        continue

	    face_directory = "../../frames/" + str(int(npcsv[i][0])).zfill(5)+"/appleFace"
	    left_eye_directory = "../../frames/"+ str(int(npcsv[i][0])).zfill(5)+"/appleLeftEye"
	    right_eye_directory = "../../frames/"+ str(int(npcsv[i][0])).zfill(5)+"/appleRightEye"
	    tflite_model_file_path = os.path.join(pp.tflite_models_folder_path, tflite_model_name)

	    # Load TFLite model and allocate tensors.
	    interpreter = tf.lite.Interpreter(model_path=tflite_model_file_path)
	    interpreter.allocate_tensors()

	    # # Get input and output tensors.
	    input_details = interpreter.get_input_details()
	    output_details = interpreter.get_output_details()
	    # Test model on random input data.
	    input_shape = input_details[0]['shape']
	    print(input_details)
	    input_shape2 = input_details[3]['shape']
	    index = (int)(npcsv[i][1])
	    face_path = os.path.join(face_directory,str(index).zfill(5)+".jpg")
	    print(face_path)
	    # command = "python3 "+ "generate_face_mask.py " + str(int(npcsv[i][2]))+" " +str(int(npcsv[i][3]))+" "+str(int(npcsv[i][4]))+" "+str(int(npcsv[i][5]))
	    # os.system(command)
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
	    left_eye_path = os.path.join(left_eye_directory,str(index).zfill(5)+".jpg")
	    right_eye_path = os.path.join(right_eye_directory,str(index).zfill(5)+".jpg")
	    input_data1 = np.reshape(cv2.imread(face_path,cv2.IMREAD_COLOR),input_shape).astype('float32')
	    input_data2 = np.reshape(cv2.imread(left_eye_path,cv2.IMREAD_COLOR),input_shape).astype('float32')
	    input_data3 = np.reshape(cv2.imread(right_eye_path,cv2.IMREAD_COLOR),input_shape).astype('float32')
	    input_data4 = np.reshape(face_mask,input_shape2)
	    interpreter.set_tensor(input_details[0]['index'], input_data1)
	    interpreter.set_tensor(input_details[1]['index'], input_data2)
	    interpreter.set_tensor(input_details[2]['index'], input_data3)
	    interpreter.set_tensor(input_details[3]['index'], input_data4)
	    start = time.process_time()
	    interpreter.invoke()
	    print(time.process_time() - start)
	    print(interpreter.get_tensor(output_details[0]['index']))
	    x1 = interpreter.get_tensor(output_details[0]['index'])[0][0]
	    y1 = interpreter.get_tensor(output_details[0]['index'])[0][1]
	    x2 = npcsv[i][6]
	    y2 = npcsv[i][7]
	    final = ((x1-x2)**2 + (y1-y2)**2)**0.5
	    sumi += final
	    count+=1
	    if final > maxim:
	    	maxim = final
	    if final < minim:
	    	minim = final
	    print(sumi)
	    print(count)
	    print("Min is ",minim,'\n')
	    print("Max is ",maxim,'\n')
	    print("Average is ",str(sumi/count))
	    # # The function `get_tensor()` returns a copy of the tensor data.
	    # # Use `tensor()` in order to get a pointer to the tensor.
	    # output_data = interpreter.get_tensor(output_details[0]['index'])
	    # print(output_data)
print("="*50+'\n')
print(sumi)
print(count)
print("Min is ",minim,'\n')
print("Max is ",maxim,'\n')
print("Average is ",str(sumi/count))