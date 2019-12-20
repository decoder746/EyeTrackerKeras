import os

project_folder_path = os.path.abspath('..')

data_folder_path = os.path.abspath('//home//anshul//Desktop//Winter//frames')
generated_folder_path = os.path.abspath(os.path.join(project_folder_path, 'generated'))
trained_models_folder_path = os.path.abspath(os.path.join(project_folder_path, 'trained models'))
resrc_folder_path = os.path.abspath(os.path.join(project_folder_path, 'resrc'))
src_folder_path = os.path.abspath(os.path.join(project_folder_path, 'src'))

inferences_folder_path = os.path.join(generated_folder_path, 'Inferences')

keras_models_folder_path = os.path.join(trained_models_folder_path, 'Keras')
tflite_models_folder_path = os.path.join(trained_models_folder_path, 'TFLite')



# Give folder path of GazeCapture Dataset here
gaze_capture_dataset_folder_path = data_folder_path

# Give metadata file path here
metadata_file_path = os.path.join(resrc_folder_path, 'reference_metadata.mat')

