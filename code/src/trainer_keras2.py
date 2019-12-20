import os
import project_path as pp
import model_keras
from tensorflow.compat.v1.keras.models import load_model
from dataloader_keras import DataBatcher
from custom_metrics import min_deviation, mean_deviation, max_deviation, std
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint
    from tensorflow.compat.v1.keras import optimizers
print('TF "Future Warnings" Suppressed!')

# Hyper Parameters
# pretrained_model_file_path = None
pretrained_model_file_path = os.path.join(pp.trained_models_folder_path, 'Instance_014', 'Keras', 'Model-017-4.506.hdf5')
NUM_EPOCHS = 30
BATCH_SIZE = 10
LEARNING_RATE = 0.01


instance = 0
while os.path.isdir(os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3))):
    instance += 1
os.mkdir(os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3)))
keras_models_folder_path = os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3), 'Keras')
tflite_models_folder_path = os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3), 'TFLite')
os.mkdir(keras_models_folder_path)
os.mkdir(tflite_models_folder_path)


# Generators
training_generator = DataBatcher(batch_size=BATCH_SIZE, type=DataBatcher.TRAIN)
validation_generator = DataBatcher(batch_size=BATCH_SIZE, type=DataBatcher.VALIDATION)

# Design model
# optimizer = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.99, amsgrad=1e-08)
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


# Saving Checkpoints
trained_model_file_path = os.path.join(keras_models_folder_path, 'Model-{epoch:03d}-{loss:.3f}.hdf5')
checkpoint = ModelCheckpoint(trained_model_file_path,
                             monitor='loss',
                             verbose=0,
                             save_best_only=False,
                             mode='auto',
                             period=1)

callbacks_list = [checkpoint]

# Train model on dataset
print('Commencing Training')
model.fit_generator(generator=training_generator,
                    epochs=NUM_EPOCHS,
                    # validation_data=validation_generator,
                    callbacks=callbacks_list)
print('Training Completed!')
