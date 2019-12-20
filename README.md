EyeTrackerKeras

After cloning the repo, change the project_path.py file to configure to your system. Mainly the data_folder_path(all others are present).
First run "python3 convert_from_metadata_to_csv.py"
This will generate the train.csv, test.csv and validation.csv in the generated folder.
After this run "python3 trainer_keras2.py" and update the parameters like epoch and BATCH_SIZE to get the required accuracy.