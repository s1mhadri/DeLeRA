from input_pipeline.process_bags import process_bag_files, create_csv_files


bagfiles_path = "bagdirs/rosbags-spike-2/"
csv_dir_path = "Data/csv-spike/"

print(process_bag_files(bagfiles_path))
print(create_csv_files(bagfiles_path, csv_dir_path))
