import os

# Local Dataset Path
data_dir = "Dataset"

# Local subfolders
subfolders = ["Fake", "Real"]

def setup_directory_and_count_files(directory, subfolders):
    total_files = 0
    for subfolder in subfolders:
        subfolder_path = os.path.join(directory, subfolder)
        total_files += sum([1 for file in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, file))])
    return total_files

train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")
validation_dir = os.path.join(data_dir, "Validation")

total_train_files = setup_directory_and_count_files(train_dir, subfolders)
total_test_files = setup_directory_and_count_files(test_dir, subfolders)
total_validation_files = setup_directory_and_count_files(validation_dir, subfolders)

total = total_train_files + total_test_files + total_validation_files
train_perc = (total_train_files / total) * 100
test_perc = (total_test_files / total) * 100
valid_perc = (total_validation_files / total) * 100
print("Total Train Files:", total_train_files)
print("Total Test Files:", total_test_files)
print("Total Validation Files:", total_validation_files)
print(f"Train Data Percentage: {train_perc:.2f}%")
print(f"Test Data Percentage: {test_perc:.2f}%")
print(f"Validation Data Percentage: {valid_perc:.2f}%")