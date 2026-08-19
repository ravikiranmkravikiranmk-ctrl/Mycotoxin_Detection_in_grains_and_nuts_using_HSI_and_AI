import h5py
import numpy as np

# This script extracts one random sample from your main dataset
# and saves it as 'sample_to_test.h5' for you to use.

MAIN_DATASET_PATH = 'data/dataset.h5'
TEST_SAMPLE_PATH = 'sample_to_test.h5'

with h5py.File(MAIN_DATASET_PATH, 'r') as f:
    num_samples = f['data'].shape[0]
    random_index = np.random.randint(0, num_samples)
    
    sample_cube = f['data'][random_index]

    with h5py.File(TEST_SAMPLE_PATH, 'w') as out_f:
        out_f.create_dataset('sample', data=sample_cube)

print(f"Success! A random sample has been extracted and saved to '{TEST_SAMPLE_PATH}'")
print("You can now use this file with 'check_sample.py'")