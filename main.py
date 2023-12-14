from functions import generate_hdf5_dataset
import time

if __name__ == '__main__':
    print('Starting generation...')
    start = time.time()
    generate_hdf5_dataset(1, '/home/pagliaro/SharedDataset')
    end = time.time()
    print(f'Process ends in {(end-start)/60} minutes')