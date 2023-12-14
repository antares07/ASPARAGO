import psutil
import socket
from datetime import datetime
import h5py
from ASPA_generator import GetASPADataset, GetSpectraDictionary

def check_available_cpus():

    #Find unused core
    cpus_percentage = psutil.cpu_percent(interval=1, percpu=True)
    less_than_10_usage_cpus = [i for i, percentage in enumerate(cpus_percentage) if percentage <= 10]
    print(f'Unused cpus: {len(less_than_10_usage_cpus)}')

    if len(less_than_10_usage_cpus)>0:
        return round(len(less_than_10_usage_cpus)/2)
    else:
        return 1
    
def check_available_memory_for_spectra():

    single_spec_memory = 5.

    #Find unused memory
    available_memory = round(psutil.virtual_memory().available/(1024**3))
    num_spec = (available_memory/single_spec_memory)/2.
    print(f'Available memory: {available_memory} GB')

    return num_spec

def get_ASPA_dataset():

    #Num cpus
    cpus = check_available_cpus()
    print(f'Using {cpus} cores...')

    #Available memory and max number of spectra
    num_spec = round(check_available_memory_for_spectra())

    #Spectra dictionary
    getspectradict = GetSpectraDictionary()
    dataset_dict = getspectradict.generate_spectra_dict(num_spec, cpus)

    #Generate ASPA
    print('Turning spectral dictionary into ASPA matrix...')
    get_aspa = GetASPADataset(dataset=dataset_dict, norm_idx_path='exogan_norm_bands_pos_10.grid')
    aspa_dataset = get_aspa.get_aspa_dataset()

    return aspa_dataset

def generate_hdf5_dataset(total_chunk=64, directory=None):
    name = socket.gethostname()+datetime.today().strftime('%d%m%Y%H%M%S')
    print(f'Creating HDF5 dataset {name}...')
    with h5py.File(f'{directory}/{name}.h5', 'w') as file:
        for chunk in range(total_chunk):
            print(f'Generating chunk {chunk}...')
            group = file.create_group(str(chunk))
            aspas = get_ASPA_dataset()
            for idx, val in enumerate(aspas):
                group.create_dataset(name=str(idx), data=val, compression='gzip')