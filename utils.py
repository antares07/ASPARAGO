from typing import Any
import mr_forecast as fc
from icecream import ic
import numpy as np
import h5py as h5
from datetime import datetime
import socket

#Select random values of parameters
class RandomParams:
    def __init__(self, all_parameters_ranges):

        self.pars = all_parameters_ranges

    def get_random_parameters(self, num_pars_lists):

        """ Order of parameters returned
            parameters = [planet_radius,
                day_temp, night_temp, deep_temp,
                H2O_day_mix, H2O_night_mix, H2O_deep_mix,
                CH4_day_mix, CH4_night_mix, CH4_deep_mix,
                CO2_day_mix, CO2_night_mix, CO2_deep_mix,
                CO_day_mix, CO_night_mix, CO_deep_mix,
                planet_mass] """
        
        print(f'- INFO - Generating parameters lists')

        pars_list = []
        for _ in range(num_pars_lists):

            #Last element is mass, not used in retrieval
            random_values = [np.random.uniform(min_val, max_val) for min_val, max_val in self.pars]
            #ic(random_values[0])
            #planet_mass = fc.Rpost2M([random_values[0]], unit='Jupiter')
            #ic(planet_mass)
            #random_values.append(planet_mass[0])
            pars_list.append(random_values)

        return np.array(pars_list)


#Wavelength grid
class WavelengthGrid:

    """ Return the wavelengths grid of a certain instrument. Accept as input a .grid file """

    def __init__(self, instrument_file):
        self.inst = instrument_file
    
    def create_wn_grid(self):

        print(f'- INFO - Creating wavelength grid')

        inst_grid = np.genfromtxt(self.inst)
        wngrid = np.sort(15000/inst_grid)

        return wngrid
    
class SpectralParameters:

    def __init__(self):
        
        #Parameters ranges
        planet_radius_range = [0.1, 2.0] #in Jupiter radii [0.05, 2.0] initially
        #planet_mass_range = fc.Rpost2M(planet_radius_range, unit='Jupiter')
        planet_mass_range = [0.1, 10.0]
        day_temp_range = [300, 3000]
        night_temp_range = [300, 3000]
        deep_temp_range = [300, 3000]
        H2O_day_mix_range = [-12, -1]
        H2O_night_mix_range = [-12, -1]
        H2O_deep_mix_range = [-12, -1]
        CH4_day_mix_range = [-12, -1]
        CH4_night_mix_range = [-12, -1]
        CH4_deep_mix_range = [-12, -1]
        CO2_day_mix_range = [-12, -1]
        CO2_night_mix_range = [-12, -1]
        CO2_deep_mix_range = [-12, -1]
        CO_day_mix_range = [-12, -1]
        CO_night_mix_range = [-12, -1]
        CO_deep_mix_range = [-12, -1]

        #Parameters list
        self.parameters_ranges_list = [planet_radius_range,
                    day_temp_range, night_temp_range, deep_temp_range,
                    H2O_day_mix_range, H2O_night_mix_range, H2O_deep_mix_range,
                    CH4_day_mix_range, CH4_night_mix_range, CH4_deep_mix_range,
                    CO2_day_mix_range, CO2_night_mix_range, CO2_deep_mix_range,
                    CO_day_mix_range, CO_night_mix_range, CO_deep_mix_range, planet_mass_range]

    def get_list(self):
        return self.parameters_ranges_list
    
class HDF5Converter:

    """ Convert a file into hdf5 """

    def __init__(self, directory_path):

        filename = socket.gethostname()+datetime.today().strftime('%d%m%Y%H%M%S')
        self.file_path = f'{directory_path}/{filename}.h5'

    def dictionary_to_hdf5(self, dictionary: dict):

        """ Convert a dictionary into h5 file """

        print(f'- INFO - Converting into h5 file')

        with h5.File(self.file_path, 'w') as file:
            for idx, pars_dict in dictionary.items():
                if isinstance(pars_dict, dict):
                    group = file.create_group(str(idx))
                    for key, value in pars_dict.items():
                        group.create_dataset(name=str(key), data=value)

    def array_to_hdf5(self, array):

        """ Convert an array into h5 file """

        print(f'- INFO - Converting into h5 file')

        with h5.File(self.file_path, 'w') as file:
            for idx in range(array.shape[0]):
                file.create_dataset(name=str(idx), data=array[idx, :, :, :])


                    
