from utils import *
from ASPA import *
from taurex.cache import OpacityCache, CIACache
from taurex.log import disableLogging
from taurex.stellar import BlackbodyStar
from spectra_generator import *
from icecream import ic
import time

#disableLogging()

#Star parameters
star_temp = 5700.
star_radius = 1.

print(f'- INFO - Loading cross sections')
opt, cia = OpacityCache(), CIACache()
opt.set_memory_mode(in_memory=True)
opt.set_opacity_path('/home/pagliaro/data/xsecs')
cia.set_cia_path('/home/pagliaro/data/cia/hitran')

opt.load_opacity_from_path(path='/home/pagliaro/data/xsecs', molecule_filter=['H2O', 'CH4', 'CO2', 'CO'])
cia.load_cia_from_path(path='/home/pagliaro/data/cia/hitran', pair_filter=['H2-H2', 'H2-He'])

#Instument grid path
inst_grid = '/home/pagliaro/ASPAGenerator/all_instruments_binned_1600.grid'
wn_grid = WavelengthGrid(inst_grid).create_wn_grid()

parameters_ranges_list = SpectralParameters().get_list()
list_of_spec_pars_lists = RandomParams(parameters_ranges_list).get_random_parameters(15000)

#Star
star = BlackbodyStar(temperature=star_temp, radius=star_radius)

if __name__ == '__main__':

    print(f'- INFO - Starting the generation')

    start = time.time()
    
    #Generate dataset
    GenerateSpectraWithInitialization(star=star).generate_multiple_spectra_dictionary(list_of_spec_pars_lists)

    end = time.time()
    print(f'- INFO - Process ends in {(end-start)} seconds')