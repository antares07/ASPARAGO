import taurex.log
import numpy as np
from utils import *
from ASPA import ASPA
from mpi4py import MPI
from taurex.mpi import get_rank, nprocs
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex_2d.temperature import Temperature2D
from taurex_2d.chemistry import Chemistry2D
from taurex_2d.chemistry import Gas2D
from taurex_2d.model import Transmission2DModel
from taurex_2d.contributions import AbsorptionContribution
from taurex_2d.contributions import CIAContribution
from taurex_2d.contributions import RayleighContribution
from taurex.binning import SimpleBinner
from icecream import ic

#Class to generate dictionary of spectra
class GenerateSpectra():

    """ parameters_dictionary = {
                'planet_radius':self.pars[0],
                'tday': self.pars[1],
                'tnight': self.pars[2],
                'tdeep': self.pars[3],
                'H2O_day_mix': self.pars[4],
                'H2O_night_mix': self.pars[5],
                'H2O_deep_mix': self.pars[6],
                'CH4_day_mix': self.pars[7],
                'CH4_night_mix': self.pars[8],
                'CH4_deep_mix': self.pars[9],
                'CO2_day_mix': self.pars[10],
                'CO2_night_mix': self.pars[11],
                'CO2_deep_mix': self.pars[12],
                'CO_day_mix': self.pars[13],
                'CO_night_mix': self.pars[14],
                'CO_deep_mix': self.pars[15]
                } """

    def __init__(self, star):

        self.star = star
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        inst_grid = '/home/pagliaro/ASPAGenerator/all_instruments_binned_1600.grid'
        self.wn_grid = WavelengthGrid(inst_grid).create_wn_grid()
        self.binner = SimpleBinner(self.wn_grid)

    #@profile
    def generate_single_spectrum(self, single_spectrum_pars):

        self.pars = single_spectrum_pars

        parameters_dictionary = {}
        parameters_dictionary['pars'] = np.array([self.pars[0],self.pars[1],self.pars[2],self.pars[3]])
        parameters_dictionary['H2O_mix'] = np.array([self.pars[4],self.pars[5],self.pars[6]])
        parameters_dictionary['CH4_mix'] = np.array([self.pars[7],self.pars[8],self.pars[9]])
        parameters_dictionary['CO2_mix'] = np.array([self.pars[10],self.pars[11],self.pars[12]])
        parameters_dictionary['CO_mix'] = np.array([self.pars[13],self.pars[14],self.pars[15]])

        #Planet
        self.planet = Planet(planet_radius=parameters_dictionary['pars'][0],
                        planet_mass=self.pars[16])
        
        #T profile
        self.temperature2D = Temperature2D(day_temp=parameters_dictionary['pars'][1],
                                      night_temp=parameters_dictionary['pars'][2],
                                      deep_temp=parameters_dictionary['pars'][3])
        
        #Gases and chemistry
        self.chemistry2D = Chemistry2D(fill_gases=['H2', 'He'], ratio=0.0196) #from arxiv1703.10834
        for gas in gases:
            self.chemistry2D.addGas(Gas2D(molecule_name=gas,
                                    day_mix_ratio=10**(parameters_dictionary[f'{gas}_mix'][0]),
                                    night_mix_ratio=10**(parameters_dictionary[f'{gas}_mix'][1]),
                                    deep_mix_ratio=10**(parameters_dictionary[f'{gas}_mix'][2])))
            
        #Model
        tm2D = Transmission2DModel(planet=self.planet,
                                   star=self.star,
                                   chemistry=self.chemistry2D,
                                   temperature_profile=self.temperature2D,
                                   atm_min_pressure=min_pressure,
                                   atm_max_pressure=max_pressure,
                                   nlayers=n_layers,
                                   nslices=n_slices,
                                   beta=beta)
        tm2D.add_contribution(AbsorptionContribution())
        tm2D.add_contribution(CIAContribution(cia_pairs=['H2-H2','H2-He']))
        tm2D.add_contribution(RayleighContribution())
        tm2D.build()

        #Binning
        bin_rprs = self.binner.bin_model(tm2D.model(wngrid=self.wn_grid))[1]

        parameters_dictionary['rprs'] = bin_rprs
    
        return parameters_dictionary
    
    #Using mpi for parallel computation
    def generate_multiple_spectra_dictionary(self, all_pars_lists):

        print(f'- INFO - Generating spectra dictionary')
        
        chunk_size = len(all_pars_lists) // self.size
        chunk_params = all_pars_lists[self.rank*chunk_size : (self.rank+1)*chunk_size]
        single_proc_spectra_dict = {}

        for idx, single_spec_parameters in enumerate(chunk_params):
            single_spectrum_dict = self.generate_single_spectrum(single_spec_parameters)
            single_proc_spectra_dict[idx] = single_spectrum_dict

        #Gather results from all processes
        all_proc_spectra_dict = self.comm.gather(single_proc_spectra_dict, root=0)

        #Collect all results on rank 0
        if self.rank == 0:
            final_spectra_dict = {}
            for rank_idx, result_dict in enumerate(all_proc_spectra_dict):
                for idx, result in result_dict.items():
                    final_spectra_dict[str(int((rank_idx*chunk_size)+idx))] = result

            #Save into h5
            HDF5Converter('/home/pagliaro/Dataset_mpi').dictionary_to_hdf5(final_spectra_dict)

            return final_spectra_dict

#Class to generate dictionary of spectra with initialization of the model
class GenerateSpectraWithInitialization:
    
    """Generate a given number of random spectra using mpi by initializing first a single spectrum"""

    """ parameters_dictionary = {
                'planet_radius':self.pars[0],
                'tday': self.pars[1],
                'tnight': self.pars[2],
                'tdeep': self.pars[3],
                'H2O_day_mix': self.pars[4],
                'H2O_night_mix': self.pars[5],
                'H2O_deep_mix': self.pars[6],
                'CH4_day_mix': self.pars[7],
                'CH4_night_mix': self.pars[8],
                'CH4_deep_mix': self.pars[9],
                'CO2_day_mix': self.pars[10],
                'CO2_night_mix': self.pars[11],
                'CO2_deep_mix': self.pars[12],
                'CO_day_mix': self.pars[13],
                'CO_night_mix': self.pars[14],
                'CO_deep_mix': self.pars[15]
                } """

    def __init__(self, star):

        self.star = star
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        inst_grid = '/home/pagliaro/ASPAGenerator/all_instruments_binned_1600.grid'
        self.wn_grid = WavelengthGrid(inst_grid).create_wn_grid()
        self.binner = SimpleBinner(wngrid=self.wn_grid)
        
        #Molecules
        self.gases = ['H2O', 'CH4', 'CO2', 'CO']

        #Others parameters
        self.max_pressure = 1e6
        self.min_pressure = 1e-0
        self.n_layers = 100
        self.n_slices = 20
        self.beta = 30

    #@profile
    def initialize_model(self):
        
        """Function to initialize the single spectrum"""

        print(f'- INFO - Initializing model {self.rank}')

        #Planet
        self.planet = Planet(planet_radius=1.0,
                             planet_mass=1.0)
        
        #T profile
        self.temperature2D = Temperature2D(day_temp=2000.0,
                                           night_temp=1000.0,
                                           deep_temp=1500.0)
        
        #Gases and chemistry
        self.chemistry2D = Chemistry2D(fill_gases=['H2', 'He'], ratio=0.0196) #from arxiv1703.10834
        for gas in self.gases:
            self.chemistry2D.addGas(Gas2D(molecule_name=gas,
                                    day_mix_ratio=10**(-2),
                                    night_mix_ratio=10**(-2),
                                    deep_mix_ratio=10**(-2)))
            
        #Model
        tm2D = Transmission2DModel(planet=self.planet,
                                   star=self.star,
                                   chemistry=self.chemistry2D,
                                   temperature_profile=self.temperature2D,
                                   atm_min_pressure=self.min_pressure,
                                   atm_max_pressure=self.max_pressure,
                                   nlayers=self.n_layers,
                                   nslices=self.n_slices,
                                   beta=self.beta)
        
        tm2D.add_contribution(AbsorptionContribution())
        tm2D.add_contribution(CIAContribution(cia_pairs=['H2-H2','H2-He']))
        tm2D.add_contribution(RayleighContribution())
        
        #Build model
        tm2D.build()

        return tm2D
    
    #Using mpi for parallel computation
    def generate_multiple_spectra_dictionary(self, all_pars_lists: list) -> dict:
        
        "Return a dictionary of a given number of spectra using mpi"

        #Initialize model
        model = self.initialize_model()

        print(f'- INFO - Generating spectra dictionary')

        
        chunk_size = len(all_pars_lists) // self.size
        chunk_params = all_pars_lists[self.rank*chunk_size : (self.rank+1)*chunk_size]
        with open('/home/pagliaro/Dataset_mpi/parameters_correct_norm.csv', 'a') as par_file:
            np.savetxt(par_file, chunk_params)
        single_proc_spectra_dict = {}

        for idx, single_spec_parameters in enumerate(chunk_params):

            single_spectrum_dict = self.update_model(model, single_spec_parameters)
            single_proc_spectra_dict[idx] = single_spectrum_dict

        #Gather results from all processes
        all_proc_spectra_dict = self.comm.gather(single_proc_spectra_dict, root=0)

        #Collect all results on rank 0
        if self.rank == 0:
            final_spectra_dict = {}
            for rank_idx, result_dict in enumerate(all_proc_spectra_dict):
                for idx, result in result_dict.items():
                    final_spectra_dict[str(int((rank_idx*chunk_size)+idx))] = result

            #Save into h5
            print(f'- INFO - Saving HDF5 file')
            HDF5Converter('/home/pagliaro/Dataset_mpi/Spectra').dictionary_to_hdf5(final_spectra_dict)

            #Convert into aspas
            aspa_dataset = ASPA(norm_idx_path='/home/pagliaro/ASPARAGO/exogan_norm_bands_pos_10.grid').get_ASPA_dataset(final_spectra_dict)
            #Save into h5
            print(f'- INFO - Saving ASPA into HDF5 file')
            HDF5Converter('/home/pagliaro/Dataset_mpi/ASPAs').array_to_hdf5(aspa_dataset)

            return final_spectra_dict
        
    def update_model(self, model, pars: list):
        
        """Update the initialized model as in TauREx"""

        parameters_dictionary = {}
        parameters_dictionary['pars'] = np.array([pars[0],pars[1],pars[2],pars[3]])
        parameters_dictionary['H2O_mix'] = np.array([pars[4],pars[5],pars[6]])
        parameters_dictionary['CH4_mix'] = np.array([pars[7],pars[8],pars[9]])
        parameters_dictionary['CO2_mix'] = np.array([pars[10],pars[11],pars[12]])
        parameters_dictionary['CO_mix'] = np.array([pars[13],pars[14],pars[15]])

        print(f'- INFO - Updating the model  {self.rank}')

        model['planet_radius'] = pars[0]
        model['planet_mass'] = pars[16]
        model['Tday'] = pars[1]
        model['Tnight'] = pars[2]
        model['Tdeep'] = pars[3]
        model['H2O_day'] = 10**pars[4]
        model['H2O_night'] = 10**pars[5]
        model['H2O_deep'] = 10**pars[6]
        model['CH4_day'] = 10**pars[7]
        model['CH4_night'] = 10**pars[8]
        model['CH4_deep'] = 10**pars[9]
        model['CO2_day'] = 10**pars[10]
        model['CO2_night'] = 10**pars[11]
        model['CO2_deep'] = 10**pars[12]
        model['CO_day'] = 10**pars[13]
        model['CO_night'] = 10**pars[14]
        model['CO_deep'] = 10**pars[15]
        
        #Binning
        bin_rprs = self.binner.bin_model(model.model(self.wn_grid))[1]

        parameters_dictionary['rprs'] = bin_rprs

        return parameters_dictionary
