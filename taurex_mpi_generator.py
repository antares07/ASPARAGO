import taurex.log
import numpy as np
from mpi4py import MPI
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
from taurex_2d.mpi import get_rank, nprocs
from icecream import ic
from utils import *
from pprint import pprint

#Molecules
gases = ['H2O', 'CH4', 'CO2', 'CO']

#Others parameters
max_pressure = 1e6
min_pressure = 1e-0
n_layers = 100
n_slices = 20
beta = 30

#Class to generate dictionary of spectra
class GenerateSpectraWithInitialization():

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
        self.rank = get_rank()
        self.size = nprocs()
        inst_grid = '/home/pagliaro/ASPAGenerator/all_instruments_binned_1600.grid'
        self.wn_grid = WavelengthGrid(inst_grid).create_wn_grid()
        self.binner = SimpleBinner(self.wn_grid)

    #@profile
    def initialize_random_single_spectrum(self):

        pprint(f'- INFO - Initializing model')

        #Planet
        self.planet = Planet(planet_radius=1.,
                        planet_mass=1.)
        
        #T profile
        self.temperature2D = Temperature2D(day_temp=1000.,
                                      night_temp=1000.,
                                      deep_temp=1000.)
        
        #Gases and chemistry
        self.chemistry2D = Chemistry2D(fill_gases=['H2', 'He'], ratio=0.0196) #from arxiv1703.10834
        for gas in gases:
            self.chemistry2D.addGas(Gas2D(molecule_name=gas,
                                    day_mix_ratio=10**(-1),
                                    night_mix_ratio=10**(-1),
                                    deep_mix_ratio=10**(-1)))
            
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

        return tm2D
    
    #Using mpi for parallel computation
    def generate_multiple_spectra_dictionary(self, spectra_per_chunk):

        #Initialize model
        model = self.initialize_random_single_spectrum()

        runs = list(range(spectra_per_chunk))

        pprint(f'{__class__.__name__} - INFO - Generating spectra dictionary')

        final_spectra_dict = {}
        
        for n in runs[self.rank::self.size]:

            pprint(f'- INFO - Index of process: {n}')
            
            pars = RandomParams(SpectralParameters().get_list()).get_random_parameters(1)[0]
            #pprint(f'Random parameters \n {pars}')

            parameters_dictionary = {}
            parameters_dictionary['pars'] = np.array([pars[0],pars[1],pars[2],pars[3]])
            parameters_dictionary['H2O_mix'] = np.array([pars[4],pars[5],pars[6]])
            parameters_dictionary['CH4_mix'] = np.array([pars[7],pars[8],pars[9]])
            parameters_dictionary['CO2_mix'] = np.array([pars[10],pars[11],pars[12]])
            parameters_dictionary['CO_mix'] = np.array([pars[13],pars[14],pars[15]])

            pprint(f'- INFO - Updating the model')

            model['planet_radius'] = pars[0]
            model['planet_mass'] = pars[-1]
            model['Tday'] = pars[1]
            model['Tnight'] = pars[2]
            model['Tdeep'] = pars[3]
            model['H2O_day'] = pars[4]
            model['H2O_night'] = pars[5]
            model['H2O_deep'] = pars[6]
            model['CH4_day'] = pars[7]
            model['CH4_night'] = pars[8]
            model['CH4_deep'] = pars[9]
            model['CO2_day'] = pars[10]
            model['CO2_night'] = pars[11]
            model['CO2_deep'] = pars[12]
            model['CO_day'] = pars[13]
            model['CO_night'] = pars[14]
            model['CO_deep'] = pars[15]
            
            #Binning
            _, bin_rprs, _, _ = self.binner.bin_model(model.model(wngrid=self.wn_grid))

            parameters_dictionary['rprs'] = bin_rprs

            #pprint(f'Parameters dictionary \n {parameters_dictionary}')

            final_spectra_dict[n] = parameters_dictionary

            pprint(f'Final dictionary \n {final_spectra_dict}')

            return final_spectra_dict

