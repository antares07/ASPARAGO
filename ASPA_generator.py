import taurex.log
import numpy as np
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.cache import OpacityCache, CIACache
from taurex_2d.temperature import Temperature2D
from taurex_2d.chemistry import Chemistry2D
from taurex_2d.chemistry import Gas2D
from taurex_2d.model import Transmission2DModel
from taurex_2d.contributions import AbsorptionContribution
from taurex_2d.contributions import CIAContribution
from taurex_2d.contributions import RayleighContribution
from taurex.binning import SimpleBinner
from multiprocessing import Pool
import mr_forecast as fc
from memory_profiler import profile

taurex.log.disableLogging()
OpacityCache().clear_cache()
opt, cia = OpacityCache(), CIACache()
opt.set_opacity_path('/home/pagliaro/data/xsecs')
cia.set_cia_path('/home/pagliaro/data/cia/hitran')

#Opacity objects for h2o, ch4, co2, co and cia
h2o_cs = opt['H2O']
ch4_cs = opt['CH4']
co2_cs = opt['CO2']
co_cs = opt['CO']
cia_h2h2 = cia['H2-H2']
cia_h2he = cia['H2-He']

#Add opacity to cache
opt.add_opacity(h2o_cs)
opt.add_opacity(ch4_cs)
opt.add_opacity(co2_cs)
opt.add_opacity(co_cs)

#Star parameters
star_temp = 5700.
star_radius = 1.

#Molecules
gases = ['H2O', 'CH4', 'CO2', 'CO']

#Others parameters
max_pressure = 1e6
min_pressure = 1e-0
n_layers = 100
n_slices = 20
beta = 30
R_nept = 24622

#Parameters ranges
planet_radius_range = [0.2, 2.0] #in Jupiter radii
planet_mass_range = fc.Mpost2R(planet_radius_range, 'Jupiter')
day_temp_range = [300, 3000]
night_temp_range = [300, 3000]
deep_temp_range = [300, 3000]
H2O_day_mix_range = [1e-12, 1e-1]
H2O_night_mix_range = [1e-12, 1e-1]
H2O_deep_mix_range = [1e-12, 1e-1]
CH4_day_mix_range = [1e-12, 1e-1]
CH4_night_mix_range = [1e-12, 1e-1]
CH4_deep_mix_range = [1e-12, 1e-1]
CO2_day_mix_range = [1e-12, 1e-1]
CO2_night_mix_range = [1e-12, 1e-1]
CO2_deep_mix_range = [1e-12, 1e-1]
CO_day_mix_range = [1e-12, 1e-1]
CO_night_mix_range = [1e-12, 1e-1]
CO_deep_mix_range = [1e-12, 1e-1]

#Parameters list
parameters = [planet_radius_range, planet_mass_range,
                day_temp_range, night_temp_range, deep_temp_range,
                H2O_day_mix_range, H2O_night_mix_range, H2O_deep_mix_range,
                CH4_day_mix_range, CH4_night_mix_range, CH4_deep_mix_range,
                CO2_day_mix_range, CO2_night_mix_range, CO2_deep_mix_range,
                CO_day_mix_range, CO_night_mix_range, CO_deep_mix_range
                ]

#Select random values of parameters
class RandomParams():
    def __init__(self, parameters):
        self.pars = parameters

    def get_random_parameters(self, num_arrays):
        pars_list = []
        for _ in range(num_arrays):
            random_values = [np.random.uniform(min_val, max_val) for min_val, max_val in self.pars]
            pars_list.append(random_values)

        return np.array(pars_list)


#Wavelength grid
class WavelengthGrid():

    """ Return the wavelengths grid of a certain instrument. Accept as input a .grid file """

    def __init__(self, instrument_file):
        self.inst = instrument_file
    
    def wn_grid(self):
        inst_grid = np.genfromtxt(self.inst)
        wngrid = np.sort(15000/inst_grid)

        return wngrid

#Generate dataset
class GenerateSpectra():

    """ parameters_dictionary = {
                'planet_radius':self.pars[0],
                'tday': self.pars[2],
                'tnight': self.pars[3],
                'tdeep': self.pars[4],
                'H2O_day_mix': self.pars[5],
                'H2O_night_mix': self.pars[6],
                'H2O_deep_mix': self.pars[7],
                'CH4_day_mix': self.pars[8],
                'CH4_night_mix': self.pars[9],
                'CH4_deep_mix': self.pars[10],
                'CO2_day_mix': self.pars[11],
                'CO2_night_mix': self.pars[12],
                'CO2_deep_mix': self.pars[13],
                'CO_day_mix': self.pars[14],
                'CO_night_mix': self.pars[15],
                'CO_deep_mix': self.pars[16]
                } """

    def __init__(self, chunk_parameters, star, wn_grid):
        self.iterable = chunk_parameters
        self.star = star
        self.wngrid = wn_grid

    #@profile
    def generate_single_spectrum(self, iterable_params):

        self.index, self.pars = iterable_params

        parameters_dictionary = {}
        parameters_dictionary['pars'] = np.array([self.pars[0],self.pars[2],self.pars[3],self.pars[4]])
        parameters_dictionary['H2O_mix'] = np.array([self.pars[5],self.pars[6],self.pars[7]])
        parameters_dictionary['CH4_mix'] = np.array([self.pars[8],self.pars[9],self.pars[10]])
        parameters_dictionary['CO2_mix'] = np.array([self.pars[11],self.pars[12],self.pars[13]])
        parameters_dictionary['CO_mix'] = np.array([self.pars[14],self.pars[15],self.pars[16]])

        #Planet
        self.planet = Planet(planet_radius=parameters_dictionary['pars'][0],
                        planet_mass=self.pars[1])
        
        #gc.collect()
        
        #T profile
        self.temperature2D = Temperature2D(day_temp=parameters_dictionary['pars'][1],
                                      night_temp=parameters_dictionary['pars'][2],
                                      deep_temp=parameters_dictionary['pars'][3])
        #Gases and chemistry
        self.chemistry2D = Chemistry2D(fill_gases=['H2', 'He'], ratio=0.0196) #from arxiv1703.10834
        for gas in gases:
            self.chemistry2D.addGas(Gas2D(molecule_name=gas,
                                    day_mix_ratio=parameters_dictionary[f'{gas}_mix'][0],
                                    night_mix_ratio=parameters_dictionary[f'{gas}_mix'][1],
                                    deep_mix_ratio=parameters_dictionary[f'{gas}_mix'][2]))
        
        #gc.collect()

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

        #gc.collect()

        #Binning
        bn = SimpleBinner(wngrid=self.wngrid)
        _, bin_rprs, _, _  = bn.bin_model(tm2D.model(wngrid=self.wngrid))

        parameters_dictionary['rprs'] = bin_rprs
        index_params_dict = {}
        index_params_dict[self.index] = parameters_dictionary

        #gc.collect()
    
        return index_params_dict
    #@profile
    def generate_dataset(self, nproc):

        #Multiprocessing
        pool = Pool(processes=nproc)#, maxtasksperchild=1)
        with pool:
            all_spectrum_dict = list(pool.map(self.generate_single_spectrum, self.iterable))

        dataset_dictionary = {
            str(list(spectrum_dict.keys())[0]): spectrum_dict[list(spectrum_dict.keys())[0]] for spectrum_dict in all_spectrum_dict
            }

        return dataset_dictionary
    
class GetASPADataset():

    def __init__(self, dataset: dict, size_spectral_matrix=40, mol_long_side=64, mol_short_side=2, up_pars_short_side=6, norm_idx_path=None):

        """ Initialize the parameters of ASPA"""

        self.dataset = dataset
        self.size_spectral_matrix = size_spectral_matrix
        self.mol_short_side = mol_short_side
        self.mol_long_side = mol_long_side
        self.up_pars_short_side = up_pars_short_side
        self.size_aspa = self.mol_long_side
        self.gases = gases
        self.spectral_matrix = np.zeros((self.size_spectral_matrix, self.size_spectral_matrix, 1))
        self.norm_spectrum = np.zeros((self.size_spectral_matrix**2))
        self.aspa = np.zeros((self.size_aspa, self.size_aspa, 1))
        self.norm_idx = np.array(np.genfromtxt(norm_idx_path), dtype=int)

    def get_aspa(self, spec_num):

        """ Puts parameters into the ASPA"""

        #Radius
        radius = np.array(self.dataset[str(spec_num)]['pars'][0])
        self.aspa[:self.size_spectral_matrix, self.up_pars_short_side*3:, 0] = radius/2.

        #Temperature
        temp = np.array(list(self.dataset[str(spec_num)]['pars'][1:]))
        for i_temp in range(len(temp)):
            self.aspa[:self.size_spectral_matrix,
                self.size_spectral_matrix+(i_temp)*self.up_pars_short_side:self.size_spectral_matrix+(i_temp+1)*self.up_pars_short_side,
                0] = temp[i_temp]/3e3

        #Molecules
        for mol_pos, mol in enumerate(self.gases):
            for pos_side, side_value in enumerate(list(self.dataset[str(spec_num)][f'{mol}_mix'])):
                self.aspa[self.size_spectral_matrix+(mol_pos*self.mol_short_side*3+(pos_side*self.mol_short_side)):self.size_spectral_matrix+(mol_pos*self.mol_short_side*3+((pos_side+1)*self.mol_short_side)),
                                    : ,0] = -np.log10(side_value) / 12.0

        #Normalized spectrum
        self.aspa[:self.size_spectral_matrix, :self.size_spectral_matrix] = self.get_normalized_spectral_matrix(spec_num)

        return self.aspa

    def get_normalized_spectral_matrix(self, spec_i):

        """ Return the normalized spectrum in matrix form"""

        #Initialize quantities
        spectrum = np.array(list(self.dataset[str(spec_i)]['rprs']))

        #Normalize quantities
        for i in range(len(self.norm_idx)-1):
            frag = np.array(spectrum[self.norm_idx[i] : self.norm_idx[i+1]])
            minf = np.min(frag)
            maxf = np.max(frag)
            if minf == maxf:
                pass
            else:
                self.norm_spectrum[self.norm_idx[i] : self.norm_idx[i+1]] = (frag-minf)/(maxf-minf)

        #Spectrum
        spectral_matrix = np.reshape(self.norm_spectrum, (40, 40, 1))

        return spectral_matrix
    
    def get_aspa_dataset(self):
        spectra_keys = list(self.dataset.keys())
        aspa_dataset = np.zeros((len(spectra_keys), self.size_aspa, self.size_aspa, 1))

        for key in spectra_keys:
            aspa = self.get_aspa(key)
            aspa_dataset[int(key), :, :, 0] = aspa[:, :, 0]

        return aspa_dataset
    
#Generate spectra dict
class GetSpectraDictionary():

    def generate_spectra_dict(self, num_spectra, ncpu):

        #Star
        star = BlackbodyStar(temperature=star_temp, radius=star_radius)

        all_inst = WavelengthGrid('all_instruments_binned_1600.grid')
        wn_grid_all_inst = all_inst.wn_grid()
        
        #Parameters list
        print(f'Generating parameters combination...')
        pars = RandomParams(parameters=parameters)
        parameters_list = np.array(list(pars.get_random_parameters(num_spectra)))
        np.random.shuffle(parameters_list)
        iterables = [(index, spectrum) for index, spectrum in enumerate(parameters_list)]
        
        #Generate dataset
        print(f'Generating {num_spectra} spectra...')
        spectra_dataset = GenerateSpectra(chunk_parameters=iterables, star=star, wn_grid=wn_grid_all_inst)
        dataset_dict = spectra_dataset.generate_dataset(nproc=ncpu)

        return dataset_dict





