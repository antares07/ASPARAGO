import numpy as np
from utils import HDF5Converter
from icecream import ic

class ASPA:

    def __init__(self, size_spectral_matrix=40, mol_long_side=64, mol_short_side=2, up_pars_short_side=6, norm_idx_path=None):

        """ Initialize the parameters of ASPA"""

        self.size_spectral_matrix = size_spectral_matrix
        self.mol_short_side = mol_short_side
        self.mol_long_side = mol_long_side
        self.up_pars_short_side = up_pars_short_side
        self.size_aspa = self.mol_long_side
        self.gases = ['H2O', 'CH4', 'CO2', 'CO']
        self.spectral_matrix = np.zeros((self.size_spectral_matrix, self.size_spectral_matrix, 1))
        self.norm_spectrum = np.zeros((self.size_spectral_matrix*self.size_spectral_matrix))
        self.aspa = np.zeros((self.size_aspa+2, self.size_aspa, 1)) #2 more rows for max and min values
        self.norm_idx = np.array(np.genfromtxt(norm_idx_path), dtype=int)

    def get_ASPA(self, spectrum):

        """ Puts spectral parameters into the ASPA"""

        #Radius
        radius = np.array(spectrum['pars'][0])
        self.aspa[:self.size_spectral_matrix, self.up_pars_short_side*3:, 0] = radius/2.

        #Temperature
        temp = np.array(list(spectrum['pars'][1:]))
        for i_temp in range(len(temp)):
            self.aspa[:self.size_spectral_matrix,
                self.size_spectral_matrix+(i_temp)*self.up_pars_short_side:self.size_spectral_matrix+(i_temp+1)*self.up_pars_short_side,
                0] = temp[i_temp]/3e3

        #Molecules
        for mol_pos, mol in enumerate(self.gases):
            for pos_side, side_value in enumerate(list(spectrum[f'{mol}_mix'])):
                self.aspa[self.size_spectral_matrix+(mol_pos*self.mol_short_side*3+(pos_side*self.mol_short_side)):self.size_spectral_matrix+(mol_pos*self.mol_short_side*3+((pos_side+1)*self.mol_short_side)),
                                    : ,0] = - side_value / 12.0

        spectral_matrix, normalization_values = self.get_normalized_spectral_matrix(spectrum)
            
        #Normalization factors
        self.aspa[-2:, :, :] = normalization_values

        #Normalized spectrum
        self.aspa[:self.size_spectral_matrix, :self.size_spectral_matrix] = spectral_matrix

        return self.aspa

    def get_normalized_spectral_matrix(self, spectrum):

        """ Return the normalized spectrum in matrix form"""

        #Initialize quantities
        spec = np.array(list(spectrum['rprs']))[::-1] #reversed
        normalization_values = np.zeros((2, self.size_aspa, 1))

        #Normalize quantities
        for i in range(len(self.norm_idx)-1):
            frag = np.array(spec[self.norm_idx[i] : self.norm_idx[i+1]])
            minf = np.min(frag)
            maxf = np.max(frag)
            normalization_values[0, i], normalization_values[1, i] = maxf, minf
            if minf == maxf:
                pass
            else:
                self.norm_spectrum[self.norm_idx[i] : self.norm_idx[i+1]] = (frag-minf)/(maxf-minf)

        #ic(normalization_values)

        #Spectrum
        spectral_matrix = np.reshape(self.norm_spectrum, (40, 40, 1))

        return spectral_matrix, normalization_values
    
    def get_ASPA_dataset(self, dataset):

        """ Convert a dictionary of spectral parameters into a dictionary of ASPAs """

        print(f'- INFO - Converting spectral dataset into ASPA dataset')

        spectra_keys = list(dataset.keys())
        aspa_dataset = np.zeros((len(spectra_keys), self.size_aspa+2, self.size_aspa, 1))

        for key in spectra_keys:
            aspa = self.get_ASPA(dataset[key])
            aspa_dataset[int(key), :, :, 0] = aspa[:, :, 0]

        return aspa_dataset
    
    def convert_ASPA_to_real_spectrum(self, aspa):

        """ Convert the normalized spectrum of an ASPA back to the real spectrum """

        spectral_matrix = aspa[:40, :40, -1]
        max_min_norm = aspa[-2:, :, -1]
        norm_spectrum = spectral_matrix.reshape(1600)

        spectrum = []
        for i in range(len(self.norm_idx)-1):
            frag = (max_min_norm[0, i]-max_min_norm[1, i])*(norm_spectrum[int(self.norm_idx[i]):(int(self.norm_idx[i+1]))])+max_min_norm[1, i]
            spectrum.append(frag)
        spectrum = np.concatenate(spectrum)

        return spectrum