import pickle
from csv import reader
from EncoderDecoder.models import PV_autoencoder
from EncoderDecoder.utils import get_dataset_analysis
import pandas as pd
import torch

"""
PV optimization using an encoder-decoder architecture (Kamila Zdybal)
Version: Load different datatypes
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: November 2025
Python version: 3.10.10
"""

class loadData:
    """ Load model, curves and metadata """

    def __init__(self, filename, suffix_filename_state_space_names = "", path_metadata = "data-files/metadata/"):
        
        self.path_metadata = path_metadata
        self.filename = filename
        self.filename_metadata = self.filename + "_metadata.pkl"
        self.metadata = self.loadMetadata()

        self.filename_state_space_names = f"Xu-state-space-names{suffix_filename_state_space_names}.csv"
        self.filename_model = self.metadata["model_name"]
        self.filename_curve = self.metadata["curve_name"]

        self.model_params = self.metadata["model_params"]

    def loadStateSpaceNames(self, path_data):
        
        with open(f"{path_data}{self.filename_state_space_names}", mode='r', newline='') as file:
            csv_reader = reader(file)
            list_species = [row[0] for row in csv_reader]

        return list_species

    def loadMetadata(self):

        with open(f'{self.path_metadata}/{self.filename_metadata}', 'rb') as f:
            loaded_dict = pickle.load(f)

        return loaded_dict
    
    def loadModel(self, filename_model = None, device = "cpu"):

        if(filename_model is None):
            filename_model = self.filename_model

        model_reloaded = PV_autoencoder(**self.model_params)
        state_dict = torch.load(
            "out/" + filename_model,
            weights_only=False,
            map_location=torch.device(device)
        )

        model_reloaded.load_state_dict(state_dict)

        return model_reloaded

    def loadCurves(self):

        df = pd.read_csv('data-files/curves/' + self.filename_curve, header=None)

        training = df.iloc[0]  #Training curve
        validation = df.iloc[1]  #Validation curve

        return training, validation

    def getInputSpecies(self, path_data): #TO MODIFY

        list_species = self.loadStateSpaceNames(path_data)
        idx_remove = self.metadata["idx species removed"]

        for index in sorted(idx_remove, reverse=True):
            del list_species[index]
        
        return list_species
    
    def updateStateSpaceNames(self, newFileStateSpaceNames):

        self.filename_state_space_names = newFileStateSpaceNames

    def getInputOutput(self, path_data, dataset_type):
        
        path_data = path_data
        general_dataset_type = self.metadata["general_dataset_type"]
        dataset_type = self.metadata["dataset_type"]
        list_species_input = self.metadata["list_species_input"]
        list_species_output = self.metadata["list_species_output"]
        input_scaling_name = self.metadata["input_scaling_name"]
        input_scaling = self.metadata["input_scaling"]
        input_bias = self.metadata["input_bias"]
        temperature_at_output = self.metadata["temperature_at_output"]
        header = self.metadata["header"]
        extra_manifold_variables = self.metadata["extra_manifold_variables"]
        range_extra_manifold_variables = self.metadata["extra_manifold_variables"]

        #get input/output
        input, output = get_dataset_analysis(path_data, general_dataset_type, dataset_type, list_species_input, list_species_output,
                                        input_scaling_name, input_scaling, input_bias, temperature_at_output,
                                        header, extra_manifold_variables, range_extra_manifold_variables)

        return input, output

    def getManifoldParameters(self, path_data, dataset_type):

        input, output = self.getInputOutput(path_data, dataset_type)

        model = self.loadModel(self.filename)

        extraVars, PV, PVsource = model.get_extraVar_PV_PVsource(input, output)

        return extraVars, PV, PVsource