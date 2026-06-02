import os
from pandas import read_csv, concat
import csv
import numpy as np
import pickle
from csv import reader
import logging
import torch
import torch.nn as nn
import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
from EncoderDecoder.models import PV_autoencoder
from PCAfold import KReg, compute_normalized_variance, normalized_variance_derivative, cost_function_normalized_variance_derivative, plot_normalized_variance_derivative
from torch import cat
from itertools import combinations

"""
PV optimization using an encoder-decoder architecture (Kamila Zdybal)
Version: Tools for the training
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: November 2025
Python version: 3.10.10
"""

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    datefmt='%Y-%m-%d %H:%M:%S'
)

#################################
#####        Loader         #####
#################################

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
            "data-files/out/" + filename_model,
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
        list_species_output = self.metadata["list_species_output_evaluation"]
        input_scaling_name = self.metadata["input_scaling_name"]
        input_scaling = self.metadata["input_species_scaling"]
        input_bias = self.metadata["input_species_bias"]
        temperature_at_output = self.metadata["temperature_output"]
        header = "infer"
        extra_manifold_variables = self.metadata["extra_manifold_parameters"]
        range_extra_manifold_variables = self.metadata["range_extra_manifold_parameters"]

        #get input/output
        input, output = get_dataset_analysis(path_data, general_dataset_type, dataset_type, list_species_input, list_species_output,
                                        input_scaling_name, input_scaling, input_bias, temperature_at_output,
                                        header, extra_manifold_variables, range_extra_manifold_variables)

        return input, output

    def getManifoldParameters(self, path_data, dataset_type):

        input, output = self.getInputOutput(path_data, dataset_type)

        model = self.loadModel(self.filename_model)

        extraVars, PV, PVsource = model.get_extraVar_PV_PVsource(input, output)

        return extraVars, PV, PVsource

#################################
#Create directories and filenames
#################################

class create_dirs:
    """ Creates directories for saving trained models and training information """

    def __init__(self, overall_dataset, dataset_type, current_time, training_id):
        self.overall_dataset = overall_dataset
        self.dataset_type = dataset_type
        self.formatted_date = current_time.strftime('%d%b%Y')
        self.formatted_time = current_time.strftime('%Hh%M')
        self.train_info_path = 'data-files/train-info/trained-models.csv'
        self.training_id = training_id

        self.path_out = "data-files/out/"
        self.path_curve = "data-files/curves/"
        self.path_metadata = "data-files/metadata/"

        self.training_name = '{}-AE-date_{}-hour_{}_{}-{}'.format(
            self.training_id,
            self.formatted_date,
            self.formatted_time,
            self.overall_dataset,
            self.dataset_type
            )
            
        self.dirout = f'{self.training_name}_model.pth'
        
        self.dircurves = f'{self.training_name}_curves.csv'

        self.dirMetadata = f'{self.training_name}_metadata.pkl'

    def create(self, train_headers):
        
        if not os.path.exists(self.path_out):
            os.makedirs(self.path_out)

        if not os.path.exists('data-files/train-info/'):
            os.makedirs('data-files/train-info/')
        
        if not os.path.exists(self.path_curve):
            os.makedirs(self.path_curve)

        if not os.path.exists(self.path_metadata):
            os.makedirs(self.path_metadata)
        
        if not os.path.exists(self.train_info_path):
            #add headers to the file if the file is not existing yet
            with open(self.train_info_path, mode='w', newline='') as file: #wrie or overwrite document
                writer = csv.DictWriter(file, fieldnames=train_headers)
                writer.writeheader()

    def save_model(self, state):
        torch.save(state, '{}{}'.format(self.path_out, self.dirout))

    def load_model(self, model_params):

        model_reloaded = PV_autoencoder(**model_params)
        model_reloaded.load_state_dict(torch.load('{}{}'.format(self.path_out, self.dirout), weights_only=False))

        return model_reloaded

    def save_train_info_model(self, data_train_info):
        with open(self.train_info_path, mode='a', newline='') as file:  #append data with "a"
            writer = csv.DictWriter(file, fieldnames=data_train_info.keys())
            writer.writerow(data_train_info)

    def save_train_val_curves(self, train_curve, val_curve):
        with open(f'{self.path_curve}{self.dircurves}', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(train_curve)  #training curve
            writer.writerow(val_curve)  #validation curve

    def save_metadata(self, metadata):
        with open(f'{self.path_metadata}{self.dirMetadata}', 'wb') as f:
            pickle.dump(metadata, f)
    
    def modify_dirout(self, epo: int):

        self.dirout = f'{self.training_name}_epo{epo}_model.pth'

        return None

######################################
#Handle input and output species lists
######################################

class Species:
    """
        Set of functions to convert name of species to indices
    """
    def __init__(self, path_data, file_species_names = "Xu-state-space-names.csv"):
        self.list_species = []

        #create a list with all the species names
        with open(path_data + file_species_names, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.list_species.append(row[0])

    def get_idx_of_species(self, species_name):
        """ Get the index of a specific species given its name. """
        try:
            idx_species_removed = self.list_species.index(species_name)
        except ValueError:
            logging.warning(f"'{species_name}' is not in the list.")

        return idx_species_removed
    
    def get_idx_from_list_species(self, list_species_name):
        """ Get the indices of a list of species given their names. """
        list_idx = []

        for species_name in list_species_name:
            list_idx.append(self.get_idx_of_species(species_name))

        return list_idx
    
    def get_list_species(self):
        """ Get list of species names. """

        return self.list_species


###################################
#Get data for training and plotting
###################################

def get_dataset_training(path_data, general_dataset_type, dataset_type, generator, perc_val, list_species_input, list_species_output,
             input_scaling, output_scaling, 
             temperature_output, header = 'infer',
             extra_manifold_variables = False, range_extra_manifold_variables = 1):
    """
        Get a train and validation input and output tensors for the chemical species and temperature
    
        Input = all chemical species + mixture fraction (last row)
        
        Output = chosen chemical species (with output_idx); temperature and all the source terms. PV-source has to be added afterwards given the source terms.
    
        Remark: data is given in torch.float64 format. This function is a short version of the get_dataloader function.
                Compared to get_dataset function, here the scaling of the output variables is already done and split into training and validation tensors.
    """

    path_data_state_space = path_data + f"{general_dataset_type}-state-space-" + dataset_type + ".csv"
    path_data_source = path_data + f"{general_dataset_type}-state-space_source-" + dataset_type + ".csv"
    
    #load the data
    data_state_space = read_csv(path_data_state_space, header = header)
    data_state_space_source = read_csv(path_data_source, header = header)
    if(temperature_output):
        path_data_temp = path_data + f"{general_dataset_type}-T-" + dataset_type + ".csv"
        data_temp = read_csv(path_data_temp, header = header)
    if extra_manifold_variables:
        list_extra_vars = []
        for variable in extra_manifold_variables:
            path_extra_var = path_data + f"{general_dataset_type}-{variable}-" + dataset_type + ".csv"
            data_extra_var = read_csv(path_extra_var, header=header)

            list_extra_vars.append(data_extra_var)
        data_extra_vars = concat(list_extra_vars, axis=1)

    #selected output species
    data_output_species = data_state_space[list_species_output]

    #selected input species
    data_state_space = data_state_space[list_species_input]
    data_state_space_source = data_state_space_source[list_species_input]

    nbr_input_species = data_state_space.shape[1]

    #combine the data for input and output
    if(extra_manifold_variables):
        data_input = concat([data_state_space, data_extra_vars], axis=1)
    else:
        data_input = data_state_space

    if(temperature_output):
        data_output = concat([data_output_species, data_temp, data_state_space_source], axis=1)
        nbr_feat_rescale = len(list_species_output)+1 #all species and temperature
    else:
        data_output = concat([data_output_species, data_state_space_source], axis=1)
        nbr_feat_rescale = len(list_species_output) #all species without temperature

    input = data_input.iloc[:,:].values
    output = data_output.iloc[:, :].values #contains only the species that do not change, the PV reaction rate has to be added during the training phase

    #convert to PyTorch tensors
    input_tensor = torch.tensor(input)
    output_tensor = torch.tensor(output)
    
    #determine number of training and validation samples
    dataset_size = input_tensor.size(0)
    train_size = int((1-perc_val) * dataset_size)

    #shuffle the indices
    indices = torch.randperm(dataset_size, generator=generator)

    #split in training and validation indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    #create input and output tensors for both training and validation
    train_input, val_input = input_tensor[train_indices], input_tensor[val_indices]
    train_output, val_output = output_tensor[train_indices], output_tensor[val_indices]


    ##############
    #RESCALE input
    ##############
    train_input_to_scale = train_input[:, :nbr_input_species]

    if(input_scaling in ["0to1", "std", "pareto", "mean-pareto", "None"]):
        mins = train_input_to_scale.min(dim=0, keepdim=True)[0]
        maxs = train_input_to_scale.max(dim=0, keepdim=True)[0]
        means = train_input_to_scale.mean(dim=0, keepdim=True)[0]
        stds = train_input_to_scale.std(dim=0, unbiased = False)
        nbr_columns = train_input_to_scale.shape[1]
        zeroes = torch.zeros(1, nbr_columns)

        if(input_scaling == "0to1"):
            input_species_bias = mins
            input_species_scaling = maxs - mins
        elif(input_scaling == "std"):
            input_species_bias = means
            input_species_scaling = train_input_to_scale.std(dim=0, unbiased = False) #compute the population standard deviation (N, unbiased False)
        elif(input_scaling == "pareto"):
            input_species_bias = zeroes
            input_species_scaling = torch.sqrt(stds)
        elif(input_scaling == "mean-pareto"):
            input_species_bias = means
            input_species_scaling = torch.sqrt(stds)
        elif(input_scaling == "None"):
            input_species_scaling = torch.ones(1, nbr_columns)
            input_species_bias = zeroes
        
        #rescale the selected training inputs
        train_input[:, :nbr_input_species] = (train_input[:, :nbr_input_species] - input_species_bias) / input_species_scaling

        #rescale the selected validation inputs
        val_input[:, :nbr_input_species] = (val_input[:, :nbr_input_species] - input_species_bias) / input_species_scaling

    elif(input_scaling=="-1to1"):
        mins = train_input_to_scale.min(dim=0, keepdim=True)[0]
        maxs = train_input_to_scale.max(dim=0, keepdim=True)[0]

        input_species_scaling = maxs - mins
        input_species_bias = mins

        #rescale the selected training inputs
        train_input[:, :nbr_input_species] = 2*(train_input[:, :nbr_input_species] - mins) / input_species_scaling -1

        #rescale the selected validation inputs
        val_input[:, :nbr_input_species] = 2*(val_input[:, :nbr_input_species] - mins) / input_species_scaling -1

    else:
        raise ValueError("get_data: input scaling not recognized")

    ###########
    #RESCALE f
    ###########
    if(range_extra_manifold_variables is not None and extra_manifold_variables):
        nbr_extra_manifold_variables = len(extra_manifold_variables)
        mins = train_input[:, -nbr_extra_manifold_variables:].min(dim=0, keepdim=True)[0]
        maxs = train_input[:, -nbr_extra_manifold_variables:].max(dim=0, keepdim=True)[0]

        #scale mf between -0.5 and 0.5, and apply it also to the validation dataset
        #also a range of 1 like PV then
        train_input[:, -nbr_extra_manifold_variables:] = range_extra_manifold_variables*(train_input[:, -nbr_extra_manifold_variables:] - mins) / (maxs - mins) - range_extra_manifold_variables/2
        val_input[:, -nbr_extra_manifold_variables:] = range_extra_manifold_variables*(val_input[:, -nbr_extra_manifold_variables:] - mins) / (maxs - mins) - range_extra_manifold_variables/2

    elif(range_extra_manifold_variables == "None"):
        logging.warning("No scaling for the extra manifold variables at the input.")

    else:
        raise ValueError("get_data: extra manifold variables scaling not recognized")

    ###############
    #RESCALE OUTPUT
    ###############
    #get min and max value of the species and temperature for the training dataset
    if(output_scaling=="-1to1"):
        mins = train_output[:, :nbr_feat_rescale].min(dim=0, keepdim=True)[0]
        maxs = train_output[:, :nbr_feat_rescale].max(dim=0, keepdim=True)[0]

        #rescale the selected training outputs
        train_output[:, :nbr_feat_rescale] = 2 * (train_output[:, :nbr_feat_rescale] - mins) / (maxs - mins) - 1

        #rescale the selected validation outputs
        val_output[:, :nbr_feat_rescale] = 2 * (val_output[:, :nbr_feat_rescale] - mins) / (maxs - mins) - 1

    elif(output_scaling=="mean-pareto"):
        means = train_output[:, :nbr_feat_rescale].mean(dim=0, keepdim=True)[0]
        std_dev = train_output[:, :nbr_feat_rescale].std(dim=0, keepdim=True)[0]

        #rescale the selected training outputs
        train_output[:, :nbr_feat_rescale] = (train_output[:, :nbr_feat_rescale] - means) / torch.sqrt(std_dev)

        #rescale the selected validation outputs
        val_output[:, :nbr_feat_rescale] = (val_output[:, :nbr_feat_rescale] - means) / torch.sqrt(std_dev)
    else:
        raise ValueError("get_data: output scaling not recognized")

    return train_input, train_output, val_input, val_output, dataset_size, input_species_scaling, input_species_bias

def get_dataset_analysis(path_data, general_dataset_type, dataset_type, list_species_input, list_species_output,
                input_scaling_name, input_scaling, input_bias, temperature_at_output = True,
                header = 'infer', extra_manifold_variables = [],
                range_extra_manifold_variables = 1):
    """
        Get the complete dataset in tensor format for the analysis once all the scalings have been computed.
    """

    path_data_state_space = path_data + f"{general_dataset_type}-state-space-{dataset_type}.csv"
    path_data_source = path_data + f"{general_dataset_type}-state-space_source-{dataset_type}.csv"
    
    #load the data
    data_state_space = read_csv(path_data_state_space, header = header)
    data_state_space_source = read_csv(path_data_source, header = header)
    if(temperature_at_output):
        path_data_temp = path_data + f"{general_dataset_type}-T-" + dataset_type + ".csv"
        data_temp = read_csv(path_data_temp, header = header)
    if extra_manifold_variables:
        list_extra_vars = []
        for variable in extra_manifold_variables:
            path_extra_var = path_data + f"{general_dataset_type}-{variable}-" + dataset_type + ".csv"
            data_extra_var = read_csv(path_extra_var, header=header)

            list_extra_vars.append(data_extra_var)
        data_extra_vars = concat(list_extra_vars, axis=1)

    nbr_extra_manifold_variables = len(extra_manifold_variables)
    if(range_extra_manifold_variables != "None"):
        mins = data_extra_vars.min(axis=0)
        maxs = data_extra_vars.max(axis=0)

        # normalize each column individually
        data_extra_vars = (data_extra_vars - mins) / (maxs - mins) * range_extra_manifold_variables - range_extra_manifold_variables/2

    #selected output species
    data_output_species = data_state_space[list_species_output]

    #remove one species from the dataframe
    data_state_space = data_state_space[list_species_input]
    data_state_space_source = data_state_space_source[list_species_input]

    #combine the data for input and output
    if(extra_manifold_variables):
        data_input = concat([data_state_space, data_extra_vars], axis=1)
    else:
        data_input = data_state_space
    
    if(temperature_at_output):
        data_output = concat([data_output_species, data_temp, data_state_space_source], axis=1)
    else:
        data_output = concat([data_output_species, data_state_space_source], axis=1)

    input = data_input.iloc[:,:].values
    output = data_output.iloc[:, :].values #contains only the species that do not change, the PV reaction rate has to be added during the training phase

    #convert to PyTorch tensors
    input_tensor = torch.tensor(input)
    output_tensor = torch.tensor(output)

    if(input_scaling_name in ["0to1", "std", "pareto", "mean-pareto"]):
        input_tensor[:, :-nbr_extra_manifold_variables] = (input_tensor[:, :-nbr_extra_manifold_variables]-input_bias)/input_scaling
    elif(input_scaling_name == "-1to1"):
        input_tensor[:, :-nbr_extra_manifold_variables] = 2*(input_tensor[:, :-nbr_extra_manifold_variables]-input_bias)/input_scaling -1
    elif(input_scaling_name == "None"):
        logging.info("No input scaling")
    else:
        logging.warning("Input scaling not recognized")

    return input_tensor, output_tensor

def get_dataset_from_np(np_state_space, np_state_space_source, np_temp, np_mf, output_idx, idx_species_removed):
    """
        Get a the complete dataset in tensor format.
    """

    #load the data
    data_state_space = pd.DataFrame(np_state_space)
    data_state_space_source = pd.DataFrame(np_state_space_source)
    data_temp = pd.DataFrame(np_temp)
    data_mf = pd.DataFrame(np_mf)

    #selected output species
    data_output_species = data_state_space.iloc[:,output_idx]

    #remove one species from the dataframe
    data_state_space = data_state_space.drop(data_state_space.columns[idx_species_removed], axis=1)
    data_state_space_source = data_state_space_source.drop(data_state_space_source.columns[idx_species_removed], axis=1)

    #combine the data for input and output
    data_input = concat([data_state_space, data_mf], axis=1)
    data_output = concat([data_output_species, data_temp, data_state_space_source], axis=1)

    input = data_input.iloc[:,:].values
    output = data_output.iloc[:, :].values #contains only the species that do not change, the PV reaction rate has to be added during the training phase

    #convert to PyTorch tensors
    input_tensor = torch.tensor(input)
    output_tensor = torch.tensor(output)

    return input_tensor, output_tensor

def get_dataset_from_np_scaled(np_state_space, np_state_space_source, np_temp, np_mf, output_idx, idx_species_removed, input_scaling, mf_scaling, path_data_state_space, path_data_mf):
    """
        Get a the complete dataset in tensor format.
    """

    #load the data
    data_state_space = pd.DataFrame(np_state_space)
    data_state_space_source = pd.DataFrame(np_state_space_source)
    data_temp = pd.DataFrame(np_temp)
    data_mf = pd.DataFrame(np_mf)

    data_state_space_scaler = read_csv(path_data_state_space)
    data_mf_scaler = read_csv(path_data_mf)

    #scaling of input
    if(input_scaling == "0to1"):
        for i in range(data_state_space.shape[1]):
            min_val = data_state_space_scaler.iloc[:, i].min()
            max_val = data_state_space_scaler.iloc[:, i].max()
            
            #Scalte 0 to 1
            data_state_space.iloc[:, i] = (data_state_space.iloc[:, i] - min_val) / (max_val - min_val)

            print(data_state_space.iloc[:, i].min())
            print(data_state_space.iloc[:, i].max())

    if(mf_scaling == "-05to05"):
        min_val = data_mf_scaler.iloc[:, 0].min()
        max_val = data_mf_scaler.iloc[:, 0].max()
        print(min_val)
        print(max_val)
        
        #Scalte 0 to 1
        data_mf.iloc[:, 0] = (data_mf.iloc[:, 0] - min_val) / (max_val - min_val)

    #selected output species
    data_output_species = data_state_space.iloc[:,output_idx]

    #remove one species from the dataframe
    data_state_space = data_state_space.drop(data_state_space.columns[idx_species_removed], axis=1)
    data_state_space_source = data_state_space_source.drop(data_state_space_source.columns[idx_species_removed], axis=1)

    #combine the data for input and output
    data_input = concat([data_state_space, data_mf], axis=1)
    data_output = concat([data_output_species, data_temp, data_state_space_source], axis=1)

    input = data_input.iloc[:,:].values
    output = data_output.iloc[:, :].values #contains only the species that do not change, the PV reaction rate has to be added during the training phase

    #convert to PyTorch tensors
    input_tensor = torch.tensor(input)
    output_tensor = torch.tensor(output)

    return input_tensor, output_tensor


###########
#Rescalings
###########

def rescale_PVsource(data, nbr_PV):
    """
    Rescale indicated variable between -1 and 1.

    Used for the source PV.
    """

    #get the feature to rescale
    features = data[:, -nbr_PV:]
    
    #initialize holder for rescaled features
    rescaled_features = []
    
    #rescale every feature
    for i in range(nbr_PV):
        column = features[:, i]
        min_val = column.min()
        max_val = column.max()
        rescaled_column = 2 * (column - min_val) / (max_val - min_val) - 1
        rescaled_features.append(rescaled_column.unsqueeze(1))
    
    # Concatenate all rescaled columns together along the last dimension
    rescaled_features = torch.cat(rescaled_features, dim=1)
    
    # Concatenate the non-PV part of the data with the rescaled PV features
    data = torch.cat((data[:, :-nbr_PV], rescaled_features), dim=1)
    
    return data

################
#Optimizer tools
################

def distance_corr(x, y):
    # n = x.shape[0]

    # x = x.view(n, -1)
    # y = y.view(n, -1)

    # a = torch.cdist(x, x, p=2)
    # b = torch.cdist(y, y, p=2)

    # A = a - a.mean(dim=0) - a.mean(dim=1, keepdim=True) + a.mean()
    # B = b - b.mean(dim=0) - b.mean(dim=1, keepdim=True) + b.mean()

    # dcov = (A * B).mean()
    # dvar_x = (A * A).mean()
    # dvar_y = (B * B).mean()

    # # Add square root to match dcor
    # dcor = torch.sqrt(dcov / torch.sqrt(dvar_x * dvar_y))
    # return dcor
    return None

def spearman_corr(pred, target):
    pred_rank = torchsort.soft_rank(pred.view(1, -1), regularization="l2").to(pred.device)
    target_rank = torchsort.soft_rank(target.view(1, -1), regularization="l2").to(target.device)

    pred_rank = (pred_rank - pred_rank.mean()) / pred_rank.std()
    target_rank = (target_rank - target_rank.mean()) / target_rank.std()

    corr = (pred_rank * target_rank).mean()
    return corr

def get_optimizer(model_parameters, optimizer_name, learning_rate, alpha = None, momentum = None):

    if(optimizer_name.lower()=="adam"):
        optimizer = torch.optim.Adam(model_parameters, lr=learning_rate)
    elif(optimizer_name.lower()=="rmsprop"):
        if(alpha is None or momentum is None):
            raise ValueError("get_optimizer: both alpha and momentum should be defined for RMSprop optimizer")
        optimizer = torch.optim.RMSprop(model_parameters, lr=learning_rate, alpha = alpha, momentum = momentum)
    else:
        raise ValueError("get_optimizer: optimizer_name not in the list")
    
    return optimizer

def get_loss_criterion(loss_name, lambda_reg=1):
    
    if(loss_name.lower()=="mse"):
        loss_criterion = nn.MSELoss()

    elif(loss_name.lower() == "mse_orth_w_multipv"):

        """
        MSE for the reconstruction with a regularization term promoting orthogonality
        between the weights of any number of PVs (columns of the encoder weight matrix).

        Pay attention: this loss function only works correctly in case the first PV is not fixed or if one PV is fixed, then without scaling.
        The loss here does not include scaling, which would influence results if the first PV is fixed.
        """

        def custom_loss(predictions, targets, model):
            mse = nn.MSELoss()(predictions, targets)

            PV_weights = model.encoder_species.weight
            n_PVs = PV_weights.shape[0]

            orth_reg = 0.0
            for i, j in combinations(range(n_PVs), 2):
                orth_reg += torch.abs(torch.dot(PV_weights[i], PV_weights[j]))

            loss = mse + lambda_reg * orth_reg
            return loss

        loss_criterion = custom_loss
    
    elif(loss_name.lower() == "mse_orth_multipv"): #slow implementation of the multipv

        """
        MSE for the reconstruction with a regularization term promoting orthogonality between any number of PVs.

        Pay attention: this loss function only works correctly in case the first PV is not fixed or if one PV is fixed, then without scaling.
        The loss here does not the scaling which would have an influence with the first PV fixed.
        """

        def custom_loss(predictions, targets, model, input):
            mse = nn.MSELoss()(predictions, targets)
            
            PV = model.get_PV(input)

            n_PVs = PV.shape[1]

            orth_reg = 0.0
            for i, j in combinations(range(n_PVs), 2):
                orth_reg += torch.abs(torch.dot(PV[:, i], PV[:, j]))

            loss = mse + lambda_reg * orth_reg
            return loss
        
        loss_criterion = custom_loss

    elif(loss_name.lower() == "mse_distcorr_multipv"):

        """
        MSE for reconstruction with a regularization term promoting decorrelation
        between any number of PV outputs using distance correlation.
        """

        def custom_loss(predictions, targets, model, input):
            mse = nn.MSELoss()(predictions, targets)

            PV = model.get_PV(input)
            n_PVs = PV.shape[1]

            dcor_reg = 0.0
            for i, j in combinations(range(n_PVs), 2):
                dcor_reg += distance_corr(PV[:, i], PV[:, j])

            loss = mse + lambda_reg * dcor_reg
            return loss

        loss_criterion = custom_loss

    elif(loss_name.lower() == "mse_spearmancorr_multipv"):

        """
        MSE for reconstruction with a regularization term promoting decorrelation
        between any number of PV outputs using spearman correlation.
        """

        def custom_loss(predictions, targets, model, input):
            mse = nn.MSELoss()(predictions, targets)

            PV = model.get_PV(input)
            n_PVs = PV.shape[1]

            spearmancorr_reg = 0.0
            for i, j in combinations(range(n_PVs), 2):
                spearmancorr_reg += spearman_corr(PV[:, i], PV[:, j])

            loss = mse + lambda_reg * (spearmancorr_reg**2)
            return loss

        loss_criterion = custom_loss
    
    elif(loss_name.lower() == "mse_orth_pvnorm"): #faster implementation of the multipv
        """
        MSE for the reconstruction with a regularization term promoting orthogonality between any number of PVs.

        Pay attention: this loss function only works correctly in case the first PV is not fixed or if one PV is fixed, then without scaling.
        The loss here does not the scaling which would have an influence with the first PV fixed.
        """

        def custom_loss(predictions, targets, model, input):
            #MSE for reconstruction error
            mse = nn.MSELoss()(predictions, targets)
            
            PV = model.get_PV(input)

            #normalize the vectors
            PV_norm = PV / (PV.norm(dim=0, p = 2, keepdim=True))

            dot_matrix = PV_norm.T @ PV_norm  # pairwise dot products
            orth_reg = torch.sum(torch.triu(torch.abs(dot_matrix), diagonal=1))

            # Combine the two losses
            loss = mse + lambda_reg * orth_reg
            return loss
            
        loss_criterion = custom_loss

    elif(loss_name.lower() == "mse_l1"):
        """MSE with L1 norm regularization applied on all weights of the model
        """

        def custom_loss(predictions, targets, model):
            # Reconstruction MSE
            mse = nn.MSELoss()(predictions, targets)

            # L1 regularization over ALL model parameters
            l1_reg = 0.0
            for param in model.parameters():
                l1_reg = l1_reg + torch.sum(torch.abs(param))

            # Combine
            loss = mse + lambda_reg * l1_reg
            return loss

        loss_criterion = custom_loss
        

    else:
        raise ValueError("get_loss_criterion: loss_name not in the list")
    
    return loss_criterion

def cosine_decay(alpha, epo, tot_epo):
    """
    Cosine decay learning rate. Start at the initial learning rate and ends at initial learning rate times alpha.
    After tot_epo, the learning rate is constant and equal to initial learning rate times alpha.
    Alpha is the multiplier for the final learning rate.
    """

    myEpo = np.min([epo,tot_epo])

    return 0.5*(1-alpha)*(1+np.cos(np.pi*myEpo/tot_epo))+alpha

# TO MODIFY, add in the loader
def load_PV(optimized_PV, data_state_space, data_state_space_source, state_space_names, scaled_PV = False, filename_optimized_PV = "", weight_inversion = False):

    if(optimized_PV):
        filename_metadata = filename_optimized_PV + "_metadata.pkl"
        path_metadata = "metadata/"
        filename_species_names = "Xu-state-space-names.csv"
        path_data = "data-files/"

        loader = loadData(filename_species_names, path_metadata, filename_metadata)
        idx_species_removed = loader.metadata["list idx species removed source"] if 'augm' in loader.metadata["dataset_type"] else loader.metadata["idx species removed"]
        model = loader.loadModel()
        id_model = loader.metadata["Training_id"]

        # inverse the PV definition
        if(weight_inversion):
            with torch.no_grad():  # Ensures we do not track gradients for this operation
                model.encoder_species.weight.mul_(-1)

        PV = model.get_PV(torch.from_numpy(np.delete(data_state_space, idx_species_removed, axis=1))).detach().numpy()
        if(scaled_PV):
            PV = (PV - PV.min())/(PV.max() - PV.min())
        PV_source = model.get_PV(torch.from_numpy(np.delete(data_state_space_source, idx_species_removed, axis=1))).detach().numpy()
        
    else:
        idx_H2O = state_space_names.index("H2O")
        idx_H2 = state_space_names.index("H2")
        idx_O2 = state_space_names.index("O2")

        PV = data_state_space[:, idx_H2O] - data_state_space[:, idx_H2] - data_state_space[:, idx_O2]
        if(scaled_PV):
            PV = (PV - PV.min())/(PV.max() - PV.min())
        PV_source = data_state_space_source[:, idx_H2O] - data_state_space_source[:, idx_H2] - data_state_space_source[:, idx_O2]

        id_model = "Heuristic PV"

    return PV, PV_source, id_model

def compute_Kreg(   path_data,
                    general_dataset_type,
                    dataset_type,
                    list_species_input,
                    list_species_output,
                    input_scaling_name,
                    input_species_scaling,
                    input_species_bias,
                    extra_manifold_parameters,
                    range_extra_manifold_variables,
                    model, device):

    """Compute the MSE of the kernel regression on the validation dataset using different number of neighbours.
    The QOI's are reconstructed given the mixture fraction and the progress variable.

    Args:
        path_data (str): Path to data files.
        general_dataset_type (str, optional): Type of dataset to be used. Defaults to "Xu" referring to the datasets linked to Xu Wen.
        dataset_type (str, optional): Type of dataset to be used. Defaults to "low" referring to the sampled DNS dataset. 
        list_species_input (list[str]): List of species to be selected for the input dataset.  
        list_species_output (list[str]): List of species to be selected for the output dataset.  
        input_scaling_name (str): Name of the scaling to be applied.  
        input_species_scaling (float): Range of the scaling to be applied on mixture fraction and progress variable.  
        input_species_bias (float): Centering applied on mixture fraction and progress variable.  
        extra_manifold_parameters (list[str]): List of the additional manifold parameters.
        range_extra_manifold_variables (float): Float indicating the range of values for the additional manifold parameters. 
        model (PV_autoencoder): Trained encoder-decoder defining the optimized progress variable.
        device (str): Indicates whether the data is on cpu or cuda. 

    Returns:
        list[float]: Return the average MSE and its standard deviation.
    """

    neighbours = [5, 10, 15, 20, 25]
    seed = 9

    mse_values_model = np.zeros(len(neighbours))

    input, output = get_dataset_analysis(   path_data, general_dataset_type, dataset_type, list_species_input, list_species_output,
                                            input_scaling_name, input_species_scaling, input_species_bias, extra_manifold_variables=extra_manifold_parameters,
                                            range_extra_manifold_variables=range_extra_manifold_variables)

    input, output = input.to(device), output.to(device)

    PV = model.get_PV(input)
    output = model.get_source_PV(output, input_species_scaling)

    f_PV = cat((input[:,-1].unsqueeze(1), PV), dim = 1)

    min_f_PV = f_PV.min(dim=0, keepdim=True)[0]  # Minimum values for each column
    max_f_PV = f_PV.max(dim=0, keepdim=True)[0]  # Maximum values for each column

    f_PV_scaled = (f_PV - min_f_PV) / (max_f_PV - min_f_PV)

    min_output = output.min(dim=0, keepdim=True)[0]  # Minimum values for each column
    max_output = output.max(dim=0, keepdim=True)[0]  # Maximum values for each column

    output_scaled = (output - min_output) / (max_output - min_output)

    #create training and validation datasets
    np.random.seed(seed)
    nbr_observations = f_PV_scaled.shape[0]
    indices = np.arange(nbr_observations)
    nbr_train = int(nbr_observations*0.8)
    sampled_indices = np.random.choice(indices, size=nbr_train, replace=False)
    validation_indices = np.setdiff1d(indices, sampled_indices)

    input_model = f_PV_scaled.detach().cpu().numpy()
    output_model = output_scaled.detach().cpu().numpy()

    for j, neighbour in enumerate(neighbours):

        query = input_model[validation_indices,:]

        kernel_model = KReg(input_model[sampled_indices, :], output_model[sampled_indices, :])
        predicted_model = kernel_model.predict(query, 'nearest_neighbors_isotropic', n_neighbors=neighbour)

        squared_error_model = (predicted_model - output_model[validation_indices,:]) ** 2
        mse_model = np.mean(squared_error_model)
        mse_values_model[j] = mse_model

    avg_mse = np.mean(mse_values_model)
    std_mse = np.std(mse_values_model)

    return [avg_mse, std_mse]

def compute_avg(costs):
    """Compute the Root mean square of all QoI costs.

    Args:
        costs (list[float]): List of all QoI costs.

    Returns:
        float: Root mean square of all QoI costs.
    """

    n = len(costs)
    sum = np.sum(costs**2)
    return 1/n*np.sqrt(sum)

def compute_cost(   path_data, general_dataset_type, dataset_type,
                    list_species_input, list_species_output,
                    input_scaling_name, input_species_scaling, input_species_bias,
                    extra_manifold_parameters, range_extra_manifold_variables,
                    depvar_names_idx, PV_dim, model, id):
    
    """Computes the cost of the mixture fraction-progress variable manifold using the PCAfold library.

    Args:
        path_data (str):  Path to data files.
        general_dataset_type (str, optional): Type of dataset to be used. Defaults to "Xu" referring to the datasets linked to Xu Wen.
        dataset_type (str, optional): Type of dataset to be used. Defaults to "low" referring to the sampled DNS dataset.
        list_species_input (list[str]): List of species to be selected for the input dataset.  
        list_species_output (list[str]): List of species to be selected for the output dataset.
        input_scaling_name (float): Range of the scaling to be applied on mixture fraction and progress variable.  
        input_species_scaling (str): Name of the scaling to be applied.  
        input_species_bias (float): Centering applied on mixture fraction and progress variable.  
        extra_manifold_parameters (list[str]): List of the additional manifold parameters.
        range_extra_manifold_variables (float): Float indicating the range of values for the additional manifold parameters.  
        depvar_names_idx (list[int]): List of indices for which the cost has to be computed.
        PV_dim (int): Indicates the dimension of the progress variable.
        model (PV_autoencoder): Trained encoder-decoder defining the optimized PV
        id (str): id of the model.     

    Returns:
        float: Compute root mean square of all QoI costs.
    """

    print("start compute cost")
    penalty_function = 'log-sigma-over-peak'
    start_bw = -6
    end_bw = 2
    nbr_points_bw = 100
    bandwidth_values = np.logspace(start_bw, end_bw, nbr_points_bw)
    power = 1
    vertical_shift = 1

    depvar_names = list_species_output + depvar_names_idx[-(PV_dim+1):]

    print("import dataset")
    #get the input (PV and f) and the output (interested Yi, T and source terms) data

    input, output = get_dataset_analysis(   path_data, general_dataset_type, dataset_type, list_species_input, list_species_output,
                                            input_scaling_name, input_species_scaling, input_species_bias,
                                            extra_manifold_variables=extra_manifold_parameters, range_extra_manifold_variables=range_extra_manifold_variables)
    PV = model.get_PV(input)
    PV_f = cat((PV, input[:, -1].reshape(-1, 1)), dim = 1) #reshape to be (5200,1) instead of (52000)
    output = model.get_source_PV(output, input_species_scaling)

    #scale every column of the PV_f tensor between 0 and 1
    min_vals = PV_f.min(dim=0, keepdim=True).values
    max_vals = PV_f.max(dim=0, keepdim=True).values
    PV_f_scaled = (PV_f - min_vals) / (max_vals - min_vals)

    indepVars = PV_f_scaled.detach().numpy()
    depVars = output.detach().numpy()

    print("compute variance data")
    variance_data = compute_normalized_variance(indepVars,
                                                    depVars,
                                                    depvar_names=depvar_names,
                                                    bandwidth_values=bandwidth_values)
    np.save(f"costs/variance_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}.npy", variance_data)

    print("compute costs")
    costs = cost_function_normalized_variance_derivative(variance_data,
                                                        penalty_function=penalty_function,
                                                        power=power,
                                                        vertical_shift=vertical_shift,
                                                        norm=None)
    np.save(f"costs/costs_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}.npy", costs)

    (derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance_data)

    plt = plot_normalized_variance_derivative(variance_data)
    plt.savefig(f"costs/plot_Dhat_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}.png")

    cost = compute_avg(np.array(costs))

    return cost

def compute_costs_from_varianceFile(name_variance, path_variance = "data-files/costs/", penalty_function = 'log-sigma-over-peak', power = 4, vertical_shift = 1):
    """Compute the costs based on the variance file.

    Args:
        name_variance (str): Name of the variance file.
        path_variance (str, optional): Path to the variance file. Defaults to "data-files/costs/".
        penalty_function (str, optional): Type of penalty applied to the derivatives. Defaults to 'log-sigma-over-peak'.
        power (int, optional): Power applied for the penalty. Defaults to 4.
        vertical_shift (int, optional): Vertical shift applied for the penalty. Defaults to 1.

    Returns:
        np.ndarray: Array containing the cost of every QoI.
    """
    variance = np.load(f"{path_variance}{name_variance}", allow_pickle=True).item()

    (derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance)

    costs = cost_function_normalized_variance_derivative(   variance,
                                                            penalty_function=penalty_function,
                                                            power=power,
                                                            vertical_shift=vertical_shift,
                                                            norm=None)
    
    return costs

def compute_derivative_from_varianceFile(name_variance, path_variance = "data-files/costs/", penalty_function = 'log-sigma-over-peak', power = 4, vertical_shift = 1):
    """Returns the derivatives, bandwith value and the bandwidth value with the highest derivative based on the variance file.

    Args:
        name_variance (str): Name of the variance file.
        path_variance (str, optional): Path to the variance file. Defaults to "data-files/costs/".
        penalty_function (str, optional): Type of penalty applied to the derivatives. Defaults to 'log-sigma-over-peak'.
        power (int, optional): Power applied for the penalty. Defaults to 4.
        vertical_shift (int, optional): Vertical shift applied for the penalty. Defaults to 1.

    Returns:
        derivative (np.ndarray): Array containing the derivative at each bandwidth value.
        bandwidth_values (np.ndarray): Array of bandwidth values at which the derivatives were computed.
        max_derivative (float): Bandwidth value at which the highest derivative is reached.
    """
    variance = np.load(f"{path_variance}{name_variance}", allow_pickle=True).item()

    (derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance)

    return derivative, bandwidth_values, max_derivative