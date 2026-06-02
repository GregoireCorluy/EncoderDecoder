"""
PV optimization using an encoder-decoder architecture (Kamila Zdybal)
Version: Definition of the model and the tools linked to it
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: November 2025
Python version: 3.10.10
"""


import torch
import torch.nn as nn
import copy
import warnings

class PV_autoencoder(nn.Module):
    """
        Encoder-decoder architecture to optimize the PV defintiion given a combustion data.

        Input: All the mass fractions of the species and the mixture fraction.

        Output: The selected species, the temperature and the PV source term.

        Remark: Input data should be in torch.float64 format.
    """

    def __init__(   self, nbr_species, PV_dim, output_dim, decoder_layers,
                    species_scaling_layer = False, activation_function = "tanh",
                    activation_function_output = "tanh", fixed_PV = False,
                    extra_manifold_parameters = [], dropout_decoder_layers = False,
                    p_dropout = -1, **kwargs):

        super(PV_autoencoder, self).__init__()

        self.fixed_PV = fixed_PV
        self.PV_dim = PV_dim
        self.extra_manifold_parameters = extra_manifold_parameters
        self.nbr_extra_manifold_parameters = len(extra_manifold_parameters)
        self.dropout_decoder_layers = dropout_decoder_layers
        self.p_dropout = p_dropout

        self.dropout_function = (nn.Dropout(self.p_dropout) if self.dropout_decoder_layers else nn.Identity())
        
        self.nbr_species = nbr_species
        n_decoder = copy.deepcopy(decoder_layers)
        n_decoder.append(0) #add the zero for the output layer
        
        self.species_scaling_layer = species_scaling_layer
        #Automatic scaling of the input species
        if(self.species_scaling_layer):
            self.scaling_weights = nn.Parameter(torch.ones(self.nbr_species, dtype=torch.float64))
            #self.scaling_biases = nn.Parameter(torch.zeros(self.nbr_species, dtype=torch.float64))

        #PV definition with all the species
        self.encoder_species = nn.Linear(self.nbr_species, PV_dim, bias = False, dtype=torch.float64)

        #Store the different decoder layers
        self.decoder = nn.ModuleList()

        #depending if mixture fraction is included as one of the manifold parameters
        if(self.extra_manifold_parameters):
            self.decoder.append(nn.Linear(PV_dim+self.nbr_extra_manifold_parameters, output_dim + n_decoder[0], dtype=torch.float64))
        else:
            self.decoder.append(nn.Linear(PV_dim, output_dim + n_decoder[0], dtype=torch.float64))

        for layer_idx in range(len(n_decoder)-1):
            self.decoder.append(nn.Linear(output_dim + n_decoder[layer_idx], output_dim + n_decoder[layer_idx+1], dtype=torch.float64))
        
        if(activation_function.lower() == "tanh"):
            self.activation_function = nn.Tanh()
        elif(activation_function.lower() == "relu"):
            self.activation_function = nn.ReLU()
        elif activation_function.lower() == "sigmoid":
            self.activation_function = nn.Sigmoid()
        elif activation_function.lower() == "linear":
            self.activation_function = nn.Identity()

        if(activation_function_output.lower() == "tanh"):
            self.activation_function_output = nn.Tanh()
        elif(activation_function.lower() == "relu"):
            self.activation_function = nn.ReLU()
        elif activation_function_output.lower() == "sigmoid":
            self.activation_function_output = nn.Sigmoid()
        elif activation_function_output.lower() == "linear":
            self.activation_function_output = nn.Identity()


    def forward(self, input_data):
        #split the input into species and mixture fraction
        
        if(self.extra_manifold_parameters):
            data_extra_manifold_parameters = input_data[:, -self.nbr_extra_manifold_parameters:]
        PV = self.get_PV(input_data)

        #Concatenate the PV and the mixture fraction horizontally to get the latent space
        #corresponds to the bottleneck
        x = torch.cat((PV, data_extra_manifold_parameters), dim=1) if self.extra_manifold_parameters else PV
        
        for i, layer in enumerate(self.decoder):
            x = layer(x)

            # output layer → use output activation
            if i == len(self.decoder) - 1:
                x = self.activation_function_output(x)
            # hidden layers → use normal activation + dropout
            else:
                x = self.dropout_function(self.activation_function(x))

        return x

    def get_PV(self, input_data):
        """
        Get the PV-value using the input data and the encoder.
        
        Remark: Can also be used to get the source PV, by giving the source terms as input data.
        """

        species = input_data[:, :self.nbr_species]  

        #scale the species
        if(self.species_scaling_layer):
            species_scaled = species * self.scaling_weights  #+ self.scaling_biases

        #Combine the species to get the PV
        if(not self.fixed_PV): #in case there is no fixed PV, use the species or scaled species for all the rows
            if(not self.species_scaling_layer):
                PV = self.encoder_species(species)
            elif(self.species_scaling_layer):
                PV = self.encoder_species(species_scaled)
        elif(self.fixed_PV): #in case the first PV is fixed
            if(not self.species_scaling_layer):
                PV = self.encoder_species(species)
            elif(self.species_scaling_layer): #do not apply the scaling to the row of the fixed PV which is already scaled
                PV_fixed = self.encoder_species(species)[:,0].unsqueeze(1) #keep the first column
                PV_others  = self.encoder_species(species_scaled)[:, 1:] #keep all the columns except the first one

                PV = torch.cat([PV_fixed, PV_others], dim = 1) #stack different columns together

        return PV

    def initialize_model_weights(self, generator, std_init_enc, std_init_dec, init_scaling = (1.0, 2.0), weights_first_PV = "None"):
        """
        Initialize the encoder and decoder weights
        """

        #encoder initialization
        #all weights equal to one
        #nn.init.ones_(model.encoder_species.weight)
        nn.init.normal_(self.encoder_species.weight, mean=0.0, std=std_init_enc, generator = generator)

        #set the encoder weights for the first PV in case the first PV is fixed
        if(self.fixed_PV):
            with torch.no_grad():
                self.encoder_species.weight[0] = weights_first_PV

        #decoder initialization
        #weights random, method has still to be investigated
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):  # Check if the layer is of type nn.Linear
                nn.init.normal_(layer.weight, mean=0.0, std=std_init_dec, generator = generator)  # Initialize weights with normal distribution
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # Initialize bias to zero
        
        if(self.species_scaling_layer):
            low, high = init_scaling
            nn.init.uniform_(self.scaling_weights, a=low, b=high, generator = generator)

        return None

    def rescale_encoder_data(self, input_data, scale_PV, always_rescale = False):
        """
        Rescale the encoder to get a PV range of 1.
        Second version of rescale_encoder function where an input tensor is used instead of a dataloader.
        """

        all_PV = self.get_PV(input_data)

        scaling_factor = torch.max(all_PV, dim = 0).values-torch.min(all_PV, dim = 0).values

        #scale for every PV separately
        #rescale now only when the range is too small, do nothing when range large enough
        for i in range(scaling_factor.numel()):
            if(scaling_factor[i]<scale_PV):
                with torch.no_grad():
                    if(self.fixed_PV and i>0 or not self.fixed_PV):
                        self.encoder_species.weight.data[i,:] /= (scaling_factor[i]/scale_PV)
            elif(always_rescale):
                with torch.no_grad():
                    if(self.fixed_PV and i>0 or not self.fixed_PV):
                        self.encoder_species.weight.data[i,:] /= (scaling_factor[i]/scale_PV)
        return None

    def get_source_PV(self, batch, input_species_scaling):
        """
        Remove the source terms from the batch and add the PV source term
        """

        source_terms = batch[:, -self.nbr_species:]/input_species_scaling.to(batch.device)
        PV_source = self.get_PV(source_terms)

        batch_without_source_terms = batch[:,:-self.nbr_species]
        batch_with_PV_source = torch.cat((batch_without_source_terms, PV_source), dim=1)

        return batch_with_PV_source
    
    def get_extraVar_PV_PVsource(self, input, output):

        PV = self.get_PV(input).detach().numpy()
        PVsource = self.get_PV(output[:,-self.nbr_species:]).detach().numpy()
        
        if(self.extra_manifold_parameters):
            data_extra_manifold_parameters = input[:,-self.nbr_extra_manifold_parameters:].numpy()
            return data_extra_manifold_parameters, PV, PVsource
        else:
            warnings.warn("No extra manifold variable is not available because 'self.extra_manifold_variables' is empty.", UserWarning)
            return PV, PVsource
    
    def get_extraVar_PV_species(self, input, speciesIdx):

        PV = self.get_PV(input).detach().numpy()
        species = input[:, speciesIdx].numpy()

        if(self.extra_manifold_parameters):
            data_extra_manifold_parameters = input[:,-self.nbr_extra_manifold_parameters:].numpy()
            return data_extra_manifold_parameters, PV
        else:
            warnings.warn("No extra manifold variable is not available because 'self.extra_manifold_variables' is empty.", UserWarning)
            return PV, species
    
    def get_scaling_weights(self):

        return self.scaling_weights.detach()
    
    def get_encoder_weights(self):

        return self.encoder_species.weight.detach()

    def get_scaled_encoder_weights(self):
        
        if(self.species_scaling_layer):
            return self.scaling_weights.detach()*self.encoder_species.weight.detach()
        else:
            print("Warning: model has no scaling layer. Only encoder weights returned")
            return self.encoder_species.weight.detach()

    def get_total_encoder_weights(self, npy = False):

        if(self.species_scaling_layer):
            total_weights = self.scaling_weights.detach()*self.encoder_species.weight.detach()
        else:
            total_weights = self.encoder_species.weight.detach()
        
        if(npy):
            total_weights = total_weights.numpy()
            
        return total_weights
    
    def reset_first_PV(self, weights_first_PV):
        """
        As it is not possible to fix partially a tensor,
        it has to be done manually after every weight update by resetting the encoder weights to the old ones.
        """

        if(self.fixed_PV):
            with torch.no_grad():
                self.encoder_species.weight[0] = weights_first_PV

        return None