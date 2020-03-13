import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from collections import namedtuple

Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])
LossFunction = namedtuple('LossFunction', ['function', 'name'])

def normalize_datafile(datafile):
    return (datafile - datafile.mean()) / datafile.std()

class AllJetsDataset(Dataset):
    def __init__(self, pickle_file):
        """
        Args:
            pickle_file (string): Path to the pickle file.
        """
        # Load the file
        datafile = pd.read_pickle(pickle_file)

        # Normalise the dataset
        normalised_df = normalize_datafile(datafile)

        self.data = normalised_df.values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        return self.data[idx]

class AE_Trainer:
    def __init__(self, autoencoder, datasets, loss_func, optimiser, num_epochs=10, batch_size=256):
        """
        Args:
            autoencoder (pytorch nn.Module): The AutoEncoder to train
            datasets (namedtuple): A namedtuple of type Datasets (refer above)
            loss_func (namedtuple): A namedtuple of type LossFunction (refer above)
            optimiser (pytorch Optimiser): An *initialised* pytorch optimiser
            num_epochs (Integer): Number of epochs to train for
            batch_size (Integer): Size of each batch while training
        """
        self.autoencoder = autoencoder
        self.loss_func, self.loss_func_name = loss_func
        self.optimiser = optimiser
        self.num_epochs = num_epochs
        self.batch_size = batch_size
            
        self.train_dataset, self.validation_dataset, self.test_dataset = datasets       
        self.train_dataloader = DataLoader(
            torch.Tensor(self.train_dataset).cuda(), 
            batch_size=self.batch_size, 
            shuffle=True)
        self.validation_dataset = torch.Tensor(self.validation_dataset).cuda()
        self.test_dataset = torch.Tensor(self.test_dataset).cuda()
        
        self.epoch_train_loss = []
        self.epoch_validation_loss = []
        self.batch_train_loss = []
        self.batch_validation_loss = []
    
    def train(self):
        # Use tqdm to obtain a progressbar for the Training
        epoch_stats = tqdm(range(self.num_epochs), desc='Training {} | Epoch'.format(self.autoencoder.__class__.__name__))

        for epoch in epoch_stats:
            for data in self.train_dataloader:
                # Forward
                output = self.autoencoder(data)
                loss = self.loss_func(data, output)
                # Backward
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

            self.epoch_train_loss.append(loss.item())
            
            validation_loss = self.calculate_dataset_loss(self.validation_dataset)
            self.epoch_validation_loss.append(validation_loss)
            
            epoch_stats.set_postfix(Train_Loss=loss.item(), Validation_Loss=validation_loss)
            # print('epoch [{}/{}], train loss:{:.4f}, validation loss:{:.4f}'
            #        .format(epoch+1, self.num_epochs, loss.item(), validation_loss))
            
    def test(self):
        self.test_output = self.autoencoder(self.test_dataset)
        
        self.test_loss = self.loss_func(self.test_dataset, self.test_output).item()
        print("Test loss is: ", self.test_loss)
        
    def plot_test_residue_graphs(self):
        self.plot_difference_graphs(self.test_dataset, self.test_output)

    def plot_test_reconstruction_distribution(self):
        self.plot_reconstruction_distribution(self.test_dataset, self.test_output)
            
    def calculate_dataset_loss(self, dataset):
        """
            Args:
                dataset (Tensor): A pytoch tensor of the dataset
            Description:
                Calculates and returns the reconstruction loss on encoding and decoding the dataset
        """
        output = self.autoencoder(dataset)
        loss = self.loss_func(dataset, output)
        return loss.item()
    
    def plot_reconstruction_loss(self):
        plt.title('Reconstruction Loss')
        plt.xlabel('Epochs')
        plt.ylabel(self.loss_func_name + ' Loss')
        plt.plot(self.epoch_train_loss, label='Train')
        plt.plot(self.epoch_validation_loss, linestyle='dashed', label='Validation')
        plt.legend()
        
    def plot_reconstruction_distribution(self, original_dataset, uncompressed_dataset):
        original_dataset = original_dataset.cpu().detach().numpy().transpose()
        uncompressed_dataset = uncompressed_dataset.cpu().detach().numpy().transpose()

        figure, [m, pt, phi, eta] = plt.subplots(4, 1)
        figure.subplots_adjust(hspace=.5)
        figure.set_figheight(12)
        figure.set_figwidth(6)

        m.set_title('Distribution of m', fontsize=16)
        m.set_ylabel('Frequency')
        m.set_xlabel('Value (Normalised)')
        m.hist(original_dataset[0], label='Original', bins=100)
        m.hist(uncompressed_dataset[0], label='Uncompressed', bins=100)
        m.legend()

        pt.set_title('Distribution of pt', fontsize=16)
        pt.set_ylabel('Frequency')
        pt.set_xlabel('Value (Normalised)')
        pt.hist(original_dataset[1], label='Original', bins=100)
        pt.hist(uncompressed_dataset[1], label='Uncompressed', bins=100)
        pt.legend()

        phi.set_title('Distribution of phi', fontsize=16)
        phi.set_ylabel('Frequency')
        phi.set_xlabel('Value (Normalised)')
        phi.hist(original_dataset[2], label='Original', bins=100)
        phi.hist(uncompressed_dataset[2], label='Uncompressed', bins=100)
        phi.legend()

        eta.set_title('Distribution of eta', fontsize=16)
        eta.set_ylabel('Frequency')
        eta.set_xlabel('Value (Normalised)')
        eta.hist(original_dataset[3], label='Original', bins=100)
        eta.hist(uncompressed_dataset[3], label='Uncompressed', bins=100)
        eta.legend()

    def plot_difference_graphs(self, original_dataset, uncompressed_dataset):
        fractional_difference = \
            ((uncompressed_dataset - original_dataset) / original_dataset).cpu().detach().numpy().transpose()
        
        figure, [[m, m_dist], [pt, pt_dist], [phi, phi_dist], [eta, eta_dist]] = plt.subplots(4, 2)
        figure.subplots_adjust(hspace=.5, wspace=.5)
        figure.set_figheight(12)
        figure.set_figwidth(12)

        m.set_title('Residue of m', fontsize=16)
        m.set_ylabel('Residue')
        m.set_xlabel('Entry #')
        m.plot(fractional_difference[0])

        m_dist.set_title('Distribution of residue of m', fontsize=16)
        m_dist.set_ylabel('Frequency')
        m_dist.set_xlabel('Residue')
        m_dist.hist(np.clip(fractional_difference[0], -1, 1), bins=100)

        pt.set_title('Residue of pt', fontsize=16)
        pt.set_ylabel('Fractional Difference')
        pt.set_xlabel('Entry #')
        pt.plot(fractional_difference[1])

        pt_dist.set_title('Distribution of residue of pt', fontsize=16)
        pt_dist.set_ylabel('Frequency')
        pt_dist.set_xlabel('Residue')
        pt_dist.hist(np.clip(fractional_difference[1], -1, 1), bins=100)

        phi.set_title('Residue of phi', fontsize=16)
        phi.set_ylabel('Fractional Difference')
        phi.set_xlabel('Entry #')
        phi.plot(fractional_difference[2])

        phi_dist.set_title('Distribution of residue of phi', fontsize=16)
        phi_dist.set_ylabel('Frequency')
        phi_dist.set_xlabel('Residue')
        phi_dist.hist(np.clip(fractional_difference[2], -1, 1), bins=100)

        eta.set_title('Residue of eta', fontsize=16)
        eta.set_ylabel('Fractional Difference')
        eta.set_xlabel('Entry #')
        eta.plot(fractional_difference[3])

        eta_dist.set_title('Distribution of residue of eta', fontsize=16)
        eta_dist.set_ylabel('Frequency')
        eta_dist.set_xlabel('Residue')
        eta_dist.hist(np.clip(fractional_difference[3], -1, 1), bins=100)

