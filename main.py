import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


class Generator(nn.Module):
    def __init__(self,z_dim =10,im_chan=1,hidden_dim = 64):
        super(Generator,self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_layer(z_dim,hidden_dim*4),
            self.make_gen_layer(hidden_dim*4,hidden_dim*2,kernel_size = 4,stride = 1),
            self.make_gen_layer(hidden_dim*2,hidden_dim),
            self.make_gen_layer(hidden_dim,im_chan,kernel_size = 4,final_layer = True)
        )
    def make_gen_layer(self,input_channels,output_channels,kernel_size=3,stride=2,final_layer = False):
        '''

        This function defines one layer in the sequential generator block. It includes a
        transposed convolutional layer, batch normalization and ReLU activation.

        Inputs:
        input_channels: The number of channels of the input feature
        output_channels: The number of channels the output feature should have
        kernel_size: the size of the convolutional filters (kernel_size,kernel_size)
        stride: the covolutional stride
        final_layer: a boolean value, True for the final layer of the network,otherwise false

        Output:
        returns a single deconvolution layer for the network
        '''

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        elif final_layer:
            return  nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride),
                nn.Tanh()
            )

    def unsqueeze_noise(self,noise):
        '''

        This function given a noise tensor returns a copy the vector with width and height = 1
        and channels = z_dim

        Inputs:
        noise: a noise tensor with dimensions (n_samples,z_dim)

        Output:
        returns a reshaped copy of the noise tensor
        '''
        return noise.view(len(noise),self.z_dim,1,1)

    def forward (self,noise):
        '''

        This function completes a forward pass of the generator. Given a noise tensor,
        returns generated images

        Inputs:
        noise: a noise tensor with dimensions (n_samples,z_dim)

        Output:
        returns fake images generated from the generated from the generator from the noise
        '''
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

def generate_noise(n_samples, z_dim, device = 'cpu'):
    '''

    Function for creating random noise tensor with dimension (n_samples,z_dim)
    creates a tensor of that shape filled with random numbers from normal distribution

    Inputs:
    n_samples: a scaler, the number of images to be generated
    z_dim: a scaler, the length of each noise vector
    device: the device type to be used for computation

    Output:
    returns a noise tensor based on the input scalers
    '''

    return torch.randn(n_samples,z_dim,device = device)



