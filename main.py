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
