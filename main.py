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


class Discriminator(nn.Module):
    def __init__(self,im_chan=1,hidden_dim=16):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan,hidden_dim),
            self.make_disc_block(hidden_dim,hidden_dim*2),
            self.make_disc_block(hidden_dim*2,1,final_layer = True)
        )

    def make_disc_block(self,input_channels,output_channels,kernel_size=4, stride = 2, final_layer = False):
        '''

        This function defines one layer of the discrimator network corresponding to the
        DCGAN. It includes a convolutional layer, batch norm and activation function(except final layer)

        Inputs:
        input_channels: the number of channels of the input feature
        output_channels: the number of channels the output feature should have
        kernel_size: kernel size of the convolutional filter, shape is (kernel_size,kernel_size)
        stride: the stride of the convolution process
        final_layer: a boolean, True if it is the final layer of the network

        Outputs:
        returns a single discriminator block
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2,inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride)
            )

    def forward(self,image):
        x= self.disc(image)
        return x.view(len(x),-1)


criterion = nn.BCEWithLogitsLoss()
z_dim = 64
display_step = 500
batch_size = 128
# A learning rate of 0.0002 works well on DCGAN
lr = 0.0002

# These parameters control the optimizer's momentum, which you can read more about here:
# https://distill.pub/2017/momentum/ but you don’t need to worry about it for this course!
beta_1 = 0.5
beta_2 = 0.999
device = 'cpu'

# You can tranform the image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)


gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

# You initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)


n_epochs = 50
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        ## Update discriminator ##
        disc_opt.zero_grad()
        fake_noise = generate_noise(cur_batch_size, z_dim, device=device)
        fake = gen(fake_noise)
        disc_fake_pred = disc(fake.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()

        ## Update generator ##
        gen_opt.zero_grad()
        fake_noise_2 = generate_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        disc_fake_pred = disc(fake_2)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ## Visualization code ##
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
