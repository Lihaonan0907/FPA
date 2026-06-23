import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
 
from torch.autograd import Variable
import torchvision.transforms as transforms

class VAE(nn.Module):
    """Convolutional VAE used as the texture generation backbone."""
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256 * 13 * 13, latent_dim)
        self.fc_logvar = nn.Linear(256 * 13 * 13, latent_dim)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)  
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def decode(self, z): 
        z = self.decoder(z)
        return z

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar
    


# class DiffusionModel(nn.Module):
#     def __init__(self):
#         super(DiffusionModel, self).__init__()
#         self.diffusion_steps = 10
#         self.sigma = 0.1
        
#         self.loss_fn = nn.MSELoss()

#     def forward(self, x):
#         target = x.clone()
#         parameters = nn.ParameterList([x])
#         self.optimizer_diffusion = optim.Adam(parameters, lr=0.001)
#         for t in range(self.diffusion_steps):
#             # Apply the reverse diffusion transformation
#             noise = torch.randn_like(x) * self.sigma
#             x = (x + noise) / np.sqrt(1.0 + self.sigma**2)
#             # Update the diffusion step specific to your desired transformation
#             # For example, you can apply denoising operations or learnable transformations

#             # Optimize the diffusion model
#             self.optimizer_diffusion.zero_grad()
#             loss = self.loss_fn(x, target)  # Define the appropriate target for the diffusion optimization
#             loss.backward(retain_graph=True)
#             self.optimizer_diffusion.step()

#         return x

def vae_loss(recon_x, x, mean, logvar):
    """Standard VAE reconstruction + KL divergence objective."""
    # print(recon_x[0], x[0])
    reconstruction_loss = nn.BCELoss(reduction='sum')(recon_x, x)
    kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence_loss

# Define a custom dataset for fractal images
class FractalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.root_dir, image_name)

        # Load the image
        image = Image.open(image_path).convert('RGB')  # Convert to grayscale
        image = self.transform(image)
        image = image.float()

        return image
# Generate new fractal images
def generate_image(reconstructed,x):
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        generated = vae.decode(z)
        generated_image = generated.view(3, 512, 512)

        # Set the number of diffusion steps
        # diffusion_steps = 10 
        #  # Apply the diffusion process using the diffusion model
        # diffused_images = generated_image   # Clone the reconstructed images
        # for t in range(diffusion_steps):
        #     # Apply the reverse diffusion transformation
        #     noise = torch.randn_like(diffused_images) * diffusion_model.sigma
        #     diffused_images = (diffused_images + noise) / np.sqrt(1.0 + diffusion_model.sigma ** 2)
        #             # Update the diffusion step specific to your desired transformation
                    # For example, you can apply denoising operations or learnable transformations
        
    # Save the generated image
    generated_image = generated_image.permute(1,2,0).cpu().numpy()
    generated_image = (generated_image * 255).astype(np.uint8)
    generated_image = Image.fromarray(generated_image)
    generated_image.save('generated_fractal.png')
    
    reconstructed = reconstructed[0]
    reconstructed = reconstructed.permute(1,2,0).detach().cpu().numpy()
    reconstructed = (reconstructed * 255).astype(np.uint8)
    reconstructed = Image.fromarray(reconstructed)
    reconstructed.save('reconstructed.png')

    origin = x[0]
    origin = origin.permute(1,2,0).detach().cpu().numpy()
    origin = (origin * 255).astype(np.uint8)
    origin = Image.fromarray(origin)
    origin.save('origin.png')
 
# if __name__ == '__main__':
#     # Set the device (CPU or GPU)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Define hyperparameters
#     latent_dim = 16
#     lr = 0.0005
#     epochs = 10000

#     # Initialize the VAE
#     vae = VAE(latent_dim).to(device)
#     model_pt = 'vae_model.pt'
#     if os.path.exists(model_pt):
#         print("loading existing parameter")
#         checkpoint = torch.load(model_pt)
#         vae.load_state_dict(checkpoint['model_state_dict'])
#     # Set the root directory of the fractal image dataset
#     root_dir = './images/'
#     # diffusion_model = DiffusionModel()
#     # Create an instance of the FractalDataset
#     dataset = FractalDataset(root_dir)

#     # Define batch size and number of workers for data loading
#     batch_size = 8
#     num_workers = 4

#     # Create a data loader for the dataset
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#     # Define the optimizer
#     optimizer = optim.Adam(vae.parameters(), lr=lr)
#     losses = []
#     loss_min = 1e9
#     # Training loop
#     for epoch in range(epochs):
#         losses = []
#         for batch_idx, data in enumerate(dataloader):
#             x = data.to(device)

#             optimizer.zero_grad()

#             reconstructed, mean, logvar = vae(x)
#             loss = vae_loss(reconstructed, x, mean, logvar)

#             loss.backward()
#             optimizer.step()
            
#             losses.append(loss.item())
#             # reinfine_image = diffusion_model(reconstructed)
#         if np.mean(losses)< loss_min:
#             print("saving the parameter")
#             loss_min = np.mean(losses)
#             generate_image(reconstructed, x)
#             torch.save({'model_state_dict': vae.state_dict()}, model_pt)

                    

#             # if batch_idx % 10 == 0:
#         print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {np.mean(losses):.4f}')



class ConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel,batch_normalization=True):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel,output_channel,3,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.batch_normalization = batch_normalization

    def forward(self,x): 
        x = self.conv1(x) 
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.batch_normalization:
            x = self.bn2(x)

        x=self.relu(x) 


        return x


class DownSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(DownSample, self).__init__()
        self.down_sample = torch.nn.MaxPool2d(factor, factor)

    def forward(self,x):
        return self.down_sample(x)


class UpSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(UpSample, self).__init__()
        self.up_sample = torch.nn.Upsample(scale_factor = factor, mode='bilinear')

    def forward(self,x):
        return self.up_sample(x)


class CropConcat(torch.nn.Module):
    def __init__(self,crop = True):
        super(CropConcat, self).__init__()
        self.crop = crop

    def do_crop(self,x, tw, th):
        b,c,w, h = x.size()
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return x[:,:,x1:x1 + tw, y1:y1 + th]

    def forward(self,x,y):
        b, c, h, w = y.size()
        if self.crop:
            x = self.do_crop(x,h,w)
        return torch.cat((x,y),dim=1)


class UpBlock(torch.nn.Module):
    def __init__(self,input_channel, output_channel,batch_normalization=True,downsample = False):
        super(UpBlock, self).__init__()
        self.downsample = downsample
        self.conv = ConvBlock(input_channel,output_channel,batch_normalization=batch_normalization)
        self.downsampling = DownSample()

    def forward(self,x):
        x1 = self.conv(x)
        if self.downsample:
            x = self.downsampling(x1)
        else:
            x = x1
        return x,x1

class DownBlock(torch.nn.Module):
    def __init__(self,input_channel, output_channel,batch_normalization=True,Upsample = False):
        super(DownBlock, self).__init__()
        self.Upsample = Upsample
        self.conv = ConvBlock(input_channel,output_channel,batch_normalization=batch_normalization)
        self.upsampling = UpSample()
        self.crop = CropConcat()

    def forward(self,x,y):
        if self.Upsample:
            x = self.upsampling(x)
        x = self.crop(y,x)
        x = self.conv(x)
        return x


class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        #Down Blocks
        self.conv_block1 = ConvBlock(3,64)
        self.conv_block2 = ConvBlock(64,128)
        self.conv_block3 = ConvBlock(128,256)
        self.conv_block4 = ConvBlock(256,512)
        self.conv_block5 = ConvBlock(512,1024)
            
        
        #Up Blocks
        self.conv_block6 = ConvBlock(1024+512, 512)
        self.conv_block7 = ConvBlock(512+256, 256)
        self.conv_block8 = ConvBlock(256+128, 128)
        self.conv_block9 = ConvBlock(128+64, 64)

        #Last convolution
        self.last_conv = torch.nn.Conv2d(64,3,1)

        self.crop = CropConcat()

        self.downsample = DownSample()
        self.upsample =   UpSample()

    def forward(self,x):
        if x.size()[1]!=3:
            x = x.permute(0,3,1,2)
        x1 = self.conv_block1(x) 
        x = self.downsample(x1) 
        x2 = self.conv_block2(x)
        x= self.downsample(x2) 
        x3 = self.conv_block3(x)
        x= self.downsample(x3) 
        x4 = self.conv_block4(x)
        x = self.downsample(x4) 
        x5 = self.conv_block5(x)

        x = self.upsample(x5) 
        x = self.crop(x4, x)
        x = self.conv_block6(x) 

        x = self.upsample(x)
        x = self.crop(x3,x)
        x = self.conv_block7(x) 

        x= self.upsample(x)
        x= self.crop(x2,x)
        x = self.conv_block8(x) 

        x = self.upsample(x)
        x = self.crop(x1,x)
        x = self.conv_block9(x) 


        x = self.last_conv(x) 

        #如果你的任务是二元分类问题，sigmoid 是合适的选择；如果是多类别分类问题，可能会使用 softmax。 
        return x

# Define a simple image generator (U-Net style)
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 2, stride=2)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        print(encoded.size())
        middle = self.middle(encoded)
        print(middle.size())
        decoded = self.decoder(middle)
        print(decoded.size())
        return decoded

class DiffusionModel(nn.Module):
    """Lightweight diffusion-style perturbation module; `t` controls the step parameter passed in from the trainer."""
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.generator = Unet()
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, x, t):
        x_t = self.generator(x)  # Generate an image at time t
        noise = torch.randn_like(x_t)  # Sample noise
        x_t = x_t + self.alpha * noise * np.sqrt(t)  # Diffusion step
        x_t = torch.sigmoid(x_t)
        return x_t

# Define the loss function (Negative Log Likelihood)
#两个tensor[10,3,640,640]
def loss_fn(x_t, x_0):
    loss =0
    x = x_t.shape[0]
    for i in range(x):
        loss = torch.mean((x_t[i] - x_0[i]) ** 2)
        loss+=loss
    loss = loss*100*1.0/x
    return loss