import torch
import torch.nn as nn
import torchvision.models as models

class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """
    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)
    

class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """
    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)

class Encoder(nn.Module):
    """
    Encoder class, mainly consisting of three residual blocks.
    """
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1) # 16 32 32
        self.BN = nn.BatchNorm2d(16)
        self.rb1 = ResBlock(16, 16, 3, 2, 1, 'encode') # 16 16 16
        self.rb2 = ResBlock(16, 32, 3, 1, 1, 'encode') # 32 16 16
        self.rb3 = ResBlock(32, 32, 3, 2, 1, 'encode') # 32 8 8
        self.rb4 = ResBlock(32, 48, 3, 1, 1, 'encode') # 48 8 8
        self.rb5 = ResBlock(48, 48, 3, 2, 1, 'encode') # 48 4 4
        self.rb6 = ResBlock(48, 64, 3, 2, 1, 'encode') # 64 2 2
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        init_conv = self.relu(self.BN(self.init_conv(inputs)))
        rb1 = self.rb1(init_conv)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        # out = torch.flatten(rb6, 1)
        return rb6

class Decoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """
    
    def __init__(self):
        super(Decoder, self).__init__()
        self.rb1 = ResBlock(64, 48, 2, 2, 0, 'decode') # 48 4 4
        self.rb2 = ResBlock(48, 48, 2, 2, 0, 'decode') # 48 8 8
        self.rb3 = ResBlock(48, 32, 3, 1, 1, 'decode') # 32 8 8
        self.rb4 = ResBlock(32, 32, 2, 2, 0, 'decode') # 32 16 16
        self.rb5 = ResBlock(32, 16, 3, 1, 1, 'decode') # 16 16 16
        self.rb6 = ResBlock(16, 16, 2, 2, 0, 'decode') # 16 32 32
        self.out_conv = nn.ConvTranspose2d(16, 3, 3, 1, 1) # 3 32 32
        self.tanh = nn.Tanh()
        
    def forward(self, inputs):
        # inputs = inputs.reshape()
        rb1 = self.rb1(inputs)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        out_conv = self.out_conv(rb6)
        output = self.tanh(out_conv)
        return output
    
class Autoencoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.adaptive_pool = nn.AdaptiveAvgPool2d( (7,7) )
        self.fc = nn.Sequential(
                            nn.BatchNorm1d(3136 ),
                            nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features = 3136 , out_features=512, bias=False),
                            nn.ReLU(inplace=True),

                            nn.BatchNorm1d(512),
                            nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features = 512, out_features=64, bias=False),
                            nn.ReLU(inplace=True),
                            
                            nn.BatchNorm1d(64),
                            nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features=64, out_features=17, bias=True)
                            )
    
    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p
    
    def forward(self, inputs):
        encoded = self.encoder(inputs)
        fv = self.adaptive_pool(encoded)
        # print(fv.shape)
        fv = torch.flatten(fv, 1)
        # print(fv.shape)
        predictions = self.fc(fv)

        reconstructions = self.decoder(encoded)
        return predictions, reconstructions
    
# model = Autoencoder()
# out = model(torch.randn(2,3,256,256))
# model, out[0].shape, out[1].shape