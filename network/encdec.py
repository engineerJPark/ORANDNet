import torch
import torch.nn as nn

# Bilinear weights deconvolution Algorithm
def bilinear_kernel_init(Cin, Cout, kernel_size):
  factor = (kernel_size + 1) // 2
  if kernel_size % 2 == 1:
    center = factor - 1
  else:
    center = factor - 0.5

  og = (torch.arange(kernel_size).reshape(-1,1), torch.arange(kernel_size).reshape(1,-1))
  filter = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)

  weight = torch.zeros((Cin, Cout, kernel_size, kernel_size))
  weight[range(Cin), range(Cout), :, :] = filter
  return weight.to(torch.float)

class EncDec(nn.Module):
  def __init__(self, num_classes=1): # class 20 + 1(bakcground)
    super().__init__() # get 1 channel image(pseudo label), input image should be (b, 1, H, W)
    self.downsample1 = nn.Sequential(
      nn.Conv2d(1,64,3,padding=100),
      nn.ReLU(inplace=True),
      nn.Conv2d(64,64,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True)
    )
    self.downsample2 = nn.Sequential(
      nn.Conv2d(64,128,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(128,128,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True)
    )
    self.downsample3 = nn.Sequential(
      nn.Conv2d(128,256,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256,256,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256,256,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True)
    )
    self.downsample4 = nn.Sequential(
      nn.Conv2d(256,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True)
    )
    self.downsample5 = nn.Sequential(
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True)
    )

    # fc layers
    self.fc6 = nn.Sequential(
      nn.Conv2d(512, 4096, kernel_size=7), 
      nn.ReLU(),
      nn.Dropout2d()
    )
    nn.init.xavier_normal_(self.fc6[0].weight)
    self.fc7 = nn.Sequential(
      nn.Conv2d(4096, 4096, kernel_size=1), 
      nn.ReLU(),
      nn.Dropout2d()
    )
    nn.init.xavier_normal_(self.fc7[0].weight)

    # fc before upsample : to num_classes
    self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
    nn.init.xavier_normal_(self.score_pool3.weight)

    self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
    nn.init.xavier_normal_(self.score_pool4.weight)

    self.score_final = nn.Conv2d(4096, num_classes, kernel_size=1)
    nn.init.xavier_normal_(self.score_final.weight)
    
    # stride s, padding s/2, kernelsize 2s -> 2 times upsampling for images
    self.upsample_make_16s = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False) # to 1/16 padding=1,
    self.upsample_make_16s.weight.data.copy_(bilinear_kernel_init(num_classes, num_classes, 4))

    self.upsample_make_8s = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4,  stride=2, bias=False) # to 1/8 padding=1,
    self.upsample_make_8s.weight.data.copy_(bilinear_kernel_init(num_classes, num_classes, 4))
    
    self.upsample_to_score = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16,  stride=8, bias=False) # to 1 padding=4,
    self.upsample_to_score.weight.data.copy_(bilinear_kernel_init(num_classes, num_classes, 16))

  def crop_(self, crop_obj, base_obj, crop=True):
      if crop:
          c = (crop_obj.size()[2] - base_obj.size()[2]) // 2
          crop_obj_out = crop_obj[:,:,c:c+base_obj.shape[2],c:c+base_obj.shape[3]]
      return crop_obj_out

  def forward(self, x):
    input_x = x.clone()
    
    h = x
    h = self.downsample1(h) # 1/2
    h = self.downsample2(h) # 1/4
    h = self.downsample3(h) # 1/8
    pool3 = h
    h = self.downsample4(h) # 1/16
    pool4 = h
    h = self.downsample5(h) # 1/32

    h = self.fc6(h) # 1/32
    h = self.fc7(h) # 1/32

    h = self.score_final(h)
    h = self.upsample_make_16s(h)
    score_fc_16s = h # 1/16

    h = self.score_pool4(pool4) # 1/16 
    h = self.crop_(h, score_fc_16s)
    score_pool4c = h # 1/16

    h = score_fc_16s + score_pool4c # 1/16
    h = self.upsample_make_8s(h)
    score_fc_8s = h # 1/8

    h = self.score_pool3(pool3)
    h = self.crop_(h, score_fc_8s)
    score_pool3c = h # 1/8

    h = score_fc_8s + score_pool3c # 1/8
    h = self.upsample_to_score(h)
    h = self.crop_(h, input_x)

    out = h
    return out

  def copy_params_from_vgg16(self, vgg16):
    print("weight copy started...")

    features = [ 
    #   self.downsample1[0], 
      self.downsample1[1],
      self.downsample1[2], self.downsample1[3],
      self.downsample1[4], 
      self.downsample2[0], self.downsample2[1],
      self.downsample2[2], self.downsample2[3],
      self.downsample2[4], 
      self.downsample3[0], self.downsample3[1],
      self.downsample3[2], self.downsample3[3],
      self.downsample3[4], self.downsample3[5], 
      self.downsample3[6], 
      self.downsample4[0], self.downsample4[1],
      self.downsample4[2], self.downsample4[3],
      self.downsample4[4], self.downsample4[5], 
      self.downsample4[6], 
      self.downsample5[0], self.downsample5[1],
      self.downsample5[2], self.downsample5[3],
      self.downsample5[4], self.downsample5[5], 
      self.downsample5[6]
    ]
    for l1, l2 in zip(vgg16.features[1:], features): # get rid of first downsample
      if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
        assert l1.weight.size() == l2.weight.size()
        assert l1.bias.size() == l2.bias.size()
        l2.weight.data.copy_(l1.weight.data)
        l2.bias.data.copy_(l1.bias.data)
    for i, name in zip([0, 3], ['fc6', 'fc7']):
      l1 = vgg16.classifier[i]
      l2 = getattr(self, name)
      l2[0].weight.data.copy_(l1.weight.data.view(l2[0].weight.size()))
      l2[0].bias.data.copy_(l1.bias.data.view(l2[0].bias.size()))