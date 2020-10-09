# torchsketch.networks.cnn


## 1. Purposes
**torchsketch.networks.cnn** submodule provides the sketch-specific CNNs (e.g., Sketch-a-Net) and the general CNNs built in **torchvision.models**.


## 2. Examples 
```
import torchsketch.networks.cnn as cnns

sketchanet = sketchanet()
sketchanet = sketchanet(num_classes = 250)
sketchanet = sketchanet(num_classes = 345)

resnet18 = cnns.resnet18()
resnet18 = cnns.resnet18(pretrained = True)

alexnet = cnns.alexnet()
alexnet = cnns.alexnet(pretrained = True)

vgg16 = cnns.vgg16()

squeezenet = cnns.squeezenet1_0()

densenet = cnns.densenet161()

inception = cnns.inception_v3()

googlenet = cnns.googlenet()

shufflenet = cnns.shufflenet_v2_x1_0()

mobilenet = cnns.mobilenet_v2()

resnext50_32x4d = cnns.resnext50_32x4d()

wide_resnet50_2 = cnns.wide_resnet50_2()

mnasnet = cnns.mnasnet1_0()
```
