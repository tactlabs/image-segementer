from PIL import Image
from torchvision import models
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2


img = Image.open(r"man.png")
 
rgb_img = img.convert('RGB')
new_image_resized = rgb_img.resize((224, 224))

new_image_resized.save('image_resized1.png')

img_resize = Image.open('image_resized1.png')


fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()


# Apply the transformations needed

trf = T.Compose([T.Resize(230),
                 T.CenterCrop(224),
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
inp = trf(img_resize).unsqueeze(0)

out = fcn(inp)['out']

om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print (om.shape)
print (np.unique(om))

# Define the helper function
def decode_segmap(image, nc=21):
    

  
    label_colors = np.array([(255,255,255),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (0,0,0),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
  
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
    rgb = np.stack([r, g, b], axis=2)

    return rgb


rgb = decode_segmap(om)


  #The pyrUp() function increases 
  # the size to double of its original size and pyrDown​() function decreases the size to half

img_small = cv2.pyrDown(rgb)

num_iter = 5
for _ in range(num_iter):
    img_small= cv2.bilateralFilter(img_small, d=9, sigmaColor=9, sigmaSpace=7)

img_rgb = cv2.pyrUp(img_small)

