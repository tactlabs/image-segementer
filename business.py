from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2
import requests
from torchvision import models
import torchvision.transforms as T
import numpy as np

# img_url = 'https://ucscgenomics.soe.ucsc.edu/wp-content/uploads/Screen-Shot-2019-09-03-at-11.27.12-AM.png'
# bg_url = 'https://image.similarpng.com/very-thumbnail/2020/08/Abstract-blue-wave-on-transparent-background-PNG.png'


def add_background(img_url,bg_url):

    img = Image.open('static/image.png')
    # img = Image.open(requests.get(img_url, stream=True).raw)
    # plt.imshow(img); 
    # plt.show()

    rgb_img = img.convert('RGB')
    new_image_resized = rgb_img.resize((224, 224))

    new_image_resized.save('static/image_resized1.png')

    img_resize = Image.open('static/image_resized1.png')
    plt.imshow(img_resize)
    # plt.show()

    fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

    trf = T.Compose([T.Resize(230),
                    T.CenterCrop(224),
                    T.ToTensor(), 
                    T.Normalize(mean = [0.485, 0.456, 0.406], 
                                std = [0.229, 0.224, 0.225])])
    inp = trf(img_resize).unsqueeze(0)

    out = fcn(inp)['out']
    # print (out.shape)

    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    # print (om.shape)
    # print (np.unique(om))

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
    plt.imshow(rgb)
    # plt.show()

    #The pyrUp() function increases the size to double of its original
    #size and pyrDown​() function decreases the size to half

    img_small = cv2.pyrDown(rgb)

    #apply bilateralfilter

    num_iter = 5
    for _ in range(num_iter):
        img_small= cv2.bilateralFilter(img_small, d=9, sigmaColor=9, sigmaSpace=7)

    img_rgb = cv2.pyrUp(img_small)

    plt.imshow(img_rgb)
    # plt.show()

    mask = img_rgb
    plt.imshow(mask)
    # plt.show()

    masked_image = np.copy(img_resize)
    masked_image[mask !=0] = [0]
    plt.imshow(masked_image)
    # plt.show()

    blur_img = cv2.imread('static/image_resized1.png')
    blur = cv2.blur(blur_img,(5,5))
    # plt.subplot(121),plt.imshow(blur_img),plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    plt.imshow(blur_rgb)
    # plt.show()

    background1 = Image.fromarray(blur_rgb)
    background1.save("static/space_background.jpg")

    background_image = cv2.imread('static/space_background.jpg')
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    crop_background = background_image[0:514, 0:816]

    crop_background[mask == 0] = [0]
    plt.imshow(crop_background)

    complete_image = masked_image + crop_background
    plt.imshow(complete_image)
    # plt.show()

    # img = Image.open(requests.get('https://image.similarpng.com/very-thumbnail/2020/08/Abstract-blue-wave-on-transparent-background-PNG.png', stream=True).raw)

    new_op = Image.open('static/background.jpg')
    rgb_new = new_op.convert('RGB')
    new_image_bg = rgb_new.resize((224, 224))

    new_image_bg.save('static/image_resized_bg.png')

    background_image_new = cv2.imread('static/image_resized_bg.png')
    background_image_new = cv2.cvtColor(background_image_new, cv2.COLOR_BGR2RGB)
    crop_background_new = background_image_new[0:514, 0:816]

    crop_background_new[mask == 0] = [0]
    plt.imshow(crop_background_new)

    complete_image_newbg = masked_image + crop_background_new
    # complete_image_newbg.save('complete_image_newbg.png')
    complete_image_newbg = cv2.cvtColor(complete_image_newbg, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/complete_image_newbg.jpg',complete_image_newbg)
    # plt.imshow(complete_image_newbg)
    # plt.show()

if __name__ == "__main__":
    add_background(0,0)