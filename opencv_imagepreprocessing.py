import os
import cv2
from PIL import Image

def is_grayscale(image_path):
    image=Image.open(image_path)
    if image.mode =='L': #for rgb RGB instead of L
        return True
    return False


def cannyedge(image_path,save_path):

    image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    canny=cv2.Canny(image,100,200)
    cv2.imwrite(save_path,canny)


def apply_thresholding(image_path,save_path):

    image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    _,binary_threshold=cv2.threshold(image,130,255,cv2.THRESH_BINARY)
    cv2.imwrite(f'{save_path}/thresh.jpg',binary_threshold)

    thresh1=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,151,15)
    thresh2=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,151,15)

    cv2.imwrite(f'{save_path}/thresh1.jpg',thresh1)
    cv2.imwrite(f'{save_path}/thresh2.jpg',thresh2)

    _,otsu=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite(f'{save_path}/otsu.jpg',otsu)

def apply_blur(image_path,save_path,kernel_size=(5,5),sigma=0):

    image=cv2.imread(image_path)
    blurred=cv2.GaussianBlur(image,kernel_size,sigma)
    cv2.imwrite(f'{save_path}/blurred.jpg',blurred)


def adjust_brightness(image_path,save_path,alpha_fact=1.2):
    image=cv2.imread(image_path)
    brighter=cv2.convertScaleAbs(image,alpha=alpha_fact)
    cv2.imwrite(f'{save_path}/bright.jpg',brighter)
    
def convert_to_grayscale(image_path, save_path):

    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_path, grayscale_image)

    

image_path='Resized_Dataset/Training/glioma/Tr-gl_0020.jpg'
print(is_grayscale(image_path))

cannyedge(image_path,save_path='/home/teena/Documents/BrainTumour_CNN/canny.jpg')
apply_thresholding(image_path,save_path='/home/teena/Documents/BrainTumour_CNN')
apply_blur(image_path,save_path='/home/teena/Documents/BrainTumour_CNN',kernel_size=(5,5),sigma=0)
adjust_brightness(image_path,save_path='/home/teena/Documents/BrainTumour_CNN',alpha_fact=1.2)
convert_to_grayscale(image_path,save_path='/home/teena/Documents/BrainTumour_CNN/gray.jpg')