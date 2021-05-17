# -*- coding: utf-8 -*-
"""
For system requirements, see https://github.com/NVlabs/stylegan

This code was built & tested in the environment faceenv in faceenv.yml, which consists of:
python 3.7.10
tensorflow-gpu 1.15
pillow 7.1.2
matplotlib 3.2.2
torchvision 0.9.1
cuda 10.0
cudnn 7.6

GPU used: NVidia GeForce GTX 1080

If the system requirements are not met, please consider using a version hosted on google colab: 
https://colab.research.google.com/drive/1Qdh623RDiqjhiYDVdhSjaGv4Uwe6GsAS?usp=sharing#scrollTo=auKnsGw9hl9O
"""

#%%
############################# 1. Prepare model (Run once) #############################
# Prepare directory
import os, shutil, sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# imports
import dnnlib, pickle
import dnnlib.tflib as tflib
import numpy as np
import random as rd
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
import matplotlib.pyplot as plt

print("*********************************")
print("*                               *")
print("*                               *")
print("* Welcome to the Face generator *")
print("* By....                        *")
print("* Thanks also to Peter Baylies  *")
print("* and NVIDIA corp               *")
print("*                               *")
print("*********************************")
print("")
print("")
print("Loading the model.  There will be a lot of warning messages due to package deprecation but you can ignore them")
print("")
print("")
print("")
print("")



if os.path.exists('data') and os.path.isdir('data'):
    shutil.rmtree('data')
os.mkdir('data')
shutil.copy("finetuned_resnet.h5", "data/finetuned_resnet.h5")

if os.path.exists('generated_images') and os.path.isdir('generated_images'):
    shutil.rmtree('generated_images')
if os.path.exists('latent_representations') and os.path.isdir('latent_representations'):
    shutil.rmtree('latent_representations')

# Load model
tflib.init_tf()
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1)

if os.path.exists('cache') and os.path.isdir('cache'):
    shutil.rmtree('cache')
os.mkdir('cache')
shutil.copy("karras2019stylegan-ffhq-1024x1024.pkl", "cache/karras2019stylegan-ffhq-1024x1024.pkl")

model_dir = 'cache/'
model_path = [model_dir+f for f in os.listdir(model_dir) if 'stylegan-ffhq' in f][0]


print("Loading StyleGAN model from %s..." %model_path)
    
with dnnlib.util.open_url(model_path) as f:
    generator_network, discriminator_network, averaged_generator_network = pickle.load(f)

print("*************************************")
print("*                                   *")
print("*                                   *")
print("StyleGAN loaded & ready for sampling!")
print("*                                   *")
print("*                                   *")
print("*************************************")

# functions to crop and mask images
## Crop images to squares in size = [stimulus_size]
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))
def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

## Crop square images into a circle with a Gaussian blur of [blur_radius] and background color of [background_color]
def mask_circle_transparent(pil_img, blur_radius, offset, bgCol):
    offset = blur_radius * 5 + offset
    mask = Image.new('L', pil_img.size, 255)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((offset, offset, pil_img.size[0] - offset, pil_img.size[1] - offset), fill=0)
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    imcopy = pil_img.copy()
    background = Image.new('RGBA', pil_img.size, bgCol)
    background.putalpha(mask)
    imcopy.paste(background, (0,0), background)
    return imcopy


# functions to create images using the StyleGAN generator
def generate_images(generator, positionMatrix, z = True):
    batch_size = 1
    if z: #Start from z: run the full generator network
        return generator.run(positionMatrix.reshape((batch_size, 512)), None, randomize_noise=False, **synthesis_kwargs)
    else: #Start from w: skip the mapping network
        return generator.components.synthesis.run(positionMatrix.reshape((batch_size, 18, 512)), randomize_noise=False, **synthesis_kwargs)

## generate images (both with and without circular crop) and save them to stylegan-faceorb/faces/
def generate_imgs(model, nimage, positionMatrix, stimsize):
    
    for i in range(nimage): #plots images in rows
        #generate one face based on a specific set of 512 dimensional coordinates
        img = generate_images(model, positionMatrix[:,i].transpose(), z = True)[0]
        #now make it the right size, and add an oval transparency mask
        thumb_width = stimsize 
        img= Image.fromarray(img) 
        im_square = crop_max_square(img).resize((thumb_width, thumb_width), Image.LANCZOS)
        im_thumb = mask_circle_transparent(im_square, blur_radius, 0, background_color)
        im_thumb = np.asarray(im_thumb)
        img = crop_max_square(img).resize((thumb_width, thumb_width), Image.LANCZOS)
        img = np.asarray(img)
        # save the original and the masked version in the faces directory
        matplotlib.image.imsave('faces/R_'+str(thisIteration)+'/face'+str(i)+'.jpg',im_thumb)
        matplotlib.image.imsave('faces/R_'+str(thisIteration)+'/squareface'+str(i)+'.jpg',img)
                     

## for circle generator: places point in cartesian plane where circle will center on
def coords():
    rad = np.arange(360)*(0.0174533) #radians
    x = np.cos(rad) #x-coordinates around a unit circle in a 2D plane
    y = np.sin(rad) #y-coordinates "

    return x,y,x**2+y**2


#%%
############################# 2. Set parameters #############################

# default values - basic
stimulus_size = 300
blur_radius = 10 
background_color = ()
bgclst = [128, 128, 128]
generation_mode = 'l'

# default values - circle 
dim_a = rd.randint(0,511)
dim_b = -1
while(dim_b==dim_a or dim_b==-1):
    dim_b = rd.randint(0,511)
circle_starting_radius = 0
circle_radius = 50
circle_stepsize = 20

# default values - line
dim = rd.randint(0,511)
line_starting_radius = 0
line_startpoint = -100
line_stoppoint = 100
line_nimage = 10
       
while True:
    commit = input("Start generating faces? [y/n]")
    if commit == 'n':
        print("Quitting face generator")
        break
            
    elif commit == 'y':
        dir_int = []
        thisIteration = 0

        for root, dirs, filenames in os.walk('faces'):
            if len(dirs) == 0: # no subfolders in sight
                thisIteration = 1
            else: 
                for dir in dirs:
                    if dir.startswith('R_'):
                        dir_int.append(int(dir[2:]))
                    if len(dir_int) == 0: # only contains irrelevant subfolders 
                        thisIteration = 1
                    else:
                        max_dir_int = max(dir_int)
                        thisIteration = max_dir_int + 1
            
            if thisIteration != 0:
                break
                    
        
        ## Basic parameters ##
        
        # stimulus_size
        stimulus_size_input = input("Enter stimulus size in pixel (default "+str(stimulus_size)+"): ")
        if len(stimulus_size_input) == 0:
            stimulus_size = stimulus_size
            print("Using default stimulus size %d" %stimulus_size)
        else:
            stimulus_size = int(stimulus_size_input)
        
        # blur_radius
        blur_radius_input = input("Enter Gaussian blur radius for circular mask in pixel (default "+str(blur_radius)+"): ")
        if len(blur_radius_input) == 0:
            blur_radius = blur_radius 
            print("Using default blur radius %d" %blur_radius)
        else:
            blur_radius = float(blur_radius_input)
        
        # background_color
        background_color = ()
        bgclst_input = list(background_color)
        bgclst_input.append(input("Enter RGB255 values for mask background - R (default "+str(bgclst[0])+"): "))
        bgclst_input.append(input("Enter RGB255 values for mask background - G (default "+str(bgclst[1])+"): "))
        bgclst_input.append(input("Enter RGB255 values for mask background - B (default "+str(bgclst[2])+"): "))
        for i in range(len(bgclst_input)):
            if len(bgclst_input[i]) == 0:
                bgclst[i] = bgclst[i]
            else:
                bgclst[i] = int(bgclst_input[i])
        background_color = tuple(bgclst)
        print("Background color set to %s" %(background_color,))
        
        # generation_mode
        generation_mode_input = input("Enter generation mode as line or circle [l/c] (default "+str(generation_mode)+"): ")
        if len(generation_mode_input) == 0:
            generation_mode = generation_mode 
            print("Using default generation mode %s" %generation_mode)
        else:
            generation_mode = generation_mode_input
        
        paramsNames = ['run', 'Stimulus size', 'Blur radius', 'Background color', 'Generation mode']        
        params = [thisIteration, stimulus_size, blur_radius, background_color, generation_mode]
        
        ## Parameters for circle generator ##
        if generation_mode == 'c':
            
            # dim_a
            dim_a_input = input("Enter first dimension (integer from 0 to 511, default "+str(dim_a)+"): ")
            if len(dim_a_input) == 0:
                dim_a = dim_a
                print("Using default first dimension #%d" %dim_a)
            else:
                dim_a = int(dim_a_input)
                
            # dim_b            
            dim_b_input = input("Enter second dimension (another integer from 0 to 511, default "+str(dim_b)+"): ")
            if len(dim_b_input) == 0:
                if int(dim_b) == dim_a:
                    if dim_a != 511:
                        dim_b = int(dim_b) + 1
                    else:
                        dim_b = int(dim_b) - 1
                else:
                    dim_b = dim_b
                print("Using default second dimension #%d" %dim_b)
            else:
                if int(dim_b_input) == dim_a:
                    if dim_a != 511:
                        dim_b = int(dim_b_input) + 1
                    else:
                        dim_b = int(dim_b_input) - 1
                    print("Second dimension should be different from first dimension. Second dimension defaulted to #%d" %dim_b)
                else:
                    dim_b = int(dim_b_input)
            
            # circle_starting_radius
            circle_starting_radius_input = input("Enter distance of circle from face 0 (default "+str(circle_starting_radius)+"): ")
            if len(circle_starting_radius_input) == 0:
                circle_starting_radius = circle_starting_radius
                print("Using default circle starting radius %d" %circle_starting_radius)
            else:
                circle_starting_radius = float(circle_starting_radius_input)
            
            # circle_radius
            circle_radius_input = input("Enter circle radius (default "+str(circle_radius)+"): ")
            if len(circle_radius_input) == 0:
                circle_radius = circle_radius
                print("Using default circle radius %d" %circle_radius)
            else:
                circle_radius = float(circle_radius_input)
            circle_radius_scaled = circle_radius * 1/25000
            
            # circle_stepsize
            circle_stepsize_input = input("Enter distance between images on circumference (must be a dividend of 360, default "+str(circle_stepsize)+"): ")
            if len(circle_stepsize_input) == 0:
                circle_stepsize = circle_stepsize
                print("Using default circle step size %d" %circle_stepsize)
            else:
                circle_stepsize = int(circle_stepsize_input)
                
            paramsNames.extend(['First dimension', 'Second dimension', 'Circle starting radius', 'Circle radius', 'Circle step size'])
            params.extend([dim_a, dim_b, circle_starting_radius, circle_radius, circle_stepsize])

        ## parameters for line generator ##
        if generation_mode == 'l':
            
            # dim
            dim_input = input("Enter dimension (integer from 0 to 511, default "+str(dim)+"): ")
            if len(dim_input) == 0:
                dim = dim
                print("Using default dimension #%d" %dim)
            else:
                dim = int(dim_input)
            
            # line_starting_radius
            line_starting_radius_input = input("Enter distance of line center from face 0 (default "+str(line_starting_radius)+"): ")
            if len(line_starting_radius_input) == 0:
                line_starting_radius = line_starting_radius
                print("Using default line starting radius %d" %line_starting_radius)
            else:
                line_starting_radius = float(line_starting_radius_input)
            
            # line_startpoint
            line_startpoint_input = input("Enter a starting point for line generation (default "+str(line_startpoint)+"): ")
            if len(line_startpoint_input) == 0:
                line_startpoint = line_startpoint
                print("Using default line start point %d" %line_startpoint)
            else:
                line_startpoint = float(line_startpoint_input)
            line_startpoint_scaled = line_startpoint * 1/50000
                
            # line_stoppoint
            line_stoppoint_input = input("Enter an ending point for line generation (default "+str(line_stoppoint)+"): ")
            if len(line_stoppoint_input) == 0:
                line_stoppoint = line_stoppoint
                print("Using default line stop point %d" %line_stoppoint)
            else:
                line_stoppoint = float(line_stoppoint_input)
            line_stoppoint_scaled = line_stoppoint * 1/50000
            
            # line_nimage
            line_nimage_input = input("Enter number of equidistant images to generate on the line (default "+str(line_nimage)+"): ")
            if len(line_nimage_input) == 0:
                line_nimage = line_nimage
                print("Using default number of images in line %d" %line_nimage)
            else:
                line_nimage = int(line_nimage_input)
                
            paramsNames.extend(['Dimension', 'Line starting radius', 'Line start point', 'Line stop point', 'Number of images in line'])
            params.extend([dim, line_starting_radius, line_startpoint, line_stoppoint, line_nimage])

        
        ############################# 3. Generate images! #############################
              
        # Prepare face directory
        print("Began face generation: run #" + str(thisIteration))
        os.mkdir('faces/R_'+str(thisIteration))
        
        # Log parameters
        f = open('faces/R_'+str(thisIteration)+'/_parametersLog_R_'+str(thisIteration)+'.txt', 'w+')
        for i in range(len(paramsNames)):
            f.write(paramsNames[i]+': '+str(params[i])+'\n')
        f.close()
            
        if generation_mode == 'c':
            
            #Circle Generator
            ndim = 512   # number of dimensions
            ndegrees = 360  # total number of steps around a circle

            if ndegrees%circle_stepsize != 0:
                    print('cirstep must be a dividend of 360')
            else:

                ## randomly select 512 points in space? 
                center = np.random.uniform(0,.5,ndim)   
                ## normalize selected points to face 0
                squares = [center[i]**2 for i in range(0,len(center))]
                t = np.sum(squares)**.5    
                center = (center/t)*circle_starting_radius 

                ## x,y are the horizontal and vertical vertical coordinates of a generic circle
                x,y,radius = coords()
                
                ## store indices of faces in matrix
                cart = np.zeros((ndim,int(ndegrees/circle_stepsize)))
                for d in range(int(ndegrees/circle_stepsize)):
                    cart[dim_a,d]=x[d*circle_stepsize]*circle_radius_scaled
                    cart[dim_b,d]=y[d*circle_stepsize]*circle_radius_scaled
                for i in range(ndim):
                    for d in range(int(ndegrees/circle_stepsize)):
                        cart[i,d] = cart[i,d] + center[i]
                
                ## call generation function
                generate_imgs(averaged_generator_network, int(ndegrees/circle_stepsize), cart, stimulus_size)
                
        if generation_mode == 'l':

            # Line Generator
            ndim = 512  # number of dimensions
            
            ## randomly select 512 points in space? 
            center = np.random.uniform(0,.5,ndim)   
            ## normalize selected points to face 0
            squares = [center[i]**2 for i in range(0,len(center))]
            t = np.sum(squares)**.5    
            center = (center/t)*line_starting_radius   
            
            ## store indices of faces in matrix
            sequence = np.linspace(line_startpoint_scaled,line_stoppoint_scaled,line_nimage)   
            cart = np.zeros((ndim,sequence.size))
            cart[dim,:] = cart[dim,:] + sequence 
            for i in range(ndim):
              for d in range(sequence.size):
                cart[i,d] = cart[i,d] + center[i]
            
            ## call generation function
            generate_imgs(averaged_generator_network, sequence.size, cart, stimulus_size)
            
        print("Completed face generation: run #" + str(thisIteration))
