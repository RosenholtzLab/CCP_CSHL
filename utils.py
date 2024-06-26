import json
import os
import cv2
import numpy as np

def readin_imglist(imgnum, eccs = [0,5,10,15,20],filepath='./ccp_search18_train_subset/'):
    '''
    Readin the uniform images at each of the eccentricities according to our filestructure

    Parameters:
        imgnum (str): the MS COCO Image number desired
        eccs (list of floats): the eccentricities in degrees desired
        filepath (str): path to the COCO Periph Directory
    Returns:
        imglist (list of numpy arrays): List of uniformly foveated images
    '''
    imglist = []
    for e in eccs:
        if(e==0):
            impath = os.path.join(filepath,f'ecc_0',f'{imgnum.zfill(12)}.jpg') 
        else:
            impath = os.path.join(filepath,f'ecc_{e}',f'{imgnum.zfill(12)}.jpg') 
        #print(impath)
        im = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2RGB)
        imglist.append(im)
    return(imglist)

def stitch_image(uniform_imlist, fixation, deglist=[0,5,10,15,20], ppd=16, blend_radius = 5):
    '''
    Funtion to stitch together uniformly tiled coco-periph images to create a new
    transformed peripheral image representing progressive loss of information from
    the chosen fixation location

    Parameters:
        uniform_imlist (list of numpy arrays): list of uniform foveated images to be stitched, starting from 0 degrees to furthest
            *Note*: All images in imlist must be the same size, and should start with an original image, then uniform transforms.
        fixation (tuple): (x,y) fixation location specified in pixels (dist from left, dist from top)
        deglist (list of floats or ints): list of eccentricities represented by each image in imlist.
            *Note*: imlist and deglist must be the same length, and start with zero and increase.
        ppd (int): how many pixels subtend 1 degree viewing angle? (This is calculated by viewing distance, and COCO-Periph uses 16ppd)
            *Note*: PPD also specifies the fovea size, or the radius of the original image in the center of fixation.
            *Note2*: Leave this at 16 if you are using COCO-Periph Images.
        blend_radius (int): How many pixels on each side of the stitching border between images do we blend with? This avoids circular borders.    

    Returns:
        pseudo_im (numpy array): Pseudofoveated Image
    
    '''
    #assume a 1 degree radius fovea
    fovea_size = ppd
    #convert degree list to pixel radius from center
    pixel_list = [ppd * d for d in deglist]
    #print(pixel_list)

    #calcualte indices for each pixel
    fx_x, fx_y = fixation
    #print(np.shape(uniform_imlist[0]))
    img_shape_y, img_shape_x, _ = np.shape(uniform_imlist[0])
    y, x = np.indices([img_shape_y, img_shape_x])
    #print(y)
    #print(x)

    #fill whole image with furthest eccentricity (this could be filled with zeros or antyihng else as it all gets over written)
    pseudo_im = uniform_imlist[-1]
    #loop through eccentricities and fill in radial 'donuts' around the center
    for i, ecc in enumerate(deglist):
        #if this is fovea, should be a circle at the center of fixation.
        if(i==0):
            ecc_range = [0,fovea_size]
        #otherwise create rings around center
        else:
            #first image is a special case, start out from fovea, then go out to halfway between first and second boundary
            if(i==1):
                #offset puts the border halfway between limit of this eccentricity and next eccentricity up.
                ecc_offset_above = (pixel_list[2] - pixel_list[1])//2
                ecc_range = [fovea_size,pixel_list[i]+ecc_offset_above]
            #last image is also a special case because we have no border above.
            elif(i==len(deglist)-1):
                # We've seeded the pseudo_im with this image already, so don't need to do this, but do it anyway for completeness.
                ecc_offset_below = (pixel_list[i] - pixel_list[i-1])//2 #this is the offset_above from the last loop
                #our range in pixels for this eccentricity is centered at this pixel eccentricity, and halfway and below to the neighboring ecc. 
                #there is no maximum here, because we don't have a COCO-Periph renders above 20 degrees, so fill the rest of the image with this eccentricity.
                ecc_range = [pixel_list[i]-ecc_offset_below,1e100]
            #all others blend bidirectionally.
            else:
                #here we need the offset both above and below
                ecc_offset_above = (pixel_list[i+1] - pixel_list[i])//2
                ecc_offset_below = (pixel_list[i] - pixel_list[i-1])//2 #this is the offset_above from the last loop
                #our range in pixels for this eccentricity is centered at this pixel eccentricity, and halfway above and below to the neighboring ecc. 
                ecc_range = [pixel_list[i]-ecc_offset_below,pixel_list[i]+ecc_offset_above]
                #print(ecc_offset_below, ecc_offset_above)
        #print(ecc, ecc_range)
        
        #Now that we've specified the range, create a mask for a ring/doughnut at this eccentricity (plus the blending region)
        normalized = ((x-fx_x),(y-fx_y)) #distance from fixation
        r = np.hypot(normalized[0], normalized[1])
        ecc_mask = np.where((r>=ecc_range[0]) & (r<ecc_range[1]+blend_radius),1., 0.).astype(np.float32)
        #print(ecc_mask.shape)
        # plt.pcolormesh(ecc_mask)
        # plt.show()
        
        #include gaussian blending at the boundaries to remove sharp edges
        if(i>0):
            start_blend_r = ecc_range[0]-blend_radius
            end_blend_r = ecc_range[0]+blend_radius
            #mask must extend lower for linear blend
            ecc_mask = np.where((r>=start_blend_r) & (r<ecc_range[1]+blend_radius),1.,0.).astype(np.float32)
            ecc_mask = np.where((r>=start_blend_r) & (r<end_blend_r),(r-start_blend_r)/(end_blend_r-start_blend_r),ecc_mask)
        
        #create masks
        ecc_mask_3d = np.concatenate((ecc_mask[:,:,None],ecc_mask[:,:,None],ecc_mask[:,:,None]),2)
        invert_mask = 1.0-ecc_mask_3d
        
        # #optional plotting of masks
        # plt.imshow(ecc_mask_3d[:,:,0])
        # plt.colorbar()
        # plt.title(f'Ecc: {ecc} degrees')
        # plt.savefig(f'./mask_ecc_{ecc}.png')
        # plt.show()
        # # plt.imshow(invert_mask[:,:,0])
        # plt.colorbar()
        # plt.show()

        pseudo_im = np.uint8(ecc_mask_3d * uniform_imlist[i] + invert_mask * pseudo_im)

    return(pseudo_im)

# def get_new_ppd(img,ppd=16):
#     '''
#     If we've resized an image, we have changed the PPD
#     '''

def resize_img_COCO_Search(img,ppd=16):
    '''Fixation locations in COCO_Search are based on images resized to 1680x1050, with zero padding to maintain aspect ratio. So for the correct fixation, we have to resize the image.
    '''
    y,x,_ = img.shape
    xf = 1680.
    yf = 1050.
    aspect_ratio_img = x/y
    aspect_ratio_final = xf/yf
    #print(aspect_ratio_img,aspect_ratio_final)
    #if img too wide, need to pad bottom
    if(aspect_ratio_img > aspect_ratio_final):
        total_padding = ((x*yf)/xf)-y
        bottom_pad = top_pad = int(round(total_padding//2))
        right_pad = left_pad = 0
    #if img too narrow, need to pad right side
    elif(aspect_ratio_img < aspect_ratio_final):
        total_padding = int(round(((y*xf)/yf)-x))
        bottom_pad = top_pad = 0
        right_pad = left_pad = int(round(total_padding//2))
        
    # Create a new image with zero padding
    padded_image = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])

    #now resize since our aspect ratios are correct.
    resized_image = cv2.resize(padded_image, (int(xf),int(yf)), interpolation = cv2.INTER_CUBIC)

    #now we've changed the PPD, report the new PPD, as we'll need it for pseudofoveation
    size_increase = xf/(x+left_pad+right_pad) #could also be yf/(y+top_pad+bottom+pad) Theese are the same because we calcuated them that way.
    new_ppd = ppd * size_increase
    
    return(resized_image, new_ppd)

