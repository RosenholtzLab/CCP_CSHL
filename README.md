# CCP_CSHL

## Coco Periph Demo for Cold Spring Harbor Course Summer 2024  

This is a Demo of using COCO-Periph [https://openreview.net/pdf?id=MiRPBbQNHv](https://openreview.net/pdf?id=MiRPBbQNHv) images to create pseudo-foveated images and a subset of COCO-Periph that has fixation data from the COCO-Search 18 dataset. [https://saliency.tuebingen.ai/datasets/COCO-Search18/](https://saliency.tuebingen.ai/datasets/COCO-Search18/)  

## Project Summary

We provide a subset of 100 images from the COCO-Periph dataset that are also from the  COCO-Search 18 training set, along with uniform eccentricity visualizations of these images and the code to create foveated visualizations for a (mostly) arbitrary fixation. Eye tracking data also exists for these images from the COCO-Search dataset from Greg Zelinsky. Observers executed these fixations while performing a search task. This combination of model predictions and eye tracking data should enable a number of interesting projects. For instance, consider the sequence of fixation locations, {(fx_i, fy_i), (fx_i+1, fy_i+1)}. Was the object at (fx_i+1, fy_i+1) likely identifiable when fixating (fx_i, fy_i), according to the peripheral vision model? In which case what might be the purpose of that saccade? 

## TL;DR
The COCO-Periph dataset provides images that simulate viewing MS-COCO images at various eccentricities, and you can stitch the same image at different eccentricities together to create new images for any fixation location. The COCO-Search dataset provides eye tracking data of where real human subjects look when performing search tasks on these same images. You can use the eye tracking data to create images that represent how much information was available to a human subject at each fixation.

## Dataset & Files to Get You Started

The main file you'll want is Demo.ipynb, which gives an example of reading in the .json file from COCO-Search, extracting fixation locations, and generating pseudofoveated images from COCO-Periph based on the fixation locations. This notebook calls functions in utils.py.  

We've selected a subset of 100 COCO-Periph images from the training split that also are included in the COCO-Search 18 training split. In other words, they have both coco_search18 fixation information and are also rendered at the 4 eccentricities of COCO-Periph. 

The COCO-Periph images are included in this repo in the ccp_searc18_train_subset folder. This folder contains ecc_0, which is the original MS-COCO images, as well as ecc_5, ecc_10, ecc_15, and ecc_20, which are COCO-Periph renders at each eccentricity in degrees. 

**Note** COCO Periph is rendered assuming 16 pixels per degree, so 5, 10, 15, and 20 degrees correspond to 80, 160, 240, and 320 pixels respectively in these images. However, COCO-Search18 uses fixations that are based on resizing images to 1680x1050 using zero padding to keep the aspect ratio. Therefore, before stitching uniform images, one must resize them in the same manner. resize_img_COCO_Search() in the utils.py file does this for you, and provides both the image and the new ppd. You must feed this new ppd to the stitch_image function so it can calculate the radial distances correctly for stitching.

The corresponding COCO Search 18 data is in the coco_search18 folder, in the coco_search_fixations_TP_train_spli1.json file. Check out the Demo.ipynb notebook for how to read in and use this file. There are some more examples in Prep_Dataset.ipynb of reading from the COCO-Search 18 JSON file. We cannot find README information for the COCO Search 18 JSON files, but they appear to be the same format at the COCO-Freeview dataset, with the README here: [https://drive.google.com/file/d/1Hj_jyK8Ml27Ge_5sogEtyI7XOjyad4aj/view](https://drive.google.com/file/d/1Hj_jyK8Ml27Ge_5sogEtyI7XOjyad4aj/view)

In addition, if you wish to work with a larger set of training split images, you can find the whole set of 990 images listed in the ccp_search_subset_train.csv file. You can download the rest of the COCO-Periph dataset at [https://data.csail.mit.edu/coco_periph/](https://data.csail.mit.edu/coco_periph/). Check out the Prep_Dataset.ipynb notebook to see how the set of 990 images was created (or if you want the validation set).
 
Finally, check out the COCO-freeview dataset, for human gaze data on the same images but in free-viewing conditions. [https://sites.google.com/view/cocosearch/coco-freeview](https://sites.google.com/view/cocosearch/coco-freeview) 

## Troubleshooting:
Q:_I see a black ring in my pseudofoveated image when calling stitch_image - did I introduce a bug?_  
A: For a small number of images in COCO Periph, the synthesis failed and produced a blank image for one eccentricity. This is a fix the authors are working on now. Just try another image!
