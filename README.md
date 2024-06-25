# CCP_CSHL

## Coco Periph Demo for Cold Spring Harbor Course Summer 2024  

This is a Demo of using COCO-Periph [https://openreview.net/pdf?id=MiRPBbQNHv](https://openreview.net/pdf?id=MiRPBbQNHv) images to create pseudo-foveated images, and a subset of COCO-Periph that has fixation data from the COCO-Search 18 dataset. [https://saliency.tuebingen.ai/datasets/COCO-Search18/](https://saliency.tuebingen.ai/datasets/COCO-Search18/)  

## Project Summary

We provide a subset of 100 images from the COCO-Periph dataset that are also from the  COCO-Search 18 training set, along with uniform eccentricity visualizations of these images and the code to create foveated visualizations for a (mostly) arbitrary fixation. Eye tracking data also exists for these images from the COCO-Search dataset from Greg Zelinsky. Observers executed these fixations while performing a search task. This combination of model predictions and eye tracking data should enable a number of interesting projects. For instance, consider the sequence of fixation locations, {(fx_i, fy_i), (fx_i+1, fy_i+1)}. Was the object at (fx_i+1, fy_i+1) likely identifiable when fixating (fx_i, fy_i), according to the peripheral vision model? In which case what might be the purpose of that saccade?   

## Dataset & Files to Get you Started

The main file you'll want is StitchPseudofoveated.ipynb, which gives an example of how to take uniform images from coco-periph and 'stitch' them together to render at arbitrary fixations.  

We've selected a subset of 100 COCO-Periph images from the trianing split that also are included in the COCO-Search 18 training split. In other words they have both coco_search18 fixation information, and are also rendered at the 4 eccentricities of COCO-Periph. 

The COCO-Periph images are included in this repo in the ccp_searc18_train_subset forlder. This folder contains ecc_0, which is the original MS-COCO images, as well as ecc_5, ecc_10, ecc_15, and ecc_20, which are COCO-Periph renders at each eccentricity in degrees. Note COCO Periph is rendered assuming 16 pixels per degree, so 5, 10, 15, and 20 degrees correspond to 80, 160, 240, and 320 pixels respectively.  

The corresponding COCO Search 18 data is in the coco_search18 folder, in the coco_search_fixations_TP_train_spli1.json file. Check out the Prep_Dataset.ipynb notebook for an example to get you started reading from the COCO-Search 18 JSON file. 

In addition, if you wish to work with a larger set of ttraining split images, you can find the whole set of 990 images listed in the ccp_search_subset_train.csv file. You can download the rest of the COCO-Periph dataset at [https://data.csail.mit.edu/coco_periph/](https://data.csail.mit.edu/coco_periph/). Check out the Prep_Dataset.ipynb notebook to see how the set of 990 images was created (of if you want the validation set).

Finally, check out the COCO-freeview dataset, for human gaze data on the same images but in free-viewing conditions. [https://sites.google.com/view/cocosearch/coco-freeview](https://sites.google.com/view/cocosearch/coco-freeview) 


