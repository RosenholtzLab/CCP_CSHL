# CCP_CSHL
Coco Periph Demo for Cold Spring Harbor Course Summer 2024

This is a Demo of using COCO-Periph images to create pseudo-foveated images, combined with data from [COCO-Search 18](https://saliency.tuebingen.ai/datasets/COCO-Search18/)

We provide a subset of the COCO-Periph dataset, images from COCO-Search 18, along with uniform eccentricity visualizations of these images and the code to create foveated visualizations for a (mostly) arbitrary fixation. Eye tracking data also exists for these images from the COCO-Search dataset from Greg Zelinsky. Observers executed these fixations while performing a search task. This combination of model predictions and eye tracking data should enable a number of interesting projects. For instance, consider the sequence of fixation locations, {(fx_i, fy_i), (fx_i+1, fy_i+1)}. Was the object at (fx_i+1, fy_i+1) likely identifiable when fixating (fx_i, fy_i), according to the peripheral vision model? In which case what might be the purpose of that saccade?
