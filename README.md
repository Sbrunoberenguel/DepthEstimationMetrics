# Depth Estimation Metrics
Code to compute depth estimation metrics for computer vision and deep learning.

In this repository we provide the code to compute different depth estiamtion metrics in an easy way. We provide in the [metrics.py](https://github.com/Sbrunoberenguel/DepthEstimationMetrics/blob/main/metrics.py) the computation of the metrics for each image and the average over the dataset used. In the [evaluation.py](https://github.com/Sbrunoberenguel/DepthEstimationMetrics/blob/main/evaluation.py) file we provide a schematic of evalaution process to use with your own dataset and network model.

For all the metricss, we deefine $pred$ as the network prediction (assumed in meters) and $gt$ as the ground truth maps (also assumed in meters). Notice that in [metrics.py](https://github.com/Sbrunoberenguel/DepthEstimationMetrics/blob/main/metrics.py) we use a $mask$ for the prediction and gt. This mask is used to take $only$ the pixels where $gt$ information is available. With this mask we avoid computing the metric on pixels where there is no ground truth information (set to 0 in the dataloader) obtaining false results in the prediction.

We can divide the proposed metrics into 2 main blocks: 

## Absolute metrics

Absolute metrics refer to those where we directly compare the difference between the predicted output of the network (assumed in meters) and the ground truth information (also assumed in meters). In this block we propose to use 4 different evaluation metrics:

### MAE: Mean Absolute Error
We compute the absolute difference between the network prediction and the ground truth for each image and average over the number of pixels in the image.

$$MAE =  \dfrac{1}{h\cdot w}\sum_{j}^{h}\sum_{i}^{w}|pred_{ij} - gt_{ij}| $$


### MSE: Mean Square Error
We compute the squared difference between the network prediction and the ground truth for each image and average over the number of pixels in the image.

$$MSE  =  \dfrac{1}{h\cdot w}\sum_{j}^{h}\sum_{i}^{w} (pred_{ij} - gt_{ij})^2 $$

### RMSE: Root Mean Square Error
We compute the root of [the square difference between the network prediction and the ground truth for each image, averaged over the number of pixels in the image]. We preset this evaluation metric with linear ($RMSE$) and logarithmic ($RMSE_{LOG}$) values of the prediction and ground truth.

$$RMSE  =  \sqrt{\dfrac{1}{h\cdot w}\sum_{j}^{h}\sum_{i}^{w} (pred_{ij} - gt_{ij})^2 }$$

$$RMSE_{LOG} = \sqrt{\dfrac{1}{h\cdot w}\sum_{j}^{h}\sum_{i}^{w} (\log pred_{ij} - \log gt_{ij})^2 }$$

## Relative metrics

Relative metrics refer to those where the error is averaged with respect the ground truth depth. 

### MARE: Mean Absolute Relative Error
We compute the absolute difference betweeen prediction and ground truth divided by the ground truth value and averaged over all the pixels in the image.

$$MARE = \dfrac{1}{h\cdot w}\sum_{j}^{h}\sum_{i}^{w}\dfrac{|pred_{ij} - gt_{ij}|}{gt_{ij}} $$

### $\delta^n$: Delta inlier ratio
We compute the ratio of pixels that are within a threshold and average it over all the image pixels.

$$\delta^n =  \dfrac{1}{h\cdot w} \left( \max \left( \dfrac{pred}{gt}, \dfrac{gt}{pred} \right) \right) < 1.25^n), n=1,2,3 $$


## Citation
If you find thin work relevant and want to evaluate your proposals following this same proceduce, please cite our work:

'''Work in Progress'''

