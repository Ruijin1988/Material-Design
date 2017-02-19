# Microstructure Representation and Reconstruction of Heterogeneous Materials via Deep Belief Network for Computational Material Design
## Ruijin Cang, Max Yi Ren
Full paper here: [https://arxiv.org/abs/1612.07401](https://arxiv.org/abs/1612.07401)
##Summary of the proposed CDBN
We 
- Set up a 5-layer network consisted by 3 layers of convolutional RBM and 2 fully connected RBM

![](images/configuration.png)

- The proposed CDBN could reduce the dimension of material microstructure images from 40,000 -> 30.
- The network can reconstruct random microstructures with properties statistically similar to the original samples.

# Sample of Original Images and Random Reconstructions
top row are four different original material microstructures samples and bottom row is their corresponded random reconstructions
![](images/imagecompare2.JPG)

## Filters extracted from each layer(take Ti-6Al-4V as an example)
1st layer:

![](images/1st_layer_filter.png)

2nd layer:

![](images/2nd_layer_filter.png)

3rd layer:

![](images/3rd_layer_filter.png)

4th layer:

![](images/4th_layer_filter.png)

5th layer:

![](images/5th_layer_filter.png)

note: It is worth noting that the visualizations from the 5th layer include images of almost all black, which is because the output images from these filters have almost uniform pixel values, appearing to be zeros after normalization. 
Also note that these filters are non-trivial, as the effect of activating multiple nodes at the output layer has nonlinear effect in the input space, i.e., the reconstructed image is not a simple addition of filter visualizations, because of the sigmoid operations throughout the network.

Such as the image shown below:

![](images/filter_activation.PNG)

## Critical Fracture Strength Comparison
We compared calcuated the fracture strength for the four different material micrsotrcuture based on three different process: original images,
original reconstructions and random reconstructions. We could see overall they're very similiar statistically(image on the left), but individually
(original compares to the original reconstruction) they're still very different, which is becuase during the training some details are lost
so that we could not reconstruct the original images with the extracted features as shown below:

![](images/compare.PNG)

# Impelementation Notice:

