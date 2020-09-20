# A1: Histogram Equalization

Histogram is a graphical representation of the intensity distribution of an image. It captures the occurrence frequency of pixel intensity values with which multiple image statistics can be calculated. Many image processing methods make use of image histograms for different purposes.

Histogram Equalization (HE) is an image processing technique that has been widely adopted to improve the contrast in images. It accomplishes this by spreading out the most frequent intensity values, i.e. stretching out the intensity range of the image. It usually increases the global contrast of images when the image pixels fall within a narrow range of intensity values. This allows for areas of lower local contrast to gain a higher contrast.

This assignment contains 2 main tasks and one optional task.

## Task 1: Implement HE algorithm

*Implement the HE algorithm in Matlab or Python or other computer programming languages, and apply your implemented HE algorithm to the 8 sample images.*

The Jupyter notebook contains 3 different implementation of the basic HE algorithm:

1. Greyscale HE
   + Entire image flattened to form an array to generate the histogram for equalization.
2. RGB HE
   + 3 histograms created from each separate R,G,B channels for HE application.
3. HSV/LAB HE:
   + Image converted from RGB to HSV color space. HE applied on V (value channel) only.
   + Image converted from RGB to LAB color space. HE applied on L (luminescence channel) only.

## Task 2: Results Discussion

*Discuss the pro and con of histogram equalization algorithm according to the enhanced sample images by your implemented HE algorithm. Discuss possible causes of some unsatisfactory contrast enhancement.*

Discussed in report using outputs of Jupyter Notebook.

## Bonus Task: Basic HE Improvement

*Discuss possible improvements of the basic HE algorithm. Implement and verify your ideas over the provided test images.* 

Discussed in report using outputs of Jupyter Notebook.

## Dependencies

+ From anaconda terminal:

```shell
conda env create --file 'env/ai6121cv_a1.yml'
```

