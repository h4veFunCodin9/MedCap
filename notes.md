#### Why N4ITK to do bias field correction?
- At higher field strengths, sometimes structural images acquire an intensity gradient across the image making some parts of the image brighter than others. This intensity gradient can influence segmentation algorithms erroneously, therefore a method has been developed to remove this intensity gradient from the image, it is known as bias field correction.
- N4ITK is based on N3 method with B-spline approximation routine and a modified hierarchical optimization scheme.

#### Why to use the method proposed by Nyul to standardize the MRI images?
- MRI lacks a standard and quantifiable interpretation of image intensities. Unlike X-ray Computerized Tomography, MRI images taken for the same patient on the same scanner at different times may apper different from each other due to a variety of scanner-dependent variations and, therefore, the absolute intensity values do not have a fixed meaning. This transformation is to make the regions with similar tissue have similar intensity.
