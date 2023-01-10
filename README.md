# TPK4960 - Efficient Perspective-n-Point With Graduated Non-Convexity for Outlier Robustness
This is the repository containing the source code written for the master's thesis of Runar Jåtun. The thesis presents a new method of robust pose estimation in the Perspective-n-Point problem, called EPnP+GNC. 

The source code for the EPnP+GNC, as well as files pertaining to the experiments, are in the "Computation and Testing"-folder. Figures that were used in the thesis are available in the "Figures"-folder, and the CAD-models from PASCAL3D+ are in the "CAD files"-folder. 


### Abstract
The Perspective-n-Point (PnP) problem is a fundamental problem in computer vision of determining the position and orientation, i.e. the pose, of a camera in a 3D environment. Furthermore, this is based on a set of known 3D points and their corresponding 2D projections in an image. Several solutions to this problem have been proposed, but the accuracy decreases when the data is noisy or full of outliers that differ significantly from the other measurements. Therefore, there is a need for more robust methods of pose estimation, that is, methods that can handle data sets containing noisy measurements and large amounts of outliers.

This research presents a new method for robust pose estimation, called EPnP+GNC. The new method is a combination of the efficient pose estimation method Effective Perspective-n-Point (EPnP) with the outlier optimization method Graduated Non-Convexity (GNC). In order to assess the new method, EPnP+GNC was implemented in a programming language and tested in a variety of scenarios. In addition to EPnP+GNC, several existing methods of pose estimation were tested such that a comparison could be made.

Overall, the results of this study shows that EPnP+GNC is an effective pose estimation method. The method correctly calculates the pose in data sets with a high percentage of outliers, and has a relatively low running time. The results are particularly good in situations with small rotations. It also shows promising results when compared to other methods. EPnP+GNC has shown potential for further development.
