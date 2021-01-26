This is a program for multiple camera calibration. It involves a kinect v2 camera and three realsense camera. Only color sensors are used excluding infrared sensors. The intrinsic matrix of each colour sensors are assumed to be known.

The schedule is following:

1. Find corners.  (Note that the kinect frame is mirrored and opencv find corner api may lead to ambiguity.) 

2. Calculate the extrinsic matrix T of each frame using opencv pnp algorithm.  The matrix from camera 1 to camera 2 is equal to 
   $$
   T_{c2,c1}=T_{c1}\cdot{}T_{c2}^{-1}
   $$
   (Because chessboard is moving when collecting the frames, the extrinsic matrix is varying. But the  matrix from camera 1 to camera 2 is invariant.)

3. Transform chessboard 3d coordinate (in world) to camera 3d coordinate.
   $$
   P_c=T\cdot{}P_w
   $$

4. Transform camera 2 3d coordinate to camera 1.
   $$
   P_{c2,c1}=T_{c2,c1}\cdot{}P_{c2}
   $$

5. The optimization object is 
   $$
   Min(P_{c1}, P_{c2,c1})
   $$
   

