import os
import numpy as np
import cv2
import math
from scipy.optimize import least_squares

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images_l = self._load_images(os.path.join(data_dir, 'image_l'))
        self.images_r = self._load_images(os.path.join(data_dir, 'image_r'))

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        self.disparities = [
            np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]
        self.fastFeatures = cv2.FastFeatureDetector_create()

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=float, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=float, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=float, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images. Shape (n, height, width)
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'10, 20th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def get_tiled_keypoints(self, img, tile_h, tile_w):
        """
        Splits the image into tiles and detects the 10 best keypoints in each tile

        Parameters
        ----------
        img (ndarray): The image to find keypoints in. Shape (height, width)
        tile_h (int): The tile height
        tile_w (int): The tile width

        Returns
        -------
        kp_list (ndarray): A 1-D list of all keypoints. Shape (n_keypoints)
        """
        # Split the image into tiles and detect the 10 best keypoints in each tile
        # Return a 1-D list of all keypoints
        # Hint: use sorted(keypoints, key=lambda x: -x.response)

        row_col_tiles = [(row, col, img[row:row+tile_h,col:col+tile_w]) for row in range(0,img.shape[0],tile_h) for col in range(0,img.shape[1],tile_w)]

        # Changes keypoints.pt to be wrt. global image rather than tile
        keypoints = []
        for row, col, tile in row_col_tiles:
            kps = self.fastFeatures.detect(tile, None)
            for i, kp in enumerate(kps):
                kps[i].pt = (kps[i].pt[0] + col, kps[i].pt[1] + row)
            sorted_kps = sorted(kps, key=lambda x: -x.response)
            keypoints = np.concatenate((keypoints,sorted_kps[0:10]),axis=None)

        return keypoints

    def track_keypoints(self, img1, img2, kp1, max_error=4):
        """
        Tracks the keypoints between frames

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error

        Returns
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """
        # Convert the keypoints using cv2.KeyPoint_convert
        # Use cv2.calcOpticalFlowPyrLK to estimate the keypoint locations in the second frame. self.lk_params contains parameters for this function.
        # Remove all points which are not trackable, has error over max_error, or where the points moved out of the frame of the second image
        # Return a list of the original converted keypoints (after removal), and their tracked counterparts

        # simon says
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)
        trackpoints2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None,  **self.lk_params)

        trackpoints1 = trackpoints1.squeeze()
        trackpoints2 = trackpoints2.squeeze()

        trackable_idx = []
        for i, (p1, p2) in enumerate(zip(trackpoints1, trackpoints2)):
            if err[i] > max_error:
                continue
            if status[i] == 0:
                continue
            if p2[0] > img2.shape[1] or p2[0] < 0 or p2[1] > img2.shape[0] or p2[1] < 0:
                continue

            trackable_idx.append(i)

        trackpoints1 = trackpoints1[trackable_idx, :]
        trackpoints2 = trackpoints2[trackable_idx, :]

        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """
        # Get the disparity for each keypoint
        # Remove all keypoints where disparity is out of bounds
        # calculate keypoint location in right image by subtracting disparity from x coordinates
        # return left and right keypoints for both frames

        q1_l = []
        q1_r = []
        q2_l = []
        q2_r = []
        #c_x = disp.shape[1]
        for pt1, pt2 in zip(q1, q2):
            dx1 = disp1[int(pt1[1]),int(pt1[0])]
            dx2 = disp2[int(pt2[1]),int(pt2[0])]
            if dx1 < max_disp and dx1 > min_disp and dx2 < max_disp and dx2 > min_disp:
                q1_l.append(pt1)
                q2_l.append(pt2)
                pt_r1 = [pt1[0] - dx1 , pt1[1]]
                pt_r2 = [pt2[0] - dx2 , pt2[1]]
                q1_r.append(pt_r1)
                q2_r.append(pt_r2)


        return np.asarray(q1_l), np.asarray(q1_r), np.asarray(q2_l), np.asarray(q2_r)


    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images

        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        # Triangulate points from both images with self.P_l and self.P_r
        hom_Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        hom_Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)

        Q1 = hom_Q1 / hom_Q1[3,:]
        Q2 = hom_Q2 / hom_Q2[3,:]

        return Q1[:-1, :].T, Q2[:-1, :].T


    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # Implement RANSAC to estimate the pose using least squares optimization
        # Sample 6 random point sets from q1, q2, Q1, Q2
        # Minimize the given residual using least squares to find the optimal transform between the sampled points
        # Calculate the reprojection error when using the optimal transform with the whole point set
        # Redo with different sampled points, saving the transform with the lowest error, until max_iter or early termination criteria met

        n = q1.shape[0]
        i = 0
        threshold = 2

        best_dof = []
        best_err = math.inf
        most_inliers = 0
        while i < max_iter:
            rnd_point_idx = np.random.choice(n, 6, replace=False)

            rq1 = q1[rnd_point_idx]
            rq2 = q2[rnd_point_idx]
            rQ1 = Q1[rnd_point_idx]
            rQ2 = Q2[rnd_point_idx]

            initial_guess = [0, 0, 0, 0, 0, 0] #dof -> transformation between points [x y z r1 r2 r3]
            res = least_squares(self.reprojection_residuals, x0=initial_guess, method='trf',x_scale="jac",ftol=1e-04, verbose=0, args=[rq1, rq2, rQ1, rQ2])

            reprojection_res = self.reprojection_residuals(res.x, q1, q2, Q1, Q2)
            inliers = np.sum(np.sqrt(np.sum(np.power(np.reshape(reprojection_res, (-1, 2), order='C'), 2), axis=1)) < threshold)
            if  inliers > most_inliers:
                best_dof = res.x
                most_inliers = inliers
            i += 1

        r = best_dof[:3]
        R, _ = cv2.Rodrigues(r)
        t = best_dof[3:]
        T = self._form_transf(R, t)

        return T

    def get_pose(self, i):
        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # Get the i-1'th image and i'th image
        img1_l, img2_l = self.images_l[i - 1:i + 1]

        # Get teh tiled keypoints
        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20) #10 , 20

        # Track the keypoints
        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        # Calculate the disparitie
        self.disparities.append(np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16))

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])

        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        return transformation_matrix


def main():
    data_dir = '../data/KITTI_sequence_1'  # Try KITTI_sequence_1 too
    vo = VisualOdometry(data_dir)

    #play_trip(vo.images_l, vo.images_r)  # Uncomment to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="poses")):
        if i < 1:
            cur_pose = gt_pose
        else:
            transf = vo.get_pose(i)
            cur_pose = np.matmul(cur_pose, transf)
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    plotting.visualize_paths(gt_path, estimated_path, "Stereo Visual Odometry")


if __name__ == "__main__":
    main()
