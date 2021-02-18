import os
import numpy as np
import cv2

from lib.visualization import plotting
from lib.visualization.video import play_trip
from matplotlib import pyplot as plt

from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'image_l'))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)


    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=float, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
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
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i'th and i-1'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i'th image
        q2 (ndarray): The good keypoints matches position in i-1'th image
        """
        # This function should detect and compute keypoints and descriptors from the i'th and i-1'th image using the class orb object
        # The descriptors should then be matched using the class flann object (knnMatch with k=2)
        # Remove the matches not satisfying Lowe's ratio test
        # Return a list of the good matches for each image, sorted such that the n'th descriptor in image i matches the n'th descriptor in image i-1
        # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
        img_i = self.images[i]
        img_i_1 = self.images[i-1]

        # find the keypoints with ORB
        kp_i = self.orb.detect(img_i, None)
        kp_i_1 = self.orb.detect(img_i_1, None)

        # compute the descriptors with ORB
        kp_i, des_i = self.orb.compute(img_i, kp_i)
        kp_i_1, des_i_1 = self.orb.compute(img_i_1, kp_i_1)

        matches = self.flann.knnMatch(des_i_1, des_i, k=2)

        q1 = []
        q2 = []

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                q1.append(kp_i[n.trainIdx].pt)
                q2.append(kp_i_1[m.trainIdx].pt)

        # draw only keypoints location,not size and orientation
        #img2 = cv2.drawKeypoints(img_i, kp_i, outImage=None, color=(0, 255, 0), flags=0)
        #plt.imshow(img2), plt.show()

        return np.asarray(q1), np.asarray(q2)

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i'th image
        q2 (ndarray): The good keypoints matches position in i-1'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Estimate the Essential matrix using built in OpenCV function
        # Use decomp_essential_mat to decompose the Essential matrix into R and t
        # Use the provided function to convert R and t to a transformation matrix T

        E, mask = cv2.findEssentialMat(q2, q1, cameraMatrix=self.K)
        right_pair = self.decomp_essential_mat(E, q1, q2)
        return VisualOdometry._form_transf(right_pair[0], right_pair[1])


    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i'th image
        q2 (ndarray): The good keypoints matches position in i-1'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.reshape(t,(3))

        right_pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
        Ts = []

        for rp in right_pairs:
            Ts.append(VisualOdometry._form_transf(rp[0],rp[1]))

        K = np.zeros((3,4))
        K[:3, :3] = self.K

        index_of_most_positive_zs = 0
        most_positive_zs = 0
        for i, T in enumerate(Ts):
            P = np.matmul(K, T)
            hom_Q1 = cv2.triangulatePoints(self.P, P, q2.T, q1.T)
            hom_Q2 = np.matmul(T, hom_Q1)

            Q1 = hom_Q1 / hom_Q1[3,:]
            Q2 = hom_Q2 / hom_Q2[3,:]

            Q1_pos_zs = list(filter(lambda z: z > 0, Q1.T[:, 2]))
            Q2_pos_zs = list(filter(lambda z: z > 0, Q2.T[:, 2]))

            if most_positive_zs < len(Q1_pos_zs) + len(Q2_pos_zs):
                most_positive_zs = len(Q1_pos_zs) + len(Q2_pos_zs)
                index_of_most_positive_zs = i

        self.P = np.matmul(K, Ts[index_of_most_positive_zs])
        return right_pairs[index_of_most_positive_zs]


def main():
    data_dir = '../data/KITTI_sequence_1'  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)

    #play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry")


if __name__ == "__main__":
    main()
