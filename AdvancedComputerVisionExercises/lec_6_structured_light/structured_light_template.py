import os
import sys
import cv2
import numpy as np
import open3d as o3d
from scipy.io import loadmat
from tqdm import tqdm

from lib.visualization.image import show_images


class Camera:
    def __init__(self, t, r, k):
        """
        Creates a Camera instance

        Parameters
        ----------
        t (ndarray): Translation vector. Shape (3, 1)
        r (ndarray): Rotation matrix. Shape (3, 3)
        k (ndarray): Camera matrix. Shape (3, 3)
        """
        self.t = t
        self.r = r
        self.k = k
        self.c = np.dot(np.linalg.inv(r), t) * -1
        self.proj = np.matmul(k, np.hstack([r, t]))

    def add_rect_proj(self, H, x_dim, y_dim):
        """
        Add the rectified projection matrix

        Parameters
        ----------
        H (ndarray): Homography matrix. Shape (3, 3)
        x_dim (ndarray): The x dimension. Shape (2)
        y_dim (ndarray): The y dimension. Shape (2)
        """
        self.H = H
        self.rect_proj = np.dot(H, self.proj)
        self.out_dim = (int(x_dim[1]), int(y_dim[1]))

    def warp_img(self, img):
        """
        Applies the rectified transformation to the image.

        Parameters
        ----------
        img (nadrray): The image. Shape (height, width)

        Returns
        -------
        img_rect (nadrray): The rectified image. Shape (rect_height, rect_width)
        """
        img_rect = cv2.warpPerspective(img, self.H, self.out_dim)
        return img_rect


class StereoCameras:
    def __init__(self, t_r, r_r, k_r, t_l, r_l, k_l):
        """
        Creates a StereoCameras instance

        Parameters
        ----------
        t_r (ndarray): Translation vector right. Shape (3, 1)
        r_r (ndarray): Rotation matrix right. Shape (3, 3)
        k_r (ndarray): Camera matrix right. Shape (3, 3)
        t_l (ndarray): Translation vector left. Shape (3, 1)
        r_l (ndarray): Rotation matrix left. Shape (3, 3)
        k_l (ndarray): Camera matrix left. Shape (3, 3)
        """
        self.cam_l = Camera(t_l, r_l, k_l)
        self.cam_r = Camera(t_r, r_r, k_r)
        self.vec_a = np.squeeze(self.cam_r.c - self.cam_l.c)  # Vector between camera centers
        self.vec_b = (self.cam_l.r[2, :] + self.cam_r.r[2, :]).T / 2  # Mean optical axis
        self.vec_c = np.cross(self.vec_b, self.vec_a)
        self.vec_a /= np.linalg.norm(self.vec_a)
        self.vec_b /= np.linalg.norm(self.vec_b)
        self.vec_c /= np.linalg.norm(self.vec_c)

    @staticmethod
    def form_hbas(va, vb, vc):
        Hbas = np.hstack([np.expand_dims(va, axis=1), np.expand_dims(vb, axis=1), np.expand_dims(vc, axis=1)])
        Hbas = np.vstack([Hbas, [0, 0, 1]])
        return Hbas

    @staticmethod
    def form_homography(proj, hbas):
        H = np.linalg.inv(np.dot(proj, hbas))
        return H / H[2, 2]

    @staticmethod
    def make_k(x, y):
        k = np.eye(3)
        k[0, 2] = x
        k[1, 2] = y
        return k

    def calc_xy_data(self, h_l, h_r, shape):
        Q = np.ones((3, 4))
        Q[0, 1], Q[0, 3], Q[1, 2], Q[1, 3] = shape[1], shape[1], shape[0], shape[0]
        q = np.hstack([np.dot(h_l, Q), np.dot(h_r, Q)])
        q /= q[2]
        out_x_dim = np.round([np.min(q[0]), np.max(q[0])])
        out_y_dim = np.round([np.min(q[1]), np.max(q[1])])
        dx = out_x_dim[0] - 1
        dy = out_y_dim[0] - 1
        out_x_dim -= dx
        out_y_dim -= dy
        k = self.make_k(-dx, -dy)
        h_l = np.dot(k, h_l)
        h_r = np.dot(k, h_r)
        return h_l, h_r, out_x_dim, out_y_dim

    def triangulate_point(self, q1, q2):
        """
        Triangulates the given point

        Parameters
        ----------
        q1 (ndarray): . Shape(2)
        q2 (ndarray): . Shape(2)

        Returns
        -------
        Q (ndarray): . Shape(3)
        """
        b = np.zeros((4, 4))
        b[0, :] = self.cam_l.proj[0, :] - q1[0] * self.cam_l.proj[2, :]
        b[1, :] = self.cam_l.proj[1, :] - q1[1] * self.cam_l.proj[2, :]
        b[2, :] = self.cam_r.proj[0, :] - q2[0] * self.cam_r.proj[2, :]
        b[3, :] = self.cam_r.proj[1, :] - q2[1] * self.cam_r.proj[2, :]
        _, _, v = np.linalg.svd(b)
        Q = v.T[:3, 3] / v.T[3, 3]
        return Q

    def calc_rect_homographies(self, shape, upscale=1.2):
        """
        Calculate the rectified homographies
        """
        img_c = np.asarray(shape) / 2
        b = np.zeros((4, 4))
        V = self.triangulate_point(img_c, img_c)

        Hbas = self.form_hbas(self.vec_a, self.vec_c, V)
        H_l = self.form_homography(self.cam_l.proj, Hbas)
        H_r = self.form_homography(self.cam_r.proj, Hbas)
        scale = np.max([np.sqrt(np.linalg.det(H_l[:2, :2])), np.sqrt(np.linalg.det(H_r[:2, :2]))]) / upscale

        Hbas = self.form_hbas(scale * self.vec_a, scale * self.vec_c, V)
        k = self.make_k(-img_c[0] * upscale, -img_c[1] * upscale)
        Hbas[:3, :] = np.dot(Hbas[:3, :], k)
        H_l = self.form_homography(self.cam_l.proj, Hbas)
        H_r = self.form_homography(self.cam_r.proj, Hbas)

        H_l, H_r, out_x_dim, out_y_dim = self.calc_xy_data(H_l, H_r, shape)
        self.cam_l.add_rect_proj(H_l, out_x_dim, out_y_dim)
        self.cam_r.add_rect_proj(H_r, out_x_dim, out_y_dim)


def load_cam_params(param_file):
    """
    Loads the camera parameters

    Parameters
    ----------
    param_file (str): The file path

    Returns
    -------
    stereoCameras (StereoCameras): The stereo cameras
    """
    cam_params = loadmat(param_file)
    t_l, r_l, k_l = cam_params['T1'], cam_params['R1'], cam_params['A1']
    t_r, r_r, k_r = cam_params['T2'], cam_params['R2'], cam_params['A2']
    stereoCameras = StereoCameras(t_r, r_r, k_r, t_l, r_l, k_l)
    return stereoCameras


def load_image_data(data_file):
    """
    Loads the image data

    Parameters
    ----------
    data_file (str): The file path

    Returns
    -------
    data_list (list): List there contains the Black_Left, White_Left, Black_Right, White_Right
                      Pos_Left, Neg_Left, Pos_Right, Neg_Right images. Shape (8)
    """
    data = loadmat(data_file)
    data_list = [data['Black_Left'], data['White_Left'], data['Black_Right'], data['White_Right']]
    keys = ['Pos_Left', 'Neg_Left', 'Pos_Right', 'Neg_Right']
    data_list.extend([np.array(np.squeeze(data[key]).tolist()) for key in keys])
    return data_list


def get_occlusion_mask(black, white):
    """
    Creates the occlusion mask

    Parameters
    ----------
    black (ndarray): . Shape (height, width)
    white (ndarray): . Shape (height, width)

    Returns
    -------
    occ_mask (ndarray): . Shape (height, width)
    """
    diff = np.abs(white - black)
    mask = diff > 0.5 * np.max(white)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((4, 4)))
    occ_mask = np.zeros(mask.shape)
    occ_mask[4:-4, 4:-4] = mask[4:-4, 4:-4]
    return occ_mask


def get_encoding_image(pos, neg, mask):
    """
    Creates the encoding image

    Parameters
    ----------
    pos (ndarray): The positive image. Shape (n, height, width)
    neg (ndarray): The negative image. Shape (n, height, width)
    mask (ndarray): The occlusion mask. Shape (height, width)

    Returns
    -------
    enc_img (ndarray): The encoding image. Shape (height, width)
    """
    diff_img = np.zeros(pos.shape)
    n_img = pos.shape[0]
    for i in range(n_img):
        diff = pos[i, :, :] - neg[i, :, :]
        diff[mask == 0] = 0
        diff_img[i, :, :] = cv2.GaussianBlur(diff, (5, 5), 0)

    enc_img = np.zeros(pos.shape[-2:])
    for i in range(n_img):
        enc_img = enc_img + 2 ** i * (diff_img[i, :, :] > 0)
    return enc_img


def scan_encoding(enc_l, enc_r):
    """
    Scans the rectified encoding images

    Parameters
    ----------
    enc_l (ndarray): The left rectified encoding image. Shape (height, width)
    enc_r (ndarray): The right rectified encoding image. Shape (height, width)

    Returns
    -------
    matchs (list): A list with the matchs. Shape (n, 3)
    """

    def sort_rows(a):
        """
        Matlabs sortrows function works similarly to the provided function

        Parameters
        ----------
        a (ndarray): The matrix to sort

        Returns
        -------
        a (ndarray): The sorted matrix. Sorted an axis 2 (unstable), 1 (stable) and then 0 (stable)
        """
        a = a[a[:, 2].argsort()]  # First sort doesn't need to be stable.
        a = a[a[:, 1].argsort(kind='mergesort')]
        return a[a[:, 0].argsort(kind='mergesort')]

    # For each line find the matches as described by H. Aanæs in chapter 9 (page 185)
    # Matlabs sortrows function works similarly to the provided function

    def edgify(enc):
        """ MADE BY US
        Finds the edges and append them to a list.

        Parameters
        ----------
        enc (ndarray): the encoded image

        Returns
        -------
        edges (ndarray): a matrix with [label1 label2 col] of shape: (height, width-1)

        xxxxxx
        xxxxxx
        xxxxxx

        x|x|x|x|x|x
        x|x|x|x|x|x
        x|x|x|x|x|x

        |: label1 (x_left), label2 (x_right), col (col of x_left)
        """
        edges = np.ndarray((enc.shape[0], enc.shape[1]-1, 3))

        for row in range(enc.shape[0]):
            for col in range(enc.shape[1]-1):
                label_1 = enc[row, col]
                label_2 = enc[row, col+1]
                edges[row, col, :] = [label_1, label_2, col]
        return edges

    # equation 9.11 as a function to compare edges
    def compare(edge_l, edge_r, b=8):
        l = edge_l[0] * pow(2,b) + edge_l[1]
        r = edge_r[0] * pow(2,b) + edge_r[1]
        if (l == r):
            return 0
        elif (l > r):
            return 1
        elif (l < r):
            return 2

    left_edges = edgify(enc_l)
    right_edges = edgify(enc_r)

    # Sort edges Table 9.2.1
    print(left_edges.shape)
    left_edges_sorted = sort_rows(left_edges)
    right_edges_sorted = sort_rows(right_edges)

    exit()
    # Remove doubles
    print(left_edges_sorted)
    print(right_edges_sorted)

    # Table 9.2.4
    matches = []
    for row in left_edges_sorted.shape[0]:
        col_idx_l = 0 # idx left image
        col_idx_r = 0 # idx right_image
        while col_idx_r < len(right_edges_sorted[row]) and col_idx_l < len(left_edges_sorted[row]):
            c = compare(right_edges_sorted[row, col_idx_r], left_edges_sorted[row, col_idx_l])
            if c == 0:
                matches.append(row, left_edges_sorted[row, col_idx_l, 2], right_edges_sorted[row, col_idx_r, 2])
            elif c == 1:
                col_idx_r = col_idx_r + 1
            elif c == 2:
                col_idx_l = col_idx_l + 1

    return np.asarray(matches)


def triangulate_matches(cams, matches):
    """
    Triangulates the matches

    Parameters
    ----------
    cams (StereoCameras): The stereo cameras
    matches (ndarray): The matches. Shape (n,3)

    Returns
    -------
    Qs (ndarray): The triangulated 3D points. Shape (n, 3)
    """
    # For each match tringulate the point using the function from StereoCams:
    # cams.triangulate_point((matches[i, 0], matches[i, 1]), (matches[i, 0], matches[i, 2]))
    pass


def visualize_points(points):
    """
    Visualizes the 3D points

    Parameters
    ----------
    points (ndarray): The 3D points. Shape (n, 3)
    """
    # Use Open3D to visualize the point cloud

    pcd = o3d.geometry()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

    pass



def main():
    # Load the data
    data_dir = '../data/StructuredLight'
    cams = load_cam_params(os.path.join(data_dir, "CameraData.mat"))
    black_l, white_l, black_r, white_r, pos_l, neg_l, pos_r, neg_r = load_image_data(
        os.path.join(data_dir, "StrlImageData.mat"))
    # Show the images
    #show_images(neg_l)
    # Calculate the rectified homographies
    cams.calc_rect_homographies(black_l.shape)

    # Get the occlusion masks
    occ_l = get_occlusion_mask(black_l, white_l)
    occ_r = get_occlusion_mask(black_r, white_r)
    #show_images(occ_l * 255, image_title="Occlusion masks")

    # Get the encoding images
    enc_l = get_encoding_image(pos_l, neg_l, occ_l)
    enc_r = get_encoding_image(pos_r, neg_r, occ_r)
    #show_images(enc_l, image_title="Encoding image")

    # Apply the rectified transformation on the occlusion masks
    occ_l_rect = cams.cam_l.warp_img(occ_l)
    occ_r_rect = cams.cam_r.warp_img(occ_r)

    # Apply the rectified transformation on the encoding images
    enc_l_rect = cams.cam_l.warp_img(enc_l)
    enc_r_rect = cams.cam_r.warp_img(enc_r)
    #show_images(enc_l_rect, image_title="Rectified encoding image")
    #show_images(enc_r_rect, image_title="Rectified encoding image")
    cv2.destroyAllWindows()

    # Apply the occlusion masks to the encoding images
    enc_l_rect[occ_l_rect == 0] = 0
    enc_r_rect[occ_r_rect == 0] = 0

    # Find the matches
    matches = scan_encoding(enc_l_rect, enc_r_rect)
    # Triangulate the matches
    points = triangulate_matches(cams, matches)
    # Visualize the points
    visualize_points(points)


if __name__ == "__main__":
    main()
