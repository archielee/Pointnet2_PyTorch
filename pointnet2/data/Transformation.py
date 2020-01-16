from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import numpy as np
import os

from scipy.spatial.transform import Rotation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Transformation:
    """
    A class used to encapsulate a transformation matrix. Can apply the transformation to points and calculate the Jacobian
    """

    def __init__(self, mat):
        """Initialize transformation object from frame 1 to frame 2
        TODO generalize this to different possible pose formats

        Parameters
        ----------
        mat : 6x1 numpy array or 4x4 numpy array
            Pose information in form [rot_x, rot_y, rot_z, x, y, z] or as
            transformation matrix
        """
        if len(mat) == 6:
            self.fromEulerXyz(mat)
        else:
            self.fromTransformationMatrix(mat)

    def fromEulerXyz(self, pose):
        """Initialize transformation from Euler angles and xyz position

        Parameters
        ----------
        pose : 6x1 numpy array
            Pose information in form [rot_x, rot_y, rot_z, x, y, z]
        """
        self.C = Rotation.from_euler(
            'xyz', pose[0:3]).as_dcm()  # rotation matrix from frame 1 to frame 2
        self.t = pose[3:]      # translation vector from frame 1 to frame 2

        self.tf_mat = np.zeros((4, 4))
        self.tf_mat[:3, :3] = self.C
        self.tf_mat[:3, 3] = self.t
        self.tf_mat[3, 3] = 1

    def fromTransformationMatrix(self, tf_mat):
        self.tf_mat = tf_mat
        self.C = self.tf_mat[:3, :3]
        self.t = self.tf_mat[3, :3]

    def invert(self):
        inverse_tf_mat = np.linalg.inv(self.tf_mat)
        return Transformation(inverse_tf_mat)

    def applyTransform(self, pt):
        """Transform the given point from frame 1 to frame 2

        Parameters
        ----------
        pt : 3x1 numpy array
            3D co-ordinate of point in frame 1

        Returns
        -------
        pt_trans
            3x1 numpy array of transformed point
        """
        pt_homogenous = np.ones(4)  # convert to homogenous form
        pt_homogenous[0:3] = pt
        pt_trans = self.tf_mat @ pt_homogenous
        return pt_trans[0:3]

    def transformAndJacobian(self, pt):
        """Transform the given point and calculate the Jacobian about that point

        Parameters
        ----------
        pt : 3x1 numpy array
            3D co-ordinate of point

        Returns
        -------
        pt_trans
            3x1 numpy array of transformed point
        J_T
            3x6 numpy array representing Jacobian w.r.t. transformation
        J_p
            3x3 numpy array representing Jacobian w.r.t. point
        """
        pt_trans = self.applyTransform(pt)
        # Jacobian w.r.t. rotation is transformed point in skew-symmetric form
        C = self.C @ pt
        # Build skew-symmetric matrix
        J_C = np.array([[0, -C[2], C[1]],
                        [C[2], 0, -C[0]],
                        [-C[1], C[0], 0]])
        # Jacobian w.r.t. translation is I
        J_t = np.eye(3)
        J_T = np.concatenate((J_C, J_t), axis=1)
        # Jacobian w.r.t. point is the rotation matrix
        J_p = self.C
        return (pt_trans, J_T, J_p)
