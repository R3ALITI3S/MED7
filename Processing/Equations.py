import numpy as np


class EyeTrackingFramework:
    """
    Implementation of equations Eq1-Eq39.
    Angles are assumed to be in radians.
    """

    # ==========================================================
    # Eq2-Eq3 : Rotation matrices
    # ==========================================================

    @staticmethod
    def Rx(phi):
        return np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

    @staticmethod
    def Ry(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

    @staticmethod
    def Rz(psi):
        return np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])

    @staticmethod
    def rotation_matrix(psi, theta, phi):
        """
        Eq3
        R = Rz(psi) @ Ry(theta) @ Rx(phi)
        """
        return (
            EyeTrackingFramework.Rz(psi)
            @ EyeTrackingFramework.Ry(theta)
            @ EyeTrackingFramework.Rx(phi)
        )

    @staticmethod
    def rotate_heading_vector(v0, psi, theta, phi):
        """
        Eq2
        v_H = R * v_H(0)
        """
        R = EyeTrackingFramework.rotation_matrix(
            psi, theta, phi
        )
        return R @ np.asarray(v0)

    # ==========================================================
    # Eq9-Eq10
    # Head rotation angles from Euler angles
    # ==========================================================

    @staticmethod
    def alpha(psi, theta, phi):
        """
        Eq9
        """
        numerator = (
            np.sin(psi) * np.sin(theta) * np.sin(phi)
            + np.cos(psi) * np.cos(phi)
        )

        denominator = (
            np.cos(psi) * np.sin(theta) * np.sin(phi)
            - np.sin(psi) * np.cos(phi)
        )

        return np.pi / 2 - np.arctan2(
            numerator,
            denominator
        )

    @staticmethod
    def beta(theta, phi):
        """
        Eq10
        """
        return np.pi / 2 - np.arccos(
            np.cos(theta) * np.sin(phi)
        )

    # ==========================================================
    # Eq6 and Eq8
    # Head coordinates on screen
    # ==========================================================

    @staticmethod
    def x_head(alpha, x_max, alpha_max):
        """
        Eq6
        """
        return (
            (x_max / 2)
            / np.tan(alpha_max / 2)
            * np.tan(alpha)
        )

    @staticmethod
    def y_head(beta, alpha,
               y_max,
               beta_max):
        """
        Eq8
        """
        return (
            (y_max / 2)
            / (
                np.tan(beta_max / 2)
                * np.cos(alpha)
            )
            * np.tan(beta)
        )

    # ==========================================================
    # Eq1, Eq11, Eq12
    # Eye-in-space estimation
    # ==========================================================

    @staticmethod
    def eye_in_space(head_signal,
                     eye_signal):
        """
        Eq1

        S_G = S_H + S_E
        """
        return head_signal + eye_signal

    @staticmethod
    def gaze_coordinates(
            x_head,
            y_head,
            x_eye,
            y_eye):
        """
        Eq11-Eq12
        """
        x_gaze = x_head + x_eye
        y_gaze = y_head + y_eye

        return x_gaze, y_gaze

    # ==========================================================
    # Eq13-Eq14
    # Object-in-space
    # ==========================================================

    @staticmethod
    def object_in_space(
            x_head, y_head,
            x_obj, y_obj):
        """
        Object coordinates transformed
        to global coordinates.
        """
        return (
            x_head + x_obj,
            y_head + y_obj
        )

    # ==========================================================
    # Eq15-Eq17
    # Velocity and acceleration
    # ==========================================================

    @staticmethod
    def velocity(x_gaze,
                 y_gaze,
                 dt):
        """
        Eq15-Eq16
        """
        vx = np.diff(x_gaze) / dt
        vy = np.diff(y_gaze) / dt

        return vx, vy

    @staticmethod
    def acceleration(vx, vy, dt):
        """
        Eq17
        """
        dvx = np.diff(vx)
        dvy = np.diff(vy)

        return np.sqrt(
            dvx**2 + dvy**2
        ) / dt

    # ==========================================================
    # Eq18
    # Saccade amplitude
    # ==========================================================

    @staticmethod
    def amplitude(x, y):
        return np.sqrt(
            (np.max(x) - np.min(x))**2
            +
            (np.max(y) - np.min(y))**2
        )

    # ==========================================================
    # Eq19-Eq20
    # Eye Euclidean distance
    # ==========================================================

    @staticmethod
    def eye_euclidean_distance(
            x_gaze,
            y_gaze):
        """
        Eq19
        """
        dx = np.diff(x_gaze)
        dy = np.diff(y_gaze)

        return np.sqrt(dx**2 + dy**2)

    @staticmethod
    def cluster_distance(ed_cluster):
        """
        Eq20
        """
        return np.sum(ed_cluster)

    # ==========================================================
    # Eq21-Eq22
    # Object Euclidean distance
    # ==========================================================

    @staticmethod
    def object_euclidean_distance(
            x_objs,
            y_objs):
        """
        Eq21
        """
        dx = np.diff(x_objs)
        dy = np.diff(y_objs)

        return np.sqrt(dx**2 + dy**2)

    @staticmethod
    def object_cluster_distance(
            ed_cluster):
        """
        Eq22
        """
        return np.sum(ed_cluster)

    # ==========================================================
    # Eq23-Eq24
    # Total achieved distance
    # ==========================================================

    @staticmethod
    def total_distance(
            d_cluster,
            mean_direction):
        """
        Eq23, Eq24
        """
        d_cluster = np.asarray(d_cluster)
        mean_direction = np.asarray(
            mean_direction
        )

        return np.sum(
            np.sqrt(
                (d_cluster * np.cos(mean_direction))**2
                +
                (d_cluster * np.sin(mean_direction))**2
            )
        )

    # ==========================================================
    # Eq25-30
    # Mean direction
    # ==========================================================

    @staticmethod
    def mean_direction(
            d_cluster,
            m_cluster):
        """
        Returns Eq25-Eq27 values.
        """

        K = len(d_cluster)

        x_m = np.sum(
            d_cluster * np.cos(m_cluster)
        ) / K

        y_m = np.sum(
            d_cluster * np.sin(m_cluster)
        ) / K

        m_total = np.arctan2(
            y_m,
            x_m
        )

        return x_m, y_m, m_total

    # ==========================================================
    # Eq31-Eq33
    # Smooth pursuit decision
    # ==========================================================

    @staticmethod
    def summation_signal(filter_outputs):
        """
        Eq31 / Eq32

        shape:
        (n_filters, n_samples)
        """
        return np.sum(
            filter_outputs,
            axis=0
        )

    @staticmethod
    def binary_signal(signal):
        return (signal >= 0).astype(int)

    @staticmethod
    def smooth_pursuit_decision(
            s_eye,
            s_obj):
        """
        Eq33
        """
        b_eye = (
            EyeTrackingFramework
            .binary_signal(s_eye)
        )

        b_obj = (
            EyeTrackingFramework
            .binary_signal(s_obj)
        )

        b = b_obj - b_eye

        smooth_pursuit = b >= 0

        return b, smooth_pursuit

    # ==========================================================
    # Eq34-Eq35
    # Standard deviation
    # ==========================================================

    @staticmethod
    def gaze_std(x_gaze,
                 y_gaze):

        sigma_x = np.std(
            x_gaze,
            ddof=0
        )

        sigma_y = np.std(
            y_gaze,
            ddof=0
        )

        return sigma_x, sigma_y

    # ==========================================================
    # Eq36-Eq39
    # Classification metrics
    # ==========================================================

    @staticmethod
    def sensitivity(tp, fn):
        return tp / (tp + fn)

    @staticmethod
    def specificity(tn, fp):
        return tn / (tn + fp)

    @staticmethod
    def precision(tp, fp):
        return tp / (tp + fp)

    @staticmethod
    def balanced_accuracy(
            tp, fn,
            tn, fp):

        sen = tp / (tp + fn)
        spec = tn / (tn + fp)

        return (sen + spec) / 2