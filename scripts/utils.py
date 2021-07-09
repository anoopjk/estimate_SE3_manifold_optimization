import numpy as np
from numpy import sqrt, trace

def skew_symmetric(v):
    return np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]]
    )

def vee(skew_sym_mat):
    return np.array([skew_sym_mat[2,1], skew_sym_mat[0,2], skew_sym_mat[1,0]])

def create_T(R,t,c=1.0):
    # create a 4x4 transform matrix
    T = np.eye(4)
    T[:3,:3] = c*R 
    T[:3,3] = t 
    return T

def quat_conjugate(q):
    qx, qy, qz, qw = q.tolist()
    return quat_to_np_array(qx=-qx, qy=-qy, qz=-qz,qw=qw)

def quat_to_np_array(qx=0, qy=0, qz=0, qw=0):
    return np.array([qx, qy, qz, qw]).reshape(4, 1)

def quat_to_R(q):
    qx, qy, qz, qw = q.tolist()
    q_vec = np.array([qx, qy, qz]).reshape(3, 1)
    mat = (qw ** 2 - q_vec.T @ q_vec) * np.eye(3) \
            + 2 * q_vec @ q_vec.T - 2 * qw * skew_symmetric(q_vec.reshape(-1, ))
    return mat.T

def R_to_quat(R):
    tr = trace(R)

    if (tr > 0): 
        S = sqrt(tr+1.0) * 2 # S=4*qw 
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S 
        qz = (R[1,0] - R[0,1]) / S 
    elif ((R[0,0] > R[1,1])&(R[0,0] > R[2,2])): 
        S = sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2 # S=4*qx 
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[1,0] + R[0,1]) / S
        qz = (R[0,2] + R[2,0]) / S 
    elif (R[1,1] > R[2,2]): 
        S = sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2 # S=4*qy
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S 
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else: 
        S = sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2  # S=4*qz
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S

    return quat_to_np_array(qx=qx, qy=qy, qz=qz, qw=qw)