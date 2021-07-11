import numpy as np
from numpy import (eye, arccos, cos, sin,
                trace, exp, log, sqrt)
from numpy.linalg import (inv, norm, solve)
from utils import (skew_symmetric, quat_to_R,
                R_to_quat, create_T, vee)


def calc_residual(T_a_w, T_b_w, T_a_b):
    """[summary]

    Args:
        T_a_w ([type]): 4x4 array, pose from a to world
        T_b_w ([type]): 4x4 array, pose from b to world
        T_a_b ([type]): 4x4 array, rel pose from a to b

    Returns:
        res: 6x1 array, tangent_error_vector
    """    
    res = inv(T_b_w) @ T_a_b @ T_a_w
    res = log_operator(res)

    return res

def build_jacobian(SE3):
    """[summary]

    Args:
        SE3 ([type]): 4x4 Transformation

    Returns:
        [type]: 6x6 Jacobian
    """    
    # Jacobian approximated by Adjoint Map
    R = SE3[:3,:3]
    t = SE3[:3,3]
    J = eye(6)
    J[:3,:3] = R 
    J[:3,3:6] = skew_symmetric(t.reshape(-1,)) @ R
    J[3:6,3:6] = R
    return J


def exp_operator(tangent_vec):
    """[summary]
    Maps twist/tangent vector delta from se3 to lie group SE3
    Args:
        delta ([type]): 6x1 twist vector [vx, vy, vz, wx, wy, wz]
    Returns:
        SE3: 4x4 Transformation
    """
    SE3 = eye(4)
    omega = tangent_vec[3:]
    omega_skew_sym = skew_symmetric(omega.reshape(-1,))
    neu = tangent_vec[:3]
    theta =  norm(omega) #sqrt(omega.T @ omega)
    if theta <= 1e-10:
        V = eye(3)
        exp_omega_skew_sym = eye(3) 
    else:
        V = eye(3) + \
         (theta**-2)*(1-cos(theta))*omega_skew_sym + \
         (theta**-3)*(theta-sin(theta))*(omega_skew_sym @ omega_skew_sym)

        exp_omega_skew_sym = eye(3) + \
                        (sin(theta)/theta)*omega_skew_sym + \
                        (theta**-2)*(1-cos(theta))*(omega_skew_sym @ omega_skew_sym)
    
     #SE3[:3,:3] = exp(omega_skew_sym)
    SE3[:3,:3] = exp_omega_skew_sym
    SE3[:3,3] = (V @ neu).reshape(-1,)

    return SE3


def log_operator(SE3):
    """[summary]
    Maps SE3 to tangential vector space
    Args:
        SE3 ([type]): 4x4 array
    Returns:
        [type]: 6x1 tangent vector
    """   
    #print('SE3 log: ', SE3)
    R = SE3[:3,:3]
    t = SE3[:3,3]
    theta = arccos(0.5*(trace(R)-1)) # radians
    lnR = 0.5*(theta/sin(theta))*(R-R.T)
    omega = vee(lnR) # vee operator
    omega_skew_sym = lnR#skew_symmetric(omega.reshape(-1,))
    
    if theta <= 1e-10:
        V = eye(3)
    else:
        V = eye(3) + \
         (theta**-2)*(1-cos(theta))*omega_skew_sym + \
         (theta**-3)*(theta-sin(theta))*(omega_skew_sym @ omega_skew_sym)
    neu = inv(V) @ t

    # if theta <= 1e-10:
    #     Vinv = eye(3)
    # else:
    #     theta_half = 0.5*theta 
    #     Vinv = eye(3) - 0.5*omega_skew_sym + \
    #         (theta**-2)*(1- (theta_half*cos(theta_half)/sin(theta_half)))*(omega_skew_sym @ omega_skew_sym)
    # neu = Vinv @ t

    return np.hstack((neu, omega)).reshape(-1,1)


def estimate_SE3_LM(T_a_w, T_b_w, T_a_b=None, iterations = 10):
    """[summary]

    Args:
        T_a_w ([type]): 4x4 array, pose from a to world
        T_b_w ([type]): 4x4 array, pose from b to world

    Returns:
        [type]: 4x4 rel transformation from a to b
    """   
    if T_a_b is None: 
        T_a_b = np.eye(4)#np.array([0,0,0,0,0,0,1.0]).reshape(-1,1)
    mu = 0.005
    prev_cost = 0
    for iteration in range(iterations):
        print('running LM iteration: ', iteration)
        H = np.zeros((6,6))
        b = np.zeros((6,1))
        cost = 0
        # compute error/residual
        for idx in range(len(T_a_w)):
            error = calc_residual(T_a_w[idx,:], T_b_w[idx,:], T_a_b)
            #print('error: ', error)
            cost += norm(error)
            J = build_jacobian(inv(T_b_w[idx,:]))
            H += J.T @ J + mu*eye(6)
            b += -J.T @  error
        #print('H and b: ', H, b)

        delta = solve(H, b)
        print('delta: ', delta)

        # if the cost increases from previous, break
        if (iteration > 0 and (prev_cost-cost)<=1e-10): break
        # if the cost reduction is high enough then reduce the lambda
        if (prev_cost-cost) > 10:
            mu += 0.001
        else:
            mu -= 0.001
        print('exp_operator(delta): ', exp_operator(delta))
        T_a_b = exp_operator(delta) @ T_a_b
        print('T_a_b: ', T_a_b)
        print('cost, prev_cost: ', cost, prev_cost)

        prev_cost = cost

    
    return T_a_b
