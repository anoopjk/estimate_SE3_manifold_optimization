import os 
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from utils import skew_symmetric, quat_to_R, R_to_quat, create_T

from umeyama import estimate_SIM3_umeyama
from manifold_optimization import estimate_SE3_LM

def P7_to_P44(P7):
    """[summary]

    Args:
        P7 ([type]): 1x7 pose in [tx ty tz qx qy qz qw]

    Returns:
        [type]: 4x4 pose in SE3 form
    """    
    P44 = np.zeros((len(P7), 4, 4))
    for i in range(len(P7)):
        P44[i,:,:] = create_T(quat_to_R(P7[i,3:]),P7[i,:3])

    return P44

def main():
    """
    https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    *Each line in the text file contains a single pose.
    *The format of each line is 'timestamp tx ty tz qx qy qz qw'
    *timestamp (float) gives the number of seconds since the Unix epoch.
    *tx ty tz (3 floats) give the position of the optical center of the color camera with respect to the world origin as defined by the motion capture system.
    *qx qy qz qw (4 floats) give the orientation of the optical center of the color camera in form of a unit quaternion with respect to the world origin as defined by the motion capture system.
    *The file may contain comments that have to start with "#".
    """
    parser = argparse.ArgumentParser(description='''
    This script computes the SE3 transform which aligns estimated trajectory to ground truth trajectory. 
    ''')
    parser.add_argument('--gt_file', help='ground-truth trajectory file (format: "timestamp tx ty tz qx qy qz qw")')
    parser.add_argument('--est_file', help='estimated trajectory file (format: "timestamp tx ty tz qx qy qz qw")')
    parser.add_argument('--method', help='umeyama or manifold', default='manifold')
    args = parser.parse_args()

    test_poses_path = args.est_file#'estPoses.txt'
    gt_poses_path = args.gt_file#'gtPoses.txt'

    gt_poses = []
    test_poses = []
    with open(gt_poses_path, 'r') as gt_file, open(test_poses_path, 'r') as test_file:
        #print('gt poses')
        for gt_line in gt_file.readlines():
            #print(gt_line)
            gt_pose = []
            for i, val in enumerate(gt_line.strip().split(' ')):
                if i==0: continue
                gt_pose.append(float(val))
            gt_poses.append(gt_pose)
        #print('test poses')
        for test_line in test_file.readlines():
            #print(test_line)
            test_pose = []
            for i, val in enumerate(test_line.strip().split(' ')):
                if i==0: continue
                test_pose.append(float(val))
            test_poses.append(test_pose)

    # convert the list of poses to array of poses
    gt_poses = np.array(gt_poses)
    test_poses = np.array(test_poses)

    # compute the SE3 transform
    if args.method == 'umeyama':
        T_rel = estimate_SIM3_umeyama(test_poses[:,:3], gt_poses[:,:3])
    elif args.method == 'manifold':
        T_rel = estimate_SE3_LM(P7_to_P44(test_poses[:,:]), P7_to_P44(gt_poses[:,:]), 
                T_a_b=None, iterations=20)
    test_poses_aligned = np.zeros(gt_poses.shape)
    quat = R_to_quat(T_rel[:3,:3])
    # calculate test_poses_aligned
    for i in range(len(test_poses)):
        test_pose_al = create_T(quat_to_R(test_poses[i,3:]),test_poses[i,:3])
        test_pose_al = T_rel @ test_pose_al
        # test_poses_aligned[i,:3] = (test_poses[i,:3] @ T_rel[:3,:3] + T_rel[:3,3].reshape(-1,))
        test_poses_aligned[i,:3] = test_pose_al[:3,3]
        test_poses_aligned[i,3:] = R_to_quat(test_pose_al[:3,:3]).reshape(-1,)


    # plot the results
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(gt_poses[:,0], gt_poses[:,1], gt_poses[:,2], 'g', label='groundtruth')
    ax.plot3D(test_poses[:,0], test_poses[:,1], test_poses[:,2], 'b', label='estimated')
    ax.plot3D(test_poses_aligned[:,0], test_poses_aligned[:,1], test_poses_aligned[:,2], 'r', label='aligned')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.show()


    # write the aligned poses to a file
    with open('alignedEstPoses'+args.method.capitalize()+'.txt', 'w') as outfile:
        for i in range(len(test_poses_aligned)):
            outfile.write(str(i) + ' ')
            for val in test_poses_aligned[i,:].tolist():
                outfile.write(str(val) + ' ')
            outfile.write('\n')


if __name__ == '__main__':
    main()
