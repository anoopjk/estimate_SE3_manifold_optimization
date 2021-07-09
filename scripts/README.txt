########  Dependencies ###########

1. python3
2. numpy
3. matplotlib

########## Running the SE3 optimization script #########
* cd to the scripts folder and then run one of the following

-for manifold optimization

python estimate_se3.py --gt_file path-to-gtPoses.txt --est_file path-to-estPoses.txt --method manifold
--------------------------------------------------------------------------------------------------------
-for umeyama

python estimate_se3.py --gt_file path-to-gtPoses.txt --est_file path-to-estPoses.txt --method umeyama


######### Running the eval script ################

python evaluate_rpe.py gtPoses.txt alignedEstPosesManifold.txt --verbose
