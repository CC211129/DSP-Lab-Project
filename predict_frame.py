import numpy as np

def motion_estimation(cur_frame, ref_frame, blocksize, search_range):
    # TODO: implement motion estimation
    mvs = None
    return mvs # return motion vectors

def motion_compensation(ref_frame, blocksize, mvs):
    # TODO: implement motion compensation
    pred_frame = None
    return pred_frame

def predict_frame(cur_frame, ref_frame, blocksize, search_range):
    """
         Main function for the prediction.
         The prediction consists of the two steps motion estimation and motion compensation.
    :param cur_frame: current frame to be predicted from ref_frame
    :param ref_frame: preceding reference frame to be used for motion estimation and compensation
    :param blocksize: block size for block-based motion estimation and compensation
    :param search_range: search range for motion estimation
    :return: predicted frame
    """

    mvs = motion_estimation(cur_frame, ref_frame, blocksize, search_range)

    pred_frame = motion_compensation(ref_frame, blocksize, mvs)

    return pred_frame