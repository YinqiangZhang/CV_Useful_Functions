import numpy as np

def calc_overlap_occlusion(bboxes1, bboxes2, idx1_list=None, idx2_list=None):
    
    """
    caculate the overlap ratios and occluded ratios between 
    bboxes1[idx1_list,:] and bboxes2[idx2_list, :]

    return: Overlap, Occlusion1, Occlusion2
    """
    # get the selected bounding boxes
    bbs1 = bboxes1 if idx1_list is None else bboxes1[idx1_list, :]
    bbs2 = bboxes2 if idx2_list is None else bboxes2[idx2_list, :]
    
    bbs1 = bbs1[np.newaxis, :] if bbs1.ndim == 1 else bbs1
    bbs2 = bbs2[np.newaxis, :] if bbs2.ndim == 1 else bbs2
    # calculate each area
    area1 = bbs1[:, 2] * bbs1[:, 3]
    area2 = bbs2[:, 2] * bbs2[:, 3]

    # reform the bounding box [x1, y1, w, h] -> [x1, y1, x2, y2]
    bbs1[:, 2] = bbs1[:, 0] + bbs1[:, 2] - 1
    bbs1[:, 3] = bbs1[:, 1] + bbs1[:, 3] - 1

    bbs2[:, 2] = bbs2[:, 0] + bbs2[:, 2] - 1
    bbs2[:, 3] = bbs2[:, 1] + bbs2[:, 3] - 1

    ov = np.empty((bbs1.shape[0], bbs2.shape[0]))
    occ1 = np.empty((bbs1.shape[0], bbs2.shape[0]))
    occ2 = np.empty((bbs1.shape[0], bbs2.shape[0]))

    # find the overlap area
    for ii, bb1 in enumerate(bbs1):
        inter_x1 = np.max(np.vstack((bb1[0]*np.ones_like(bbs2[:, 0]), bbs2[:, 0])), axis=0)
        inter_y1 = np.max(np.vstack((bb1[1]*np.ones_like(bbs2[:, 1]), bbs2[:, 1])), axis=0)
        inter_x2 = np.min(np.vstack((bb1[2]*np.ones_like(bbs2[:, 2]), bbs2[:, 2])), axis=0)
        inter_y2 = np.min(np.vstack((bb1[3]*np.ones_like(bbs2[:, 3]), bbs2[:, 3])), axis=0)
        
        # calculate and verify the width and height of intersection region 
        inter_w = inter_x2 - inter_x1 + 1
        inter_h = inter_y2 - inter_y1 + 1
        inter_w[inter_w < 0] = 0
        inter_h[inter_h < 0] = 0
        
        # calculate the area of the intersection region
        inter_area = inter_w * inter_h
        # calcualte the area of the union region
        union_area = area1 + area2 - inter_area

        # calculate overlap and occlusion
        ov[ii, :] = inter_area / union_area
        occ1[ii, :] = inter_area / area1
        occ2[ii, :] = inter_area / area2
    
    return ov, occ1, occ2


if __name__ == "__main__":
    
    a = np.array([[1,1,10,10], [2,3,6,7]])
    b = np.array([[5,5,10,10],[7,7,9,9],[3,3,2,2]])

    ov, occ1, occ2 = calc_overlap_occlusion(a,b, idx2_list=[0, 1])

    print('ov:{}'.format(ov))
    print('occ1:{}'.format(occ1))
    print('occ2:{}'.format(occ2))