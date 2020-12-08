import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import maxflow

files = [str(i) for i in range(5, 10)]
val_dir = os.getcwd()+'/val_set/'
for file in files:
    files_in_direc = os.listdir(val_dir+file+"/")
    #print(files_in_direc)
    img_ = cv2.imread(val_dir+file+'/'+files_in_direc[1])
    frac = 0.4
    dim = (int(img_.shape[1]*frac), int(img_.shape[0]*frac))
    img_ = cv2.resize(img_, dim, interpolation = cv2.INTER_AREA)
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img = cv2.imread(val_dir+file+'/'+files_in_direc[0])
    dim = (int(img.shape[1]*frac), int(img.shape[0]*frac))
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # images = [img_, img]
    # stitcher = cv2.createStitcher()
    # (status, stitched) = stitcher.stitch(images)
    # cv2.imwrite('val_set/4/output.jpg',stitched)

    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.75*m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H1, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        raise AssertionError("Can’t find enough keypoints.")
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2,des1, k=2)

    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.75*m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    if len(matches[:,0]) >= 4:
        src = np.float32([ kp2[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp1[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H2, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        raise AssertionError("Can’t find enough keypoints.")

    dst = cv2.warpPerspective(img_,H1,(img.shape[1]+img_.shape[1], img.shape[0]+img_.shape[0]))
    min_row, min_col = min(img.shape[:2], img_.shape[:2])
    # #print(np.count_nonzero(dst[:int(0.1*img.shape[0]), :img.shape[1]//2]==0))
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0]])
    if np.count_nonzero(dst[:int(0.1*min_row), :min_col//2]==0) > 0.75*(int(0.05*min_col*min_row)) or \
        np.count_nonzero(dst[:int(min_row//2), :int(0.1*min_col)]==0) > 0.5*(int(0.05*min_col*min_row)):
        g = maxflow.Graph[float]()
        nodes = g.add_grid_nodes(img.shape[:2])
        weights = np.linalg.norm(dst[:img.shape[0], :img.shape[1]]-img, axis=2)
        g.add_grid_edges(nodes, weights=weights, structure=structure, symmetric=True)
        temp1, temp2 = np.zeros(img.shape[:2]), np.zeros(img.shape[:2])
        temp1[:, 0] = np.inf
        temp2[:, -1] = np.inf
        g.add_grid_tedges(nodes, temp1, temp2)
        g.maxflow()
        result = g.get_grid_segments(nodes)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if not result[i, j] or dst[i, j].all() == 0:
                    dst[i, j] = img[i, j]
        cv2.imwrite(val_dir+file+'/output.jpg',dst)
    else:
        print("dir 2")
        dst = cv2.warpPerspective(img,H2,(img.shape[1]+img_.shape[1], img.shape[0]+img_.shape[0]))
        g = maxflow.Graph[float]()
        nodes = g.add_grid_nodes(img_.shape[:2])
        weights = np.linalg.norm(dst[:img_.shape[0], :img_.shape[1]]-img_, axis=2, ord=2)
        g.add_grid_edges(nodes, weights=weights, structure=structure, symmetric=True)
        temp1, temp2 = np.zeros(img_.shape[:2]), np.zeros(img_.shape[:2])
        temp1[:, 0] = np.inf
        temp2[:, -1] = np.inf
        g.add_grid_tedges(nodes, temp1, temp2)
        g.maxflow()
        result = g.get_grid_segments(nodes)
        for i in range(img_.shape[0]):
            for j in range(img_.shape[1]):
                if not result[i, j] or dst[i, j].all() == 0:
                    dst[i, j] = img_[i, j]
        cv2.imwrite(val_dir+file+'/output.jpg',dst)