import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import maxflow

files = [str(i) for i in range(2, 3)]
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
    structure1 = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])
    # structure2 = np.array([[0, 0, 0],
    #                       [0, 0, 0],
    #                       [0, 1, 0]])
    if np.count_nonzero(dst[:int(0.1*min_row), :min_col//2]==0) > 0.75*(int(0.05*min_col*min_row)) or \
        np.count_nonzero(dst[:int(min_row//2), :int(0.1*min_col)]==0) > 0.5*(int(0.05*min_col*min_row)):
        g = maxflow.Graph[float]()
        nodes = g.add_grid_nodes(img.shape[:2])
        weights = np.zeros(img.shape[:2])
        weights1 = np.zeros(img.shape[:2])
        weights2 = np.zeros(img.shape[:2])
        count = 0
        last = -1
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                norm = sum((dst[i, j]-img[i, j])**2)**0.5
                weights[i, j] = norm
                # if dst[i, j].all() != 0:
                #     dst[i, j] = img[i, j]
                # try:
                #     norm1 = sum((dst[i, j]-dst[i-1, j])**2)**0.5
                #     norm2 = sum((dst[i-1, j]-dst[i-2, j])**2)**0.5
                #     norm3 = sum((img[i, j]-img[i-1, j])**2)**0.5
                #     norm4 = sum((img[i-1, j]-img[i-2, j])**2)**0.5
                #     norm = (norm1+norm2+norm3+norm4)
                #     if norm != 0:
                #         weights2[i-1, j] = (weights[i, j]+weights[i-1, j])/norm
                #     else:
                #         weights2[i-1, j] = np.inf
                # except:
                #     pass
                # try:
                #     norm1 = sum((dst[i, j]-dst[i, j-1])**2)**0.5
                #     norm2 = sum((dst[i, j-1]-dst[i, j-2])**2)**0.5
                #     norm3 = sum((img[i, j]-img[i, j-1])**2)**0.5
                #     norm4 = sum((img[i, j-1]-img[i, j-2])**2)**0.5
                #     norm = (norm1+norm2+norm3+norm4)
                #     if norm != 0:
                #         weights2[i, j-1] = (weights[i, j]+weights[i, j-1])/norm
                #     else:
                #         weights2[i, j-1] = np.inf
                # except:
                #     pass
        g.add_grid_edges(nodes, weights=weights, structure=structure1)
        # g.add_grid_edges(nodes, weights=weights2, structure=structure2, symmetric=True)
        temp1, temp2 = np.zeros(img.shape[:2]), np.zeros(img.shape[:2])
        temp1[:, 0] = np.inf
        temp2[:, -1] = np.inf
        g.add_grid_tedges(nodes, temp2, temp1)
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
        # #print(np.argmax(weights))
        last = -1
        weights = np.zeros(img_.shape[:2])
        weights1 = np.zeros(img_.shape[:2])
        weights2 = np.zeros(img_.shape[:2])
        count = 0
        last = -1
        for i in range(img_.shape[0]):
            for j in range(img_.shape[1]):
                # if dst[i, j].all() != 0:
                norm = sum((dst[i, j]-img_[i, j])**2)**0.5
                weights[i, j] = norm
                # else:
                #     weights[i, j] = np.inf
                # try:
                #     weights1[i-1, j] = weights[i, j]+weights[i-1, j]
                # except:
                #     pass
                # try:
                #     weights2[i, j-1] = weights[i, j]+weights[i, j-1]
                # except:
                #     pass
        g.add_grid_edges(nodes, weights=weights, structure=structure1, symmetric=True)
        # g.add_grid_edges(nodes, weights=weights2, structure=structure2, symmetric=True)
        temp1, temp2 = np.zeros(img_.shape[:2]), np.zeros(img_.shape[:2])
        temp1[:, 0] = np.inf
        temp2[:, -1] = np.inf
        g.add_grid_tedges(nodes, temp1, temp2)
        g.maxflow()
        result = g.get_grid_segments(nodes)
        for i in range(img_.shape[0]):
            for j in range(img_.shape[1]):
                if result[i, j] or dst[i, j].all() == 0:
                    dst[i, j] = img_[i, j]
        cv2.imwrite(val_dir+file+'/output.jpg',dst)