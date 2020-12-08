import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
import maxflow
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generation Script for segmentation of Gall Bladder Images')
    parser.add_argument('-i', '--input_path', type=str, default='img', required=True, help="Path for the input image folder")
    
    args = parser.parse_args()
    files_in_direc = os.listdir(args.input_path)
    img_ = cv2.imread(args.input_path+'/'+files_in_direc[1])
    frac = 0.4
    dim = (int(img_.shape[1]*frac), int(img_.shape[0]*frac))
    img_ = cv2.resize(img_, dim, interpolation = cv2.INTER_AREA)
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img = cv2.imread(args.input_path+'/'+files_in_direc[0])
    dim = (int(img.shape[1]*frac), int(img.shape[0]*frac))
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
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
    structure = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0]])
    c1 = np.dot(H1, np.array([img.shape[0]//2, img.shape[1]//2, 1]))
    c2 = np.dot(H2, np.array([img_.shape[0]//2, img_.shape[1]//2, 1]))
    print(c1, c2)
    if sum(c1)>sum(c2) or sum(c2)-sum(c1) < 0.1*sum(c1):
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
        cv2.imwrite(args.input_path+'/output.jpg',dst)
    else:
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
        cv2.imwrite(args.input_path+'/output.jpg',dst)