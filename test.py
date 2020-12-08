import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import graph_tool.all as gt

files = [str(i) for i in range(7, 8)]
val_dir = os.getcwd()+'/val_set/'
for file in files:
    files_in_direc = os.listdir(val_dir+file+"/")
    print(files_in_direc)
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
        if m[0].distance < 0.5*m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        #print H
    else:
        raise AssertionError("Can’t find enough keypoints.")

    dst = cv2.warpPerspective(img_,H,(img.shape[1]+img_.shape[1], img.shape[0]+img_.shape[0]))
    plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
    plt.show()
    # plt.figure()
    g = gt.Graph()
    edge_weights = g.new_edge_property('double')
    g.edge_properties['weight'] = edge_weights
    first = -1
    last = -1
    print(img.shape)
    src_val = img.shape[0]*img.shape[1]
    tgt_val = src_val+1
    for i in range(0, img.shape[0]):
        last = -1
        for j in range(0, img.shape[1]):
            if(dst[i, j].all() == 0):
                dst[i, j] = img[i, j]
            elif i < img.shape[0]-1 and j < img.shape[1]-1:
                if first<i:
                    first = i
                    e = g.add_edge(src_val, i*img.shape[1]+j)
                    edge_weights[e] = np.inf
                e = g.add_edge(i*img.shape[1]+j, (i+1)*img.shape[1]+j)
                norm = sum(abs(img[i, j]-dst[i, j])) + \
                        sum(abs(img[i+1, j]-dst[i+1, j]))
                edge_weights[e] = norm
                # e1 = g.add_edge(i*img.shape[1]+j, i*img.shape[1]+j+1)
                # norm = ((sum((img[i:i+32, j:j+32]-dst[i:i+32, j:j+32])**2))**0.5) + ((sum((img[i:i+32, j+32:j+64]-dst[i:i+32, j+32:j+64])**2))**0.5)
                # edge_weights[e1] = norm
                last = j
            if last!=-1:
                e = g.add_edge(i*img.shape[1]+last, tgt_val)
                edge_weights[e] = np.inf
    print("done")
    src_index = g.vertex_index[src_val]
    tgt_index = g.vertex_index[tgt_val]
    src, tgt = g.vertex(src_index), g.vertex(tgt_index)
    res = gt.boykov_kolmogorov_max_flow(g, src, tgt, edge_weights)
    part = gt.min_st_cut(g, src, edge_weights, res)
    # mc, part = gt.min_cut(g, edge_weights)
    # pos = g.vertex_properties['pos']
    for i in range(g.num_vertices()):
        if int(g.vertex(i)) < img.shape[0]*img.shape[1]:
            if part[g.vertex(i)]:
                dst[i//img.shape[1], i%img.shape[1]] = img[i//img.shape[1], i%img.shape[1]]
                
    # dst[0:img.shape[0], 0:img.shape[1]] = img
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    _, contour, hei = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contour, key=cv2.contourArea)
    x, y, w, h =  cv2.boundingRect(cnt)
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(val_dir+file+'/output.jpg',dst)