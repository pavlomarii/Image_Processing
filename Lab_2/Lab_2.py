import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def match(des1, des2, ratio=0.98):
    match1=[]
    match2=[]
    distances = {}
    for i in range(des1.shape[0]):
        if np.std(des1[i,:])!=0:
            d = np.zeros(des2.shape[0])
            for j in range(des2.shape[0]):
                d[j]=cv.norm(des1[i, :], des2[j, :], cv.NORM_HAMMING)
            orders =np.argsort(d).tolist()
            if d[orders[0]]/d[orders[1]]<=ratio:
                match1.append((i,orders[0]))
            distances[f'{i}-{orders[0]}'] = d[orders[0]] 
        
    for i in range(des2.shape[0]):
        if np.std(des2[i,:])!=0:
            d = np.zeros(des1.shape[0])
            for j in range(des1.shape[0]):
                d[j]=cv.norm(des2[i, :], des1[j, :], cv.NORM_HAMMING)
            orders =np.argsort(d).tolist()
            if d[orders[0]]/d[orders[1]]<=ratio:
                match2.append((orders[0],i))
            distances[f'{orders[0]}-{i}'] = d[orders[0]] 
            
    ##find good matches in rotation tests both ways
    match = list(set(match1).intersection(set(match2)))
    return [(pair[0], pair[1], distances[f'{pair[0]}-{pair[1]}']) for pair in match]

images = []
images.append( [cv.imread('./image1.jpg', cv.IMREAD_GRAYSCALE), cv.imread('./image2.jpg', cv.IMREAD_GRAYSCALE)])
images.append( [cv.imread('./image3.jpg', cv.IMREAD_GRAYSCALE), cv.imread('./image4.jpg', cv.IMREAD_GRAYSCALE)])

for i in images:
    image1, image2 = i

    star = cv.xfeatures2d.StarDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    temp1 = star.detect(image1, None)
    temp2 = star.detect(image2, None)

    image1_keypoints, image1_descriptor = brief.compute(image1, temp1)
    image2_keypoints, image2_descriptor = brief.compute(image2, temp2)

    matches = match(image1_descriptor, image2_descriptor)
    matches = sorted([cv.DMatch(*i) for i in matches], key=lambda x: x.distance)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches_bf = bf.match(image1_descriptor, image2_descriptor)
    matches_bf = sorted(matches_bf, key = lambda x:x.distance)

    fig, axis = plt.subplots(1, 2)
    img3 = cv.drawMatches(image1, image1_keypoints, image2, image2_keypoints, matches[:30], None, **dict( 
        matchColor = (255, 0, 0),
        singlePointColor = (255,0,0),
        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    ))
    img3_bf = cv.drawMatches(image1, image1_keypoints, image2, image2_keypoints, matches_bf[:30], None, **dict( 
        matchColor = (255, 0, 0),
        singlePointColor = (255,0,0),
        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    ))

    axis[0].imshow(img3)
    axis[0].set_title("Custom matcher")
    axis[1].imshow(img3_bf)
    axis[1].set_title("BF Matcher")
    plt.show()
