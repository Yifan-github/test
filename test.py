# draw keypoints,scale,rotation...

import cv2
import pylab
import matplotlib.pyplot as plt
img = cv2.cvtColor(cv2.imread('/nfs/volume-315-5/zhaoyifan/projects/PoseCorr-main/data/scannet_dataset/scans_test/scene0754_00/color/2190.jpg'), cv2.COLOR_BGR2RGB)

det = cv2.ORB_create(500)
kps, descs = det.detectAndCompute(img, None)

out_img = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(out_img)
plt.imsave("/nfs/volume-315-5/zhaoyifan/projects/PoseCorr-main/0754_00-002190.jpg",out_img)
pylab.show()
print("0000")




