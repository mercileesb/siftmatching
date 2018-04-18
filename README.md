# siftmatching
SIFT matching for searching near duplicate images
-


The purpose of this project is to find the near duplicate image.

At first, I extracted sift feature upon all image feature, dumping features to pickle file.

Secondly, I experimented 3 cases for matching speed.

    - [1] Linear search on each indexed feature
    - Linear search with FLANN method on each indexed feature  
    - [2] Linear search on prefetching indexed features
    - [3] Parallel search on prefetching indexed features

### Files
    - img_augmentor.py: To make query images, augment all original image
    - utils.py: parsing, prefetching, ranking
    - sift_controller.py: SIFT warpper of cv2.SIFT()
    - main.py: main



## Performance

CPU : i7-8700K CPU @  3.70GHz
Number of Test image: 3,334

    - [1] : 784ms
    - [2] : 56ms
    - [3] : 8ms


## Author

Seungbin Lee: merci.leesb@gmail.com 




