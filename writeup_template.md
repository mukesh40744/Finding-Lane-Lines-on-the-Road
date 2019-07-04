

---

I used python and open cv to draw line 



#### 1. Pipeline. 

Below are pipeline i have used to drwa line on lane

1-apply canny algo on image to convert into gray and then show the edges into image
2-then i select a region using poly to above images. ref- - 
     -[Canny Edge Detection OpenCV Theory](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html)
    - [cv2.Canny OpenCV API Reference](http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html)

3-There are multiple lines detected for a lane line.  We should come up with an averaged line for that.
  also, some lane lines are only partially recognized.  We should extrapolate the line to cover full lane line length.
  We want two lane lines: one for the left and the other for the right.  The left lane should have a positive slope, and the right    laneshould have a negative slope.  Therefore, we'll collect positive slope lines and negative slope lines separately and take averages.

4-then drwan line on image using avove avg lines 

5-combined the both origanl image and point 4 image

final source code file is -FindingLaneInVideo.py

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
