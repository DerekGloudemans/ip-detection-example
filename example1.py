import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
import torch
import sys,os

sys.path.insert(-1,"PyTorch_YOLOv4")
from PyTorch_YOLOv4.detect import detect, load_classes

#%% First, let's open and view frames of a video one at a time from a file

# create a VideoCapture object - this reads frames from either a file or a stream
cap = cv2.VideoCapture('Falls9.mov')


if cap.isOpened():
    ret,frame = cap.read() # ret will be False if there's no frame, so can use it to check for last frame

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while ret:

 
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit, or any other key to advance a frame
    key = cv2.waitKey(0)    # Need to call waitKey for each frame as this pauses until the frame is displayed - a value other than 0 will cause a pause of value milleseconds and then it will continue with future frames
    if key == ord('q'): # you can use this to do different things for different keys pressed - in this case pressing q quits
      break
 
    # Load the next frame
    ret, frame = cap.read()
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()


#%% Now, let's do the same thing but for an RTSP (AoT camera) stream

cap = cv2.VideoCapture('rtsp://aot:K%K4@lNEqW@10.32.136.8/defaultPrimary-1?streamType=u',cv2.CAP_FFMPEG)
# the syntax for this is rtsp://<username>:<password>@<ip address>/defaultPrimary-<camera index on head>?streamType=u

cap = cv2.VideoCapture('rtsp://aot:K%K4@lNEqW@10.32.144.14/defaultPrimary-2?streamType=u',cv2.CAP_FFMPEG)


if cap.isOpened():
    ret,frame = cap.read() # ret will be False if there's no frame, so can use it to check for last frame

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while ret:

 
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    key = cv2.waitKey(1) 
    if key == ord('q'):
      break
 
    ret, frame = cap.read()
 
# When everything done, release the video capture object 
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()


#%% Let's take a look at one frame from the stream
cap = cv2.VideoCapture('rtsp://aot:K%K4@lNEqW@10.32.136.8/defaultPrimary-1?streamType=u',cv2.CAP_FFMPEG)

if cap.isOpened():
    ret,frame = cap.read() # ret will be False if there's no frame, so can use it to check for last frame

    print("Frame is an np array of integers with shape: {}".format(frame.shape))
    
    # just as an example, let's plot a rectangle on the image before displaying it
    cv2.rectangle(frame,(100,200),(400,500),(255,0,0),thickness = 2)
    
    # Display the resulting frame
    cv2.imshow('Frame',frame)
    cv2.waitKey(0)
    
    
 
# When everything done, release the video capture object 
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()


#%%  Let's take an RTSP stream and save it to a file as we receive frames


"""
The OpenCV video codecs are pretty slow and to my knowledge there's no way to write a frame without first decoding it
which mean that saving a video by first loading it into python via OpenCV is quite slow.
Instead, you can use FFmpeg to save chunks of video. But anyway here's how you do it in OpenCV.
This works well if you want to get a frame from the stream, process it (e.g. object detector/ tracking), 
display the outputs on the frame, and then save it
"""

# Create a VideoCapture object
cap = cv2.VideoCapture('rtsp://aot:K%K4@lNEqW@10.32.136.8/defaultPrimary-1?streamType=u',cv2.CAP_FFMPEG)#cap = cv2.VideoCapture('http://root:worklab@192.168.1.10/mjpg/video.mjpg') 
cap = cv2.VideoCapture('rtsp://aot:K%K4@lNEqW@10.32.144.14/defaultPrimary-0?streamType=u',cv2.CAP_FFMPEG)#cap = cv2.VideoCapture('http://root:worklab@192.168.1.10/mjpg/video.mjpg') 


# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(5))

    
# Define the codec and create VideoWriter object.
out = cv2.VideoWriter('test_output.mp4',cv2.VideoWriter_fourcc(*'H264'),frame_rate,(frame_width,frame_height))
start = time.time()
ret, frame = cap.read()

# load the class names that the detector was trained with
names = "PyTorch_YOLOv4/data/coco.names"
class_names = load_classes(names)

# make some random colors for each unique object class
colors = (np.random.rand(len(class_names),3)*255).astype(int)

frame_idx = 0
skip = 4  # process every 5 frames
while ret:
  
    
    # detect objects with YoloV4
    if frame_idx % skip == 0:
    
        # downsample because the GPU doesn't have enough memory otherwise
        downsample = 2
        im_det = cv2.resize(frame,(frame.shape[1]//downsample,frame.shape[0]//downsample))
        detections = detect(im_det,"PyTorch_YOLOv4/cfg/yolov4.cfg","PyTorch_YOLOv4/weights/yolov4.weights")[0].data.cpu().numpy()
        torch.cuda.empty_cache()
        detections[:,:4] *= downsample # rescale 
        
        # plot each of the detections
        for bbox in detections:
            label = "{}: {:.2f}".format(class_names[int(bbox[5])],bbox[4])
            color = colors[int(bbox[5])]
            color = (int(color[0]),int(color[1]),int(color[2]))
            
            c1 =  (int(bbox[0]),int(bbox[1]))
            c2 =  (int(bbox[2]),int(bbox[3]))
            cv2.rectangle(frame,c1,c2,color,thickness = 1)
            
            # plot label
            text_size = 0.8
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN,text_size , 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(frame, c1, c2,color, -1)
            cv2.putText(frame, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,text_size, [225,255,255], 1)
    
    
        # Write the frame into output file
        out.write(frame)
        
        # Display the resulting frame    
        cv2.imshow('frame',frame)
        
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break


    # get next frame
    ret, frame = cap.read()
    frame_idx += 1
    

# release video write and read objects
n_frames = cap.get(1) -1
cap.release()
out.release()

end = time.time()
elapsed = end - start

print('{} frames in {} seconds -- approx. FPS: {}'.format(n_frames,elapsed,n_frames/elapsed))
print('Stream dimensions:{} x {}'.format(frame_width,frame_height))

# close all frames
cv2.destroyAllWindows() 
print("Capture and write objects closed.")


#%% Ok, now let's consider the same example above but try to track objects


"""
We'll implement a very basic bipartite matching-based tracker. This means that for each new detection, we'll
select at most one existing object and match the detection-object pair. We need three things for this:
    
    1.) A record of existing objects and their locations
    2.) A similarity or dissimilarity function that scores a detection-object pair
    3.) An algorithm that assigns matches based on the scores defined in 2)
    
    1. We'll store the record as a dictionary:
        { id1: [position for frame k, position for frame k+1, ... , position for frame n-1], 
          id2: [position for frame k2, position for frame k2+1, ... , position for frame n-1], 
          ...
        }    
    where k is the first frame at which that object was visible
    
    2. For similarlity metric, we'll start with the most basic metric so we can see where it works and where it fails
    metric = Euclidean Distance 
    
    3. We'll use a greedy algorithm. That means we'll select one object, assign the closest detection, 
       remove that detection from the remaining set of detections, and continue until all objects have
       detections associated or no detection was close enough (a manually defined threshold prevents bad matches)
       Then, remaining detections are assigned as new objects
    
"""

# Create a VideoCapture object
cap = cv2.VideoCapture('rtsp://aot:K%K4@lNEqW@10.32.144.14/defaultPrimary-0?streamType=u',cv2.CAP_FFMPEG)
cap = cv2.VideoCapture('rtsp://aot:K%K4@lNEqW@10.32.136.8/defaultPrimary-1?streamType=u',cv2.CAP_FFMPEG)


# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(5))

    
# Define the codec and create VideoWriter object.
out = cv2.VideoWriter('test_output.mp4',cv2.VideoWriter_fourcc(*'H264'),frame_rate,(frame_width,frame_height))
start = time.time()
ret, frame = cap.read()

# load the class names that the detector was trained with
names = "PyTorch_YOLOv4/data/coco.names"
class_names = load_classes(names)

# make some random colors for each unique object class
colors = (np.random.rand(1000,3)*255).astype(int)

frame_idx = 0
skip = 3  # process every f frames
downsample = 2 # downsample because the GPU doesn't have enough memory otherwise


# To store objects
objects = {}
next_new_id = 0

# To count frames since each object was last detected
fsld = {}
fsld_limit = 2
threshold = 600

while ret:
  
    print("\rOn frame {}".format(frame_idx), end = '\r', flush = True)

    
    # detect objects with YoloV4
    if frame_idx % skip == 0:
    
        im_det = cv2.resize(frame,(frame.shape[1]//downsample,frame.shape[0]//downsample))
        detections = detect(im_det,"PyTorch_YOLOv4/cfg/yolov4.cfg","PyTorch_YOLOv4/weights/yolov4.weights")[0].data.cpu().numpy()
        torch.cuda.empty_cache()
        detections[:,:4] *= downsample # rescale 
        
        
        
        # compute similarity score for each detection-object pair
        
        # get most recent location from all objects
        boxes = []
        ids = []
        for id in objects:
            obj = objects[id][-1] # use the last known position
            boxes.append(obj)
            ids.append(id)
        boxes = np.array(boxes)
        
        # create array of distances
        distance = np.zeros([len(ids),len(detections)])
        for i in range(len(ids)):
            for j in range(len(detections)):
                # calculate Euclidean distance for object-detection pair
                distance[i,j] = np.sqrt(((boxes[i,0]+boxes[i,2]) - (detections[j,0]+ detections[j,2]))**2 +  ((boxes[i,1]+boxes[i,3]) - (detections[j,1]+ detections[j,3]))**2)
        
        
        # for each object, select best (min distance) detection
        matched_det_idxs = []
        for i in range(len(ids)):
            min_idx = np.argmin(distance[i])
            if distance[i,min_idx] < threshold:
                
                # store detection with object positions
                objects[ids[i]].append(detections[min_idx])
                
                # keep track that a detection was matched to this object 
                fsld[ids[i]] = 0
                
                # make sure no other objects get matched to this detection
                distance[:,min_idx] = np.infty
                
                matched_det_idxs.append(min_idx)
                
            else:
                fsld[ids[i]] += 1
        
        # now, let's deal with detections, initializing new objects for unmatched detections
        for j in range(len(detections)):
            if j not in matched_det_idxs:
                objects[next_new_id] = [detections[j]]
                fsld[next_new_id] = 0
                next_new_id += 1
        
        # lastly, let's remove all objects that haven't been detected for a while
        removals = []
        for id in fsld:
            if fsld[id] > fsld_limit:
                removals.append(id)

        for removal in removals:
            del objects[removal] # note that if you want to do something with these objects besides visualize them, you would want to store them in an inactive list rather than just deleting them
            del fsld[removal]
            
        
        
        # plot each of the objects with a unique color per object
        for id in objects:
            bbox = objects[id][-1]
            label = "{} {}".format(class_names[int(bbox[5])],id)
            color = colors[id]
            color = (int(color[0]),int(color[1]),int(color[2]))
            
            c1 =  (int(bbox[0]),int(bbox[1]))
            c2 =  (int(bbox[2]),int(bbox[3]))
            cv2.rectangle(frame,c1,c2,color,thickness = 1)
            
            # plot label
            text_size = 1.6
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN,text_size , 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(frame, c1, c2,color, -1)
            cv2.putText(frame, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,text_size, [225,255,255], 1)
    
    
        # Write the frame into output file
        out.write(frame)
        
        # Display the resulting frame    
        cv2.imshow('frame',frame)
        
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break


    # get next frame
    ret, frame = cap.read()
    frame_idx += 1
    

# release video write and read objects
n_frames = cap.get(1) -1
cap.release()
out.release()

end = time.time()
elapsed = end - start

print('{} frames in {} seconds -- approx. FPS: {}'.format(n_frames,elapsed,n_frames/elapsed))
print('Stream dimensions:{} x {}'.format(frame_width,frame_height))

# close all frames
cv2.destroyAllWindows() 
print("Capture and write objects closed.")