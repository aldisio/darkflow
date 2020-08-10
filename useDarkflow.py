import sys
import cv2
import numpy as np
from darkflow.net.build import TFNet

##################################################### 
# AM: Set YOLO init configuration
##################################################### 

options = {
	"model": "Yolo_COCO/yolov2.cfg", 
	"load": "Yolo_COCO/yolov2.weights", 
	"threshold": 0.60, 
	"gpu":0.20, 
	"labels":"Yolo_COCO/coco.names"
	}

#AM: Load YOLO Model.
tfnet = TFNet(options)

##################################################### 
# AM: Set input
##################################################### 
# AM: Uncomment lines below to test video file
# uri = 'video_sample.mp4'
# cap = cv2.VideoCapture(uri)

# AM: Uncomment line below to test webcam
cap = cv2.VideoCapture(0)

# AM: Read next frame
ret, frame = cap.read()

##################################################### 
# AM: Get parameters to save output video
##################################################### 
fr_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))            
fr_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))            
fr_fps = int(cap.get(cv2.CAP_PROP_FPS))

# AM: Define the codec and create VideoWriter object para mp4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4",fourcc, fr_fps, (fr_width,fr_height))


#####################################################
# AM: Main Loop - Predictions
#####################################################

while(cap.isOpened()):
	ret, frame = cap.read()
	frame_orig = frame.copy()

	# AM: Predict darkflow	
	result = tfnet.return_predict(frame)

	if len(	result ) > 0:	

		for i in range(len(result)):	
			print('-------------------')
			print('Result Yolo:', result)
			print('-------------------')
			print('topleft', result[0]['topleft'])
			print('bottomright',result[0]['bottomright'])
		
			tlx = result[i]['topleft']['x']
			tly = result[i]['topleft']['y']
			brx = result[i]['bottomright']['x']
			bry = result[i]['bottomright']['y']
	
			cX = int((tlx + brx) / 2.0)
			cY = int((tly + bry) / 2.0)		
			
			#if result[i]['label'] == 'car':
			cv2.rectangle(frame,(tlx,tly),(brx,bry),(255,0,0),2)
			text = "ACC {0:.6f}".format(result[i]['confidence'])
			print(result[i]['label'], "ACC: ", text)

			cv2.putText(frame, text, (tlx - 5, tly + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)
			text = "{}".format(result[i]['label']).upper()
			cv2.putText(frame, text, (tlx - 5, tly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)			

	# AM: write processed video
	out.write(frame)	

	# AM: show processed frame
	cv2.imshow('frame',frame)		

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()
