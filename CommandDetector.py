import cv2
import numpy as np
import time
import os
import Command

path = os.path.dirname(os.path.abspath(__file__))
train_commands = Command.load_commands( path + '/Commands_Imgs/')

image = cv2.imread('1.jpeg')

pre_proc = Command.preprocess_image(image)
cv2.imwrite(path+"/testePre.jpeg", pre_proc);

commands = Command.find_commands(pre_proc, image, train_commands)
   
for i in range(len(commands)):
    temp_cnts = []
    temp_cnts.append(commands[i].contour)
    cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
    cv2.putText(image,commands[i].best_command_match,(commands[i].center[0]-60,commands[i].center[1]+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3,cv2.LINE_AA)
    cv2.imwrite(path+"/testePreasdsadsad.jpeg", image);