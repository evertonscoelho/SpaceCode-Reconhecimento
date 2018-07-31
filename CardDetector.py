import cv2
import numpy as np
import time
import os
import Cards


# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_commands = Cards.load_commands( path + '/Commands_Imgs/')

image = cv2.imread('3.jpeg')

# Pre-process camera image (gray, blur, and threshold it)
pre_proc = Cards.preprocess_image(image)
cv2.imwrite(path+"/testePre.jpeg", pre_proc);

# Find and sort the contours of all cards in the image (query cards)
cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

if len(cnts_sort) != 0:
    cards = []
    k = 0
    # For each contour detected:
    for i in range(len(cnts_sort)):
        if (cnt_is_card[i] == 1):
            cards.append(Cards.preprocess_card(cnts_sort[i],image))
            cards[k].best_command_match, cards[k].diff = Cards.match_card(cards[k],train_commands)
            k = k + 1
   
    for i in range(len(cards)):
        temp_cnts = []
        temp_cnts.append(cards[i].contour)
        cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
        cv2.imwrite(path+"/testePreasdsadsad.jpeg", image); 

# Close all windows and close the PiCamera video stream.
cv2.destroyAllWindows()