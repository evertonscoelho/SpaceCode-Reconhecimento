import numpy as np
import cv2
import time

### Constants ###

# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125


COMMAND_DIFF_MAX = 2000

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 40

font = cv2.FONT_HERSHEY_SIMPLEX

class Query_command:
    def __init__(self):
        self.contour = [] 
        self.width, self.height, self.x, self.y = 0, 0, 0, 0
        self.corner_pts = [] 
        self.center = [] 
        self.warp = []
        self.command_img = [] 
        self.best_command_match = "Unknown"
        self.diff = 0 
        

class Train_command:
    def __init__(self):
        self.img = [] 
        self.name = "Placeholder"

def load_commands(filepath):
    train_commands = []
    i = 0
    for Command in ['Down','F1','F2','F3','Left','Right','Up']:
        train_commands.append(Train_command())
        train_commands[i].name = Command
        filename = Command + '.png'
        train_commands[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1
    return train_commands

def preprocess_image(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def find_cards(thresh_image):
    dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt_is_card = np.zeros(len(cnts),dtype=int)
    
    for i in range(len(cnts)):
        size = cv2.contourArea(cnts[i])
        peri = cv2.arcLength(cnts[i],True)
        approx = cv2.approxPolyDP(cnts[i],0.01*peri,True)
        if len(approx) == 4 and (size < CARD_MAX_AREA) and (size > CARD_MIN_AREA):
            cnt_is_card[i] = 1
    
    return cnts, cnt_is_card

def preprocess_card(contour, image):
    qCommand = Query_command()

    qCommand.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCommand.corner_pts = pts

    # Find width and height of card's bounding rectangle
    qCommand.width, qCommand.height,qCommand.x, qCommand.y = cv2.boundingRect(contour)

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCommand.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCommand.warp = flattener(image, pts, qCommand.width, qCommand.height)

    # Grab corner of warped card image and do a 4x zoom
    Qcorner = qCommand.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)

    # Sample known white pixel intensity to determine good threshold level
    white_level = Qcorner_zoom[15,int((CORNER_WIDTH*4)/2)]
    thresh_level = white_level - CARD_THRESH
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)
    
    
    Qcommand = query_thresh[20:185, 0:128]

    # Find rank contour and bounding rectangle, isolate and find largest contour
    dummy, Qcommand_cnts, hier = cv2.findContours(Qcommand, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qcommand_cnts = sorted(Qcommand_cnts, key=cv2.contourArea,reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(Qcommand_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(Qcommand_cnts[0])
        Qcommand_roi = Qcommand[y1:y1+h1, x1:x1+w1]
        Qcommand_sized = cv2.resize(Qcommand_roi, (RANK_WIDTH,RANK_HEIGHT), 0, 0)
        qCommand.command_img = Qcommand_sized

    return qCommand

def match_card(qCard, train_command):
    best_command_match_diff = 10000
    best_command_match_name = "Unknown"
    i = 0

    if (len(qCard.command_img) != 0):
        for Tcommand in train_command:
            diff_img = cv2.absdiff(qCard.command_img, Tcommand.img)
            command_diff = int(np.sum(diff_img)/255)
                
            if command_diff < best_command_match_diff:
                best_command_match_diff = command_diff
                best_command_name = Tcommand.name

    if (best_command_match_diff < RANK_DIFF_MAX):
        best_command_match_name = best_command_name

    return best_command_match_name, best_command_match_diff
    
    
def draw_results(image, qCommand):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCommand.center[0]
    y = qCommand.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)

    rank_name = qCommand.best_rank_match
    suit_name = qCommand.best_suit_match

    # Draw card name twice, so letters have black outline
    print(rank_name)
    print(suit_name)
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(50,200,200),2,cv2.LINE_AA)

    cv2.putText(image,suit_name,(x-60,y+25),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,suit_name,(x-60,y+25),font,1,(50,200,200),2,cv2.LINE_AA)
    
    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    #r_diff = str(qCommand.rank_diff)
    #s_diff = str(qCommand.suit_diff)
    #cv2.putText(image,r_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)
    #cv2.putText(image,s_diff,(x+20,y+50),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image

def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

        

    return warp