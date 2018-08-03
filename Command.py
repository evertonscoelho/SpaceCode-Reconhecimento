import numpy as np
import cv2
import time
import os

### Constants ###

# Adaptive threshold levels
CARD_THRESH = 30

# Dimensions of rank train images
COMMAND_WIDTH = 168
COMMAND_HEIGHT = 148

RANK_DIFF_MAX = 200000000

COMMAND_MAX_AREA = 120000
COMMAND_MIN_AREA = 40

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
        filename = Command + '.jpg'
        train_commands[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1
    return train_commands

def preprocess_image(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    retval, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    cv2.imwrite("/home/everton/OpenCV-Playing-Card-Detector/teste/oi.jpeg", thresh);
    return thresh

def find_cnts_commands(thresh_image):
    dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts_return = []    
    print(len(cnts))
    for i in range(len(cnts)):
        size = cv2.contourArea(cnts[i])
        peri = cv2.arcLength(cnts[i],True)
        approx = cv2.approxPolyDP(cnts[i],0.01*peri,True)
        if len(approx) == 4 and (size < COMMAND_MAX_AREA) and (size > COMMAND_MIN_AREA):
              cnts_return.append(cnts[i])  
    return cnts_return

def find_commands(cnts, image, train_commands):
    if len(cnts) == 0:
        return []
    cnts = sort_contours(cnts, method="top-to-bottom")
    #ordena os y; verifica faixa aceitavel de y; do menor, percorre e vai separando em linhas pelo y; dentro de cada linha, ordena o x;    
    commands = []
    for i in range(len(cnts)):
        qCommand = Query_command()
        qCommand = preprocess_command(cnts[i],image)
        qCommand.teste = i
        best_command_match, diff = match_command(qCommand,train_commands, i)
        if(best_command_match != "Unknown"):
            qCommand.best_command_match, qCommand.diff = best_command_match, diff
            commands.append(qCommand)
    return commands

def match_command(qCommand, train_command, a):
    best_command_match_diff = 1000000000
    best_command_match_name = "Unknown"
    i = 0
    retval, qCommand.command_img = cv2.threshold(qCommand.command_img, 100, 255, cv2. THRESH_BINARY)
    cv2.imwrite("/home/everton/OpenCV-Playing-Card-Detector/teste/imagemQuadrado"+str(a)+".jpeg", qCommand.command_img);
    if (len(qCommand.command_img) != 0):
        for Tcommand in train_command:
            cv2.imwrite("/home/everton/OpenCV-Playing-Card-Detector/teste/imagemReconhecer.jpeg", Tcommand.img);
            diff_img = cv2.absdiff(Tcommand.img, qCommand.command_img)
            command_diff = int(np.sum(diff_img)/255)
            if command_diff < best_command_match_diff:
                best_command_match_diff = command_diff
                best_command_name = Tcommand.name
    if (best_command_match_diff < RANK_DIFF_MAX):
        best_command_match_name = best_command_name
    return best_command_match_name, best_command_match_diff

def preprocess_command(contour, image):
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
    qCommand.command_img = flattener(image, pts, qCommand.width, qCommand.height)
    return qCommand

def sort_contours(cnts, method="left-to-right"):
    i = 0
    if method == "top-to-bottom":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=False))
    return cnts

def flattener(image, pts, w, h):
    temp_rect = np.zeros((4,2), dtype = "float32")
    s = np.sum(pts, axis = 2)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    if w <= 0.8*h: 
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl
    if w >= 1.2*h: 
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br
    if w > 0.8*h and w < 1.2*h: 
        if pts[1][0][1] <= pts[3][0][1]:
            temp_rect[0] = pts[1][0] 
            temp_rect[1] = pts[0][0] 
            temp_rect[2] = pts[3][0] 
            temp_rect[3] = pts[2][0] 
        if pts[1][0][1] > pts[3][0][1]:
            temp_rect[0] = pts[0][0] 
            temp_rect[1] = pts[3][0] 
            temp_rect[2] = pts[2][0] 
            temp_rect[3] = pts[1][0]         
    maxWidth = 168
    maxHeight = 148
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    return warp