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

RANK_DIFF_MAX = 17000

COMMAND_MAX_AREA = 120000
COMMAND_MIN_AREA = 40

LIMIT_Y_LINE = 100



font = cv2.FONT_HERSHEY_SIMPLEX

class Query_command:
    def __init__(self):
        self.contour = [] 
        self.width, self.height, self.x, self.y = 0, 0, 0, 0
        self.rect = []
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
    for Command in ['Down','F1','F2','F3','Left','Right','Up', 'A', 'B', 'C']:
        train_commands.append(Train_command())
        train_commands[i].name = Command
        filename = Command + '.jpg'
        train_commands[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1
    return train_commands

def preprocess_image(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    retval, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    return thresh

def find_cnts_commands(thresh_image):
    dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts_return = [] 
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

    cnts_order = []
    line_cnts  = []
    center, pts = define_center(cnts[0])
    limitY = center[1] + LIMIT_Y_LINE    
    for i in range(len(cnts)):
        center, pts = define_center(cnts[i])
        if(center[1] < limitY):
            line_cnts.append(cnts[i])
        else:
            limitY = center[1] + LIMIT_Y_LINE
            line_cnts = sort_contours(line_cnts)
            cnts_order.append(line_cnts)
            line_cnts = []
            line_cnts.append(cnts[i])
    line_cnts = sort_contours(line_cnts)
    cnts_order.append(line_cnts)        

    commands = []
    for y in range(len(cnts_order)): 
        line_commands  = []
        for x in range(len(cnts_order[y])):  
            qCommand = Query_command()
            qCommand = preprocess_command(cnts_order[y][x],image)
            best_command_match, diff = match_command(qCommand,train_commands)
            if(best_command_match != "Unknown"):
                qCommand.best_command_match, qCommand.diff = best_command_match, diff
                line_commands.append(qCommand)
        check_line(line_commands, image)
        if len(line_commands) > 0:
            commands.append(line_commands)
    
    return commands 

def check_line(line_commands, img):
    for x in range(len(line_commands)):  
        for y in range(x+1, len(line_commands)):
            if intersection(line_commands[x].x, line_commands[y].x, line_commands[x].y, line_commands[y].y, line_commands[x].height, line_commands[y].height, line_commands[x].width, line_commands[y].width):
                line_commands.remove(line_commands[x])           
                break

def intersection(X1, X2, Y1, Y2, H1, H2, W1, W2):
    if X1+W1<X2 or X2+W2<X1 or Y1+H1<Y2 or Y2+H2<Y1:
        return False
    else:
        return True


def match_command(qCommand, train_command):
    best_command_match_diff = 1000000000
    best_command_match_name = "Unknown"
    i = 0
    retval, qCommand.command_img = cv2.threshold(qCommand.command_img, 100, 255, cv2. THRESH_BINARY)
    if (len(qCommand.command_img) != 0):
        for Tcommand in train_command:
            err = np.sum((Tcommand.img.astype("float") - qCommand.command_img.astype("float")) ** 2)
            err /= float(Tcommand.img.shape[0] * qCommand.command_img.shape[1]) 
            if err < best_command_match_diff:
                best_command_match_diff = err
                best_command_name = Tcommand.name
    if (best_command_match_diff < RANK_DIFF_MAX):
        best_command_match_name = best_command_name
    return best_command_match_name, best_command_match_diff

def responseCommands(commands):
    response = ""
    if len(commands) == 0:
        return "Unknown"
    for y in range(len(commands)): 
        for x in range(len(commands[y])): 
            if x != len(commands[y])-1:
                response = response + commands[y][x].best_command_match +","
            else:
                response = response + commands[y][x].best_command_match
        if y != len(commands)-1:
            response = response + ",NEXT,"
    return response
    

def preprocess_command(contour, image):
    qCommand = Query_command()
    qCommand.contour = contour
    # Find width and height of card's bounding rectangle
    qCommand.width, qCommand.height,qCommand.x, qCommand.y = cv2.boundingRect(contour)
    qCommand.rect = (qCommand.width, qCommand.height,qCommand.x, qCommand.y)
    # Find center point of card by taking x and y average of the four corners.
    qCommand.center, pts = define_center(contour)
    qCommand.command_img = flattener(image, pts, qCommand.width, qCommand.height)
    return qCommand

def define_center(contour):
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    return [cent_x, cent_y], pts

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
    
    temp_rect[0] = tl
    temp_rect[1] = tr
    temp_rect[2] = br
    temp_rect[3] = bl

    maxWidth = 168
    maxHeight = 148
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
 
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    return warp