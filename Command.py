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

RANK_DIFF_MAX = 20000

COMMAND_MAX_AREA = 120000
COMMAND_MIN_AREA = 5000

LIMIT_Y_LINE = 80

font = cv2.FONT_HERSHEY_SIMPLEX

path = os.path.dirname(os.path.abspath(__file__))

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
        self.imgs = [] 
        self.name = "Placeholder"

def load_commands(filepath):
    train_commands = []
    i = 0
    for Command in ['2','3','4','5','6','7','8','9','circle','left','right']:
        imgs = []
        train_commands.append(Train_command())
        train_commands[i].name = Command
        filename = Command + '.jpg'
        imgs.append(cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE))
        train_commands[i].img = imgs
        i = i + 1

    for Command in ['loop','loop2']:
        imgs = []
        train_commands.append(Train_command())
        train_commands[i].name = 'loop'
        filename = Command + '.jpg'
        imgs.append(cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE))
        train_commands[i].img = imgs
        i = i + 1

    for Command in ['move', 'move2','move3','move4']:
        imgs = []
        train_commands.append(Train_command())
        train_commands[i].name = 'move'
        filename = Command + '.jpg'
        imgs.append(cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE))
        train_commands[i].img = imgs
        i = i + 1

    for Command in ['star', 'star2','star3','star4']:
        imgs = []
        train_commands.append(Train_command())
        train_commands[i].name = 'star'
        filename = Command + '.jpg'
        imgs.append(cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE))
        train_commands[i].img = imgs
        i = i + 1
    
    for Command in ['triangle', 'triangle2','triangle3','triangle4']:
        imgs = []
        train_commands.append(Train_command())
        train_commands[i].name = 'triangle'
        filename = Command + '.jpg'
        imgs.append(cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE))
        train_commands[i].img = imgs
        i = i + 1
           
    return train_commands

train_commands = load_commands( path + '/Commands_Imgs/')

def preprocess_image(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def find_cnts_commands(thresh_image):
    dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts_return = [] 
    for i in range(len(cnts)):
        size = cv2.contourArea(cnts[i])
        peri = cv2.arcLength(cnts[i],True)
        approx = cv2.approxPolyDP(cnts[i],0.01*peri,True)
        if len(approx) == 4 and (size < COMMAND_MAX_AREA) and (size > COMMAND_MIN_AREA):
                #print(size)
                cnts_return.append(cnts[i])
    return cnts_return, len(cnts), len(cnts_return)

def find_commands(cnts, image):
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
            if intersection(line_commands[x].contour, line_commands[y].contour):
                line_commands.remove(line_commands[x])           
                break

def intersection(cnt1, cnt2):
    leftmost_cnt1 = tuple(cnt1[cnt1[:,:,0].argmin()][0])
    rightmost_cnt1 = tuple(cnt1[cnt1[:,:,0].argmax()][0])
    topmost_cnt1 = tuple(cnt1[cnt1[:,:,1].argmin()][0])
    bottommost_cnt1 = tuple(cnt1[cnt1[:,:,1].argmax()][0])

    leftmost_cnt2 = tuple(cnt2[cnt2[:,:,0].argmin()][0])
    rightmost_cnt2 = tuple(cnt2[cnt2[:,:,0].argmax()][0])
    topmost_cnt2 = tuple(cnt2[cnt2[:,:,1].argmin()][0])
    bottommost_cnt2 = tuple(cnt2[cnt2[:,:,1].argmax()][0])
    
    if leftmost_cnt1[0] < leftmost_cnt2[0] and rightmost_cnt1[0] > leftmost_cnt2[0]:
        return True
    else:
        return False


def match_command(qCommand, train_command):
    best_command_match_diff = 1000000000
    best_command_match_name = "Unknown"
    best_command_name = "Unknown"
    i = 0
    retval, qCommand.command_img = cv2.threshold(qCommand.command_img, 100, 255, cv2. THRESH_BINARY)
    if (len(qCommand.command_img) != 0):
        for Tcommand in train_command:
            for img in Tcommand.img:
                err = np.sum((img.astype("float") - qCommand.command_img.astype("float")) ** 2)
                err /= float(img.shape[0] * img.shape[1]) 
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
    qCommand.width, qCommand.height,qCommand.x, qCommand.y = cv2.boundingRect(contour)
    qCommand.rect = (qCommand.width, qCommand.height,qCommand.x, qCommand.y)
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

    maxWidth = 185
    maxHeight = 185
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
 
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    return warp