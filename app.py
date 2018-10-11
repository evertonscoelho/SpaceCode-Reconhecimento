#!flask/bin/python
from flask import Flask
from flask import request
import numpy as np
import Command
import cv2

app = Flask("Card-Detector")

@app.route('/', methods=['POST'])
def post():
    content = request.get_json()
    return getCommandByImage(content['image'])

def readb64(base64_string):
	nparr = np.fromstring(base64_string.decode('base64'), np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	return img

def getCommandByImage(content):
	image = readb64(content)
	r = 1100.0 / image.shape[1]
	dim = (1100, int(image.shape[0] * r))
	image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
	pre_proc = Command.preprocess_image(image)
	cnts, qntd_found, qtnd_squard = Command.find_cnts_commands(pre_proc)
	commands = Command.find_commands(cnts, image)
	return Command.responseCommands(commands)

app.run(host='0.0.0.0', port=80)
