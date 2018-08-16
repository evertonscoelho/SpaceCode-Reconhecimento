import cv2
import numpy as np
import time
import os
import Command

path = os.path.dirname(os.path.abspath(__file__))
pathTeste = path+"/test/"
train_commands = Command.load_commands( path + '/Commands_Imgs/')

arqInput = open(pathTeste+"testInput.txt", 'r')
textInput = arqInput.readlines()
textOutput =""

for linha in textInput:
	split = linha.split(";")
	nameFile = split[0]
	resultExpected = split[1]
	resultExpected = resultExpected.replace("\n", "")
	image = cv2.imread(pathTeste + nameFile)

	r = 1100.0 / image.shape[1]
	dim = (1100, int(image.shape[0] * r))
	image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
	pre_proc = Command.preprocess_image(image)
	cnts = Command.find_cnts_commands(pre_proc)
	commands = Command.find_commands(cnts, image, train_commands)
	response = Command.responseCommands(commands)

	if response.lower() == resultExpected.lower():
		textOutput = textOutput + "1;"
	else:
		textOutput = textOutput +"0;"
	textOutput = textOutput + nameFile +";"
	textOutput = textOutput + response +"\n"
	
	temp_cnts = []
	for y in range(len(commands)):
		for x in range(len(commands[y])):
			temp_cnts.append(commands[y][x].contour)
			cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
			cv2.putText(image,commands[y][x].best_command_match,(commands[y][x].center[0]-60, commands[y][x].center[1]+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3,cv2.LINE_AA)
			cv2.imwrite(path+"/testeResultado"+ nameFile +".jpeg", image);     

arqOutput = open(pathTeste+"testOutput.txt", 'w')
arqOutput.write(textOutput)
arqOutput.close()

