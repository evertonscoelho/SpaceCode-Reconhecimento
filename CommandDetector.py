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

	image = cv2.imread(pathTeste + nameFile)
	pre_proc = Command.preprocess_image(image)
	cnts = Command.find_cnts_commands(pre_proc)
	commands = Command.find_commands(cnts, image, train_commands)
	if len(commands) > 0:
		response = Command.responseCommands(commands)
	else:
		train_commands = Command.load_commands( path + '/Commands_Imgs_Black_Backgroud/')
		commands = Command.find_commands(cnts, image, train_commands)
		response = Command.responseCommands(commands)
	textOutput = textOutput + nameFile +";"
	textOutput = textOutput + response +";"
	if response.equals(resultExpected):
		textOutput = textOutput + "1\n"
	else:
		textOutput = textOutput +"0\n"

arqOutput = open(pathTeste+"testOutput.txt", 'w')
arqOutput.write(textOutput)
arqOutput.close()