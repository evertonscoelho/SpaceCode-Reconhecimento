import cv2
import numpy as np
import time
import os
import Command

path = os.path.dirname(os.path.abspath(__file__))
pathTeste = path+"/teste/"
train_commands = Command.load_commands( path + '/Commands_Imgs/')

arqInput = open(pathTeste+"testeOutput.txt", 'w')
textInput = arqInput.readlines()
textOutput = []

for linha in textInput:
	split = linha.split(";")
	nomeArquivo = split[0]
	retornoEsperado = split[1]

	image = cv2.imread(pathTeste + nomeArquivo)
	pre_proc = Command.preprocess_image(image)
	cnts = Command.find_cnts_commands(pre_proc)
	commands = Command.find_commands(cnts, image, train_commands)

	textOutput.append(nomeArquivo +";")
	textOutput.append(commands +";")
	if commands.equals(retornoEsperado):
		textOutput.append("1\n")
	else:
		textOutput.append("0\n")


arqOutput = open(pathTeste+"testeOutput.txt", 'w')
arqOutput.write(teste)
arqOutput.close()