def printToFile(str, filePath):
    file = open(filePath, "a")
    file.write(str + "\n")
    file.close()
