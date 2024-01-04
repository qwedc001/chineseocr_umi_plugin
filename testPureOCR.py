from model import *
from PIL import Image
ocr = OcrHandle()
res = ocr.text_predict(Image.open("15.png"),960)
for item in res:
    text = item[1].split(" ")[1]
    accuracy = item[2]
    position = item[0]
    print(text)