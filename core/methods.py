from core.utils import processText, processImage, processVideo

def text(query : str):
    return processText(query)

def vid(yt_link : str):
    return processVideo(yt_link)

def img(img_link : str):
    return processImage(img_link)