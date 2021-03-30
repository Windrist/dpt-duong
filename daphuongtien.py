import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import scipy.misc
from skimage.io import imshow
from skimage.color import rgb2ycbcr,ycbcr2rgb
from scipy.fftpack import dct,idct
import math
import huffman
from collections import Counter
import time

def myYcbcr2rgb(ycbcr):
    return (ycbcr2rgb(ycbcr).clip(0,1)*255).astype("uint8")

def toBlocks(img,yLen,xLen,h,w):
    blocks = np.zeros((yLen,xLen,h,w,3),dtype="int16")
    for y in range(yLen):
        for x in range(xLen):
            blocks[y][x]=img[y*h:(y+1)*h,x*w:(x+1)*w]
    return np.array(blocks)

def plotBlocks(blocks,gray=False):
    xLen=blocks.shape[1]
    yLen=blocks.shape[0]

def ycbcrBlock2rgb(block):
    return (ycbcr2rgb(block)*(255/ycbcr2rgb(block).max())).astype("uint8")

def dctOrDedctAllBlocks(blocks,type,yLen,xLen,h,w):
    f=dct if type=="dct" else idct
    dedctBlocks = np.zeros((yLen,xLen,h,w,3))
    for y in range(yLen):
        for x in range(xLen):
            d = np.zeros((h,w,3))
            for i in range(3):
                block=blocks[y][x][:,:,i]
                d[:,:,i]=f(f(block.T, norm = 'ortho').T, norm = 'ortho')
                if (type!="dct"):
                    d=d.round().astype("int16")
            dedctBlocks[y][x]=d
    return dedctBlocks

def blocks2img(blocks,yLen,xLen,h,w):
    W=xLen*w
    H=yLen*h
    img = np.zeros((H,W,3))
    for y in range(yLen):
        for x in range(xLen):
            img[y*h:y*h+h,x*w:x*w+w]=blocks[y][x]
    return img

def zigZag(block,h,w):
    lines=[[] for i in range(h+w-1)] 
    for y in range(h): 
        for x in range(w): 
            i=y+x 
            if(i%2 ==0): 
                lines[i].insert(0,block[y][x]) 
            else:  
                lines[i].append(block[y][x]) 
    return np.array([coefficient for line in lines for coefficient in line])

def huffmanCounter(zigZagArr,bitBits,runBits,rbBits):
    rbCount=[]
    run=0
    for AC in zigZagArr[1:]:
        if(AC!=0):
            AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
            if(run>2**runBits-1):
                runGap=2**runBits
                k=run//runGap
                for i in range(k):
                    rbCount.append('1'*runBits+'0'*bitBits)
                run-=k*runGap
            run=min(run,2**runBits-1) 
            bitSize=min(int(np.ceil(np.log(abs(AC)+0.000000001)/np.log(2))),2**bitBits-1)
            rbCount.append(format(run<<bitBits|bitSize,'0'+str(rbBits)+'b'))
            run=0
        else:
            run+=1
    rbCount.append("0"*(rbBits))
    return Counter(rbCount)

def runLengthReadable(zigZagArr,lastDC,bitBits,runBits,rbBits):
    rlc=[]
    run=0
    newDC=min(zigZagArr[0],2**(2**bitBits-1)-1)
    DC=newDC-lastDC
    bitSize=max(0,min(int(np.ceil(np.log(abs(DC)+0.000000001)/np.log(2))),2**bitBits-1))
    rlc.append([np.array(bitSize),DC])
    code=format(bitSize, '0'+str(bitBits)+'b')+"\n"
    if (bitSize>0):
        code=code[:-1]+","+(format(DC,"b") if DC>0 else ''.join([str((int(b)^1)) for b in format(abs(DC),"b")]))+"\n"
    for AC in zigZagArr[1:]:
        if(AC!=0):
            AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
            if(run>2**runBits-1):
                runGap=2**runBits
                k=run//runGap
                for i in range(k):
                    code+='1'*runBits+'0'*bitBits+'\n'
                    rlc.append([runGap-1,0])
                run-=k*runGap
            bitSize=min(int(np.ceil(np.log(abs(AC)+0.000000001)/np.log(2))),2**bitBits-1)
            #VLI encoding (next 2 lines of codes)
            code+=format(run<<bitBits|bitSize,'0'+str(rbBits)+'b')+','
            code+=(format(AC,"b") if AC>=0 else ''.join([str((int(b)^1)) for b in format(abs(AC),"b")]))+"\n"
            rs=np.zeros(1,dtype=object)
            rs[0]=np.array([run,bitSize])
            rs= np.append(rs,AC)
            rlc.append(rs)
            run=0
        else:
            run+=1
    rlc.append([0,0])
    code+="0"*(rbBits)#end
    return np.array(rlc),code,newDC

def runLength(zigZagArr,lastDC,hfm,bitBits,runBits,rbBits):
    rlc=[]
    run=0
    newDC=min(zigZagArr[0],2**(2**bitBits-1))
    DC=newDC-lastDC
    bitSize=max(0,min(int(np.ceil(np.log(abs(DC)+0.000000001)/np.log(2))),2**bitBits-1))
    code=format(bitSize, '0'+str(bitBits)+'b')
    if (bitSize>0):
        code+=(format(DC,"b") if DC>0 else ''.join([str((int(b)^1)) for b in format(abs(DC),"b")]))
    for AC in zigZagArr[1:]:
        if(AC!=0):
            AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
            if(run>2**runBits-1):
                runGap=2**runBits
                k=run//runGap
                for i in range(k):
                    code+=('1'*runBits+'0'*bitBits)if hfm == None else  hfm['1'*runBits+'0'*bitBits]#end
                run-=k*runGap
            run=min(run,2**runBits-1) 
            bitSize=min(int(np.ceil(np.log(abs(AC)+0.000000001)/np.log(2))),2**bitBits-1)
            rb=format(run<<bitBits|bitSize,'0'+str(rbBits)+'b') if hfm == None else hfm[format(run<<bitBits|bitSize,'0'+str(rbBits)+'b')]
            code+=rb+(format(AC,"b") if AC>=0 else ''.join([str((int(b)^1)) for b in format(abs(AC),"b")]))
            run=0
        else:
            run+=1
    code+="0"*(rbBits) if hfm == None else  hfm["0"*(rbBits)]#end
    return code,newDC

def runLength2bytes(code):
    return bytes([len(code)%8]+[int(code[i:i+8],2) for i in range(0, len(code), 8)])

def huffmanCounterWholeImg(blocks,yLen,xLen,h,w,bitBits,runBits,rbBits):
    rbCount=np.zeros(xLen*yLen*3,dtype=Counter)
    zz=np.zeros(xLen*yLen*3,dtype=object)
    for y in range(yLen):
        for x in range(xLen):
            for i in range(3):
                zz[y*xLen*3+x*3+i]=zigZag(blocks[y, x,:,:,i],h,w)
                rbCount[y*xLen*3+x*3+i]=huffmanCounter(zz[y*xLen*3+x*3+i],bitBits,runBits,rbBits)
    return np.sum(rbCount),zz
    rbCount,zz=huffmanCounterWholeImg(qDctBlocks,yLen,xLen,h,w,bitBits,runBits,rbBits)

def savingQuantizedDctBlocks(blocks,yLen,xLen,useHuffman,img,bitBits,runBits,rbBits,h,w):
    rbCount,zigZag=huffmanCounterWholeImg(blocks,yLen,xLen,h,w,bitBits,runBits,rbBits)
    hfm=huffman.codebook(rbCount.items())
    sortedHfm=[[hfm[i[0]],i[0]] for i in rbCount.most_common()]
    code=""
    DC=0
    for y in range(yLen):
        for x in range(xLen):
            for i in range(3):
                codeNew,DC=runLength(zigZag[y*xLen*3+x*3+i],DC,hfm if useHuffman else None,bitBits,runBits,rbBits)
                code+=codeNew
    savedImg=runLength2bytes(code)
    print(str(code[:100])+"......")
    print(str(savedImg[:20])+"......")
    print("Image original size:    %.3f MB"%(img.size/(2**20)))
    print("Compression image size: %.3f MB"%(len(savedImg)/2**20))
    print("Compression ratio:      %.2f : 1"%(img.size/2**20/(len(savedImg)/2**20)))
    return bytes([int(format(xLen,'012b')[:8],2),int(format(xLen,'012b')[8:]+format(yLen,'012b')[:4],2),int(format(yLen,'012b')[4:],2)])+savedImg,sortedHfm


def run(img):
    img = scipy.misc.face()
    w=8 #modify it if you want, maximal 8 due to default quantization table is 8*8
    w=max(2,min(8,w))
    h=w
    xLen = img.shape[1]//w
    yLen = img.shape[0]//h
    runBits=1 #modify it if you want
    bitBits=3  #modify it if you want
    rbBits=runBits+bitBits ##(run,bitSize of coefficient)
    useYCbCr=True #modify it if you want
    useHuffman=True #modify it if you want
    quantizationRatio=1 #modify it if you want, quantization table=default quantization table * quantizationRatio

    originalImg=img.copy()
    ycbcr=rgb2ycbcr(img)
    rgb=myYcbcr2rgb(ycbcr)
    if (useYCbCr):
        img=ycbcr
    np.allclose(rgb,originalImg,atol=1)
    blocks = toBlocks(img,yLen,xLen,h,w)
    x1=int(0.47*xLen)#x1,x2,y1,y2 is location of the face
    y1=int(0.21*yLen)
    plt.figure(figsize=(5,5))
    if(useYCbCr):
        plotBlocks(np.array(list(map(ycbcrBlock2rgb,blocks[y1:y1+40,x1:x1+40]))))
    else:
        plotBlocks(blocks[y1:y1+40,x1:x1+40])
    dctBlocks=dctOrDedctAllBlocks(blocks,"dct",yLen,xLen,h,w)
    newImg=blocks2img(dctBlocks,yLen,xLen,h,w)

    #quantization table
    QY=np.array([[16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]])
    QC=np.array([[17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]])
    QY=QY[:w,:h]
    QC=QC[:w,:h]
    qDctBlocks=dctBlocks.copy()
    Q3 = np.moveaxis(np.array([QY]+[QC]+[QC]),0,2)*quantizationRatio if useYCbCr else np.dstack([QY*quantizationRatio]*3)#all r-g-b/Y-Cb-Cr 3 channels need to be quantized
    Q3=Q3*((11-w)/3)
    qDctBlocks=(qDctBlocks/Q3).round().astype('int16') 

    qDctImg=blocks2img(qDctBlocks,yLen,xLen,h,w).astype('int16')

    Qrandom=np.arange(200,200+w*h).reshape(w,h)
    qRandDctBlocks=dctBlocks.copy()
    Qr3 = np.dstack([Qrandom]*3)
    qRandDctBlocks/=Qr3
    qRandDctBlocks=qRandDctBlocks.round().astype('int16')

    dedctBlocks=dctOrDedctAllBlocks(qDctBlocks*Q3,"idct",yLen,xLen,h,w)

    zigZag(qDctBlocks[0][0][:,:,0],h,w)

    rbCount=np.zeros(3,dtype=Counter)
    rbCount[0]=huffmanCounter(zigZag(qDctBlocks[0][0][:,:,0],h,w),bitBits,runBits,rbBits)
    rbCount[1]=huffmanCounter(zigZag(qDctBlocks[0][0][:,:,1],h,w),bitBits,runBits,rbBits)
    rbCount[2]=huffmanCounter(zigZag(qDctBlocks[0][0][:,:,2],h,w),bitBits,runBits,rbBits)
    rbCount=np.sum(rbCount)

    #Show run-length in a readable way
    b=np.zeros(64,dtype='int16')
    b[0]=222
    b[4]=9
    b[11]=33
    b[12]=25
    b[14]=-129
    b[17]=77
    b[27]=12
    b[47]=82
    arr,code,DC=runLengthReadable(b,0,bitBits,runBits,rbBits)

    code1,DC=runLength(zigZag(qDctBlocks[0][0][:,:,0],h,w),0,None,bitBits,runBits,rbBits)
    code2,DC=runLength(zigZag(qDctBlocks[0][0][:,:,1],h,w),DC,None,bitBits,runBits,rbBits)
    code3,DC=runLength(zigZag(qDctBlocks[0][0][:,:,2],h,w),DC,None,bitBits,runBits,rbBits)
    codeBlock=code1+code2+code3
    print(codeBlock+"\nCompresion size of this block: "+str(len(codeBlock)/8)+"KB\nOriginal size of one block: "+str(w*h*3)+"KB")

    #Following shows example of the saving and loading process of a block
    code2bytes=runLength2bytes(codeBlock)
    bytes2code="".join([format(i,'08b') for i in list(code2bytes)])

    hfm=huffman.codebook(rbCount.items())
    sortedHfm=[[hfm[i[0]],i[0],rbCount[i[0]]] for i in rbCount.most_common()]

    # print("(run,bit) occupies: "+str(np.sum(np.array(rbCount.most_common(),dtype='int32')[:,1]*rbBits)/8/2**10)+"KB")
    # print("(run,bit) after Huffman Coding occupies: "+str(np.sum([rbCount[k]*len(v) for k,v in hfm.items()])/8/2**10)+"KB")


    t1=time.time()
    savedImg,sortedHfmForDecode=savingQuantizedDctBlocks(qDctBlocks,yLen,xLen,useHuffman,img,bitBits,runBits,rbBits,h,w)
    t2=time.time()
    print("Encoding: "+str(t2-t1)+" seconds")
    save = open("img.bin", "wb")
    save.write(savedImg)
    save.close()

    qr=5
    qDctBlocks5=dctBlocks.copy()
    Q3_5 = Q3*qr
    qDctBlocks5=(qDctBlocks5/Q3_5).round().astype('int16')

    plt.figure()
    plt.title("Decompression of original DCT")
    dedctBlocks=dctOrDedctAllBlocks(dctBlocks,"idct",yLen,xLen,h,w)
    # imshow(myYcbcr2rgb(blocks2img(dedctBlocks,yLen,xLen,h,w)) if useYCbCr else blocks2img(dedctBlocks,yLen,xLen,h,w).astype('int16'))
    plt.figure()
    plt.title("Decompression of quantized DCT (quantization ratio = "+str(qr)+")")
    dedctBlocks=dctOrDedctAllBlocks(qDctBlocks5*Q3_5,"idct",yLen,xLen,h,w)
    # imshow(myYcbcr2rgb(blocks2img(dedctBlocks,yLen,xLen,h,w)) if useYCbCr else blocks2img(dedctBlocks,yLen,xLen,h,w).astype('int16'))

    savedImg1,rbCount1=savingQuantizedDctBlocks(qDctBlocks5,yLen,xLen,useHuffman,img,bitBits,runBits,rbBits,h,w)
    save = open("img5foldCompression.bin", "wb")
    save.write(savedImg1)
    save.close()

    return myYcbcr2rgb(blocks2img(dedctBlocks,yLen,xLen,h,w)) if useYCbCr else blocks2img(dedctBlocks,yLen,xLen,h,w)
