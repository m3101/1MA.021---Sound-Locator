import cv2
import numpy as np
from NPFields import *
import scipy.signal
import wave

def normalise(field:np.ndarray):
    return (((field-field.min())/(field.max()-field.min()))*255).astype(np.uint8)
def posnegnor(field:np.ndarray,nam="Un",mi=-1,ma=1,maxi=False):
    r = np.ones(field.shape)/2
    neg = field[field<0]
    if len(neg)>0:
        mi=mi if maxi else neg.min()
        r[field<0] = .5*((neg-mi)/(-mi))
    pos = field[field>0]
    if len(pos)>0:
        ma=ma if maxi else pos.max()
        r[field>0] += .5*((pos)/(ma))
    return (r*255).astype(np.uint8)

size = 300

field = np.zeros((size,size))
#field = np.random.rand(size,size)

delfield = np.zeros(field.shape)

mx=0
my=0
drawing = False
drawingField=False
def click(event,x,y,flags,param):
    global pts,vels,field,size,drawing,mx,my,drawingField
    mx=x
    my=y
    if event == cv2.EVENT_LBUTTONDOWN:
        drawingField=True
    elif event == cv2.EVENT_LBUTTONUP:
        drawingField=False
    elif event == cv2.EVENT_MBUTTONDOWN:
        drawing=True
    elif event == cv2.EVENT_MBUTTONUP:
        drawing=False

cv2.namedWindow("Field",cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Analysis",cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback("Field",click)

c = .005

running = False
step = False


#fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#out = cv2.VideoWriter('output.avi',fourcc, 40.0, (300,300), True)

def graph(seq,width,height):
    img = np.zeros((height,width))
    positions = np.linspace(0,len(seq),width,False,dtype=int)
    pi=positions[0]
    mi = seq.min()
    ma = seq.max()
    if ma==mi:
        ma=1
    for x,i in enumerate(positions[1:]):
        cv2.line(img,(x-1,height-int(height*((seq[pi]-mi)/(ma-mi)))),(x,int(height-height*((seq[i]-mi)/(ma-mi)))),(1,1,1),1)
        pi=i
    return img

def polarmap(side):
    img=np.zeros((side,side))
    xx,yy = np.meshgrid(np.arange(side),np.arange(side))
    phi = np.abs(np.arctan2(xx-side/2,side/2-yy))
    phi=(phi-phi.min())/(phi.max()-phi.min())
    return phi.T

def graph180(seq:np.ndarray,map,normalise=False):
    if normalise:
        return (seq[map]-seq.min())/(seq.max()-seq.min())
    else:
        return seq[map]


"""
Distance of the mics/speed of sound
Number of frames between mic pulses
Time it takes for the sound to travel
"""
length=250

s_every = 3
s = 0

l=(length//s_every)
mic_a = np.zeros((l))
a=0
mic_b = np.zeros((l))
b=0
mic_c = np.zeros((l))

mic = np.array([140,150])
mic2 = np.array([160,150])

psize = 50
pmap = (polarmap(psize)*((2*(l-1)))).astype(int)
print(pmap.min(),pmap.max())

while True:
    #frame = posnegnor(field,"field",
                        #-1 if field.min()>-1 else field.min(),
                        #1 if field.max()<1 else field.max(),True)
    frame = posnegnor(field,"field")
    gradient = ScalarField.gradient(field)
    #samples.append(frame[mic[1],mic[0]])

    #samples.append(frame[mic2[1],mic2[0]])
    cv2.circle(frame,(mic[0],mic[1]),3,(255,255,255),3)
    cv2.circle(frame,(mic2[0],mic2[1]),3,(255,255,255),3)
    #cv2.imshow("Field",frame)
    cv2.imshow("Field",cv2.applyColorMap(frame,cv2.COLORMAP_JET))
    #out.write(cv2.applyColorMap(frame,cv2.COLORMAP_JET))
    
    div = VectorField.divergent(gradient)

    if running:
        s+=1
        if s>s_every:
            s=0
            mic_a=np.roll(mic_a,-1)
            mic_a[-1]=field[mic[1],mic[0]]
            a+=1
            a%=len(mic_a)
            mic_b=np.roll(mic_b,-1)
            mic_b[-1]=field[mic2[1],mic2[0]]
            b=1
            b%=len(mic_b)

            cross = scipy.signal.correlate(mic_a,mic_b)
            
            img = np.zeros((150,300))
            img[:50,:]=graph(mic_a,300,50)
            img[50:100,:]=graph(mic_b,300,50)
            img[100:,:]=graph(cross,300,50)
            cv2.imshow("Analysis",img)
            
            #cv2.imshow("Analysis",graph180(cross,pmap)/.3)

        field*=.99
        field += delfield
        #Borders
        field[0,:]=0
        field[-1,:]=0
        field[:,0]=0
        field[:,-1]=0
        delfield += div*c
        if step:
            step=False
            running=False

    if drawingField:
        xx,yy = np.meshgrid(np.arange(300),np.arange(300))
        d = ((xx-mx)**2 + (yy-my)**2)**.5
        o=d>10
        i=d<10
        field[i]=1

    key = cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    elif key==ord('r'):
        fc=0
        running=~running
    elif key==ord('z'):
        pts=[]
        vels=[]
    elif key==ord('s'):
        step=True
        running=True
    elif key==ord('a'):
        field = np.zeros((size,size))
        delfield = np.zeros((size,size))
#out.release()
#cv2.destroyAllWindows()

#sout = wave.open("out.wav",'wb')
#sout.setnchannels(2)
#sout.setframerate(40)
#sout.setsampwidth(1)
#sout.writeframes(bytes(samples))
#sout.close()