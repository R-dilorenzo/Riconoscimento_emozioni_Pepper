from treelib import Node, Tree
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import string
import qi
import argparse
import sys
import time
import librosa
import librosa.display
import numpy as np
import os
import sys
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import platform
import os
from threading import Thread
import Queue

PathFace = os.getcwd()
PathVoice = os.getcwd()

# salvo img e audio nel percorso dove si trova il file
if platform.system() == "Windows":
    PathFace1 = os.getcwd() + r'\image.jpg'
    PathVoice1 = os.getcwd() + r'\output.wav'
else:
    PathFace1 = os.getcwd() + r'/image.jpg'
    PathVoice1 = os.getcwd() + r'/output.wav'


# classe Domanda utilizzata per attributo data di treelib
class Domanda(object):
    def __init__(self, domanda):
        self.domanda = domanda

    def __repr__(self):
        return self.domanda


# funzione che passando un immagine restituisce emozione e % su emozione predetta
def EmotionImage(img):
    # parameters for loading data and images
    detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
    emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'
    img_path = img

    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    # reading the frame
    orig_frame = cv2.imread(img_path)
    frame = cv2.imread(img_path, 0)
    faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
        print "FACE emotion detected " + label
        print "FACE value of emotion detected " + str(emotion_probability)
    cv2.imshow('test_face', orig_frame)
    cv2.imwrite('test_output/' + img_path.split('/')[-1], orig_frame)
    if (cv2.waitKey(2000) & 0xFF == ord('q')):
        sys.exit("Thanks")
    cv2.destroyAllWindows()
    print "Face Array"
    print "angry - disgust - scared - happy - sad - surprised - neutral"
    print preds  # array con % di tutte le classi emozione
    return label, emotion_probability


# funzione che passando un audio restituisce emozione e % su emozione predetta
def VoiceEmotion(fileAudio):
    warnings.filterwarnings("ignore")
    classLabels = ('angry', 'scared', 'disgust', 'happy', 'sad', 'surprised', 'neutral')
    numLabels = len(classLabels)
    in_shape = (39, 216)
    model = Sequential()
    model.add(Conv2D(8, (13, 13), input_shape=(in_shape[0], in_shape[1], 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (13, 13)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(8, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(numLabels, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    # print(model.summary(), file=sys.stderr)
    model.load_weights('VoiceEmo.h5')

    file = fileAudio

    X, sample_rate = librosa.load(file, res_type='kaiser_best', duration=2.5, sr=22050 * 2, offset=0.5)
    # X, sample_rate = librosa.load(file, res_type='kaiser_best',duration=8.5,sr=22050*2,offset=0.5)

    sample_rate = np.array(sample_rate)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=39)
    feature = mfccs
    feature = feature.reshape(39, 216, 1)
    # feature = feature.reshape(39, 733, 1)
    #   classLabels[np.argmax(model.predict(np.array([feature])))]
    EmozioneVoce = classLabels[np.argmax(model.predict(np.array([feature])))]
    #   print "VOICE emotion detected " +str(EmozioneVoce)
    valEmozioneMax = np.argmax(model.predict(np.array([feature])))
    ArrayEmozione = model.predict(np.array([feature]))
    valNumericoEmoz = ArrayEmozione[0][valEmozioneMax]
    print "VOICE emotion detected " + str(EmozioneVoce)
    print "VOICE value of emotion detected " + str(valNumericoEmoz)

    print "Voice Array"
    print "'angry', 'scared', 'disgust', 'happy', 'sad', 'surprised', 'neutral'"
    print str(ArrayEmozione)  # array con % di tutte le classi emozione

    return EmozioneVoce, valNumericoEmoz

# classe thread per instanziare un nuovo thread che si occupa dell' analisi delle funzioni biometriche
class myThreads(Thread):
    def __init__(self, img, voice, queue):
        Thread.__init__(self)
        self.voice = voice
        self.queue = queue
        self.img = img

    def run(self):
        faceEm, valEmozFace = EmotionImage(self.img)
        voiceEm, valEmozVoice = VoiceEmotion(self.voice)
        self.queue.put(faceEm)
        self.queue.put(valEmozFace)
        self.queue.put(voiceEm)
        self.queue.put(valEmozVoice)


# funzione getID,joinString,lastParent utilizzata per creare automaticamente il parent e id nella funzione createNodo
def getID(e):
    if e == "happy":
        return "1"
    elif e == "neutral":
        return "2"
    elif e == "sad":
        return "3"
    elif e == "surprised":
        return "4"
    elif e == "scared":
        return "5"
    elif e == "disgust":
        return "6"
    elif e == "angry":
        return "7"


def joinString(parent1):
    listWord = []

    data = parent1.split("-")
    for temp in data:
        listWord.append(temp)

    listInt = []
    for temp1 in listWord:
        listInt.append(getID(temp1))

    new = ""
    for x in listInt:
        new += x

    return new


def lastParent(parent):
    data = parent.split("-")

    temp1 = []
    for temp in data:
        temp1.append(temp)

    return string.join(temp1[-1:])


def createRoot(domanda):
    return tree.create_node("root", "root", data=Domanda(domanda))

# funzione per creare il nodo passando emozione,livello,domanda e parent (per i livelli successivi al livello 0)
# che crea il nodo utilizzando la funzione create_node di treelib automatizzando id
def createNodo(emozione, livello, domanda, parent=None):
    if emozione in ['happy', 'neutral', 'sad', 'surprised', 'scared', 'disgust', 'angry']:
        if livello == 0 and parent is None:
            return tree.create_node(emozione, emozione + str(livello) + getID(emozione), "root", data=Domanda(domanda))
        elif livello == 1 and parent is not None:
            parent = parent + str(livello - 1) + getID(parent)
            return tree.create_node(emozione, emozione + str(livello) + str(parent[(-livello):]) + getID(emozione),
                                    parent, data=Domanda(domanda))
        elif livello >= 2 and parent is not None:
            x = lastParent(parent)
            y = joinString(parent)
            parent1 = str(x) + str(livello - 1) + str(y)
            return tree.create_node(emozione, emozione + str(livello) + str(parent1[(-livello):]) + getID(emozione),
                                    parent1, data=Domanda(domanda))
        else:
            print "errore creazione nodo"
            sys.exit()
    else:
        print "nella creazione del nodo inserisci un emozione tra 'happy/neutral/sad/scared/surprised/disgusted/angry'"
        sys.exit()


def risultatoFace(valF):
    # res= (%Predict + AccModelFace)/2
    res = (valF + 0.8928) / 2
    return res


def risultatoVoice(valV):
    # res= (%Predict + AccModelFace)/2
    res = (valV + 0.9697) / 2
    return res


class HumanGreeter(object):
    """
    inizializzo i moduli necessari per Pepper
    """

    def __init__(self, app):
        """
            Initialisation of qi framework and event detection.
        """
        super(HumanGreeter, self).__init__()
        app.start()
        session = app.session
        # Get the service ALMemory.
        self.memory = session.service("ALMemory")

        # Connetto con evento di callback.
        self.subscriber = self.memory.subscriber("FaceDetected")
        self.subscriber.signal.connect(self.on_human_tracked)

        # Get the services ALTextToSpeech and ALFaceDetection.
        self.tts = session.service("ALTextToSpeech")
        self.asr_service = session.service("ALSpeechRecognition")
        self.face_detection = session.service("ALFaceDetection")
        self.audioRecorder = session.service("ALAudioRecorder")
        #       self.asr_service.setLanguage("Italian")
        self.photoCaptureProxy = session.service("ALPhotoCapture")
        self.face_detection.subscribe("HumanGreeter")
        self.got_face = False
        self.asr_service.setParameter("NbHypotheses", 1)
        self.ba_service = session.service("ALBasicAwareness")
        self.face = False
        self.animation_service = session.service('ALAnimationPlayer')
        self.posture_service = session.service('ALRobotPosture')

    # evento per controllare se rileva un volto
    # quando rilevato l'array viene riempieto con alcuni valori del volto rilevato
    def on_human_tracked(self, value):
        # quando il volto scompare il valore dell array e' vuoto
        if value == []:
            self.got_face = False
        # quando appare un volto setto la variabile a true
        elif not self.got_face and self.face == False:
            self.got_face = True

    def takePhoto(self):
        # prendo 1 immagine in VGA e la salvo in /home/nao/recordings/cameras/
        self.photoCaptureProxy.setResolution(2)
        self.photoCaptureProxy.setPictureFormat("jpg")
        # self.photoCaptureProxy.takePictures(1, "/home/nao/recordings/cameras", "image")
        # AGGIORNAMENTO: il percorso nel robot e' stato cambiato, e dopo la cartella cameras e'
        #               stata creata una nuova cartella di nome 'image.jpg' e in questa cartella
        #               verra' salvata la foto di estensione .jpg
        self.photoCaptureProxy.takePictures(1, "/home/nao/recordings/cameras/image.jpg", "image")

        # prendo immagine e la salvo nel pc
        # utilizzo comando pscp per scaricare attraverso PuTTY (ssh)
        # altrimenti utilizzo il comando scp attraverso ssh inserendo anche il path della private key
        cmd = 'pscp.exe nao@pepper.local:/home/nao/recordings/cameras/image.jpg/image.jpg ' + PathFace

        p = os.popen(cmd, "w")       # pipe (popen) con "w" per scrittura su pipe
        # \n per simulare invio
        p.write("***\n") # INSERIRE PASSWORD PEPPER
        print "take picture eseguito"

        path = PathFace1  # secondo path di cmd aggiungendo immagine per sovrasciverla
        f = path
        return f

    def takeVoice(self):
        record_path = '/home/nao/recordings/output.wav'
        # sample rate di 16000 con 1 solo canale attivo oppure mettere 44000
        # con tutti e 4 i canali (di startMicrophoneRecording) settati a 1
        RATE = 16000

        self.tts.say("start recording")
        time.sleep(1)
        self.audioRecorder.startMicrophonesRecording(record_path, "wav", RATE, [0, 0, 1, 0])
        time.sleep(4)
        self.audioRecorder.stopMicrophonesRecording()
        self.tts.say("done recording")
        # prendo audio e la salvo nel pc
        # utilizzo comando pscp per scaricare attraverso PuTTY (ssh)
        # altrimenti utilizzo il comando scp attraverso ssh inserendo anche il path della private key
        cmd = 'pscp.exe nao@pepper.local:/home/nao/recordings/output.wav ' + PathVoice

        p = os.popen(cmd, "w")
        p.write("***\n")  # INSERIRE PASSWORD PEPPER
        print "eseguito"

        path = PathVoice1  # secondo path di cmd aggiungendo il file audio per sovrascriverlo
        f1 = path

        return f1

    # Partendo dal nodo radice, Pepper dice la frase associata al nodo radice, scatta foto (takePhoto()) e
    # registra audio (takeVoice()).
    # Tra i figli del nodo radice viene scelto quello con tag=emozione rilevata e viene visto:
    # -se il nodo ha figli si esegue ricorsivamente la funzione passando il nodo trovato come argomento della funzione
    # -se il nodo non ha figli viene detta la frase associata al nodo e termina
    def myfunct(self, root):
        point = tree.get_node(root)
        self.tts.say(str(point.data))
        npoint = tree.children(point.identifier)

        # solo se e' presente il volto inizia la funzione takePhoto e takeVoice
        # se non sono presenti volti Pepper rimane in attesa
        # e cerca di trovare un volto mentre si trova in BasicAwareness
        boolP = False
        while boolP == False:
            if self.got_face == True:
                boolP = True
                photo = human_greeter.takePhoto()
                voice = human_greeter.takeVoice()
            elif self.got_face == False:
                time.sleep(2)
            time.sleep(2)

        # inializzo oggetto coda che viene passato al thread per mantenere i valori delle funzioni biometriche
        queue = Queue.Queue()
        myThread = myThreads(photo, voice, queue)

        myThread.start()

        # vengono effettuate animazioni e conversazioni  mentre in myThread vengono eseguite le funzioni biometriche
        self.animation_service.run("animations/Stand/Gestures/Hey_3", _async=False)
        self.tts.say('Ok, now i detect your emotion...')
        self.animation_service.run("animations/Stand/Gestures/Thinking_8", _async=True)
        self.tts.say("Sorry, wait a few second!")
        self.posture_service.goToPosture('Stand', 0.5)
        myThread.join()

        # ottengo i valori delle funzioni dalla coda
        faceEm = queue.get()
        valEmozFace = queue.get()
        voiceEm = queue.get()
        valEmozVoice = queue.get()

        if faceEm == voiceEm:
            emozione = faceEm
        else:
            ResFace = risultatoFace(valEmozFace)
            ResVoice = risultatoVoice(valEmozVoice)

            print "ResFace= " + str(ResFace)
            print "ResVoice= " + str(ResVoice)

            if ResFace > ResVoice:
                emozione = faceEm
            elif ResVoice > ResFace:
                emozione = voiceEm

        print "dopo aver confrontato le emozioni e' stato scelto " + str(emozione)

        res = []
        # si cicla l'array (npoint) contente tutti i figli del nodo root e viene scelto il nodo con tag=emozione
        for n in npoint:
            if n.tag == emozione:
                res.append(n)
                point1 = tree.get_node(res[0].identifier)
                if tree.children(point1.identifier) == []:
                    self.tts.say(str(point1.data))
                else:
                    time.sleep(2)
                    # rieseguo la funzione sul nuovo nodo, livello successivo dell'albero
                    self.myfunct(point1.identifier)

    def run(self):

        print "Starting Metodo RUN"
        try:
            self.tts.say("start Basic Awareness")
            self.ba_service.setEngagementMode("FullyEngaged")
            self.ba_service.startAwareness()

            # solo se e' presente il volto inizia la funzione myfunct
            # se non sono presenti volti Pepper rimane in attesa
            # e cerca di trovare un volto mentre si trova in BasicAwareness
            boolW = False
            while boolW == False:
                if self.got_face == True:
                    boolW = True
                    human_greeter.myfunct("root")
                    time.sleep(1)
                    self.tts.say("ok,i finish")
                elif self.got_face == False:
                    time.sleep(2)
            #  time.sleep(2)

            self.tts.say("end basic awareness")
            self.ba_service.stopAwareness()

            self.tts.say("end client")
            print "fine main"

        except KeyboardInterrupt:
            print "Interrupted by user, stopping HumanGreeter"
            self.face_detection.unsubscribe("HumanGreeter")
            print "disattivo basic awareness"
            self.ba_service.stopAwareness()
            # stop
            sys.exit(0)


if __name__ == "__main__":
    # Connessione a Pepper
    # creazione dell' oggetto ArgumentParser() che contiene le informazioni necessarie per connettersi al robot
    parser = argparse.ArgumentParser()
    # e' stato inserito come valore di default i valori di Pepper utilizzato
    parser.add_argument("--ip", type=str, default="pepper.local",
                        help="Robot IP address. On robot or Local Naoqi: use 'pepper.local'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")
    args = parser.parse_args()
    try:
        # inizializzo qi framework.
        connection_url = "tcp://" + args.ip + ":" + str(args.port)
        app = qi.Application(["HumanGreeter", "--qi-url=" + connection_url])
    except RuntimeError:
        print("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
                                "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    tree = Tree()

    # createRoot(domanda)
    createRoot("Hello,i'll detect your emotion")
    # createNodo(emozione,livello,domanda,parent)
    # iniziare dal livello 0 non inserendo parent
    # al livello 1 inserire il parent
    # dal livello 2 inserire il path dal livello emozione 0 con "-" tra le emozioni
    # es. livello happy-sad se si vuole inserire il nodo al livello 2 con parent sad e granparent happy
    createNodo("happy", 0, "i detect Happy")
    createNodo("sad", 0, "i detect Sad")
    createNodo("surprised", 0, "i detect Surprised")
    createNodo("scared", 0, "i detect Scared")
    createNodo("neutral", 0, "i detect Neutral")
    createNodo("disgust", 0, "i detect Disgust")
    createNodo("angry", 0, "i detect Angry")

    createNodo("happy", 1, "lvl2 happy", "happy")
    createNodo("sad", 1, "lvl2 sad", "happy")
    createNodo("surprised", 1, "lvl2 surprised", "happy")
    createNodo("scared", 1, "lvl2 scared", "happy")
    createNodo("neutral", 1, "lvl2 neutral", "happy")
    createNodo("disgust", 1, "lvl2 disgust", "happy")
    createNodo("angry", 1, "lvl2 angry", "happy")

    createNodo("happy", 1, "lvl2 happy", "sad")
    createNodo("sad", 1, "lvl2 sad", "sad")
    createNodo("surprised", 1, "lvl2 surprised", "sad")
    createNodo("scared", 1, "lvl2 scared", "sad")
    createNodo("neutral", 1, "lvl2 neutral", "sad")
    createNodo("disgust", 1, "lvl2 disgust", "sad")
    createNodo("angry", 1, "lvl2 angry", "sad")

    createNodo("happy", 1, "lvl2 happy", "surprised")
    createNodo("sad", 1, "lvl2 sad", "surprised")
    createNodo("surprised", 1, "lvl2 surprised", "surprised")
    createNodo("scared", 1, "lvl2 scared", "surprised")
    createNodo("neutral", 1, "lvl2 neutral", "surprised")
    createNodo("disgust", 1, "lvl2 disgust", "surprised")
    createNodo("angry", 1, "lvl2 angry", "surprised")

    createNodo("happy", 1, "lvl2 happy", "scared")
    createNodo("sad", 1, "lvl2 sad", "scared")
    createNodo("surprised", 1, "lvl2 surprised", "scared")
    createNodo("scared", 1, "lvl2 scared", "scared")
    createNodo("neutral", 1, "lvl2 neutral", "scared")
    createNodo("disgust", 1, "lvl2 disgust", "scared")
    createNodo("angry", 1, "lvl2 angry", "scared")

    createNodo("happy", 1, "lvl2 happy", "neutral")
    createNodo("sad", 1, "lvl2 sad", "neutral")
    createNodo("surprised", 1, "lvl2 surprised", "neutral")
    createNodo("scared", 1, "lvl2 scared", "neutral")
    createNodo("neutral", 1, "lvl2 neutral", "neutral")
    createNodo("disgust", 1, "lvl2 disgust", "neutral")
    createNodo("angry", 1, "lvl2 angry", "neutral")

    createNodo("happy", 1, "lvl2 happy", "disgust")
    createNodo("sad", 1, "lvl2 sad", "disgust")
    createNodo("surprised", 1, "lvl2 surprised", "disgust")
    createNodo("scared", 1, "lvl2 scared", "disgust")
    createNodo("neutral", 1, "lvl2 neutral", "disgust")
    createNodo("disgust", 1, "lvl2 disgust", "disgust")
    createNodo("angry", 1, "lvl2 angry", "disgust")

    createNodo("happy", 1, "lvl2 happy", "angry")
    createNodo("sad", 1, "lvl2 sad", "angry")
    createNodo("surprised", 1, "lvl2 surprised", "angry")
    createNodo("scared", 1, "lvl2 scared", "angry")
    createNodo("neutral", 1, "lvl2 neutral", "angry")
    createNodo("disgust", 1, "lvl2 disgust", "angry")
    createNodo("angry", 1, "lvl2 angry", "angry")

    # createNodo("happy",2,"lvl3 granparent happy parent happy","happy-happy")

    """
    #es.secondo livello happy-emotion
    createNodo("happy",2,"lvl3 granparent happy parent happy","happy-happy")
    createNodo("happy",2,"lvl3 granparent happy parent neutral","happy-neutral")
    #es.terzo livello 
    createNodo("sad",3,lvl4 il sottoalbero risulta root-happy-neutral-happy","happy-neutral-happy")
    """

    # visualizzo albero creato con funzione .show
    # tree.show(idhidden=False)

    human_greeter = HumanGreeter(app)
    human_greeter.run()
