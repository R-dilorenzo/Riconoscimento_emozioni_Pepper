# Riconoscimento_emozioni_Pepper

Il progetto consiste nella creazione di un sistema multibiometrico per il riconoscimento delle emozioni attraverso due reti neurali, in particolare volto e voce, utilizzando il robot Pepper di Softbank Robotics

## Requisiti:
- Python 2.7.16 (32 bit)
- NAOqi
- OpenCV 4.1.26 
- LibROSA 0.7.1 
- NumPy 1.16.4 
- Keras 2.2.4 
- Theano 1.0.4
- treelib 1.5.5 
- PuTTY

## Architettura del sistema:

<img src="https://github.com/R-dilorenzo/Riconoscimento_emozioni_Pepper/blob/master/img/architettura_proposta.png" alt="architettura del sistema proposto" width="550px" height="650px">

Inizialmente bisogna connettersi al robot, percui si ha bisogno di un url del tipo tcp://nomeIP:porta. E' stato automatizzato questo processo creando l'oggetto ArgumentParser() e passando IP e la porta di Pepper. 

```Python
    # Connessione a Pepper
    # creazione dell' oggetto ArgumentParser() che contiene le informazioni necessarie per connettersi al robot
    parser = argparse.ArgumentParser()
    # e' stato inserito come valore di default i valori di Pepper utilizzato
    parser.add_argument("--ip", type=str, default="pepper.local",
                        help="Robot IP address. On robot or Local Naoqi: use 'pepper.local'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")
    args = parser.parse_args()
```

Viene eseguito il metodo run() e Pepper interagisce con l'utente rilevando un volto con la funzione di callback

```Python
    # evento per controllare se rileva un volto
    # quando rilevato l'array viene riempieto con alcuni valori del volto rilevato
    def on_human_tracked(self, value):
        # quando il volto scompare il valore dell array e' vuoto
        if value == []:
            self.got_face = False
        # quando appare un volto setto la variabile a true
        elif not self.got_face and self.face == False:
            self.got_face = True
```

Il robot per determinare l'emozione scatta una foto attraverso il metodo takePictures 

```Python
        # prendo 1 immagine in VGA e la salvo in /home/nao/recordings/cameras/
        self.photoCaptureProxy.setResolution(2)
        self.photoCaptureProxy.setPictureFormat("jpg")
        # self.photoCaptureProxy.takePictures(1, "/home/nao/recordings/cameras", "image")
        # AGGIORNAMENTO: il percorso nel robot e' stato cambiato, e dopo la cartella cameras e'
        #               stata creata una nuova cartella di nome 'image.jpg' e in questa cartella
        #               verra' salvata la foto di estensione .jpg
        self.photoCaptureProxy.takePictures(1, "/home/nao/recordings/cameras/image.jpg", "image")

```

registra l'audio per 3 secondi attraverso il metodo startMicrophonesRecording 

```Python
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
```

I file vengono scaricati sul pc attraverso PuTTY e si inizializza un nuovo thread in cui vengono passati i file alle funzioni per le reti neurali

```Python
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
```

Mentre nel thread principale il robot esegue animazioni e conversazioni

```Python
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

```

