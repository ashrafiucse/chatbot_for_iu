import serial
import time
import speech_recognition as sr
import pyttsx3
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)
def write_read(x):
	   arduino.write(bytes(x, 'utf-8'))
	   time.sleep(0.05)


def recognize_speech():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Capture audio from the microphone
    with sr.Microphone() as source:
        print("Please start speaking...")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)

    try:
        # Recognize speech using Google Speech Recognition
        text = recognizer.recognize_google(audio_data)

        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        print("Error occurred; {0}".format(e))

#--------------------------------------------------------------



#---------------------------------
print('Hello! I am a Robot Rufaidah. How can I help you today? Type "bye bye" to exit.')
alexa = pyttsx3.init()
voices = alexa.getProperty('voices')
alexa.setProperty('voice',voices[1].id)
alexa.say("Hello! I am a Robot Rufaidah. How can I help you today? say terminate to exit.")
alexa.runAndWait()
while True:
    audio_text = recognize_speech()
    print(audio_text)
    # user_input = input('> ')
    if audio_text is not None:
        if audio_text.lower() == 'terminate':
            break
        elif audio_text.lower() == 'stand':
            #str = "Hello Sir I am saluting you"
            #alexa.say(str)
            #time.sleep(1.0)
            num = '0'  # Taking input from user
            value = write_read(num)
            #time.sleep(2.0)
            #str = "Thank you sir"
            #alexa.say(str)
        elif audio_text.lower() == 'sit':
            #str = "Hello Sir I am saluting you"
            #alexa.say(str)
            #time.sleep(1.0)
            num = '1'  # Taking input from user
            value = write_read(num)
            #time.sleep(2.0)
            #str = "Thank you sir"
            #alexa.say(str)

        elif audio_text.lower() == 'salute':
            str = "Hello Sir I am saluting you"
            alexa.say(str)
            time.sleep(1.0)
            num = '4'  # Taking input from user
            value = write_read(num)
            time.sleep(2.0)
            str = "Thank you sir"
            alexa.say(str)

            alexa.runAndWait()
            #print(value)  # printing the value