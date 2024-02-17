import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import speech_recognition as sr
import pyttsx3
import random
import json
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
import os
import pickle

#--------------------------------------------------
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
        print(text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        print("Error occurred; {0}".format(e))


vectorizer = "TfidfVectorizer"
best_model = "Decision Tree"

# Traing Data
intents = {
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey", "Good day"],
      "responses": ["Hello!", "Good to see you!", "Hi there, how can I help?"],
      "context_set": ""
    },

  {
      "tag": "greeting_about",
      "patterns": ["How are you?"],
      "responses": ["I am fine.Thank you.How can I help you?"],
      "context_set": ""
    },

    {
      "tag": "farewell",
      "patterns": ["Goodbye", "Bye", "See you later", "Talk to you later"],
      "responses": ["Sad to see you go :(", "Goodbye!", "Come back soon!"],
      "context_set": ""
    },

    {
      "tag": "creator",
      "patterns": ["Who created you?", "Who is your developer?", "Who made you?"],
      "responses": ["I was created by Ashraf Ali."],
      "context_set": ""
    },

    {
      "tag": "identity",
      "patterns": ["What is your name?", "What should I call you?", "Who are you?"],
      "responses": ["You can call me Soofia. I'm a Chatbot."],
      "context_set": ""
    },

    {
      "tag": "university_identity",
      "patterns": ["What is the name of our university?", "The name of our university?", "Our university name?"],
      "responses": ["Our university name is Islamic University, Bangladesh."],
      "context_set": ""
    },


    {
    "tag": "university_details",
      "patterns": ["Tell me something about Islamic University", "Tell me something about IU"],
      "responses": ["Islamic University is one of the renowned public universities in Bangladesh. Established in 1985, it is located in Kushtia district.The university was founded on the principles of Islamic education and values, aiming to combine modern education with traditional Islamic teachings."],
      "context_set": ""
    },

    {
      "tag": "department_identity",
      "patterns": ["What is the name of our department?", "The name of our department?", "Our department name?"],
      "responses": ["Our department name is Computer Science and Engineering."],
      "context_set": ""
    },

    {
      "tag": "department_details",
      "patterns": ["Tell me something about our department", "Tell something about CSE department", "Tell something about Computer Science and Engineering department"],
      "responses": ["Computer Science and Engineering department started its academic activities from 1996 under the Faculty of Applied Science and Technology. The department provides an outstanding opportunity to students to get quality education in CSE. The students from the department are heavily recruited by both academia and industry. We invite you to explore the information here to learn more about our department."],
      "context_set": ""
    },

    {
      "tag": "department_chairman",
      "patterns": ["Who is the chairman of CSE department?", "Who is the chairman of Computer Science Engineering department?", "Chairman name of CSE department"],
      "responses": ["The name of Chairman of Computer Science and Engineering department is Prof. Dr. Md. Robiul Hoque"],
      "context_set": ""
    },

    {
      "tag": "hours",
      "patterns": ["What are the University timings?", "When the university open?", "What are your hours of operation?"],
      "responses": ["The university is open from 9am to 5pm, Saturday to Wednesday."],
      "context_set": ""
    },
    {
      "tag": "contact",
      "patterns": ["How can I contact the university?", "What is the university telephone number?", "Can I get your contact number?"],
      "responses": ["You can contact the university at 01715-351226."],
      "context_set": ""
    },
    {
      "tag": "location",
      "patterns": ["Where is the university located?", "What is the university address?", "How can I reach the university?"],
      "responses": ["The university is located at Kushtia. You can find the location on Google Maps."],
      "context_set": ""
    },
    {
      "tag": "happiness",
      "patterns": ["Are you happy here?", "Do you enjoy being at this University?"],
      "responses": ["As an AI, I don't have emotions, but I'm here to assist and provide information about the University."],
      "context_set": ""
    }
  ]
}


def chatbot_response(user_input):
  print("$$$$$"+user_input)
  input_text = vectorizer.transform([user_input])

  print("@@@@@"+input_text)
  predicted_intent = best_model.predict(input_text)[0]
  for intent in intents['intents']:
    if intent['tag'] == predicted_intent:
      response = random.choice(intent['responses'])
      break
  print(intents)
  return response




if not os.path.exists('model'):
    os.makedirs('model')

if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Save the trained model
with open('model/chatbot_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save the vectorizer
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save the intents to the "dataset" folder
with open('dataset/intents1.json', 'w') as f:
    json.dump(intents, f)

#--------------------------------------------

print('Hello! I am a chatbot. How can I help you today? Type "bye bye" to exit.')
alexa = pyttsx3.init()
voices = alexa.getProperty('voices')
alexa.setProperty('voice',voices[1].id)
alexa.say("Hello! I am a chatbot. How can I help you today? Say terminate to exit")
alexa.runAndWait()
while True:
    audio_text = recognize_speech()
    print("******"+audio_text)
    # user_input = input('> ')
    if audio_text is not None:
        if audio_text.lower() == 'terminate':
            break

        response = chatbot_response(audio_text)

        print("####"+response)
        alexa.say(response)
        alexa.runAndWait()

