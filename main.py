import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
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
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        print("Error occurred; {0}".format(e))

#--------------------------------------------------------------


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
      "patterns": ["How about you?"],
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
      "patterns": ["Who is the chairman of CSE department?","what is the name of our chairman?", "Who is the chairman of Computer Science Engineering department?", "Chairman name of CSE department"],
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


#synonym replacement-------------------------------
# Function to perform synonym replacement
def synonym_replacement(tokens):
    augmented_sentences = []
    for i in range(len(tokens)):
        synonyms = []
        for syn in wordnet.synsets(tokens[i]):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if len(synonyms) > 0:
            for synonym in synonyms:
                augmented_tokens = tokens[:i] + [synonym] + tokens[i+1:]
                augmented_sentences.append(' '.join(augmented_tokens))
    return augmented_sentences


#--------------------------------------
text_data = []
labels = []
stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

for intent in intents['intents']:
    for example in intent['patterns']:
        tokens = nltk.word_tokenize(example.lower())
        filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords and token.isalpha()]
        if filtered_tokens:
            text_data.append(' '.join(filtered_tokens))
            labels.append(intent['tag'])

            augmented_sentences = synonym_replacement(filtered_tokens)
            for augmented_sentence in augmented_sentences:
                text_data.append(augmented_sentence)
                labels.append(intent['tag'])

#-----------------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_data)
y = labels

#---------------------------------------
def find_best_model(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100)


    models1 = [
        ('Logistic Regression', LogisticRegression(), {
            'penalty': ['l2'],
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear'],
            'max_iter': [100, 1000, 10000]
        }),
        ('Multinomial Naive Bayes', MultinomialNB(), {'alpha': [0.1, 0.5, 1.0]}),
        ('Linear SVC', LinearSVC(), {
            'penalty': ['l2'],
            'loss': ['hinge', 'squared_hinge'],
            'C': [0.1, 1, 10],
            'max_iter': [100, 1000, 10000]
        }),
        ('Decision Tree', DecisionTreeClassifier(), {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }),
        ('Random Forest', RandomForestClassifier(), {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        })
    ]
    models = [
        ('Decision Tree', DecisionTreeClassifier(), {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        })
    ]
    for name, model, param_grid in models:
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f'{name}: {score:.4f} (best parameters: {grid.best_params_})')

    best_model = max(models, key=lambda x: GridSearchCV(x[1], x[2], cv=3, n_jobs=-1).fit(X_train, y_train).score(X_test, y_test))
    print(f'\nBest model: {best_model[0]}')

    # Fit the best model to the full training data
    best_model[1].fit(X, y)

    return best_model[1]

#-------------------------------------------------
best_model = find_best_model(X, y)

#--------------------------------------------------
def chatbot_response(user_input):
    input_text = vectorizer.transform([user_input])
    predicted_intent = best_model.predict(input_text)[0]
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            break

    return response

#---------------------------------------------
import os
import pickle


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
import serial
import time
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)
def write_read(x):
	   arduino.write(bytes(x, 'utf-8'))
	   time.sleep(0.05)

#---------------------------------
print('Hello! I am a robot Rufaidah. How can I help you today? Type "terminate" to exit.')
alexa = pyttsx3.init()
voices = alexa.getProperty('voices')
alexa.setProperty('voice',voices[1].id)
alexa.say('Hello! I am a robot Rufaaiidaahh. How can I help you today? say "terminate" to exit.')
alexa.runAndWait()
while True:
    #audio_text = recognize_speech()
    audio_text = input('> ')
    if audio_text is not None:
        if audio_text.lower() == 'terminate':
            break
        elif audio_text.lower() == 'salute':
            num = '4'  # Taking input from user
            write_read(num)
            alexa.say("Hello Sir, I am saluting you")
            alexa.runAndWait()
            #print(value)  # printing the value

        elif audio_text.lower() == 'stand':
            #str = "Hello Sir I am saluting you"
            #alexa.say(str)
            #time.sleep(1.0)
            num = '0'  # Taking input from user
            write_read(num)
            #time.sleep(2.0)
            #str = "Thank you sir"
            #alexa.say(str)
        elif audio_text.lower() == 'sit':
            #str = "Hello Sir I am saluting you"
            #alexa.say(str)
            #time.sleep(1.0)
            num = '1'  # Taking input from user
            write_read(num)
            #time.sleep(2.0)
            #str = "Thank you sir"
            #alexa.say(str)
        elif audio_text.lower() == 'listen':
            str = "yes dear!"
            print(str)
            alexa.say(str)
            alexa.runAndWait()
            while True:
                audio_text = input('> ')
                #audio_text = recognize_speech()
                if audio_text == "bye":
                    response = chatbot_response(audio_text)
                    print(response)
                    alexa.say(response)
                    alexa.runAndWait()
                    break;
                response = chatbot_response(audio_text)
                print(response)
                alexa.say(response)
                alexa.runAndWait()
