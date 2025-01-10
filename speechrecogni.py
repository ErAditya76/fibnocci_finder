try:
    import speech_recognition as sr
except ImportError:
    print("Please install speech_recognition using: "
          "pip install SpeechRecognition")
    print("You may also need to install PyAudio using: pip install PyAudio")
    exit(1)
import pyttsx3
import datetime
import webbrowser

# Initialize the speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

def speak(text):
    """Function to make the assistant speak."""
    engine.say(text)
    engine.runAndWait()

def take_command():
    """Capture voice input."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.pause_threshold = 1  # Wait for 1 second before considering the speech is finished
        try:
            audio = recognizer.listen(source)
            print("Recognizing...")
            # Recognize speech using Google Web Speech API
command = recognizer.recognize_google(audio, language='en-US')
            print(f"You said: {command}")
        except Exception as e:
            print("Could not understand your voice. Please try again!")
            return None
    return command.lower()

def assistant():
    """Voice assistant main logic."""
    speak("Hello, how can I assist you?")
    while True:
        query = take_command()
        if query is None:
            continue
        
        # Perform actions based on commands
        if "time" in query:
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            speak(f"The time is {current_time}.")
        elif "open google" in query:
            webbrowser.open("https://www.google.com")
            speak("Opening google")
        elif "quit" in query or "exit" in query:
            speak("Goodbye!")
            break
        else:
            speak("I am still learning and cannot handle that command yet.")

# Run the voice assistant
if __name__ == "__main__":
    assistant()
