import requests
import base64
import numpy as np

def simesaba_voice(text):
    url = "http://<YOUR_FLASK_APP_IP>:8080/simesaba-voice"
    headers = {
        'Content-Type': 'text/plain'
    }

    response = requests.post(url, data=text, headers=headers)
    data = response.json()

    audio = base64.b64decode(data['audio_base64'])

    audio = np.load(audio)

    return audio

def main():
    None

if __name__ =="__main__":
    main()