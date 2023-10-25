import requests
import base64
import numpy as np
import io

def simesaba_voice(text):
    url = "https://simesaba-voice-zrc662eobq-an.a.run.app/"
    headers = {
        'Content-Type': 'text/plain; charset=utf-8'  # UTF-8を指定
    }

    response = requests.post(url, data=text.encode('utf-8'), headers=headers)
    data = response.json()

    audio = base64.b64decode(data['audio_base64'])

    with io.BytesIO(audio) as buffer:
        audio = np.load(buffer, allow_pickle=True)

    return audio

def main():
    None

if __name__ =="__main__":
    main()