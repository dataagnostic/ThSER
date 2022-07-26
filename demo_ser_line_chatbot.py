from flask import Flask, request
from linebot import (LineBotApi)
from linebot.exceptions import (InvalidSignatureError)
import librosa
import os
from dotenv import load_dotenv
from joblib import load
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import json
import requests
import pydub


app = Flask(__name__)
load_dotenv() 
channel_access_token = os.getenv('CHANNEL_ACCESS_TOKEN')
line_bot_api = LineBotApi(channel_access_token)
sc = load('model/sc.joblib')
model = load_model('model/model_ser_nrm_07.hdf5', compile=True)
emotions=['Neutral 😐', 'Angry 😡', 'Happy 😄', 'Sad 😭', 'Frustrated 😒']
def extract_feature(file_name): 
    X, sample_rate = librosa.load(file_name)
    result=np.array([])
    stft = np.abs(librosa.stft(X))
    
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    result=np.hstack((result, mfccs))
    
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    result=np.hstack((result, chroma))
    
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    result=np.hstack((result, mel))
    
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    result=np.hstack((result, contrast))
    
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    result=np.hstack((result, tonnetz))
    return result


@app.route("/")
def hello():
    return "Hello World ThaiSER!"

@app.route("/webhook", methods=['POST', 'GET'])
def webhook():
    payload = request.json
    print(payload)
    if len(payload['events']) >0:
        reply_token = payload['events'][0]['replyToken']
        try:
            result = ''
            if payload['events'][0]['message']['type'] == 'audio':
                print('Get audio :)')
                message_id  = payload['events'][0]['message']['id']
                message_content = line_bot_api.get_message_content(message_id)
                temp = 'data/temp_'+reply_token
                audio_file = 'data/audio_'+reply_token+'.wav'
                with open(temp, 'wb') as fd:
                    for chunk in message_content.iter_content():
                        fd.write(chunk)
                pydub.AudioSegment.from_file(temp).set_frame_rate(44100).export(audio_file, format='wav')
                test = extract_feature(audio_file)
                prediction = model.predict(sc.transform([test]).tolist())
                maxindex = int(np.argmax(prediction))   
                print('!!! Result Emotion: '+emotions[maxindex])
                result = emotions[maxindex]
                if os.path.exists(temp):
                    os.remove(temp)
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            else:
                result = 'รองรับแค่ Audio ครับ'
            ReplyMessage(reply_token,result, channel_access_token)
        except: 
            ReplyMessage(reply_token,'ERROR', channel_access_token)
    return 'OK',200
def ReplyMessage(Reply_token, TextMessage, Line_Acees_Token):
    LINE_API = 'https://api.line.me/v2/bot/message/reply'
    Authorization = 'Bearer {}'.format(Line_Acees_Token)
    print(Authorization)
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization':Authorization
    }
    data = {
        "replyToken":Reply_token,
        "messages":[{
            "type":"text",
            "text":TextMessage
        }]
    }
    data = json.dumps(data)
    r = requests.post(LINE_API, headers=headers, data=data)
    return 200
    


if __name__ == "__main__":
    app.run()