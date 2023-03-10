import os
import numpy as np
import tensorflow as tf
from keras import layers
from numpy import array
from PIL import Image
from pickle import load
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.layers import add
from keras.applications.inception_v3 import InceptionV3
import keras.utils as image

from keras.models import Model
from keras import Input
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.utils import pad_sequences
from flask import Flask, render_template, request
from keras.models import load_model

    
def load_descriptions(doc):
    mapping = dict()
    
    for line in doc.split('\n'):
        
        tokens = line.split()
        if len(line) < 2:
            continue
           
        image_id, image_desc = tokens[0], tokens[1:]
        
        image_id = image_id.split('.')[0]
        
        image_desc = ' '.join(image_desc)
        
        if image_id not in mapping:
            mapping[image_id] = list()
          
        mapping[image_id].append(image_desc)
    return mapping


descriptions = load_descriptions("descriptions.txt")
def to_vocabulary(descriptions):
    
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


vocabulary = to_vocabulary(descriptions)

def preprocess(image_path):
   
    img = image.load_img(image_path, target_size=(299, 299))
    
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x)
    return x

model = InceptionV3(weights='imagenet')
model.save("inception.h5")

model=load_model('inception.h5')
model_new = Model(model.input, model.layers[-2].output)

def encode(image):
    image = preprocess(image) 
    fea_vec = model_new.predict(image) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) 
    return fea_vec

with open('vocab.pkl','rb') as file:
    vocab = load(file)

ixtoword = {} 
wordtoix = {} 

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1
"""
"""
vocab_size = len(ixtoword) + 1 
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc


max_length =34
embedding_dim = 200

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(1,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.load_weights('model.h5')

def imageSearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final




print("Image with Caption:",)
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():

    

    img = request.files['file1']

    img.save('static/file.jpg')


    
    image = encode('static/file.jpg')
    image = image.reshape((1,2048))
    res = imageSearch(image)
    
    
    return render_template('after.html', data=res)

if __name__ == "__main__":
    app.run(debug=True)