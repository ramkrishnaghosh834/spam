from flask import Flask,request,jsonify
import pickle
import numpy as np
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer






def process(text):
    # lowercase it
    text = text.lower()
    # remove punctuation
    text = ''.join([t for t in text if t not in string.punctuation])
    # remove stopwords
    text = [t for t in text.split() if t not in stopwords.words('english')]
    # stemming
    st = Stemmer()
    text = [st.stem(t) for t in text]
    # return token list
    return text


model=pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')

def home():
    return "hello"
@app.route('/predict',methods=['POST'])
def predict():
    email=request.form.get('email')
    
    result=model.predict([email])
   
        # return jsonify({'Email':'True'})
    return jsonify({'Email':str(result)})
if __name__ == '__main__':
    app.run(debug=True)
