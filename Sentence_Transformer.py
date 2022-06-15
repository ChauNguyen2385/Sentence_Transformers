from flask import request
from flask import abort
from flask import Flask, jsonify
import requests
import pickle
from sentence_transformers import SentenceTransformer, InputExample, losses
import sentence_transformers
from torch.utils.data import DataLoader
#response = requests.post("http://127.0.0.1:5000/hello", data = {'fact':'anh Thanh dep trai'})

# requests.put('http://127.0.0.1:5000/hello', data = {'fact':'anh Thanh dep trai'})

# with open('new_v2_3/pytorch_model.bin','wb') as f:
#     pickle.dump(test_model,f)
model = SentenceTransformer('./new_v2_3')
# model = pickle.load(open('new_v2_3/pytorch_model.bin','rb'))
app = Flask(__name__)

@app.route('/hello',methods = ['POST','GET'])
def add_string():
    # if not request.json or not 'title' in request.json:
    #     abort(400)
    print('1111111')
    data = {
        'fact':'Anh thanh dep trai',
        'id':123    
    }
    return jsonify({'data': data})

@app.route('/info', methods = ['POST','GET'])
def get_info():
    fact = request.args.get('fact', default = 'The Earth is not flat',type = str)
    id = request.args.get('id',default = 1,type = int)
    print(id,': ' + fact)
    return str(id)

@app.route('/predict',methods = ['POST','GET'])
def predict():
    text1 = request.args.get('text1',default = 'recruiter text',type = str)
    text2 = request.args.get('text2',default ='employee text',type = str)
    r1 = model.encode(text1)
    e1 = model.encode(text2)
    return str(sentence_transformers.util.cos_sim(r1,e1).item())
if __name__ == '__main__':
    app.run(debug = True)