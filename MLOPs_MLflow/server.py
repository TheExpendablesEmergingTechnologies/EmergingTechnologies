import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
#from fastai.data.all import *
#from fastai.optimizer import *
#from fastai.callback.core import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from hierarchical_model import *
import pathlib
from nltk.corpus import stopwords
import string
from random import randrange

export_file_url = 'https://www.googleapis.com/drive/v3/files/1-RcEJYzZ2g3k51QLZZ-6RzDXbdJL6BRJ?alt=media&key=AIzaSyCiLpxA2j60lVqzx-kehWiISSn_Lsgk0CE'
export_file_name = 'epochs:2-lstm_dim:100-lstm_layers:1-devacc:0.444.pth'

classes = ['1', '2', '3', '4']
path = pathlib.Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        model = Hierarchical_Model(vocab,
                                   new_matrix,
                                   tag_to_ix,
                                   num_labels,
                                   task2label2id,
                                   embedding_dim,
                                   hidden_dim,
                                   1,
                                   train_embeddings=train_embeddings)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

#model = torch.load(path / export_file_name)
import torch
import mlflow
import mlflow.pytorch
# Set values
model_path_dir = "/Users/vigneshkumarthangarajan/Documents/special/project/multitask_negation_for_sa/models/mlruns/1/56733ffe135d416fa98b709fbe0856a5/artifacts/model/"
run_id = "96771d893a5e46159d9f3b49bf9013e2"
model = mlflow.pytorch.load_model(model_path_dir)

model.eval()
pdict = pickle.load(open('app/params.pkl', 'rb'))
vocab = SetVocab(pdict[0])

datadir = "../data/datasets/en/SST/fine"
sst = SSTDataset(vocab, False, datadir)

@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

def predict_sentiment(review, vocab, model):
    datadir = "../data/datasets/en/SST/fine"
    sst = SSTDataset(vocab, False, datadir)
    data_file = open(datadir+'/single.txt', 'w')
    data_file.write("1\t"+review)
    maintask_dev_iter = sst.get_split("single")
    model.eval()
    pdict = pickle.load(open('app/params.pkl', 'rb'))
    vocab = SetVocab(pdict[0])
    f1, acc, preds, ys = model.eval_sent(maintask_dev_iter,
                                     batch_size=1)
    maintask_dev_iter = None
    print(preds)
    return preds[0]

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    text_data = await request.form()
    print(text_data)
    text_bytes = text_data['file']
    data = torch.LongTensor(vocab.ws2ids(text_bytes))
    model.eval()
    preds = predict_sentiment(text_bytes, vocab, model)
    mapdict = {0 : "strong negative", 1 : "negative", 2 : "neutral", 3 : "positive", 4 : "strong positive"}
    return JSONResponse({'result': str(mapdict[preds])})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5001, log_level="info")
