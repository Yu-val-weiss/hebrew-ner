# Hebrew NER: Part II Project

## Forked YAP Parser

### New Command

`yap/heb/joint/lattice` takes in a JSON object with the following structure

```JSON
{"amb_lattice": "0\t1\tב\tב\tPREPOSITION\tPREPOSITION\t_\t1\n0\t3\tבגן\tבגן\tNNP\tNNP\tgen=M|num=S\t1\n0\t3\tבגן\tבגן\tNN\tNN\tgen=M|num=P|num=S\t1\n0\t3\tבגן\tבגן\tNN\tNN\tgen=M|num=S\t1\n0\t3\tבגן\tבגן\tNNP\tNNP\tgen=F|num=S\t1\n0\t3\tבגן\tבגן\tNNP\tNNP\tgen=F|gen=M|num=S\t1\n0\t3\tבגן\tבגן\tNNP\tNNP\t_\t1\n0\t3\tבגן\tבגן\tNN\tNN\tgen=M|num=P\t1\n0\t3\tבגן\tבגן\tNN\tNN\tgen=F|num=S\t1\n0\t3\tבגן\tבגן\tNN\tNN\tgen=F|num=P\t1\n1\t3\tגן\tגן\tNN\tNN\tgen=M|num=S\t1\n1\t3\tגן\tגן\tNNT\tNNT\tgen=M|num=S\t1\n1\t2\tה\tה\tDEF\tDEF\t_\t1\n2\t3\tגן\tגן\tNNT\tNNT\tgen=M|num=S\t1\n2\t3\tגן\tגן\tNN\tNN\tgen=M|num=S\t1\n\n"}
```

#### Notes

- There **MUST** be two whitespace characters at the end
- The MA step returns it as `{"ma_lattice" : ""}`
- Go version must be 1.15

### Rebuilding

If a new version has been pushed to GitHub run the following:

```zsh
docker compose build --no-cache
```

```zsh
docker compose up
```

### To use Colab

First do:

```sh
ngrok start yap 
```

Note: password is on google colab and in `/Users/yuval/.config/ngrok/ngrok.yml`

```Python
import requests
from requests.auth import HTTPBasicAuth

url = NGROKURL
json_data = {"amb_lattice": "...."}
username = "user"
password = "pass"

response = requests.get(url, json=json_data, auth=HTTPBasicAuth(username, password))

if response.status_code == 200:
    # Request was successful, and the response is stored in 'response.text'
    print("Response:")
    print(response.json())
else:
    print(f"Request failed with status code: {response.status_code}")
```

## FastText Embeddings

Bin allows for outoftext words to be predicted

### Download bin using Python

```Py
import fasttext.util
fasttext.util.download_model('he', if_exists='ignore')
```

```zsh
mkdir -p fasttext && sudo mv cc.* fasttext/
```

### Load from bin

```Python
import fasttext
import fasttext.util

ft = fasttext.load_model('fasttext/cc.he.300.bin')
ft.get_dimension() # 300
fasttext.util.reduce_model(ft, 100)
ft.get_dimension() # 100
embed = ft.get_word_vector('שלום')
```

### Load from text

```Py
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data
```

## Moving data to HPC

```zsh
make xxx-archive
make upload-xxx # where xxx can be code fasttext archive
```

Can clean with

```zsh
make clean
```

### Now on HPC

```bash
tar -xvf archive.tar.gz
```

## Seeds

### List of seeds

*Token single*: 46
*Token multi*: 50
*Morph*: 52

### Updating seed

```bash
perl -pi -e 's/seed_num = [0-9]*/seed_num = 46/' /home/yw580/hebrew-ner/ncrf_main.py
```
