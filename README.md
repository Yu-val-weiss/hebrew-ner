# Hebrew NER: Part II Project, How to Run The App

## Requirements

- A working `Python 3.8` installation (the app might be forwards compatible, but not tested)
- The Docker daemon running
- `docker-compose` installed

## .env file

Users should create a `.env` file with the following parameters that should be filled accordingly. Defaults are shown where applicable. `ABSOLUTE_PATH_HEBREW_NER` is the absolute path to the current directory.

```config
ABSOLUTE_PATH_HEBREW_NER=
YAP_HOST=127.0.0.1
YAP_PORT=8000
```

## Set up Python `venv`

To create the virtual environment

```zsh
python3.8 -m venv venv
```

To activate it in the terminal

```zsh
source venv/bin/activate
```

To install the library requirements

```zsh
pip install -r requirements.txt
```

All further instructions assume that the virtual environment is active.

### Deactivating the `venv`

Simply execute the following

```zsh
deactivate
```

## Installing fastText

It is now necessary to install the fastText binary.

We will place it in a folder called fastText. 

```zsh
mkdir fasttext && cd fasttext
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.he.zip
unzip wiki.he.zip
```

### Testing the installation

To test that the installation worked, start a Python interpreter

Now run the following

```Python
import fasttext

ft = fasttext.load_model('fasttext/wiki.he.bin')
```

Note: the following warning may appear.

```zsh
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
```

This can be safely ignored.

## Installing the models

Now my extension models must be downloaded.

Run the following to download them (from Figshare)

```zsh
wget -O trained_models.zip https://figshare.com/ndownloader/articles/25773039?private_link=ab195c4231927a669e0e
```

Once this has downloaded run

```zsh
mkdir -p ncrf_hpc_configs/transformer && unzip trained_models.zip -d ncrf_hpc_configs/transformer
```

## Running

Use the following to run the app alongisde YAP using Docker compose

```zsh
docker compose up
```

### Force rebuild

Use the following to force-rebuild

```zsh
docker compose build --no-cache
```

then run

```zsh
docker compose up
```

### Running the app natively

Use the command

```zsh
python ner_app.py
```

If making changes to the app, can enable hot reload by running

```zsh
python ner_app.py --reload
```

### Running just YAP using Docker

Use the command (this will run it in the background)

```zsh
docker-compose start yap
```

To monitor the process of it loading can run the following

```zsh
docker ps
```

And find the container whose image is called 'hebrew-ner_yap', and copy the container id. Now run

```zsh
docker logs *CONTAINER_ID*
```

If it says `All models loaded!' then Yap is up and running (if not just wait a little longer).

## Making a request

Once the app is running a request can be made.

Here is an example one that can be run from the command line using `curl`.

```zsh
curl --request POST \
  --url http://127.0.0.1:5000/predict \
  --header 'Content-Type: application/json' \
  --data '{
        "text": "גנו גידל דגן בגן.",
        "model": "token_single"
}'
```

## Available endpoints

The app has two available endpoints `/tokenize` and `/predict`, both of which are `POST` requests. 

### `/tokenize`

Used to tokenize a string input into sentences. Requires a JSON in the form:

```json
{
    "text": "string"
}
```

### `/predict`

```json
{
    "text": "text",
    "model": "token_single"
}
```

### Documentation

Further documentation (auto generated by FastAPI) is available at <http://127.0.0.1:5000/docs>, or the new host/port if they are modified.
