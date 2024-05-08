# Hebrew NER: Part II Project, How to Run The App

## Requirements

- A working `Python 3.8` installation
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

To test that the installation worked, start a Python interpreter by typing `Python`

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
unzip trained_models.zip
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

### Running just YAP using Docker

Use the command

```zsh
docker-compose start yap
```
