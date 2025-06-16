#!/bin/bash
# bash setup_wronai_data.sh

# Utwórz środowisko wirtualne
python -m venv venv
source venv/bin/activate

# Zainstaluj zależności
pip install --upgrade pip
pip install datasets huggingface_hub
pip install beautifulsoup4 requests lxml
pip install pandas pyarrow orjson
pip install tqdm rich
pip install langdetect fasttext
pip install datasketch text-dedup
pip install zstandard ftfy regex
pip install unidecode

# Pobierz model identyfikacji języka FastText
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

echo "Środowisko WronAI gotowe do użycia!"