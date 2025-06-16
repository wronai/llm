#!/bin/bash
# WronAI Data Collection Setup Script
# Usage: bash setup.sh

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Rozpoczynanie konfiguracji środowiska WronAI..."

# Sprawdź, czy Python jest zainstalowany
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 nie został znaleziony. Zainstaluj Python 3.8 lub nowszy."
    exit 1
fi

# Utwórz środowisko wirtualne
echo "🔧 Tworzenie środowiska wirtualnego..."
python3 -m venv venv
source venv/bin/activate

# Zainstaluj zależności
echo "📦 Instalowanie zależności..."
pip install --upgrade pip
pip install -r requirements.txt

# Pobierz model identyfikacji języka FastText
echo "📥 Pobieranie modelu FastText dla identyfikacji języka..."
if [ ! -f "lid.176.bin" ]; then
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
else
    echo "Model FastText już istnieje."
fi

# Sprawdź, czy wszystkie zależności zostały zainstalowane
echo "✅ Weryfikacja instalacji..."
python3 -c "import datasets, fasttext, datasketch, ftfy, regex, tqdm, pandas, requests" || {
    echo "❌ Nie wszystkie zależności zostały poprawnie zainstalowane."
    exit 1
}

echo "✨ Środowisko WronAI gotowe do użycia!"
echo "Aby aktywować środowisko, użyj: source venv/bin/activate"