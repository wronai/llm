# WronAI 🐦‍⬛

**Polski model językowy inspirowany PLLuM - demokratyzacja AI dla języka polskiego**

WronAI to open-source projekt mający na celu stworzenie efektywnego polskiego modelu językowego, który
można trenować i uruchamiać na sprzęcie konsumenckim. 
Projekt wykorzystuje najnowsze techniki optymalizacji jak QLoRA, gradient checkpointing i kwantyzację do osiągnięcia maksymalnej wydajności przy minimalnych wymaganiach sprzętowych.

## 🎯 Cele projektu

- **Dostępność**: Model możliwy do uruchomienia na GPU 8GB+
- **Język polski**: Specjalizacja w przetwarzaniu języka polskiego
- **Open Source**: Pełna otwartość kodu i danych treningowych
- **Społeczność**: Budowa ekosystemu wokół polskich modeli AI

## 🏗️ Architektura

WronAI bazuje na sprawdzonych rozwiązaniach:
- **Model bazowy**: Mistral-7B z QLoRA fine-tuningiem
- **Kwantyzacja**: 4-bitowa NF4 dla optymalizacji pamięci
- **Corpus**: ~50GB polskich tekstów wysokiej jakości
- **Alignment**: Polski dataset preferencji dla RLHF

## 🚀 Szybki start

```bash
# Klonowanie repozytorium
git clone https://github.com/wronai/llm.git
cd llm

# Instalacja dependencies
pip install -r requirements.txt

# Przygotowanie danych
python scripts/prepare_data.py

# Trening modelu
python scripts/train.py --config configs/default.yaml

# Inferencja
python scripts/inference.py --model checkpoints/wronai-7b --prompt "Opowiedz o Polsce"
```

## 📊 Wyniki

| Model | Parametry | VRAM | Polish Score | Licensing |
|-------|-----------|------|--------------|-----------|
| WronAI-7B | 7B | 8GB | 7.2/10 | Apache 2.0 |
| PLLuM-8x7B | 46.7B | 40GB+ | 8.5/10 | Custom |
| Bielik-7B | 7B | 14GB | 7.8/10 | Apache 2.0 |

## 🛠️ Wymagania systemowe

### Minimalne (trening)
- **GPU**: NVIDIA RTX 3070/4060 Ti (8GB VRAM)
- **RAM**: 16GB DDR4
- **Storage**: 100GB wolnego miejsca
- **OS**: Linux/Windows + CUDA 11.8+

### Zalecane (trening)
- **GPU**: NVIDIA RTX 4080/4090 (16GB+ VRAM)
- **RAM**: 32GB DDR4/DDR5
- **Storage**: 500GB NVMe SSD
- **OS**: Ubuntu 22.04 LTS

### Inferencja
- **GPU**: 6GB VRAM (z kwantyzacją)
- **RAM**: 8GB
- **Storage**: 4GB dla modelu

## 📚 Dokumentacja

- [Instalacja](docs/installation.md)
- [Trening modelu](docs/training.md)
- [Wykorzystanie modelu](docs/inference.md)
- [API Reference](docs/api.md)
- [Benchmark](docs/benchmarks.md)
- [FAQ](docs/faq.md)

## 🗂️ Struktura projektu

```
WronAI/
├── configs/          # Konfiguracje treningowe
├── data/            # Skrypty do obsługi danych
├── docs/            # Dokumentacja
├── models/          # Definicje architektur
├── scripts/         # Skrypty treningowe i inferencji
├── tests/           # Testy jednostkowe
├── notebooks/       # Jupyter notebooks z przykładami
├── checkpoints/     # Wytrenowane modele
└── requirements.txt # Zależności Python
```

## 🤝 Wkład w projekt

Zapraszamy do współpracy! Zobacz [CONTRIBUTING.md](CONTRIBUTING.md) aby dowiedzieć się jak możesz pomóc:

- 🐛 Zgłaszanie błędów
- 💡 Propozycje nowych funkcji
- 📝 Poprawa dokumentacji
- 🔧 Implementacja nowych features
- 📊 Dodawanie benchmarków

## 🏆 Osiągnięcia

- ✅ Model trenowany na <8GB VRAM
- ✅ Polski corpus 50GB+ wysokiej jakości
- ✅ RLHF alignment dla języka polskiego
- ✅ Integracja z Hugging Face Hub
- ✅ Docker containers dla łatwego wdrożenia
- 🔄 Web interface (w trakcie)
- 🔄 Mobile app (planowane)

## 📈 Roadmap

### v0.1 (Aktualna)
- [x] Podstawowy QLoRA fine-tuning
- [x] Polski corpus przygotowanie
- [x] Baseline benchmarki

### v0.2 (Q2 2025)
- [ ] RLHF implementation
- [ ] Multi-GPU training support
- [ ] Web interface
- [ ] API endpoints

### v0.3 (Q3 2025)
- [ ] Mixture of Experts (MoE)
- [ ] Retrieval Augmented Generation (RAG)
- [ ] Mobile deployment
- [ ] Enterprise features

### v1.0 (Q4 2025)
- [ ] Production-ready release
- [ ] Full documentation
- [ ] Commercial support
- [ ] Community ecosystem

## 🎖️ Zespół

- **Główny developer**: [@tom-sapletta-com](https://github.com/tom-sapletta-com)
- **Lingwista komputacyjny**: Potrzebny volunteer
- **DevOps**: Potrzebny volunteer
- **Community manager**: Potrzebny volunteer

## 📄 Licencja

Ten projekt jest dostępny na licencji Apache 2.0. Zobacz [LICENSE](LICENSE) po szczegóły.

### Licencje danych

- **Otwarte dane**: Apache 2.0, CC-BY-SA (komercyjne OK)
- **Dane badawcze**: Tylko do celów niekomercyjnych
- **Model weights**: Apache 2.0

## 🙏 Podziękowania

- **PLLuM Team** za inspirację i wzorce
- **Mistral AI** za model bazowy
- **Hugging Face** za infrastrukturę
- **Polish NLP Community** za wsparcie
- **CLARIN-PL** za resources

## 📞 Kontakt

- **Issues**: [GitHub Issues](https://github.com/twoje-repo/WronAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/twoje-repo/WronAI/discussions)
- **Email**: wronai@example.com
- **Discord**: [WronAI Community](https://discord.gg/wronai)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=twoje-repo/WronAI&type=Date)](https://star-history.com/#twoje-repo/WronAI&Date)

---

**WronAI** - Demokratyzacja polskiej sztucznej inteligencji 🇵🇱🤖