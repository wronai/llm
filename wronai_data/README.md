# Kompletny plan pozyskiwania danych dla WronAI

Polski model językowy WronAI może zostać zbudowany na solidnej podstawie dostępnych polskich korpusów, które łącznie oferują ponad 150GB wysokiej jakości tekstów. Plan obejmuje strategiczne podejście do pobierania danych z priorytetyzacją jakości, praktyczne narzędzia implementacyjne oraz szacunkowy koszt infrastruktury ~50 zł miesięcznie.

## Najlepsze polskie korpusy językowe i datasety

### Korpusy open-source najwyższej jakości

**OSCAR Polish** stanowi fundament - 49-109GB polskiego tekstu z internetu po deduplikacji i filtrowaniu jakości. **OSCAR 23.01** oferuje najnowsze dane z ulepszonym filtrowaniem, dostępne przez HuggingFace pod licencją CC0. Dostęp wymaga rejestracji, ale dane są kompletnie otwarte dla szkolenia AI.

**Wikipedia Polish** dostarcza 4GB+ wysokiej jakości treści encyklopedycznych z miesięcznymi aktualizacjami. Dostępna przez oficjalne dumpy Wikimedia pod licencjami Creative Commons, oferuje czyste, zredagowane treści z cytowaniami.

**Common Crawl CC-100 Polish** zawiera 12GB skompresowanego tekstu internetowego po filtrowaniu, reprezentując szeroką różnorodność źródeł internetowych pod licencją Common Crawl Terms of Use.

### Zasoby akademickie instytucjonalne

**Narodowy Korpus Języka Polskiego (NKJP)** stanowi referencyjny korpus o rozmiarze 1.5 miliarda słów, z 300 milionami w korpusie zbalansowanym i 1 milionem w korpusie ręcznie anotowanym. Zawiera literaturę, gazety, czasopisma, rozmowy i teksty internetowe pod licencją CC-BY dla 1M podkorpusu.

**CLARIN-PL** udostępnia multiple zasoby językowe w tym Spokes (2.3M słów rozmów), Paralela (teksty polsko-angielskie), ParlaMint Polish (1,010 godzin przemówień parlamentarnych) i datasets do desambiguacji znaczenia słów.

**Korpus Współczesnego Języka Polskiego (KWJP)** to reprezentatywny korpus polszczyzny lat 2010. z anotacjami morfologicznymi i syntaktycznymi dla zastosowań badawczych.

### Zbiory literackie i kulturowe

**Wolne Lektury** oferuje 6,714+ dzieł literackich w tym obowiązkowe lektury szkolne przez otwarte API w formatach XML, HTML, PDF, EPUB, MOBI, FB2, TXT. Licencja public domain i Creative Commons umożliwia swobodne wykorzystanie profesjonalnie zredagowanych tekstów z przypisami.

### Publikacje techniczne i naukowe

**Polish Library of Science Corpus (PLSC)** zawiera 100K+ rekordów naukowych z bibliotekanauki.pl obejmując 8 dziedzin i 47 dyscyplin. **SpeakLeash Polish Corpus** udostępnia 90+ miliardów tokenów polskiego tekstu kuracyjnego dla projektów LM.

### Conversational datasets

**SpokesBiz Corpus** zawiera 650+ godzin nagrań rozmów biznesowych z wysokiej jakości transkrypcją i anotacjami. **DiaBiz Corpus** oferuje anotowane rozmowy call center. **Polish Parliamentary Corpus** dostarcza 3,000+ plików XML oficjalnych transkrypcji parlamentarnych.

## Struktura folderów i organizacja danych

### Hierarchiczna struktura jakościowa

```
wronai_data/
├── raw_data/
│   ├── high_quality/          # Wikipedia, książki, papers
│   │   ├── wikipedia_pl/
│   │   ├── wolne_lektury/
│   │   └── academic_papers/
│   ├── medium_quality/        # Wiadomości, fora, Q&A
│   │   ├── news_portals/
│   │   ├── forums_discussions/
│   │   └── qa_platforms/
│   └── low_quality/           # Media społecznościowe, komentarze
│       ├── social_media/
│       ├── web_comments/
│       └── unfiltered_crawl/
├── processed_data/
│   ├── filtered/              # Po filtrowaniu jakości
│   ├── deduplicated/         # Po deduplikacji
│   ├── tokenized/            # Po tokenizacji
│   └── balanced/             # Po balansowaniu domen
└── splits/
    ├── train/                # 80% danych
    ├── validation/           # 10% danych
    └── test/                # 10% danych
```

### Konwencje nazewnictwa i metadane

**Format nazw plików:**
```
{źródło}_{jakość}_{domena}_{wersja}_{split}_{shard}.{format}
```

**Przykłady:**
- `wikipedia_high_general_v2.1_train_00001.jsonl`
- `oscar_medium_web_v23.01_val_00042.parquet`
- `nkjp_high_literature_v1.5_test_00001.jsonl`


## Algorytm pobierania w kolejności priorytetowej

### Strategia Quality-First z balansowaniem domen

**Faza 1: Fundament wysokiej jakości (15GB)**
1. **Wikipedia Polish** (4GB) - pobierz najnowszy dump, wyodrębnij czyste teksty
2. **Wolne Lektury** (3GB) - użyj API do pobrania wszystkich dostępnych dzieł
3. **NKJP zbalansowany** (5GB) - pobierz dostępne podkorpusy
4. **CLARIN-PL akademicki** (2GB) - zbierz wysokiej jakości zbiory specjalistyczne
5. **Parliamentary Corpus** (1GB) - oficjalne transkrypcje

**Faza 2: Główny korpus internetowy (30GB)**
1. **OSCAR Polish deduplicated** (30GB) - pobierz najnowszą wersję 23.01
2. Zastosuj dodatkowe filtrowanie jakości (perplexity < 500)
3. Wykonaj deduplikację na poziomie dokumentów i zdań

**Faza 3: Uzupełnienie i balansowanie (5GB)**
1. **Common Crawl CC-100** - dla zwiększenia różnorodności internetowej
2. **Conversational data** - SpokesBiz i DiaBiz dla języka mówionego
3. **Specialized domains** - techniczne, naukowe, biznesowe teksty

### Strategie deduplikacji i czyszczenia

**Multi-poziomowa deduplikacja:**
- **Dokument-level**: MinHash LSH z progiem 0.8 (usuwa 70-80% duplikatów)
- **Zdanie-level**: Usuwanie powtarzających się fraz
- **Line-level**: Deduplikacja w ramach losowych bucket'ów

**Pipeline czyszczenia:**
1. **Encoding normalization**: Konwersja do UTF-8, naprawa ftfy
2. **Language detection**: FastText z 95% confidence threshold
3. **Content filtering**: Usuwanie HTML, boilerplate, offensive content
4. **Quality scoring**: Perplexity-based + heuristic features
5. **Length filtering**: Min 100 znaków, max 50K znaków na dokument

### Balansowanie domen językowych

**Docelowy rozkład 50GB datasetu:**
- **OSCAR (web content)**: 35GB (70%) - różnorodność internetowa
- **Wikipedia**: 4GB (8%) - wysoka jakość, encyklopedyczność
- **NKJP**: 5GB (10%) - zbalansowana reprezentacja gatunków
- **Wolne Lektury**: 3GB (6%) - język literacki, kultura
- **CLARIN-PL**: 2GB (4%) - domeny specjalistyczne
- **Parliamentary**: 1GB (2%) - język formalny, polityczny

## Konkretny skrypt do automatycznego pobierania


### Skrypt Docker dla reprodukowalnego środowiska


## Szacunkowe rozmiary i wymagania

### Rozmiary źródeł danych

**Źródła pierwszego priorytetu (15GB):**
- **Wikipedia Polish**: 4GB nieskompresowany tekst
- **Wolne Lektury**: 3GB literatury wysokiej jakości  
- **NKJP zbalansowany**: 5GB zbalansowanego korpusu
- **CLARIN-PL**: 2GB zbiorów akademickich
- **Parliamentary Corpus**: 1GB oficjalnych transkrypcji

**Główny korpus (30GB):**
- **OSCAR Polish deduplicated**: 49GB dostępny, 30GB po dodatkowym filtrowaniu
- **Common Crawl CC-100**: 12GB skompresowany, uzupełnienie różnorodności

**Uzupełnienie (5GB):**
- **Conversational datasets**: 1GB rozmów i dialogów
- **Specialized domains**: 4GB technicznych i naukowych tekstów

### Wymagania infrastrukturalne

**Przestrzeń dyskowa:**
- **Dane surowe**: 200GB (4x rozmiar docelowy na przetwarzanie)
- **Cache i temp**: 100GB dla przetwarzania tymczasowego
- **Backupy**: 150GB dla redundancji i wersjonowania
- **Łącznie**: 450GB zalecanej przestrzeni NVMe SSD

**Pamięć RAM:**
- **Minimum**: 32GB dla podstawowego przetwarzania
- **Zalecane**: 64GB dla efektywnego przetwarzania równoległego
- **Optymalne**: 128GB dla dużych korpusów bez swappingu

**Procesor:**
- **Minimum**: 8 rdzeni dla sekwencyjnego przetwarzania
- **Zalecane**: 16 rdzeni dla przetwarzania równoległego
- **Optymalne**: 32 rdzenie (AMD Ryzen 7000 series lub Intel 13th gen)

**Łącze internetowe:**
- **Minimum**: 100 Mbps dla pobierania większych korpusów
- **Zalecane**: 1 Gbps dla efektywnego pobierania OSCAR (2-4 godziny)
- **Transfer**: ~150GB łącznego pobierania danych

### Szacunkowe koszty (miesięcznie)

**Cloud storage (AWS/Azure/GCP):**
- **50GB dataset**: $1-2/miesiąc storage
- **200GB z przetwarzaniem**: $4-8/miesiąc  
- **Backup i redundancja**: +$2-4/miesiąc

**Obliczenia (processing):**
- **Occasional processing**: $10-20/miesiąc
- **Regular reprocessing**: $30-50/miesiąc

**Łączny szacunkowy koszt:**
- **Mała skala (50GB)**: $15-30/miesiąc (~50-100 zł)
- **Średnia skala (200GB)**: $40-80/miesiąc (~150-300 zł)
- **Duża skala (1TB+)**: $100-200/miesiąc (~400-800 zł)

### Optymalizacja kosztów

**Strategie oszczędności:**
- **Lifecycle policies**: Automatyczne przenoszenie starszych danych do tańszego storage
- **Spot instances**: Użycie tanich instancji do przetwarzania niekrytycznego  
- **Regional optimization**: Utrzymanie danych i obliczeń w tym samym regionie
- **Kompresja**: Efektywna kompresja zarchiwizowanych danych (zstd, lz4)

Ten kompleksowy plan zapewnia solidną podstawę do budowy polskiego modelu językowego WronAI z możliwością skalowania w zależności od potrzeb i dostępnych zasobów. Implementacja pipeline'u w Docker'ze gwarantuje reprodukowalność, a szczegółowe instrukcje pozwalają na łatwe dostosowanie do specific requirements projektu.