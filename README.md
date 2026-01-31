# ASL Sign Language Translator (MediaPipe Landmarks)

Projekt rozpoznawania liter alfabetu języka migowego (ASL) w czasie rzeczywistym z użyciem kamery internetowej, **MediaPipe Hands** oraz **sieci neuronowej opartej o landmarki dłoni**, a nie obrazy.

---

## Funkcjonalności

- rozpoznawanie liter ASL w czasie rzeczywistym (kamera)
- wysoka stabilność predykcji
- odporność na:
  - tło
  - oświetlenie
  - położenie dłoni w kadrze
- poprawne rozróżnianie podobnych liter (`M`, `N`, `V`, `W`, `P`, `Z`)
- brak zależności od orientacji obrazu (brak flip / rotate)
- buforowanie predykcji (wygładzenie)

---

## Jak to działa (obecne rozwiązanie)

Projekt wykorzystuje **MediaPipe Hands**, który wykrywa **21 punktów anatomicznych dłoni**.  
Każdy punkt posiada współrzędne:

- `x`
- `y`
- `z`

Daje to **63 cechy numeryczne**, które są:
- normalizowane względem nadgarstka
- niezależne od skali i położenia dłoni
- wrażliwe na **układ i orientację palców**

Model uczy się **geometrii dłoni**, a nie pikseli obrazu.

---

## Wcześniejsze podejścia (dlaczego NIE działały)

### 1. Klasyfikacja na podstawie obrazów (CNN)

Pierwsze wersje projektu wykorzystywały klasyczną sieć CNN uczoną na obrazach dłoni.

**Problemy:**
- różne proporcje dłoni → resize zniekształcał obraz
- tło z datasetu ≠ tło z kamery
- oświetlenie i cień wpływały na predykcję
- MediaPipe przycinał dłoń inaczej niż dataset
- podobne litery (`M/N/V/W/P`) miały niemal identyczny obraz

Model uczył się **artefaktów obrazu**, nie gestu.

---

### 2. Augmentacje (obrót, flip)

Dodanie augmentacji (rotacje, odbicia):

**Problemy:**
- orientacja dłoni w ASL MA znaczenie
- obrót zmieniał znaczenie znaku
- model uczył się sprzecznych przykładów

Accuracy rosło, ale predykcje LIVE były błędne.

---

### 3. Accuracy ≠ jakość

Model osiągał ~95% accuracy na zbiorze walidacyjnym, ale:

- w LIVE większość znaków była klasyfikowana jako `P`
- predykcja „klejała się” do jednej klasy

Klasy były niezbalansowane i zbyt podobne wizualnie.

---

## Dlaczego landmarki rozwiązują problem

|      Problem      | Obrazy | Landmarki |
|-------------------|--------|----------|
| tło                |  ❌  |    ✅    |
| światło            |  ❌  |    ✅    |
| proporcje          |  ❌  |    ✅    |
| orientacja palców  |  ⚠️  |    ✅    |
| podobne litery     |  ❌  |    ✅    |

Landmarki opisują **strukturę dłoni**, nie jej wygląd.

---

## 📁 Struktura projektu

```
.
├── train_model.py
├── live_predict.py
├── model_landmarks.h5
├── classes.npy
├── label_encoder.pkl
└── asl_alphabet_train/
```

---

## 🔧 Wymagania

- Python 3.9 – 3.11
- Kamera internetowa

### Biblioteki

```
pip install opencv-python mediapipe tensorflow scikit-learn numpy
```

---

## Jak uruchomić projekt

### Pobranie repozytorium

```
git clone https://github.com/TWOJE_REPOZYTORIUM/asl-translator.git
cd asl-translator
```

### Wypakuj pliki

Wypakuj zapakowane pliki

### Trenowanie modelu

```
python train_model.py
```

### Uruchomienie LIVE

```
python live_predict.py
```

---

## 🎯 Wskazówki

- pokazuj jedną dłoń
- nie zakrywaj palców
- wykonuj znaki zgodnie z ASL
- kamera nie jest odwracana ani flipowana

