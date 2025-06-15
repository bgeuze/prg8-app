# Hand Gesture Quiz App

Dit is de hoofdapplicatie waarmee je een quiz kunt spelen door handgebaren te maken voor de antwoorden A, B, C of D. Je traint eerst je eigen model in de trainer-tool, downloadt het model, en laadt het vervolgens in deze app.

## Features

- Speel een quiz met 10 willekeurige vragen
- Antwoord met je eigen getrainde handgebaren (A/B/C/D)
- Live webcamherkenning met Mediapipe
- Score en feedback na afloop

## Installatie & Gebruik

1. **Clone deze repository:**
   ```bash
   git clone https://github.com/bgeuze/prg8-app.git
   ```

2. **Start een lokale webserver** (anders werkt de webcam niet!):

   - Met Python (werkt op elke Mac/Windows/Linux):
     ```bash
     python3 -m http.server 8000
     ```
     Ga naar [http://localhost:8000](http://localhost:8000) in je browser.

   - Of gebruik de Live Server extensie in VSCode.

3. **Open `index.html` in je browser via de lokale server.**

4. **Laad je getrainde model in** (zie hieronder).

## Model trainen

1. Ga naar de trainer (andere repo en live deployment)
2. Train voor elke letter (A, B, C, D) een uniek handgebaar.
3. Sla het model op via de knop "Save Model".
4. Laad het model in de hoofdapp via "Load Model".

## Benodigdheden

- Een moderne browser (Chrome, Edge, Firefox, Safari)
- Webcam
- Geen installatie van extra packages nodig (alle dependencies via CDN)

## Licentie

MIT
