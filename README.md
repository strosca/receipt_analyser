# Kassenbon-Auswertung

Selbstgehostete Web-App, die Kassenbons und Rechnungen (Fotos und PDFs) einliest, jede einzelne Position
per KI erfasst und daraus Auswertungen erstellt („REWE im November: 34 € Süßigkeiten, 120 € Lebensmittel …“).

**Stand:** Minimalversion: Upload → Texterkennung → KI → Positionen anzeigen.
Kategorien, Korrekturen und Dashboard folgen.

## So funktioniert es

1. **Upload** über die Weboberfläche (Foto, Galerie oder PDF, mehrere auf einmal).
2. **Texterkennung**
   - PDFs mit Textebene (eBons, Amazon-Rechnungen): Der Text wird direkt ausgelesen (`pdftotext`).
   - Fotos und gescannte PDFs: Das Bild wird aufbereitet (Bon finden und entzerren, Schatten entfernen,
     Kontrast, Schärfen, Begradigen) und dann mit Tesseract (Deutsch und Englisch) gelesen.
3. **Positionen per Regel** (schnell, ohne KI): Zeilen der Form „Name … Preis“ werden direkt gelesen,
   Mengenzeilen („2 Stk x 1,29 … 2,58“) dem richtigen Artikel zugeordnet. **Nur wenn die Positionen
   exakt die SUMME ergeben**, wird dieses Ergebnis übernommen. Dann ist der Bon in Sekunden fertig.
4. **KI-Auswertung**, wenn die Regeln nicht aufgehen: über das vorhandene Ollama (`qwen-paperless`). Es läuft immer nur **eine** Anfrage
   gleichzeitig, und eine Anfrage darf bis zu 20 Minuten dauern.
5. **Plausibilitätsprüfung**: Ergeben die Positionen nicht die Gesamtsumme (±5 ct), oder fehlen Datum
   oder Händler, bekommt der Bon den Status **„Prüfen“** mit Begründung.

Status eines Bons: *In Warteschlange → Wird verarbeitet → Fertig / Prüfen / Fehler*

## Installation auf wgserver

Voraussetzung: Docker mit Compose, und das Docker-Netzwerk `ki` mit Ollama läuft bereits.

```bash
cd ~                                   # oder ein anderer Ordner deiner Wahl
git clone <REPO-URL> receipt_analyser
cd receipt_analyser
cp .env.example .env
nano .env                              # mindestens APP_PASSWORD ändern!
docker compose up -d --build
```

Der erste Build dauert einige Minuten. Danach ist die App erreichbar unter:
**<http://192.168.68.118:8095>** (bzw. `http://wgserver:8095`).

### Auf dem iPhone als App

Seite in Safari öffnen → Teilen-Symbol → **„Zum Home-Bildschirm“**. Die App startet dann im Vollbild ohne Adressleiste.

## Update

```bash
cd ~/receipt_analyser
git pull
docker compose up -d --build
docker image prune -f                  # entfernt alte, unbenutzte Images dieses Builds (spart Platz)
```

Die Datenbank wird beim Start automatisch auf den neuesten Stand gebracht.

> Hinweis: `docker image prune -f` löscht nur Images **ohne Namen**, die von keinem Container benutzt
> werden. Laufende Dienste in anderen Stacks sind davon nicht betroffen.

## Backup

Alle Daten liegen in `./data`:

| Pfad | Inhalt |
|---|---|
| `data/db/receipts.sqlite3` | Datenbank (Bons, Positionen) |
| `data/uploads/` | Original-Fotos und PDFs |
| `data/logs/` | Protokolle (nicht sichernswert) |
| `data/secret_key` | Schlüssel für das Login-Cookie |

Sicherung im laufenden Betrieb:

```bash
cd ~/receipt_analyser
sqlite3 data/db/receipts.sqlite3 ".backup 'data/backup.sqlite3'"   # konsistente Kopie der DB
tar czf ~/receipt-backup-$(date +%F).tar.gz data/backup.sqlite3 data/uploads .env
```

(Falls `sqlite3` fehlt: `sudo apt install sqlite3`, oder die App kurz mit `docker compose stop` anhalten und
`data/` einfach kopieren.)

## Konfiguration (`.env`)

| Variable | Standard | Bedeutung |
|---|---|---|
| `APP_PORT` | `8095` | Port im Heimnetz |
| `APP_USERNAME` / `APP_PASSWORD` | `admin` / – | Zugangsdaten (**Passwort muss gesetzt sein**) |
| `SECRET_KEY` | leer | Leer = wird automatisch erzeugt |
| `LLM_PROVIDER` | `ollama` | `ollama` oder `openai` (jede OpenAI-kompatible API) |
| `LLM_BASE_URL` | `http://ollama:11434` | Adresse der KI. Für OpenAI z. B. `https://api.openai.com/v1` |
| `LLM_MODEL` | `qwen-paperless` | Modellname |
| `LLM_API_KEY` | leer | Nur für OpenAI-kompatible APIs |
| `LLM_TIMEOUT` | `1200` | Maximale Wartezeit pro KI-Anfrage in Sekunden |
| `LLM_NUM_CTX` | leer | Kontextlänge für Ollama (siehe unten) |
| `LLM_MAX_INPUT_CHARS` | `6000` | Längere Bontexte werden gekürzt (Bon wird dann „Prüfen“) |
| `OWN_NAMES` | leer | Dein Name und deine Adresse(n), mit `;` getrennt. Werden nie als Händler übernommen. |
| `OCR_LANG` | `deu+eng` | Sprachen für Tesseract |
| `RULE_PARSER` | `true` | Positionen zuerst per Regel lesen; `false` = immer die KI fragen |
| `SUM_TOLERANCE` | `0.05` | Erlaubte Abweichung Positionen ↔ Summe in Euro |

Nach Änderungen an der `.env`: `docker compose up -d` (ohne `--build` reicht).

### Kontextlänge (`LLM_NUM_CTX`)

Das Modell kann nur eine begrenzte Textmenge auf einmal lesen. Leer bedeutet: Es gilt der Standard von Ollama.
Das ist empfohlen, weil Paperless dasselbe Modell nutzt und ein abweichender Wert Ollama zwingt, das Modell neu zu laden.
Wenn bei langen Bons am Ende Positionen fehlen, kannst du `LLM_NUM_CTX=8192` setzen. Das kostet dann etwas mehr RAM und Zeit,
wenn sich die App und Paperless abwechseln.

## Fehlersuche

- **Bon bleibt „In Warteschlange“**: Ein anderer Bon wird gerade verarbeitet, oder Ollama ist durch
  Paperless belegt. Der Status in der Liste zeigt den aktuellen Schritt.
- **Status „Fehler“**: Die Fehlermeldung steht am Bon. Häufig ist Ollama nicht erreichbar.
  Prüfen mit: `docker compose exec receipt-analyser python -c "import httpx; print(httpx.get('http://ollama:11434/api/tags').text)"`
- **Positionen falsch**: In der Bon-Ansicht „Erkannter Rohtext“ aufklappen. Ist schon der Text schlecht,
  liegt es an Foto oder OCR. Ist der Text gut, liegt es an der KI. Danach „Neu verarbeiten“.
- **Logs**: `docker compose logs -f` oder `data/logs/app.log`

## Entwicklung

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     # Windows: .venv\Scripts\pip
.venv/bin/pytest
```

Aufbau:

```
app/
  main.py         App-Start, Login
  web.py          Seiten (Upload, Liste, Detail)
  worker.py       Warteschlange im Hintergrund (ein Bon zur Zeit)
  pipeline.py     Ablauf pro Bon: Text → KI → Prüfung → Speichern
  ocr.py          pdftotext, Bildaufbereitung, Tesseract
  extraction.py   Prompt, Einlesen der KI-Antwort, Plausibilitätsprüfung
  lineparser.py   Positionen per Regel (ohne KI), Ladenname/Filiale/Zahlungsart
  llm/            Austauschbare KI-Anbindung (Ollama, OpenAI-kompatibel)
  db.py           SQLite-Schema und Migrationen
tests/            pytest, inkl. Beispiel-Bontexte von REWE und dm
```

## Rechnungen statt Kassenbons

- **Netto-Rechnungen** (Handwerker, Firmen): Sind die Positionen netto und wird die MwSt erst darunter
  aufgeschlagen, rechnet die App die Positionen anteilig auf brutto hoch. So zeigt die Auswertung,
  was du wirklich bezahlt hast. Am Bon steht dann ein Hinweis „Netto-Rechnung: …“.
- **Absender statt Empfänger**: Händler ist immer der Aussteller. Die Absenderzeile über dem Adressfeld
  („Firma · Straße · PLZ Ort“) hat Vorrang vor der KI. Trag zusätzlich deinen Namen in `OWN_NAMES` ein.
