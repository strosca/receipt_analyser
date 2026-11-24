# Kassenbon-Auswertung

Selbstgehostete Web-App, die Kassenbons (Fotos und PDFs) einliest, jede Position per KI erfasst
und kategorisiert und daraus Auswertungen erstellt.

> Die README wird mit den nächsten Schritten vervollständigt (Installation, Update, Backup, Konfiguration).

## Schnellstart (auf wgserver)

```bash
cp .env.example .env      # Passwort usw. eintragen
docker compose up -d --build
```

Danach im Browser: <http://192.168.68.118:8095>

## Entwicklung (lokal)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     # Windows: .venv\Scripts\pip
.venv/bin/pytest
```
