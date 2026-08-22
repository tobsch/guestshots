# guestshots

Gib dem Tool YouTube-Links einer Podcast-Folge, es liefert N Screenshots, auf denen **der Gast** (nicht der Host) gut und aktiv aussieht — Augen offen, Lächeln oder Gestik, scharf, frontal.

```
output/<id>_<titel>/shots/01_18m28s_score0.72.jpg
                          02_27m13s_score0.77.jpg
                          ...
```

## Setup auf einem frischen Mac (Apple Silicon oder Intel)

```bash
# 1. Homebrew (falls nicht da): https://brew.sh
# 2. Systemtools
brew install ffmpeg uv git

# 3. Repo holen + Python-Umgebung bauen (uv lädt Python 3.12 selbst, nichts weiter nötig)
git clone https://github.com/tobsch/guestshots.git
cd guestshots
uv sync

# 4. OpenRouter-Key für die LLM-Stufe (optional, aber empfohlen — ~3,5 Cent pro Video)
echo 'OPENROUTER_API_KEY=<dein-key>' > .env

# 5. Host-Foto(s) ablegen, damit das Tool weiß, wer NICHT der Gast ist
cp ~/Pictures/ich.jpg hosts/
```

Beim ersten Lauf lädt InsightFace einmalig seine Modelle (~300 MB) nach `~/.insightface/`.

Das war's. Kein Docker, keine GPU, kein CUDA — läuft auf CPU (Apple Silicon nutzt CoreML automatisch).

## Benutzung

```bash
uv run guestshots 'https://www.youtube.com/watch?v=XXXX'
uv run guestshots 'https://youtu.be/XXXX' -n 8 --solo-only
uv run guestshots URL1 URL2 URL3            # mehrere Folgen in einem Rutsch
```

**URL immer in Quotes** — zsh stolpert sonst über `?` und `&`.

| Option | Default | Bedeutung |
|---|---|---|
| `-n` | 5 | Screenshots pro Video |
| `--solo-only` | aus | Nur Frames, in denen der Gast allein (groß) im Bild ist |
| `--min-gap` | 20 | Mindestabstand in Sekunden zwischen zwei Shots |
| `--fps` | 1 | Analyse-Frames pro Sekunde (höher = genauer, langsamer) |
| `--host-dir` | `hosts/` | Ordner mit Referenzfotos des Hosts |
| `--guest-id N` | — | Gast-Identität erzwingen (siehe `identities/`-Crops) |
| `--llm / --no-llm` | an | LLM-Re-Ranking über OpenRouter |
| `--llm-model` | `anthropic/claude-sonnet-5` | beliebiges OpenRouter-Vision-Modell, z.B. `google/gemini-2.5-pro` |
| `--llm-pool` | 4 | Kandidaten pro finalem Shot, die ans LLM gehen (n × pool Bilder) |

Dauer: ~10 min für eine 80-min-Folge beim ersten Lauf (Download + Gesichtserkennung), danach Sekunden — Video und Detektionen werden in `cache/` gecached.

## Output

```
output/<id>_<titel>/
  shots/01_12m34s_score0.81.jpg   ← die Screenshots, volle Auflösung
  contact_sheet.jpg               ← alle Shots auf einen Blick
  candidates/                     ← Gesichts-Crops, die das LLM gesehen hat
  identities/id0_host_x1423.jpg   ← wer wurde als wer erkannt (host / cand)
  report.json                     ← Timestamps, Scores, LLM-Notizen
```

## Wie es funktioniert

1. `yt-dlp` lädt das Video (≤1080p), `ffmpeg` zieht 1 Frame/s.
2. InsightFace erkennt Gesichter, Embeddings und Landmarks; Gesichter werden zu Identitäten geclustert.
3. **Host vs. Gast**
   - Referenzfotos in `hosts/` → Identität mit Cosine-Similarity ≥ 0.45 ist Host.
   - Ohne Referenz, aber ≥2 URLs → die Person, die in mehreren Videos vorkommt, ist der Host.
   - Sonst: häufigste Person = Gast (mit Warnung). Korrigieren per `--guest-id N`.
4. **Scoring pro Gast-Frame**, kalibriert auf die Verteilung im jeweiligen Video: Augen offen (InsightFace **und** MediaPipe müssen zustimmen), frontal, Lächeln oder moderat sprechender Mund, Bewegung im Gesichtsbereich (Gestik), Schärfe, Licht, Größe, allein im Bild. Blinzler, Lach-Kniffe, weit aufgerissener Mund und weggedrehte Köpfe fliegen raus.
5. Kandidaten werden in voller Auflösung geholt und **verifiziert** (Gesicht im Frame muss per Embedding der Gast sein) — der 1-fps-Analyse-Frame liegt bis 0,5 s neben dem exakten Timestamp, was an Kameraschnitten sonst den Host liefert.
6. **LLM-Stufe** (optional): die besten `n × pool` Crops gehen über OpenRouter an das Vision-Modell, das erst die Augen klassifiziert (`open|squint|closed` — nur `open` ist brauchbar) und dann „vorteilhaft" und „aktiv" 1–10 bewertet. Landmark-Modelle erkennen Lach-Kniffe nicht zuverlässig, ein Vision-LLM mit Full-Res-Crop schon.
7. Top-N mit Mindestabstand → `shots/`.

## Kosten

Nur die LLM-Stufe kostet: gemessen **$0.035 pro Video** (24 Bilder, `anthropic/claude-sonnet-5`). Der Rest läuft lokal. Die Usage-Zeile wird bei jedem Lauf ausgegeben.

## Hinweise / Troubleshooting

- **Audio-Folgen mit Standbild** (z.B. ältere alphalist-Uploads, 720×720 Cover) → Warnung „video is (almost) static". Funktioniert nur mit echtem Interview-Video.
- **„could not determine the guest"** → in `output/.../identities/` nachsehen, wer erkannt wurde, dann `--guest-id N` oder ein Host-Foto nach `hosts/`.
- **Host ist in den Shots** → Host-Foto nach `hosts/` legen (am einfachsten: den Host-Crop aus `identities/` kopieren) oder `--solo-only`.
- `mediapipe` ist bewusst auf 0.10.14 gepinnt — neuere Versionen crashen auf macOS (Metal-Delegate). Nicht hochziehen.
- `.env`, `cache/`, `output/` und `hosts/*` sind gitignored — Key und Fotos landen nicht im Repo.
