# Storico dei giri di training — Distacco

Vedi [../CLAUDE.md](../CLAUDE.md) per il quadro generale. Modello: U-Net, encoder resnet50
(pesi ImageNet), input solo BaseColor (RGB, 3 canali), risoluzione 1024×1024,
`segmentation_models_pytorch`. Log completi in `training-logs/train*.log`.

| Giro | Data | Dati (train/test) | Ricetta | Risultato | Checkpoint |
|---|---|---|---|---|---|
| 1 | 2026-10-06 | 71 / 11 | Adam, ExponentialLR, 80 epoche, no color jitter | IoU 0.611 | — (scartato) |
| 2 | 2026-10-06 | 68 / 8 (dopo pulizia, vedi sotto) | stessa ricetta | **IoU 0.779** | `models_distacco_best.pth` |
| 3 | 2026-10-06 | 68 / 8 | AdamW (weight_decay 1e-4), CosineAnnealingLR, ColorJitter, 150 epoche, da zero | **IoU 0.8366** | `models_distacco_best_v2.pth` — **ultimo validato con test reale** |
| 4 | 2026-10-06 | 76 / 0 (test unito al train) | stessa ricetta del giro 3 | IoU 0.92 **sul train stesso, non affidabile** — vedi nota overfitting | `models_distacco_v3_76img.pth` — **in produzione nel bucket, non validato** |
| 5 | pianificato | da definire (con nuovo test set) | + RandomResizedCrop, RandomPerspective, RandomErasing | non ancora eseguito | — |

## Pulizia dati tra giro 1 e giro 2

Confrontando maschera vera e predetta (vedi `compare_predictions.py`), trovate 6 immagini (3 nel
train, 3 nel test) con **maschera vuota salvata sopra un distacco reale ed evidente** (texture di
intonaco mancante ben visibile, ma nessuna area segnata) — probabilmente confermate con la
maschera iniziale di SAM 3 senza correggerle. Eliminate dal bucket. Il salto di IoU dal giro 1 al
giro 2 (0.611 → 0.779, **a parità di ricetta**) conferma che il problema principale era la qualità
di alcune annotazioni, non il modello.

## Giro 4: la lezione sull'overfitting

Su richiesta di Lorenzo, le 8 immagini di test del giro 3 sono state rimesse in train (erano
annotate a mano da zero, "le più difficili") per usare tutti i 76 dati disponibili, con l'idea di
validare in seguito su foto nuove. Risultato: **nessuna metrica affidabile durante questo giro**
(l'IoU calcolato ogni epoca finisce per essere quello sul train stesso, con le augmentation
attive — train.py lo segnala esplicitamente come "solo per debug"). A fine training, IoU 0.921 e
recall 0.982 sul train: tipico segnale di overfitting.

Confermato in pratica con `compare_checkpoints.py` su foto mai viste (le 15 di
`distacchi-non-annotati`): risultati **misti**, non un miglioramento netto. Caso più chiaro:
`IMG_9306`, dove il modello del giro 3 (0.8366) cattura bene una macchia grande ed evidente, e
quello del giro 4 la perde quasi del tutto. Altri casi (`IMG_9317`) mostrano invece buon accordo
tra i due modelli.

**Decisione presa**: da ora in poi, tenere sempre un 10-15% dei dati come test reale, anche a
costo di un training set più piccolo — il costo di non vedere l'overfitting mentre accade è
maggiore del beneficio di qualche immagine in più nel train.

**Checkpoint attualmente in produzione**: quello del giro 4 (`gs://architecture-degradi-annotation/models/distacco.pth`),
caricato prima di aver notato il problema di overfitting. Non ancora sostituito — valutare se
tornare al checkpoint del giro 3 (0.8366, validato) in attesa di un giro 5 con test set vero.

## Augmentation usate

**Dal giro 3**: `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomAffine` (rotazione ±20°,
traslazione 10%), `ElasticTransform`, `ColorJitter` (luminosità/contrasto/saturazione/tonalità).

**Aggiunte per il giro 5** (scritte e testate su tensori sintetici il 2026-10-09, non ancora
allenate per davvero — serve un server GPU nuovo):
- `RandomResizedCrop` (scale 0.5-1.0): moltiplica le "viste" diverse per immagine, aiuta con le
  macchie piccole isolate che il modello tendeva a perdere.
- `RandomPerspective` (distortion_scale 0.3): simula angolazioni di scatto diverse.
- `RandomErasing`/cutout (scale 0.02-0.15, solo sull'immagine non sulla maschera): forza il
  modello a non affidarsi a un solo indizio evidente — mirato proprio al tipo di errore visto in
  `IMG_9306`.

Scartate deliberatamente: MixUp/CutMix (pensate per classificazione, su segmentazione creano
maschere innaturali). Non ancora implementate ma utili in futuro: copy-paste augmentation (incollare
ritagli di distacco reale su muri puliti diversi — tecnica più efficace in letteratura per
segmentazione con pochi esempi positivi, ma più complessa da implementare bene).
