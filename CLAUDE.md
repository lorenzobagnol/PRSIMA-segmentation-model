# PRISMA — segmentazione dei degradi (beni culturali, UNI 11182)

Leggi questo file per intero prima di agire: è il punto di ingresso per qualunque sessione nuova
su questo progetto. Dettagli operativi → [docs/runbook.md](docs/runbook.md). Storico dei
risultati di training → [docs/training-history.md](docs/training-history.md). Ricerca iniziale su
pseudo-labeling (Gemini/OpenAI/SAM 3) → [docs/pseudo-label-sam3.md](docs/pseudo-label-sam3.md)
(solo storico, lo stato attuale è qui).

## Cosa fa il progetto

Rilevare automaticamente i degradi su materiali lapidei/intonaci di beni culturali (normativa UNI
11182), a partire da foto di facciate. Obiettivo finale: un modello leggero per degrado (U-Net),
da mettere in produzione, che segua lo schema già usato con successo per le "cavillature" (crepe).

**Il lavoro attuale è concentrato solo su "Distacco"** (perdita di intonaco/strato superficiale).
Altri degradi (Rigonfiamento, Crepa) sono predisposti nella pipeline ma non ancora lavorati.

Esistono anche 3 servizi Cloud Run più vecchi e **non collegati a questo lavoro**
(`cavillature-service`, `distacco-service`, `rigonfiamento-service`, progetto
`architecture-degradi`, regione `europe-west12`, architettura efficientnet-b3): secondo Lorenzo
davano risultati scarsi su distacco/rigonfiamento, ed è il motivo per cui è partito questo nuovo
giro di lavoro. Non toccarli senza che sia chiesto esplicitamente.

## Come siamo arrivati qui (riassunto)

1. **Dataset quasi assente** → scartate due strade per generare pseudo-label: Gemini (poligoni
   grossolani, 10-40 punti) e OpenAI gpt-image-2.5 (rigenera l'immagine, maschera disallineata).
   **SAM 3** (`facebook/sam3`, pesi gated su Hugging Face) zero-shot con prompt testuali si è
   rivelato il migliore — a volte perfetto su Distacco, inadeguato su Rigonfiamento (macchie
   bianco su bianco, troppo sottili per i suoi prompt testuali). Dettagli in
   [docs/pseudo-label-sam3.md](docs/pseudo-label-sam3.md).
2. Costruito uno **strumento di annotazione web** (`annotation/`) che usa SAM 3 per proporre una
   maschera di partenza, corretta a mano nel browser (pennello, flood-fill su mappa di
   probabilità, clic SAM puntuale). Deployato su Cloud Run.
3. Annotate ~85 foto di Distacco. Allenata una **U-Net leggera** (resnet50, solo RGB) su quei
   dati, in più giri, migliorando via via la ricetta di training (vedi
   [docs/training-history.md](docs/training-history.md) per i numeri).
4. **Integrato il modello allenato nello strumento di annotazione stesso**: ora la maschera di
   partenza per una foto nuova è la predizione della U-Net (~1-4s su CPU), molto più veloce di
   SAM 3 (25-45s) — SAM 3 resta disponibile come aiuto extra su richiesta. Questo chiude il
   cerchio: annota → allena → il modello migliora l'annotazione successiva → repeat.

## Stato attuale (vedi training-history.md per la cronologia completa)

- **Modello in produzione nel bucket** (`gs://architecture-degradi-annotation/models/distacco.pth`):
  quello del giro "76 immagini" (train+test uniti, 150 epoche, nessuna metrica di validazione
  affidabile — vedi sotto). **Non ancora validato con un vero test set.**
- **Ultimo checkpoint validato con test set reale**: IoU 0.8366 (giro a 68 train + 8 test),
  salvato in locale come `models_distacco_best_v2.pth` e sul vecchio server RunPod (ora spento)
  in `models_v2_76test8/`.
- **Lezione imparata (importante)**: il giro "76 immagini" ha rimosso l'unico test set reale per
  usare più dati in training. Risultato: IoU altissimo sul train (0.92, recall 0.98 — overfitting
  da manuale) e un caso concreto di regressione su una foto mai vista (`IMG_9306`: il modello
  vecchio trovava una macchia ovvia, il nuovo l'ha persa quasi del tutto). **Da ora in poi, tenere
  sempre un 10-15% di dati come test reale, anche con pochi dati disponibili** — senza quello non
  si vede l'overfitting mentre accade.
- **In corso**: aggiunte a `train.py` tre nuove augmentation (`RandomResizedCrop`,
  `RandomPerspective`, `RandomErasing`/cutout), motivate proprio dai problemi osservati (macchie
  piccole isolate perse, overfitting). Codice scritto e testato con tensori sintetici su CPU, **non
  ancora allenato per davvero** — serve un server GPU (quello RunPod precedente è stato spento il
  2026-10-09, va richiesto un nuovo accesso SSH a Lorenzo).
- **15 foto nuove non annotate** caricate nello strumento (`C:\Users\loren\Downloads\distacchi-non-annotati`),
  in attesa di validazione manuale da parte di Lorenzo — diventeranno probabilmente il prossimo
  test set reale.

## Architettura (dettagli completi in docs/runbook.md)

- **Strumento di annotazione**: `annotation/server.py` (FastAPI) + `annotation/static/index.html`
  (pagina singola, canvas). Motori: `annotation/sam3_engine.py` (SAM 3) e
  `annotation/unet_engine.py` (U-Net allenata). Deployato su Cloud Run, URL
  `https://annotation-429808723098.europe-west1.run.app`, protetto da IAP (login Google, solo
  `lorenzobgl@gmail.com` e `danielepanerai@gmail.com` autorizzati).
- **Dati**: tutto vive in un bucket GCS (`gs://architecture-degradi-annotation`), montato come
  volume nel container Cloud Run. Struttura: `images/`, `meta/`, `masks/<Categoria>/`, `cache/`
  (mappe SAM 3 + U-Net), `exports/` (zip per il training), `models/distacco.pth` (checkpoint
  servito dall'app). Non c'è versioning sul bucket (scelta esplicita di Lorenzo): una
  sovrascrittura è persa per sempre.
- **Pipeline di training**: script alla radice della repo (`train.py`, `config.py`, `models.py`,
  `pbr_maps.py`, `create_dataset.py`, `compare_predictions.py`, `compare_checkpoints.py`,
  `configs/distacco.yaml`). Gira su un server GPU **affittato on-demand** (RunPod finora), **non
  persistente**: a ogni nuova sessione di training serve chiedere a Lorenzo le coordinate SSH di
  un server (nuovo o riacceso) e rifare il setup da zero.

## Convenzioni e trappole note (leggi prima di ripetere un errore già fatto)

Vedi [docs/runbook.md](docs/runbook.md) per la lista completa (percorsi Windows/Git Bash, auth
verso l'app via impersonificazione service account, pattern `setsid nohup` per il training in
background, cold start di Cloud Run, ecc.). Non rifare da zero ricerche su Gemini/OpenAI per il
pseudo-labeling: già valutati e scartati, motivazioni in
[docs/pseudo-label-sam3.md](docs/pseudo-label-sam3.md).

## Stile di lavoro con Lorenzo

- Preferisce che si proceda in autonomia su task infrastrutturali/ripetitivi (deploy, training,
  upload) senza chiedere conferma a ogni passo, ma vuole essere avvisato chiaramente di ogni esito
  e di ogni scelta con un trade-off reale (es. quale checkpoint tenere in produzione).
- Legge e capisce bene le metriche — dargli numeri precisi (IoU/precision/recall), non solo
  giudizi qualitativi.
- Fa domande di verifica tecnica dirette (es. "hai ripreso dal checkpoint precedente?",
  "sono tutte le augmentation possibili?") — rispondere con precisione tecnica, ammettendo
  limiti/scelte quando rilevante, non solo rassicurando.
