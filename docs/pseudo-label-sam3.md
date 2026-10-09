# Pseudo-label per Distacco e Rigonfiamento: stato e prossimi passi

> **Nota:** questo documento copre la fase di ricerca iniziale (confronto Gemini/OpenAI/SAM 3 per
> il pseudo-labeling) ed è rimasto fermo al 2026-09-24. Per lo stato attuale del progetto, i
> risultati di training e le procedure operative, vedi [../CLAUDE.md](../CLAUDE.md),
> [training-history.md](training-history.md) e [runbook.md](runbook.md). Le conclusioni di
> questo documento (SAM 3 scelto, Gemini/OpenAI scartati) restano valide. **Il codice citato qui
> sotto** (`pseudo_label.py`, `compare/gemini25_test.py`, `compare/openai_test.py`,
> `compare/sam3_test.py`) **è stato eliminato il 2026-10-09**: serviva solo per questo confronto
> una tantum, già concluso, e la sua logica utile (prompt curati, filtro sulle istanze troppo
> grandi) è confluita in `annotation/sam3_engine.py`, tuttora in uso. Questo documento resta come
> traccia di cosa è stato provato e perché, non come riferimento a codice ancora presente.

Aggiornato al 2026-09-24.

## Cosa abbiamo provato

Confronto su `dataset_no_annotation/` (foto RGB, senza mappe PBR), con i risultati in
`compare/review.html` e in `compare/confronto_pseudo_label.pdf`.

| Metodo | Esito |
|---|---|
| Gemini 3.x (`pseudo_label.py`) | Poligoni con 10-40 punti, quindi bordi grossolani. Restituisce `[y, x]` invece di `[x, y]` (lo script lo corregge). Spesso mancano alcune zone. |
| Gemini 2.5 Flash (maschere per pixel) | Non disponibile per le chiavi API nuove (404). Non si sblocca dall'account. |
| OpenAI gpt-image-2.5 | Bordi fini, ma il modello ridisegna l'immagine: le maschere risultano spostate, scalate, ruotate e distorte rispetto alla foto. Non affidabile. |
| **SAM 3, prompt testuali** (`compare/sam3_test.py`) | **Il più promettente.** Su Distacco le maschere sono precise al pixel e allineate, in alcuni casi perfette ("missing plaster", "peeling plaster"). Su Rigonfiamento quasi sempre non trova nulla o prende il muro intero; funziona solo sulle bolle di pittura evidenti ("bubbling paint"). |

Lezioni apprese su SAM 3:
- Alcune frasi prendono il muro intero ("detached render", "swollen plaster"). Per questo si scartano le istanze che coprono più del 50% dell'immagine.
- Resta un'ambiguità su Distacco: a volte segna anche lo strato di intonaco ancora attaccato intorno alla lacuna. Va deciso cosa conta come "distacco".
- VRAM: circa 4GB in inferenza su una singola immagine.

## Idea (Lorenzo): logits + correzione manuale + fine-tuning di SAM 3

1. Salvare le **mappe di probabilità / logits** di SAM 3 invece delle maschere già sogliate, includendo anche le istanze sotto soglia.
2. Creare un **programma semplice di correzione**: si clicca su una zona mancante e si "accendono" i pixel connessi al clic con probabilità sopra una soglia simile.
3. Con le maschere corrette costruire un **dataset migliore**.
4. Fare il **fine-tuning di SAM 3** su quel dataset.

## Piano proposto

Stato al 2026-09-24: i punti 1 e 2 sono fatti. Lo strumento (`annotation/server.py`, `annotation/sam3_engine.py`, `annotation/static/index.html`) gestisce anche upload di immagini, categorie nuove (degradi o materiali) con i loro prompt, prompt scritti a mano ed export del dataset in zip. Si parte con il punto 3.

### Hosting (un solo utente, budget minimo)

Misure di SAM 3 **solo CPU, 8 core**: preparazione di un'immagine nuova 25s (una volta), clic SAM 0.07s, un prompt 2s, RAM di picco 7.3GB. Il vision encoder viene calcolato una volta e riusato da tutti i prompt (risultato identico al calcolo completo). La GPU quindi non serve: l'opzione prevista è Cloud Run solo CPU (8 vCPU / 16GiB, scala a zero), con i dati su un bucket GCS montato come volume e l'accesso protetto da IAP.

Struttura dei dati (una cartella locale, oppure un bucket GCS montato come volume):
- ogni foto è salvata una volta sola in `images/<id>`, dove `<id>` sono i primi 12 caratteri dell'hash SHA-256 del file; i doppioni vengono riconosciuti;
- le maschere stanno in `masks/<Categoria>/<id>.png`, quindi elencare la cartella di una categoria dà le immagini che la contengono; una maschera vuota salvata è un negativo verificato;
- i metadati stanno in `meta/<id>.json` e fanno fede; all'avvio vengono caricati in un indice in memoria;
- mappe SAM 3 e copia di lavoro (lato lungo massimo 2048px) stanno in `cache/`;
- gli snapshot per il training stanno in `exports/`, con `manifest.jsonl`.

Niente versioning del bucket e niente metadati aggiuntivi (decisione di Lorenzo). Su Cloud Run i file grandi vanno direttamente dal browser al bucket tramite URL firmato, per aggirare il limite di 32MB per richiesta.

Su RunPod il template PyTorch ha un nginx che espone la porta 7861 inoltrandola alla 7860: lo strumento, che non ha login, gira sulla 7870 per non passarci.

1. **Salvare le mappe di probabilità** in `sam3_test.py`: per ogni frase, il massimo pesato per score sulle query (`sigmoid(mask_logits) * score`), alla risoluzione originale e in PNG a 8/16 bit. Non si salvano tutte le query grezze, che occuperebbero decine di MB per immagine.
2. **Strumento di correzione** (Gradio sulla GPU del server, raggiunto dal browser tramite tunnel SSH), con tre azioni:
   - clic + soglia sulla mappa di probabilità: *flood fill* delle regioni connesse (l'idea sopra);
   - clic come **prompt a punto per SAM 3**: segmentazione interattiva nativa, utile dove la mappa testuale è vuota, come per Rigonfiamento;
   - gomma per rimuovere i falsi positivi, poi salvataggio nel formato `Maschere/<Degrado>.jpg` che `create_dataset.py` già legge.
3. **Annotare Distacco** (23 foto) e **Rigonfiamento** (18 foto), tenendo da parte 4-5 foto per degrado come test set, mai viste in training.
4. **Training**, in due varianti confrontate sullo stesso test set:
   - U-Net / SegFormer piccoli sulle maschere corrette: la pipeline esistente, leggera da mettere in produzione;
   - fine-tuning di SAM 3 con backbone congelato e solo decoder/head addestrati (o LoRA), per stare nei 16GB. Con circa 40 immagini c'è rischio di overfitting: diventa più interessante man mano che il dataset corretto cresce.

### Deploy (2026-09-24)

- URL: https://annotation-429808723098.europe-west1.run.app (IAP: login Google; accedono solo gli account autorizzati, che devono essere anche utenti di test della schermata di consenso OAuth)
- Progetto `architecture-degradi`, regione `europe-west1`, servizio Cloud Run `annotation` (8 vCPU / 16GiB, da 0 a 1 istanza), bucket `gs://architecture-degradi-annotation` montato su `/data`, service account `annotation-app@`.
- Build: `gcloud builds submit annotation --config annotation/cloudbuild.yaml --region=europe-west1 --substitutions=_IMAGE=europe-west1-docker.pkg.dev/architecture-degradi/annotation/annotation:<tag>`; il token HF è nel secret `hf-token`.
- Autorizzare un utente: `gcloud iap web add-iam-policy-binding --member=user:<email> --role=roles/iap.httpsResourceAccessor --region=europe-west1 --resource-type=cloud-run --service=annotation`, e aggiungerlo come utente di test in console (Google Auth Platform → Audience).

### Semplificazione: un degrado alla volta (2026-10-02)

Il 2026-10-02, su richiesta di Lorenzo, il bucket è stato svuotato (immagini/maschere/mappe; `categories.json` conservato) per ripartire da zero, e l'app è stata semplificata per velocità:
- **Il degrado annotato è fisso lato server**, non scelto in UI: variabile d'ambiente `ACTIVE_CATEGORY` sul servizio Cloud Run (default `Distacco`), letta da `server.py` all'avvio. Deve già esistere in `categories.json` (il server si rifiuta di partire altrimenti). Per cambiare degrado: `gcloud run services update annotation --region=europe-west1 --update-env-vars=ACTIVE_CATEGORY=<Nome>` e nuovo deploy.
- **Niente prompt manuali in UI**: tolto `/api/prompt` e il pannello "Prompt manuale". I prompt di ogni categoria restano solo quelli in `categories.json` (curati a mano), così SAM 3 calcola sempre e solo quelli, mai query impreviste.
- **Niente editor di categorie in UI**: tolti `/api/categories` e `/api/image-category`. Le categorie si modificano editando `categories.json` nel bucket direttamente.
- **Upload più diretto**: ogni immagine caricata viene etichettata subito con la categoria attiva (niente checkbox) e le mappe SAM 3 partono in automatico dopo l'upload.
- `categories.json` resta una lista generica (più categorie, più prompt ciascuna) per un'eventuale versione futura con più degradi per immagine; questa versione ne legge e scrive solo una.
- Prompt di Distacco ridotti ai 3 validati in `compare/sam3_test.py`: "peeling plaster", "missing plaster", "flaking paint" (tolti "exterior plaster wall" e "detached plaster", aggiunti nel frattempo da Daniele, perché il primo rischia di prendere il muro intero).
- Immagine `annotation:v3` su Artifact Registry.
