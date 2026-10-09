# Runbook operativo

Procedure concrete e trappole già incontrate. Vedi [../CLAUDE.md](../CLAUDE.md) per il quadro
generale.

## Risorse GCP

- Progetto: `architecture-degradi` (numero progetto `429808723098`), regione `europe-west1`.
- Bucket dati: `gs://architecture-degradi-annotation` (montato su `/data` nel container Cloud Run).
- Servizio Cloud Run: `annotation` — `https://annotation-429808723098.europe-west1.run.app`.
- Repository immagini Docker: `europe-west1-docker.pkg.dev/architecture-degradi/annotation/annotation`
  (tag `v1`...`v9` finora, incrementare a ogni build).
- Service account app: `annotation-app@architecture-degradi.iam.gserviceaccount.com`.
- Secret Manager: `hf-token` (token Hugging Face con accesso a `facebook/sam3`, usato da
  `annotation/cloudbuild.yaml` per scaricare i pesi durante la build).
- `gcloud` installato in locale in `%LOCALAPPDATA%\Google\google-cloud-sdk\bin` (non nel PATH di
  default: aggiungerlo a ogni sessione nuova, es.
  `export PATH="$PATH:/c/Users/loren/AppData/Local/Google/google-cloud-sdk/bin"` in Git Bash, o
  `$env:Path += ";$env:LOCALAPPDATA\Google\google-cloud-sdk\bin"` in PowerShell).

## Deploy di una nuova versione dello strumento di annotazione

```bash
# da Git Bash, nella root della repo
export PATH="$PATH:/c/Users/loren/AppData/Local/Google/google-cloud-sdk/bin"
gcloud builds submit annotation --config annotation/cloudbuild.yaml --region=europe-west1 \
  --substitutions=_IMAGE=europe-west1-docker.pkg.dev/architecture-degradi/annotation/annotation:vN
```

Il deploy vero e proprio va fatto da **PowerShell**, non Git Bash: Git Bash riscrive i percorsi
assoluti che iniziano con `/` (es. `/data`) in percorsi Windows, rompendo gli argomenti di
`gcloud run deploy`.

```powershell
$env:Path += ";$env:LOCALAPPDATA\Google\google-cloud-sdk\bin"
gcloud run services update annotation --region=europe-west1 `
  --image=europe-west1-docker.pkg.dev/architecture-degradi/annotation/annotation:vN --quiet
```

Se il comando contiene argomenti con virgole (es. `--add-volume=name=...,type=...`), mettili tra
virgolette in PowerShell, altrimenti li spezza sulla virgola.

`annotation/cloudbuild.yaml` include uno smoke test (importa i moduli e carica la config di SAM 3
offline, senza scaricare i pesi) che fa fallire la build *prima* del push se manca una dipendenza
critica — non saltarlo.

**Aggiornare solo il modello U-Net** (senza rebuild): carica il nuovo checkpoint su
`gs://architecture-degradi-annotation/models/Distacco.pth` (un file per categoria, `models/<Categoria>.pth`), poi **forza un riavvio
dell'istanza** (il container tiene il modello in RAM, non lo rilegge da solo):

```powershell
gcloud run services update annotation --region=europe-west1 --image=<stessa immagine attuale> --quiet
```

## Autorizzare un nuovo utente

```bash
gcloud iap web add-iam-policy-binding --member=user:<email> --role=roles/iap.httpsResourceAccessor \
  --region=europe-west1 --resource-type=cloud-run --service=annotation
```

Va **anche** aggiunto come "utente di test" nella schermata di consenso OAuth (Google Auth
Platform → Audience in console), altrimenti IAP lo respinge comunque.

## Chiamare le API dello strumento da script (upload/compute massivi)

L'app è dietro IAP con un client OAuth personalizzato: un token utente normale (`gcloud auth
print-identity-token`) **non funziona** (audience sbagliata). Serve impersonare il service
account dell'app, che va autorizzato una volta sola:

```bash
# una tantum
gcloud iap web add-iam-policy-binding --member=serviceAccount:annotation-app@architecture-degradi.iam.gserviceaccount.com \
  --role=roles/iap.httpsResourceAccessor --region=europe-west1 --resource-type=cloud-run --service=annotation
gcloud iam service-accounts add-iam-policy-binding annotation-app@architecture-degradi.iam.gserviceaccount.com \
  --member=user:lorenzobgl@gmail.com --role=roles/iam.serviceAccountTokenCreator
```

Poi, a ogni script:

```python
CID = ...  # da .env, chiave "AP_CLIENT_ID" (sic, manca la "I" iniziale nel nome della variabile) o "IAP_CLIENT_ID"
token = subprocess.run([GCLOUD, "auth", "print-identity-token",
    "--impersonate-service-account=annotation-app@architecture-degradi.iam.gserviceaccount.com",
    f"--audiences={CID}", "--include-email"], capture_output=True, text=True, check=True).stdout.strip()
# poi Authorization: Bearer <token> su ogni chiamata a /api/...
```

Endpoint utili: `/api/upload-urls` + `/api/ingest` (upload diretto su bucket, aggira il limite di
32MB per richiesta di Cloud Run), `/api/compute` (calcola la predizione; `with_sam3: false` per
usare solo la U-Net, molto più veloce), `/api/export` (zip del dataset annotato, train/test
secondo il flag `test` di ogni immagine), `/api/delete-image`, `/api/test-flag`.

Da **Python nativo Windows** (non Git Bash), il path di `gcloud` va passato in stile Windows con
l'estensione `.cmd`, non in stile Git Bash (`/c/...`) — `subprocess.CreateProcess` non lo capisce:

```python
GCLOUD = r"C:\Users\loren\AppData\Local\Google\google-cloud-sdk\bin\gcloud.cmd"
```

## Server GPU per il training (non persistente)

Il server (finora RunPod) **non è un servizio stabile**: va chiesto a Lorenzo ogni volta che
serve, con le coordinate SSH (`ssh root@<ip> -p <porta> -i ~/.ssh/id_ed25519`), e **viene spento
tra una sessione e l'altra** — ricreare l'ambiente da zero ogni volta:

```bash
# venv che eredita il torch già installato sull'immagine del pod (di solito torch 2.8 + cu128)
python -m venv --system-site-packages .venv
. .venv/bin/activate
pip install segmentation_models_pytorch==0.5.0 pyyaml matplotlib tqdm

# copiare dal repo locale: config.py models.py pbr_maps.py create_dataset.py train.py
# generate.py compare_predictions.py compare_checkpoints.py configs/distacco*.yaml

# dataset: esportarlo dall'app (vedi sopra /api/export) e scaricarlo dal bucket, oppure
gcloud storage cp gs://architecture-degradi-annotation/exports/<ultimo>.zip dataset_export.zip
# poi scp sul server ed estrarre in images-and-masks/annotated/{train,test}
```

Lanciare il training **sempre con `setsid nohup`**, altrimenti muore alla disconnessione SSH:

```bash
setsid nohup python -u train.py --config configs/distacco.yaml > train.log 2>&1 < /dev/null &
```

Per aspettare la fine senza bloccare la sessione:

```bash
while pgrep -f "python -u train.py" > /dev/null; do sleep 20; done; echo finito; tail -40 train.log
```

Questo comando, lanciato in background, viene periodicamente interrotto dal limite di tempo
dell'harness (non è un errore): controllare con `pgrep -af "python -u train.py" | grep -v while`
se il processo remoto è ancora vivo (quasi sempre sì, grazie a `setsid nohup`), e rilanciare lo
stesso comando di attesa.

**Prima di ogni nuovo giro di training**, spostare altrove i checkpoint del giro precedente (es.
`mv models/distacco.pth models/distacco_best.pth models_vN_backup/`), altrimenti `train.py`
riprende automaticamente da lì invece di ripartire da zero — utile a volte, ma rende impossibile
un confronto pulito tra una ricetta e l'altra.

**Prima di spegnere il server**, scaricare in locale: i checkpoint (`models/*.pth`), i log
(`*.log`) — tutto il resto (tensori, dataset estratto) è rigenerabile dal bucket.

## Altre trappole

- `pkill -f <pattern>` lanciato dentro un comando SSH la cui riga di comando contiene essa stessa
  quel pattern **uccide anche se stesso**: usare `pkill -f "[s]erver.py"` (il carattere in
  parentesi quadre rompe il self-match) o uno script separato.
- Cloud Run scala a zero istanze quando inattivo: la prima richiesta dopo una pausa impiega
  30-60s per ripartire (cold start). Lo strumento ora ha un timeout di 20s con un retry sui
  caricamenti immagine proprio per questo, ma se serve zero attesa durante una sessione di
  annotazione intensiva si può impostare `--min-instances=1` (costa di più quando acceso).
- Il bucket **non ha versioning**: cancellare o sovrascrivere un file lì è definitivo.

## Formato dati nel bucket (dal 2026-10-09)

`meta/<id>.json` piatto (`category`, `reviewed`, `computed`, `model_computed`, `prompts`, `test`),
`cache/<id>/` senza sottocartella di categoria, `categories.json` con `model` per categoria.
La migrazione dal vecchio formato è in `annotation/migrate_bucket.py` (già eseguita; salva i vecchi
file in `--backup`). Nota: il classificatore dei permessi di Claude Code blocca scritture massive
sul bucket di produzione, quindi gli script di migrazione vanno lanciati dall'utente o autorizzati
esplicitamente. Le copie vecchie in `cache/<id>/Distacco/` e `models/distacco.pth` sono ancora lì
(si possono cancellare quando si è sicuri).
