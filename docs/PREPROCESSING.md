# Preprocesarea video brut

`inference.py` așteaptă input deja preprocesat: clipuri `.avi` la rezoluția
**160x160** cu fața deja croppată și centrată pe gură. Acest format este
identic cu cel folosit de [MultiVSR](https://github.com/Sindhu-Hegde/multivsr).

Pentru video brut (de exemplu, un clip YouTube, sau o filmare proprie),
trebuie întâi extrasă regiunea gurii. Recomandăm pipeline-ul MultiVSR
(adaptat din [SyncNet](https://github.com/joonson/syncnet_python)).

## Pași

```bash
# 1. Clonează MultiVSR
git clone https://github.com/Sindhu-Hegde/multivsr.git
cd multivsr

# 2. Instalează dependențe (separat de mediul nostru)
pip install -r requirements.txt

# 3. Rulează preprocesarea
cd preprocess
python run_pipeline.py \
    --videofile /cale/catre/video.mp4 \
    --reference my_clip \
    --data_dir /tmp/processed
```

Output-ul (clipuri 160x160) ajunge în `/tmp/processed/my_clip/pycrop/*.avi`.

## Apoi rulează inferența

```bash
cd /cale/catre/ro-vsr
python inference.py --fpath /tmp/processed/my_clip/pycrop/00000.avi
```

## Limitări

- Pipeline-ul MultiVSR detectează automat fețe; dacă videoclipul are mai
  mulți vorbitori, va produce un clip per persoană (face track).
- Clipurile prea scurte (<1s) sau cu ocluziune pe gură vor da rezultate
  slabe.
- Modelul a fost antrenat pe **podcast-uri în limba română** cu o singură
  persoană în cadru — performanța pe alte tipuri de conținut (filme,
  conferințe, conținut multilingual) nu a fost evaluată.
