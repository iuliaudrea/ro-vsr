# Fișiere de copiat din MultiVSR

Pentru ca codul de inferență să funcționeze, ai nevoie de **2 fișiere** din
[Sindhu-Hegde/multivsr](https://github.com/Sindhu-Hegde/multivsr), copiate
direct în acest folder (`ro_vsr/`):

- `models.py` — definiția arhitecturii encoder-decoder + visual encoder
- `tokenizer.py` — tokenizer Whisper

Cea mai simplă cale este să rulezi:

```bash
bash scripts/setup.sh
```

care le clonează automat din repo-ul lor și le copiază aici.

## De ce nu sunt incluse direct în repo?

Repo-ul MultiVSR nu are fișier `LICENSE`, deci redistribuția codului lor
este într-o zonă gri legal. Soluția curată este să clonăm la setup.

## Modificare posibil necesară după copiere

`tokenizer.py` din MultiVSR conține o variabilă `tokenizer = ...` la nivel
de modul. Codul nostru îl importă printr-o funcție `get_tokenizer()`. Dacă
aceasta nu există încă în versiunea pe care o copiezi, adaug-o la sfârșit:

```python
def get_tokenizer():
    return tokenizer
```

Fără alte modificări, restul fișierelor sunt folosite ca atare.

## Modulul nostru `beam_search_ngram.py`

Beam search-ul cu n-gram blocking este complet original și NU depinde de
fișierul `search.py` din MultiVSR. Conține tot ce e nevoie pentru decodare,
inclusiv `subsequent_mask` (extras separat în `dataloader_utils.py`).
