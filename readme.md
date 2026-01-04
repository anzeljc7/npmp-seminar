# SBML – Validacija in simulacija bioloških modelov

To orodje omogoča enostavno delo z modeli v formatu **SBML**:

- validacija SBML strukture s knjižnico **libSBML**
- numerična simulacija z **Tellurium**
- shranjevanje rezultatov simulacije v **CSV**
- samodejni izris grafa in generiranje **PNG slike** 


Orodje je primerno za delo z LLM-generiranimi modeli in validacijo proti referenčnim modelom.

---

## Namestitev

Najprej namestite potrebne odvisnosti:

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```
---

## Zagon validacije in simulacije

SBML modele (`.xml` ali `.sbml`) shranite v mapo `models/`.

Primer zagona:

```bash
python sbml_validate_and_simulate.py models/generated_model.xml
```

Po zagonu skripta:

- preveri veljavnost SBML
- izvede simulacijo
- ustvari rezultate:
  - `simulation.csv` – numerični izvoz podatkov  
  - `simulation.png` – graf časovnih potekov  
  - `JSON povzetek` – izpisan v terminalu

Rezultati simulacije se lahko primerjajo z referenčnimi modeli


## JSON  – struktura izpisa

| Polje | Tip | Opis |
|------|------|------|
| `sbml_valid` | boolean | Ali je SBML dokument veljaven. |
| `validation_errors` | list | Napake, zaznane med validacijo SBML. |
| `validation_warnings` | list | Opozorila validatorja libSBML. |
| `simulation_ok` | boolean | Ali je simulacija uspela. |
| `simulation_message` | string | Dodatne informacije o poteku simulacije. |
| `columns` | list | Imena simuliranih spremenljivk (vključno s časom). |
| `final_values` | object | Končne vrednosti spremenljivk ob zaključku simulacije. |
| `plot_file` | string | Pot do ustvarjenega PNG grafa. |
| `csv_file` | string | Pot do ustvarjene CSV datoteke. |

### Primer JSON izpisa

```json
{
  "sbml_valid": true,
  "validation_errors": [],
  "validation_warnings": [],
  "simulation_ok": true,
  "simulation_message": "OK",
  "columns": ["time", "ATP", "ADP", "Glucose"],
  "final_values": {
    "ATP": 1.45,
    "ADP": 0.22,
    "Glucose": 0.01
  },
  "plot_file": "simulation123.png",
  "csv_file": "simulation123.csv"
}
```