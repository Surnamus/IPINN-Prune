# Fajlovi za objavljivanje

Fajlovi u ovom direktorijumu sada obuhvataju trening, CSV, grafikone, statistiku i izvoz konačnog modela. Originalni repo i aktivni trening nisu menjani.

## Obavezno za kompletan osnovni pipeline

- pruningalg.py: OG struktura sa dokumentovanim izmenama.
- train_dst.py: zajednički Adam+LBFGS trening; doslovna OG IPINN klasa.
- cfdsolver.py: originalni solver i učitavanje dataset-a.
- experiment_config.py: trenutni puni dizajn, seed-ovi 2022–2033, uglovi 0/45/90/135, proređenosti 0.1–0.9 i 0.95.
- run_experiments.py: sweep, dva radnika uz --workers 2; prvo Model1, zatim Model2, random i dense.
- main.py: evaluacija i CSV.
- plot.py: grafikoni iz CSV-a.
- mask_analysis.py: provera maski i memorijskih veličina.
- stats_analysis.py: Friedman i Wilcoxon sa Holm korekcijom i osnovni agregati.
- postprocess_sweep.py: pokreće CSV, osnovnu statistiku i grafikone.
- export_model.py: stvarni izvoz konačnih modela, veličine datoteka i provera rekonstruisanog modela.
- final_analysis.py: analiza na nivou seed-a, intervali, kriterijumi i ekvivalencija.
- angle_analysis.py: dodatna analiza uticaja uglova.
- analysis_criteria.json: granice za analizu.
- power_pilot.py: zavisnost final_analysis za eksploratorni proračun snage; može se preskočiti sa --skip-power.
- README.md, requirements.txt, .gitignore: uputstva i zavisnosti.

## Dodatne dijagnostike

gradient_diagnostics.py za odnose normi komponentnih gradijenata; convergence_check.py za nastavak treninga sa fiksnom maskom. Sačuvaj ih ako u radu koristiš ove rezultate.

## Minimalan redosled pokretanja

Iz ovog direktorijuma, u kompatibilnom PyTorch/ROCm okruženju:

```bash
python run_experiments.py --workers 2 --dataset datasets/static_dataset.pt --output runs
python postprocess_sweep.py --output runs --dataset datasets/static_dataset.pt --device cpu
python export_model.py --results runs
python final_analysis.py --results runs --criteria analysis_criteria.json --pilot-source runs --skip-power
python angle_analysis.py --results runs --criteria analysis_criteria.json
```

Ove komande su uputstvo za buduće pokretanje; novi sweep nije pokrenut ovim pakovanjem. --skip-power preskače proračun snage; za pilot analizu zadaj poseban pilot skup rezultata.

Konfiguracija prati puni izvedeni eksperimet, uključujući 0.95. Izveštaj koji isključuje 0.95 jeste zasebna podanaliza; za reprodukovanje tog izveštaja potrebni su filtrirani rezultati i odgovarajuća konfiguracija. Nemoj menjati opis već izvedenog punog dizajna u devet nivoa.

U repo dodaj dataset ako licenca i veličina dozvoljavaju ili javni link, generator, tačnu komandu i SHA256. Dataset korišćen u eksperimentima ima SHA256 f7be7ecba8c5148fab4410c14f43814451293ad040982ee68c07866eff740a43.

Originalne model1.py/model2.py/LBFGS skripte možeš zadržati u postojećem repo-u. Ulazi model1.py/model2.py u ovom paketu su tanki ulazi u train_dst; nisu minimalno izmenjene cele OG petlje. Ne moraš njima prepisati svoje OG fajlove: run_experiments poziva train_dst direktno.

Radni main.py treba koristiti za sadašnji format rezultata. Ako zadržavaš OG main.py pod istim imenom, obradu izdvoji u poseban direktorijum sa svim navedenim međusobnim zavisnostima.

Za rezultate rada dodaj finalne CSV/statističke tabele, konfiguraciju i informacije o verzijama okruženja. Ne dodavati venv, GPU core dump-ove ili nepotrebne Adam cache-eve. Pruning diff i verification.json dokumentuju promene i kratke CPU provere; puna ROCm ekvivalencija nije proveravana.

## Ograničene izmene u pruningalg.py

1. `_rigl_step`: drop kandidati su samo trenutno aktivne veze, a growth kandidati samo veze neaktivne pre ažuriranja. Broj zamena ograničen je brojem neaktivnih veza. Nove težine počinju od nule; Adam stanje se briše na izmenjenim pozicijama.
2. `reset_momentum`: maskira i Adam-ove tenzore stanja, pored SGD momentum-a.
3. `__init__`: čuva objekte koji nose score gradijente, ali ne registruje hook koji bi prerano nulirao gradijente neaktivnih veza. Maskiranje je eksplicitno u petlji. Automatski omotač optimizer.step nije uključen. Podrazumevano se linearni slojevi proređuju, kao u postojećoj radnoj verziji; OG model1/model2 su već eksplicitno birali False.
4. `apply_mask_to_gradients`: podržava gradijent None.
5. `__call__`: ne preskače Adam korak na ažuriranju topologije; fallback koristi gusti gradijent težine. Trenutna petlja koristi due/update, ne ovaj stariji ulaz.
6. `due/update`: mali adapter prema postojećoj petlji; update koristi originalni `_rigl_step` i originalne score objekte.
7. Checkpoint uključuje broj promenjenih veza i prihvata i postojeći ravni format. Kosinus koristi isti math.cos kao dosadašnja radna verzija.

Formule ostaju D=|W·g_drop| i G=|g_growth|. Za Model1: g_drop=cos(theta)·g_phys+sin(theta)·g_data, g_growth=g_phys; g_phys obuhvata IC+BC+PDE. Za Model2 oba score gradijenta koriste g_phys+g_data.

Odsečena veza može ponovo da poraste pri sledećem ili kasnijem ažuriranju, ali ne u istom ažuriranju.


## Sadržaj arhive

Ova arhiva sadrži novi pipeline i uputstva. Ne sadrži dataset niti originalne model1/model2 skripte: zadrži ih iz postojećeg repo-a. main.py i ostali fajlovi koji već postoje pod istim imenom zamenjuju se radnim verzijama samo u novoj grani. README objašnjava koji ulazi reprodukuju sadašnji protokol. Prilagodi --dataset stvarnoj putanji dataset-a u repo-u.
