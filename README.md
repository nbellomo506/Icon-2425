# ICON-24-25
### Esame di ingegneria della conoscenza, UniBa
Realizzato da: Carbone Giuseppe Emanuele  
               Bellomo Nicolas  
               Lastella Nicola  
***
## Setup iniziale dell'ambiente di lavoro: 
**-Clonare**  
Per quanto riguarda l' esecuzione del progetto, bisogna innanzitutto clonare il repository eseguendo il seguente comando:  
```
git clone https://github.com/nbellomo506/Icon-2425
```
Il progetto è stato sviluppato interamente in Python nella versione **3.13** . Sarà necessario quindi Python.  
***
**-Requirements**  
Successivamente dopo aver installato Python, bisognerà installare le dipendenze necessarie all'esecuzione del progetto. Per far ciò eseguire il comando:  
```
pip install -r requirement.txt
```
che installerà automaticamente tutte le dipendenze elencate nel file `requirement.txt` garantendo il corretto funzionamento del progetto.  
***
## Esecuzione
All'interno del cmd, recarsi nella directory del progetto, eseguire i seguenti comandi:  
```
python main.py eda --grafici
```
Questo comando fa l’analisi esplorativa del dataset: statistiche, valori mancanti, distribuzione del target; con `--grafici` mostra anche istogrammi e matrice di correlazione. 
***
```
python main.py kb
```
Questo comando crea la Knowledge Base mappando feature vocali a nomi canonici, imposta soglie (raffinate dai dati) e salva regole in `knowledge_base.json`.  
***
```
python main.py train
```
Questo comando addestra il modello Random Forest, calibra le probabilità, trova le soglie ottimali (F1 di default) oppure inserire --thr youden, valuta su test e salva tutto in `parkinson_model.joblib`.
***
```
python main.py random_patient
```
Effettuabile solamente dopo aver eseguito il train. Il comando random_patient serve per creare in automatico un file JSON con i dati di un paziente generati in modo casuale ma plausibile, in modo tale da testare il modello senza dover inserire manualmente tutti i valori.
***
```
python main.py predict --json paziente_random.json --thr youden
```
Effettuabile solamente dopo aver eseguito il train, usa il modello salvato per predire lo stato di un paziente da file JSON; permette di scegliere la soglia (F1, Youden, o fixed).
***
```
python main.py reason --json paziente_random.json
```
Effettuabile solamente dopo aver eseguito il train, applica regole KB e calcola score fuzzy per un paziente da JSON.
***
```
python main.py all-in-one --csv parkinsons_disease_dataset.csv --json paziente_random.json --model parkinson_model.joblib --thr youden --out report.html
```
Il comando, effettuabile solo dopo aver eseguito il train e il predict, effettua una valutazione completa del rischio di Parkinson per un paziente: carica il dataset, il modello e i dati clinici, calcola la probabilità e la diagnosi usando la soglia di Youden, stima il rischio fuzzy e genera un report HTML con risultati e spiegazioni.
