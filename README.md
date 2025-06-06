## Über das Projekt

*i2pnodesniffer* ist ein Forschungs-Repository, das den Traffic von **I2P-Knoten** mitschneidet und versucht, mit maschinellem Lernen einen Service zu de-anonymisieren.

Unter anderem bietet es das **Convolutional Neural Network (CNN)**, um charakteristische Sequenzen im I2P-Datenstrom automatisch zu erkennen.

### Analytischer Fokus  
* **Datenexport**: Rohdaten werden in strukturierte CSV-Formate überführt, die direkt als Eingabe für das CNN dienen.  
* **Modelle-Trainieren**: Ein PyTorch-/Keras-Modul trainiert unter anderem ein mehrschichtiges CNN auf den exportierten Sequenzen und ermöglicht sofortige Klassifikation.  
* **Notebook basierte Auswertung**: Jupyter-Notebooks ermöglicht eine Schrittweise Analyse der Rohdaten

### Projektstruktur
* ip2_network_traffic_patterns_*: Per Jupyter Notebook eine Schrittweise Analyse der aufgezeichneten Netzwerkdaten.
* csv_export_for_*: Aufbereitung der Rohdaten und Generierung von CSV-Dateien für das Training.
* ml/*: Enthält die Trainingskripte für die CNN-Modelle als auch Validierungs- und Evaluierungscode
Unter dem Ordner ml sind einerseits die Scripts zum Trainieren der Modelle, sowie auch für die validation.

### Einbettung in die I2P-Laborumgebung  
Das Tool wurde ergänzend für [`i2pd-testnet-kubernetes`](https://github.com/h-phil/i2pd-testnet-kubernetes) entwickelt, das eine komplette I2P-Testumgebung in Kubernetes bereitstellt. Dadurch lassen sich reproduzierbare Experimente fahren, deren Traffic mit diesem Repository analysiert werden kann.