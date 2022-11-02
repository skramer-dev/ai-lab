## Model objective
The mode gets an image of a pokemon and has to predict it's primary type according to the metatadata from the list below

## Data
Our training data is a mix of two datasets from Kaggle:

https://www.kaggle.com/datasets/lantian773030/pokemonclassification

The first dataset was fully used. We only used data from the second one from Pokemon that weren't already in the first (basically everything newer than gen 1 Pokemon)

https://www.kaggle.com/datasets/aaronyin/oneshotpokemon

The following was used to match metadata with the pokemon from the other two datasets:

https://www.kaggle.com/datasets/rounakbanik/pokemon

## Setup info
| type | value |
| --- | --- |
| without data augmentation | ~12.000 images |
| with data augementation | ~25.000 images |
| training duration | heavily depends on available hardware. On the 1070 that we used, anything from a few minutes to 750 mins for the longest run |

## Ordner

| Odner | Inhalt |
| ------------------ | ----------------- |
| *data-operations* | Alles was wir an Vorbereitung gebraucht haben (von Bilder Normalisiern und skalieren zu Liste mit Trainingsbeispielen erstellen) |
| *failed-approach* | Der komplette erste Ansatz mit dem Framework, welches wir initial benutzt hatten |
| *metadata* | csv Dateien mit den Infos über alle Pokemon (von Kaggle) und unsere Listen für alle Trainingsbeispiele einmal mit und einmal ohne data augmentation |
| *saved-models* | checkpoints von den fertig trainierten Modellen. Werden für Inferenz geladen und benutzt |
| *training-metrics* | csv style Historien von den jeweiligen Trainings. Namen sind gleich mit den zugehörigen saved-models |

## Notebooks

| Notebook | Inhalt |
| ------------------ | ----------------- |
| *Metric-visualization* | Visualisierung vom Verlauf der verschiedenen Trainingsdurchläufe |
| *Model-structure-exploration* | Wurde nur benutzt um Modellstruktur in Pytorch zu ermitteln (um die BatchNorm layer einzufrieren) |
| *ai-lab* | dump von unserem Conda env |
| *Inference* | Modelle laden und Beispielbilder klassifizieren lassen |
| *output* | Modellstruktur als txt |
| *torch-transfer-learning* | Herzstück des Projektes. Trainingsloop + early stopping + perfomance Metriken tracken und abspeichern |
| *visualize-dataAug-difference* | Visualisierung speziell für den Unterschied zwischen data aug und baseline |
