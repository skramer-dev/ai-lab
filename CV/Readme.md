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
