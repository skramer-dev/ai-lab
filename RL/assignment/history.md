# Ablauf der Versuchten Ansätze

## Aller erster Ansatz: Pytorch Implementierung
Alle online Tutorials, die Pytorch verwendet haben waren veraltet und haben nicht funktioniert.
Um nicht unnötig Zeit zu verlieren sind wir nach recht kurzer Zeit auf Keras/TF umgestiegen (obwohl unsere Expertise eher bei Pytorch liegt)

## TF Implementierung

Erster Ansatz Atlantis mit der ersten funktionierenden Version (50 Epochen, 50.00 Replay buffer start, linearer decay)
![grafik](https://user-images.githubusercontent.com/81179144/211892268-2148586e-15ff-423d-9a38-05bb225a4087.png)

Zweiter Ansatz mit leicht anderen Hyperparametern (Training war auch auf 50 Epochen angesetzt, 5.000 Replay buffer, nicht linearer decay)
![grafik](https://user-images.githubusercontent.com/81179144/211892381-363f29c5-b373-49ae-ae55-041b6882011a.png)

## Wechsel auf Pong mit der Hoffnung, dass das Training schneller ist

Verhalten bei Pong war quasi identisch zu Atlantis
![grafik](https://user-images.githubusercontent.com/81179144/211892508-54f0200c-f218-4414-8afd-0b2c624a57a4.png)

## Suche nach human scores und Erfahrungswerten bezüglich Training/Konvergenz von anderen Implementationen
https://arxiv.org/pdf/1710.02298.pdf als Referenz (im Appendix)

![grafik](https://user-images.githubusercontent.com/81179144/211894366-b953780b-5594-4b63-b4e9-d376e544674e.png)
![grafik](https://user-images.githubusercontent.com/81179144/211894450-b2a66feb-1f87-471e-afb8-8cddc76f43e1.png)

Pong sollte also recht schnell und auch unabhängig von der verwendeten Methode konvergieren. 
Atlantis hingegen ist ein komplizierteres Spiel und nicht alle Ansätze konnten es gleich gut und oder gleich schnell lösen.


## Ein paar Tips für die Implementierung
https://towardsdatascience.com/learnings-from-reproducing-dqn-for-atari-games-1630d35f01a9
(von den dort erwähnten hat jedoch keiner geholfen)

## Helper Funktionen die schon gestellt waren mitbenutzen
Training konnte gestartet werden, Trainingsverlauf war jedoch ähnlich schlecht.
Implementierung hat zumindest bei Atlantis zu sehr komischem Verhalten von dem env geführt (Score plot nicht vergleichbar mit anderen runs)

![grafik](https://user-images.githubusercontent.com/81179144/211895372-e8e5da91-68db-48d6-9f39-4674a3690737.png)

Beim Zuschauen ist aufgefallen, dass die Episoden sehr schnell vorbei sind (Spiel vorbei bei 1/7 zerstörten Gebäuden).
Scores sehr niedrig wegen verkürzten Episoden.

## Nach weiterer Suche ist aufgefallen, dass alle Angaben in Anzahl Lernschritte und nicht gespielten Episoden sind
Umstellung des Plots auf Trainingsschritte anstatt Episoden. Training dauert immer noch Ewigkeiten ohne erkennbaren Fortschritt.

![grafik](https://user-images.githubusercontent.com/81179144/211896824-edede87f-eb8f-46d1-be6e-2f7624758fbc.png)

## ...


