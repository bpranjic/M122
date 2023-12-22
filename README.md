# M122

In der Aufgabe muss man ein Modell trainieren, welches anhand von 3 Bildern von Essen, A, B und C, entscheiden muss, ob A im Geschmack näher zu B oder C ist.

1. Die Grösse aller Bilder anpassen
2. Das vortrainierte InceptionResNetV2 Model von Keras nutzen, um die Features von den Bildern zu extrahieren
3. Tensoren für Training und Testing erstellen
4. Um die Genauigkeit des Modells zu erhöhen werden die Features von B und C (inklusive) Label vertauscht
5. Tensorflow Model wird mit 5 Dense Layers erstellt.
6. Dropout Funktion um Overfitting in dem Model zu verhindern.
7. Relu Aktivierungsfunktion
8. Sigmoid Output Funktion
9. Das Model verhält sich wie eine Binäre Klassifikation, wenn der Wert kleiner als 0.5 ist, wird B gewählt, sonst C
