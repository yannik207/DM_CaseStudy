# DM_CaseStudy

Aufgabenstellung
Ziel der Analyse ist es, mit Hilfe des Datensatzes ein Modell zu trainieren, das dazu geeignet ist, Betrugsversuche
zu erkennen. Die Vorhersage Ihres Modells wird am Ende mit Hilfe eines weiteren Datensatzes
mit 100.000 Einkäufen geprüft, für den Sie die Zielvariable nicht kennen. Mit diesen Datensatz
(self_checkout_scoring.csv) wird mittels der Gesamtkosten bzw. des Gesamtertrags bewertet, wie gut
die Vorhersage Ihres Modells funktioniert. Das bedeutet, dass Sie sicherstellen müssen, dass beim Training
Ihres Modells kein Overfitting vorliegt, da die Vorhersage auf neuen Datensätzen ansonsten zu schlechten
Ergebnissen führen wird. Hierzu sollten Sie Ihren Datensatz in Trainings- und Testdaten aufteilen oder
eine geeignete Resampling Methode verwenden, um Overfitting zu vermeiden.
Gehen Sie dazu wie folgt vor:
1. Verschaffen Sie sich einen Überblick und ein Verständnis der vorliegenden Daten durch deskriptive
Analysen und grafische Darstellungen.
2. Säubern Sie die Daten falls notwendig und leiten Sie neue “schlaue” Variablen her, die für die
Vorhersagen genutzt werden können.
3. Nutzen Sie die Ihnen bekannten geeigneten Klassifkationsalgorithmen und erstellen Sie Vorhersagen.
Tunen Sie gegebenenfalls die Hyperparameter des Modells.
4. Messen Sie die Güte des Modells bzw. vergleichen Sie die Güte der Modelle und wählen ein finales
Modell. Nutzen Sie dazu statistische Kennzahlen aber vor allem die Gesamtkosten bzw. den
Gesamtertrag.
5. Ermitteln Sie, welche Merkmale sich gut zur Vorhersage eignen.
6. Veranschaulichen Sie Ihre Ergebnisse durch Tabellen und Abbildungen und interpretieren Sie diese.


Ergebnisse:
Bei der Bewertung unseres Modells haben wir uns für den Gewinn, der sich aus der Kosten- und Confusion Matrix ergibt, als primären Zielparameter entschieden, da dieser die größte betriebswirtschaftliche Relevanz aufweist. Als sekundäre Zielparameter haben wir zusätzlich die ROC und die AUC betrachtet. (s. Abbildung 1)
Folgende Ergebnisse sind vor bzw. nach Hyperparameter Tuning entstanden. Dabei wurden die optimalen Grenzwerte für die jeweils berechnete Kostenmatrix eingesetzt: (s. Tabelle 1)

Die optimalen Grenzwerte für die jeweiligen Kostenmatrizen (auch “Cost Optimal Cutoff” genannt) haben wir durch eine Funktion herausgefunden, die sowohl den Wert für den Cost Optimal Cutoff, als auch einen Graphen ausgibt, der die graphische Bestimmung des Cost Optimal Cutoffs ermöglicht. (s. Abbildung 2)

Das XGBoost Modell schneidet hinsichtlich der Zielparameter am besten ab und ist somit das Modell, das auf den Testdaten die beste Prognose liefert. Bei der Prognose auf den Testdaten unterlaufen dem Modell lediglich 239 Fehler bei 119.844 untersuchten Fällen. (s. Tabelle 2) Das entspricht einer Accuracy von 99,8%. Außerdem ergibt sich für das Modell eine Sensitivität von 97,6% und eine Spezifität von 99,9%. Die im Vergleich zur Sensitivität deutlich höhere Spezifität lässt sich damit erklären, dass falsch positive Vorhersagen mit einer deutlich höheren Strafe belegt werden, als falsch negative Vorhersagen.

Hyperparameter Tuning führt bei diesem Modell zu keiner signifikanten Verbesserung, was darauf hindeutet, dass bei erhöhten Hyperparametern Overfitting vorliegen könnte. Overfitting beschreibt das Problem, das sich das errechnete Modell zu stark auf die Daten angepasst hat, mit denen es trainiert wurde. Ein Versuch Overfitting zu vermeiden besteht darin, die Daten zufällig in Trainings- und Testdaten aufzuteilen. Das Modell wird mit den Trainingsdaten trainiert und anschließend mit den Testdaten getestet. Dieses Verfahren haben wir bei allen Modellen angewendet. Trotzdem können wir Overfitting bei keinem Modell gänzlich ausschließen. Schließlich bildet auch der gesamte Datensatz mit 400.000 Beobachtungen nur einen Ausschnitt aus der Gesamtheit ab. Die Daten könnten nur in einer bestimmten Filiale oder nur in einem bestimmten Zeitraum gesammelt worden sein und das Kundenverhalten könnte in anderen Filialen oder Zeiträumen abweichen.

Ein wichtiger Tuning-Parameter bei allen Boosting-Verfahren ist die Anzahl an durchgeführten Iterationen. Die Modellgüte steigt in der Regel mit der Anzahl der durchgeführten Iterationen. Allerdings bergen zu viele Iterationen auch die Gefahr des Overfittings. Daher haben wir uns den Trainings- und Testfehler der Modelle in Abhängigkeit von der Anzahl der Iterationen anzeigen lassen und einen Punkt gewählt, bei dem der Testfehler möglichst niedrig ist. (s. Abbildung 7)

Interessanterweise ergibt das nicht-hyperparameter optimierte GBM in der Kostenmatrix einen Verlust und ist damit das Modell mit dem geringsten Gewinn. Das liegt an dem relativ zu den anderen Modellen erhöhten Alpha Fehler (einen Kunden fälschlicherweise als Betrüger zu bezeichnen), welcher mit fünfmal so hohen Kosten bestraft wird, wie der Beta Fehler (einen Kunden nicht als Betrüger zu erkennen). Aus betriebswirtschaftlicher Sicht kann das durchaus Sinn machen, da solch eine Anschuldigung tendenziell eher dazu führt, dass ein Kunde gar nicht mehr in dem betrachteten Supermarkt einkauft und somit einen zukünftigen Customer Lifetime Value von 0 hat. Nach Parameter Tuning wird dieses Modell zur zweitbesten Option basierend auf den vorher definierten Zielparametern. Dieses Ergebnis unterstreicht das Potenzial von Parameter Tuning.

Beim Betrachten der Variable Importance der Prognosemodelle fällt auf, dass die Kovariable trustLevel im GBM ebenso wie im XGBoost-Modell eine große Rolle spielt. (s. Abbildungen 3, 5) Das bedeutet, dass unsere beiden besten Modelle die Ausprägung dieser Kovariable bei der Berechnung der Prognose ziemlich stark berücksichtigen. Weiterhin fällt auf, dass das trustLevel in den Modellen DecisionTree und ADABoosting eine deutlich kleinere Rolle spielt. (s. Abbildungen 4, 6) Es wäre sicherlich interessant zu wissen, weshalb das trustLevel speziell im ADABoosting Modell fast gar keine Rolle spielt und ob sich das Modell durch einen stärkeren Einbezug der Kovariable verbessern ließe.
Die Kovariablen scannedLineItemsPerSecond, valuePerItem und totalScanTimeInSeconds weisen in allen Modellen eine hohe Variable Importance auf. Daraus lässt sich schließen, dass vor allem diese Kovariablen besonders wichtig sind, um zu bestimmen, ob es sich in einem vorliegenden Fall um einen Betrug handeln könnte.
