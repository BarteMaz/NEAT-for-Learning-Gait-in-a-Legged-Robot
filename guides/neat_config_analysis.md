# Analiza konfiguracji NEAT po przebiegach z 25.11.2026

Dokument podsumowuje, dlaczego wszystkie dotychczasowe przebiegi treningu (pierwszy przebieg z 21.11 i trzy przebiegi z 25.11) utknęły w stagnacji, i co w konfiguracji `configs/a1.ini` za to odpowiada. Wnioski opierają się na kodzie źródłowym neat-python 2.0.0 oraz na pomiarach na zapisanych przebiegach.

## 1. Skąd pochodzą dane

Przebiegi (wszystkie: ziarno 1, populacja 100, 100 pokoleń, 12 procesów, ta sama konfiguracja `a1.ini`):

| przebieg | funkcja celu | filtr akcji α | najlepszy fitness | pokolenia, w których najlepszy wzrósł |
|---|---|---|---|---|
| `1st_major_run_1_2026-11-21_20-16-30` | v1 | 1,0 (brak) | 5,60 | 17, 25, 31, 35, 45 |
| `v1_filter_1_2026-11-25_12-12-42` | v1 | 0,5 | 5,04 | 17, 22, 31 |
| `v2_1_2026-11-25_12-19-11` | v2 | 1,0 | 5,63 | 22, 23, 24, 28, 32, 36, 70 |
| `v2_filter_1_2026-11-25_12-25-13` | v2 | 0,5 | 5,42 | 7, 11, 15, 34 |

Wspólny obraz:
- najlepszy fitness rośnie tylko 3-7 razy w ciągu 100 pokoleń, a od pokolenia 31-70 stoi w miejscu;
- średni fitness populacji przez cały przebieg jest bliski zera albo ujemny (−0,8 na początku, ok. −0,2 do +0,05 na końcu), mimo że najlepszy osobnik ma ponad 5;
- w `speciation.csv` przez cały przebieg jest jedna kolumna, czyli jeden gatunek.

Pomiary w tym dokumencie:
- odległości genetyczne i rozmiary sieci liczone na populacji z checkpointu pokolenia 100 każdego przebiegu (`neat-checkpoint-100`, 300 losowych par genomów);
- test potomków: 30 potomków zwycięzcy, ocenionych tym samym epizodem co w treningu (deterministycznie), dla kilku ustawień mutacji.

## 2. Problem A: specjacja nie działa

### Jak neat-python liczy odległość między genomami

`DefaultGenome.distance` (neat-python 2.0, `genome.py`) składa się z dwóch części: węzłów i połączeń. Dla połączeń:

- **gen wspólny** (ten sam klucz w obu genomach): `|w1 − w2|`, plus 1, jeśli jeden jest włączony, a drugi nie; całość razy `compatibility_weight_coefficient` (u nas 0,5);
- **gen rozłączny** (jest tylko w jednym genomie): `compatibility_disjoint_coefficient` (u nas 1,0);
- suma jest **dzielona przez liczbę genów połączeń w większym genomie**.

Część węzłowa wygląda analogicznie: różnica biasu i `response`, dzielona przez liczbę węzłów.

Wynik to więc **średnia różnica na gen**, a nie suma. Nasze genomy mają ok. 430 genów połączeń, bo start `full_direct` tworzy wszystkie 36 × 12 = 432 połączenia wejście-wyjście. Nawet duże różnice w kilkudziesięciu wagach rozkładają się na 430 genów i dają małą odległość.

### Pomiar

Odległości między parami genomów w populacji z pokolenia 100:

| przebieg | min | mediana | 90. percentyl | max | próg |
|---|---|---|---|---|---|
| 1st | 0,65 | 1,39 | 1,69 | 1,98 | 3,0 |
| v1 + filtr | 0,79 | 1,44 | 1,65 | 2,03 | 3,0 |
| v2 | 0,79 | 1,42 | 1,69 | 2,05 | 3,0 |
| v2 + filtr | 0,73 | 1,38 | 1,61 | 1,79 | 3,0 |

**Żadna para nie przekracza progu `compatibility_threshold = 3.0`**, więc każdy nowy genom trafia do istniejącego gatunku.

### Skutki

1. **Brak ochrony innowacji.** Specjacja w NEAT ma chronić nowe struktury (np. świeżo dodany neuron ukryty, który na początku obniża fitness), bo genom konkuruje tylko w swoim gatunku. Przy jednym gatunku każdy nowy pomysł konkuruje od razu z najlepszym osobnikiem całej populacji i zwykle przegrywa.
2. **Mechanizm stagnacji jest wyłączony.** `max_stagnation = 20` usuwa gatunki, które nie poprawiły się przez 20 pokoleń, ale `species_elitism = 2` chroni dwa najlepsze gatunki przed usunięciem. Przy jednym gatunku jest on zawsze chroniony, więc stagnacja nigdy nie prowadzi do odnowienia populacji.
3. **Brak różnorodności w reprodukcji.** Liczba potomków przypada na gatunki proporcjonalnie do ich dopasowania; przy jednym gatunku całą reprodukcją rządzi jeden zbiór 20% najlepszych (`survival_threshold = 0.2`).

### Zalecenie

- Obniżyć próg tak, żeby powstało kilka gatunków. Mediana odległości to ok. 1,4, więc próg rzędu **1,0-1,3** powinien dać kilka-kilkanaście gatunków. Dokładną wartość dobrać pomiarem (liczba gatunków w `speciation.csv` po kilku pokoleniach krótkiego przebiegu), z celem rzędu 5-10 gatunków przy populacji 100.
- Uwaga: po zmianie problemu B (poniżej) liczba włączonych połączeń i rozkład odległości się zmienią, więc próg trzeba dobrać **po** tamtej zmianie.
- W tej konfiguracji próg jest stały przez cały przebieg; jeśli liczba gatunków będzie się mocno zmieniać w czasie, do rozważenia są inne wartości `compatibility_weight_coefficient` i `compatibility_disjoint_coefficient`.

## 3. Problem B: zanik połączeń (zapadka wyłączania genów)

### Co zmierzyliśmy

| przebieg | geny połączeń (średnia) | **włączone** (średnia) | wyjścia z co najmniej jednym wejściem | używane wejścia (z 36) | zwycięzca: włączone |
|---|---|---|---|---|---|
| 1st | 428,6 | **16,6** | 8,2 z 12 | 10,7 | 60 |
| v1 + filtr | 429,4 | **13,6** | 7,0 z 12 | 9,9 | 106 |
| v2 | 420,4 | **8,0** | 4,6 z 12 | 6,3 | **11** |
| v2 + filtr | 426,9 | **13,5** | 6,5 z 12 | 9,4 | 110 |

Geny prawie nie są usuwane (ok. 430 z 432), ale **96-98% z nich jest wyłączonych**. Typowy genom w populacji steruje 5-8 stawami, a pozostałe dostają stałe wyjście `tanh(bias)`. Zwycięzca przebiegu v2 ma 11 włączonych połączeń i 2 z 12 wyjść bez żadnego wejścia; to dlatego jego biodra FR i RR oraz udo RR stoją w miejscu.

### Mechanizm (kod neat-python 2.0)

1. **Reguła 75% przy krzyżowaniu** (`genes.py`, `BaseGene.crossover`): jeśli którykolwiek z rodziców ma dany gen wyłączony, potomek dostaje go wyłączonego z prawdopodobieństwem 75%, niezależnie od tego, od którego rodzica wziął pozostałe atrybuty. To reguła z oryginalnego artykułu NEAT.
2. **Mutacja `enabled` jest słaba i symetryczna** (`attributes.py`, `BoolAttribute.mutate_value`): z prawdopodobieństwem `enabled_mutate_rate` (u nas 0,01) gen dostaje **losową** wartość (50% włączony). Ponowne włączenie konkretnego genu ma więc szansę ok. 0,5% na mutację.
3. **Dodanie połączenia, które już istnieje, nic nie robi** (`genome.py`, `mutate_add_connection`), chyba że `structural_mutation_surer = true`; wtedy istniejące połączenie jest włączane. U nas jest wartość domyślna (`default`, czyli równoważna `single_structural_mutation = false`), więc nie. Przy starcie `full_direct` prawie każda próba dodania połączenia wejście-wyjście trafia w istniejący gen, czyli `conn_add_prob = 0.5` w praktyce prawie nie dodaje połączeń.
4. Do tego `node_add` wyłącza dzielone połączenie (to też reguła NEAT).

Wyłączenie jest łatwe (75% przy każdym krzyżowaniu z rodzicem, który ma gen wyłączony), a włączenie prawie niemożliwe. Przy 432 genach od startu każde przypadkowe wyłączenie rozprzestrzenia się przez krzyżowanie i nie wraca. W oryginalnym NEAT start jest minimalny (kilka genów), a nowe połączenia powstają przez `add_connection`, więc ta asymetria nie ma takiego znaczenia.

### Skutki dla ewolucji

- Najlepsze genomy (60-110 włączonych połączeń) przeżywają dzięki elitaryzmowi, ale ich potomkowie z krzyżowania z typowym członkiem populacji mają medianę **22-24** włączonych połączeń. W teście potomków zwycięzców v1 + filtr i v2 + filtr (krzyżowanie z losowym członkiem populacji + mutacja) mediana fitness potomków wynosiła **0,00**: robot stoi albo przewraca się bez postępu.
- Stąd średni fitness populacji bliski zeru: większość populacji to sieci, które nie mogą chodzić, bo nie sterują wszystkimi stawami.

### Możliwe rozwiązania (do decyzji)

| rozwiązanie | co zmienia | koszt lub ryzyko |
|---|---|---|
| `structural_mutation_surer = true` | `add_connection` na istniejącym genie go włącza, więc `conn_add_prob` zaczyna działać jak „włącz połączenie” | w neat-python 2.0 przełącznik działa w dwóch miejscach: to opisane obok oraz `add_node` w genomie bez połączeń (zamiast nic nie robić, dodaje połączenie); przy `conn_add_prob = 0.5` włączanie może przeważyć nad regułą 75% i sieci zrobią się gęste, więc liczbę włączonych połączeń trzeba obserwować |
| mniejsze `conn_delete_prob` | mniej usuwania genów | nie dotyka głównej przyczyny (wyłączanie, nie usuwanie) |
| start rzadki: `initial_connection = partial_direct <p>` albo połączenie tylko wejść fazy | bliżej idei NEAT „od minimum w górę”; mniej genów do wyłączenia | ewolucja musi sama dodać połączenia od obserwacji; dłuższy start |
| `enabled_mutate_rate` + `enabled_rate_to_true_add` | dodatnie przesunięcie w stronę włączania | trzeba dobrać tak, żeby zrównoważyć regułę 75% |

Proponowany pierwszy krok: `structural_mutation_surer = true` i sprawdzenie pomiarem, czy średnia liczba włączonych połączeń w populacji przestaje spadać.

## 4. Problem C: mutacja niszczy rodziców

### Mechanizm (kod neat-python 2.0)

`FloatAttribute.mutate_value` dla **każdego** genu (wag i biasów, także genów wyłączonych) losuje r:
- r < `weight_mutate_rate` (0,8): waga += N(0; `weight_mutate_power`) = N(0; 0,5),
- r < `weight_mutate_rate` + `weight_replace_rate` (0,8 + 0,1): waga = nowa wartość z rozkładu początkowego N(0; 0,1),
- w pozostałych 10% bez zmian.

Dla zwycięzcy z ok. 110 włączonymi połączeniami jedna mutacja to średnio:
- ok. 88 wag przesuniętych o N(0; 0,5),
- ok. 11 wag **zastąpionych** wartością bliską zera.

Wagi zwycięzców mają medianę |w| ≈ 0,5-1,1 i maksimum 2,9-5,7. Zastąpienie oznacza więc skasowanie wyuczonych wag, a przesunięcie o 0,5 jest porównywalne z samą wagą. Do tego dochodzi ruchliwa struktura: `conn_add_prob` i `conn_delete_prob` po 0,5, `node_add_prob` i `node_delete_prob` po 0,2, a przy `single_structural_mutation = false` każda z nich może zajść w tej samej mutacji.

### Pomiar: 30 potomków zwycięzcy v2 + filtr (fitness rodzica 5,42), sama mutacja bez krzyżowania

| ustawienia mutacji wag i biasów | mediana fitness | ≥ 90% rodzica | ≥ rodzic | upadek lub ściana |
|---|---|---|---|---|
| **obecne**: rate 0,8, power 0,5, replace 0,1 | −0,01 | 0% | 0% | 50% |
| rate 0,8, power 0,1, replace 0,1 | 0,00 | 7% | 3% | 43% |
| rate 0,8, power 0,1, **replace 0** | 0,83 | 23% | 7% | 33% |
| rate 0,1, power 0,5, replace 0 | 0,49 | 3% | 0% | 30% |
| **rate 0,1, power 0,1, replace 0** | **1,56** | 23% | 7% | **10%** |
| jak wyżej i bez mutacji struktury | 1,04 | 30% | 13% | 33% |

To samo dla zwycięzcy v1 + filtr (obecne ustawienia): mediana −0,89, 0% potomków w 90% rodzica, 87% upadków; przy power 0,1: 13% potomków w 90% rodzica.

### Interpretacja

- Przy obecnych ustawieniach **żaden potomek nie zbliża się do rodzica**. Ewolucja nie może stopniowo poprawiać dobrego rozwiązania; poprawy zdarzają się tylko wtedy, gdy przypadkowa duża zmiana trafi w lepszy punkt. To tłumaczy rzadkie, skokowe poprawy i długie plateau.
- Największy pojedynczy winowajca to `weight_replace_rate = 0.1` (porównaj wiersze 2 i 3).
- Nawet przy łagodnej mutacji tylko ok. 23-30% potomków zostaje w 90% rodzica. Krajobraz fitness jest bardzo nierówny: robot w pętli sprzężenia zwrotnego z kontaktami jest układem, w którym mała zmiana wagi może zamienić chód w upadek. Mutacja musi więc być łagodna, a ocena deterministyczna (jest), żeby selekcja w ogóle widziała drobne poprawy.

### Zalecenie (punkt startowy do sprawdzenia przebiegiem)

- `weight_replace_rate = 0.0`, `bias_replace_rate = 0.0` (ewentualnie 0,01);
- `weight_mutate_power` i `bias_mutate_power` ok. 0,1-0,2;
- `weight_mutate_rate` i `bias_mutate_rate` niższe niż 0,8 (np. 0,1-0,3), bo przy ok. 100 aktywnych wagach nawet 10% to ok. 10 zmienionych wag na mutację.

## 5. Jak to się składa w obraz stagnacji

1. Mutacja (problem C) sprawia, że potomkowie dobrego genomu prawie nigdy nie są dobrzy.
2. Krzyżowanie (problem B) przenosi wyłączone geny z populacji na potomków elity, więc nawet oni tracą sterowanie częścią stawów.
3. Brak specjacji (problem A) sprawia, że nic nie chroni nowych struktur, a mechanizm stagnacji nie może odnowić populacji.

W efekcie populacja to głównie sieci, które nie chodzą (średni fitness ok. 0), elita przeżywa tylko dzięki `elitism = 2`, a najlepszy fitness rośnie wyłącznie przez rzadkie, przypadkowe trafienia.

## 6. Plan sprawdzenia zmian

Kolejność, bo problemy na siebie wpływają:

1. **Mutacja wag** (problem C): zmiana parametrów, test potomków (jak w tabeli w punkcie 4) na zwycięzcy jednego z przebiegów. Cel: znaczący odsetek potomków blisko rodzica i mało upadków.
2. **Zanik połączeń** (problem B): zmiana jednego z mechanizmów, krótki przebieg (np. 20 pokoleń) i średnia liczba włączonych połączeń oraz wyjść z wejściem w kolejnych checkpointach. Cel: liczba włączonych połączeń w populacji nie spada do kilkunastu.
3. **Próg zgodności** (problem A): dobór po krokach 1-2, na podstawie rozkładu odległości w nowej populacji i liczby gatunków w `speciation.csv`.
4. **Przebieg porównawczy:** ten sam wariant (np. v2 + filtr), to samo ziarno, nowa konfiguracja `a1.ini`. Porównać: liczbę popraw najlepszego fitness, pokolenie ostatniej poprawy, przebieg średniego fitness, liczbę gatunków, średnią liczbę włączonych połączeń.

Metryki z punktu 4 warto zapisywać na bieżąco w każdym przebiegu (planowany minimalny reporter CSV: liczba gatunków, średnia liczba neuronów i włączonych połączeń w populacji).

## 7. Zastrzeżenia

- Każdy wariant to jedno ziarno, a pomiary populacji pochodzą z jednego checkpointu na przebieg (pokolenie 100). Wnioski o mechanizmach (kod biblioteki, deterministyczne testy potomków) są mocne; dokładne wartości progów i parametrów trzeba potwierdzić na kilku ziarnach (blok 5).
- Test potomków ocenia potomków jednego genomu. Przy innym zwycięzcy liczby będą inne, ale kierunek (replace i duża moc mutacji niszczą) jest ten sam dla obu sprawdzonych zwycięzców.
