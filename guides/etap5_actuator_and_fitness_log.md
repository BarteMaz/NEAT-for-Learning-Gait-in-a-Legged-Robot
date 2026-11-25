# Etap 5: od exploitu „wibracji” do filtru akcji i fitness_v2

Zapis kolejnych problemów, pomiarów, wniosków i poprawek w sterowaniu robotem (model napędu) i w funkcji celu, od pierwszego pełnego przebiegu (21.11.2026) do przebiegów porównawczych 2 × 2 (25.11.2026). Kolejność jest chronologiczna. Każdy krok: co zauważyliśmy, jak to zmierzyliśmy, co wyszło, co zmieniliśmy.

Układ wspólny dla wszystkich pomiarów: robot Unitree A1, tor płaski 6 × 3 m, dystans do mety D = 4 m, limit czasu T = 10 s, krok fizyki dt = 1/240 s, sterowanie co k = 4 kroki (60 Hz), `POSITION_CONTROL` z momentem ograniczonym do `effort` z URDF. Baseline: kłus otwartopętlowy f = 3 Hz, amplituda uda 0,4, łydki 0,3 (w jednostkach akcji).

---

## 1. Problem: zwycięzca pierwszego przebiegu nie chodzi, tylko „wibruje”

**Obserwacja (21.11).** Pierwszy przebieg (v1, 100 × 100) dał zwycięzcę z fitness 5,6 (baseline 4,52). W `replay.py` robot nie stawiał kroków, tylko bardzo szybko drgał nogami i dojeżdżał do mety po 2,0 s (ok. 2 m/s, prawie 4 razy szybciej niż baseline).

**Funkcja celu v1:** postęp wzdłuż x obcięty do [0, D] + przy mecie premia 2 · (T − t)/T; przy upadku lub ścianie stała kara 1.
- baseline: 3,99 + 2 · (10 − 7,38)/10 = 4,52,
- zwycięzca: 4,0 + 2 · (10 − 2,0)/10 = 5,6.

**Pytanie:** czy to luka w funkcji celu, czy w symulacji (nierealistyczny napęd albo artefakt solvera kontaktu)? Od odpowiedzi zależy, co poprawiać.

## 2. Narzędzie: skrypt diagnostyczny

**Co zbudowaliśmy** (`scratch/etap5/diagnose.py`): odtworzenie epizodu w trybie `DIRECT` z nagrywaniem przez opakowanie polityki `policy(obs, t)`, bez zmian w `run_episode`. Metryki na każdy staw:
- max |prędkość| (z obserwacji × 10) wobec limitu prędkości z URDF i odsetek przekroczeń,
- średnia |Δa| między krokami sterowania,
- dominująca częstotliwość akcji (FFT po odjęciu średniej) i amplituda piku,
- udział energii widma powyżej 10 Hz,
- oraz liczba zmian kontaktu stopy z podłożem na sekundę.

**Zasady metody:**
- `np.fft.rfftfreq(n, d = 1/60)` z liczbą próbek n, nie z długością wyniku FFT; rozdzielczość Δf = 60/n (zwycięzca 2 s: 0,5 Hz; baseline 7,38 s: 0,135 Hz), górna granica 30 Hz (Nyquist przy 60 Hz).
- Bez odjęcia średniej koszyk 0 Hz przykrywa pik (dla 0,3 + 0,5·sin(2π·3t), n = 120: 36 wobec 30).
- Pik czytany razem z amplitudą (pik 30 Hz o amplitudzie 0,006 to szum).
- Udział energii powyżej 10 Hz jako odporniejsza miara (sinusoida 3 Hz ≈ 0, przerzut co krok = 1, szum ≈ 0,67).
- Narzędzie najpierw sprawdzone na sygnałach o znanej odpowiedzi, potem na baseline'ie, dopiero potem na zwycięzcy.

**Błędy znalezione w pierwszej wersji skryptu** (warte zapamiętania): długość wyniku FFT zamiast liczby próbek, średnia |Δa| odejmowana zamiast średniej akcji, prędkości bez skalowania, `lista * 10` (powtórzenie listy zamiast mnożenia), `argparse` z `type=bool`, czas epizodu wpisany na sztywno.

## 3. Wynik diagnozy

| | baseline | zwycięzca pierwszego przebiegu |
|---|---|---|
| pik widma ud i łydek | 2,98 Hz, amplituda 0,38 | 7 stawów: **30 Hz**, amplituda 0,8-1,0 |
| średnia \|Δa\| | 0,08 (uda) | 1,6-2,0 na 7 stawach (maksimum możliwe to 2) |
| udział energii > 10 Hz | ≈ 0 | 0,98-1,0 |
| pozostałe stawy | biodra stoją (z założenia) | FR_upper, FL_hip, RR_hip amplituda < 0,04; RL_upper stały |
| zmiany kontaktu na sekundę | 6,1-6,2 (kłus 3 Hz: postawienie i oderwanie w każdym cyklu) | 20,5-26,5 |
| max \|prędkość stawu\| | 2,7-3,0 rad/s | ≤ 19 rad/s, brak przekroczeń limitu |

**Wniosek:** robot przerzuca 7 stawów pełnym zakresem z częstotliwością 30 Hz, czyli w każdym kroku sterowania, a pozostałe trzyma prawie nieruchomo.

## 4. Hipoteza: napęd przekracza limit prędkości

**Rozumowanie:** pełny przerzut 1,05 rad w 1/60 s wymaga średnio ok. 63 rad/s, a limit uda i łydki z URDF to 28,6 rad/s. `POSITION_CONTROL` ograniczał tylko moment, nie prędkość.

**Implementacja:** `RobotSpec.apply_velocity_limit` (domyślnie `False`), limity prędkości z URDF, `setJointMotorControl2(..., maxVelocity=...)` na każdy staw (`setJointMotorControlArray` nie ma `maxVelocity`). Odrzucona alternatywa: ograniczenie zmiany celu |Δq| ≤ v_max·k·dt, bo to filtr komendy, a nie limit fizyki (cel przesunięty o 0,48 rad osiągany w 1-2 krokach fizyki daje chwilowo ok. 58 rad/s).

**Pomiar (krok 4):** zwycięzca z limitem z URDF — wynik **identyczny co do bitu** jak bez limitu.

**Wyjaśnienie:** pomiar prędkości w każdym kroku fizyki przy akcji przerzucanej co krok na stojącym A1:

| wariant | max \|q̇\| [rad/s] |
|---|---|
| bez limitu | 16,6-22,7 |
| limit z URDF (28,6 i 52,4) | identycznie 16,6-22,7 |
| limit sztucznie 5 rad/s | 5,0-6,5 |

Prędkość docelowa regulatora `POSITION_CONTROL` to ok. kp·Δq/dt (dokumentacja PyBullet, str. 27). Przy domyślnym kp = 0,1 i Δq = 1,05 rad: 0,1 · 1,05 · 240 ≈ 25 rad/s. Domyślne wzmocnienie samo działa jak ukryty limit prędkości, poniżej limitów z URDF. **Hipoteza odrzucona:** przyczyną nie jest przekroczenie limitu prędkości. Pierwotne przypuszczenie, że pomiar co 4 kroki ukrywa przekroczenia, też się nie potwierdziło.

## 5. Test kroku fizyki: regulator czy solver kontaktu?

**Krok 5:** dt = 1/480, k = 8 (ta sama częstotliwość sterowania), kp = 0,1 bez zmian. Zmienia się dokładność fizyki, ale też regulator (ukryty limit prędkości rośnie do ok. 50 rad/s).

| | baseline | zwycięzca |
|---|---|---|
| wynik | meta po 6,38 s (było 7,38), prędkości stawów prawie 2 razy wyższe | kontakty o ponad 10/s mniej, obrót, koniec na ścianie, przekroczenia limitów prędkości |

Baseline zmienia się **ilościowo**, zwycięzca **jakościowo**. Baseline był tu kontrolą: bez niego można by wnioskować, że symulacja przy 1/240 jest niedokładna albo że każdy chód jest tak wrażliwy.

**Krok 5b:** dt = 1/480, k = 8 i **kp = 0,05**, czyli ta sama prędkość docelowa regulatora kp·Δq/dt, a tylko dokładniejsza fizyka. Baseline: meta po 7,22 s. Zwycięzca: znowu meta po 2,0 s.

**Wniosek:** kruchość zwycięzcy wynika z **dynamiki regulatora** (w praktyce modelu napędu), a nie z solvera kontaktu. Rozwiązanie znalezione przez ewolucję jest dopasowane do symulatora, a nie do zadania.

**Poprawka po drodze:** `RobotSpec.position_gain` (domyślnie 0,1, czyli domyślna wartość PyBullet). Jawne 0,1 daje ruch identyczny co do bitu z pominięciem argumentu.

## 6. Staw jako filtr dolnoprzepustowy

**Spostrzeżenie:** równanie stawu w `POSITION_CONTROL` bez nasycenia to q[n+1] = q[n] + kp · (q_cel − q[n]), czyli filtr wykładniczy (EMA) z α = kp, liczony 240 razy na sekundę. Stała czasowa ok. 39,5 ms, częstotliwość odcięcia ok. 4 Hz.

**Pomiar** (A1 zawieszony, udo FR, sinusoidalny cel o amplitudzie 0,1 rad):

| f celu | kp = 0,1: zmierzone | kp = 0,1: wzór EMA | kp = 0,05: zmierzone | kp = 0,05: wzór EMA |
|---|---|---|---|---|
| 1 Hz | 0,971 | 0,971 | 0,891 | 0,891 |
| 3 Hz | 0,802 | 0,802 | 0,547 | 0,547 |
| 10 Hz | 0,374 | 0,374 | 0,193 | 0,193 |
| 30 Hz | 0,136 | 0,136 | 0,067 | 0,067 |

Zgodność do trzeciego miejsca po przecinku. To wyjaśniło dwie wcześniejsze obserwacje:
- baseline dostaje 80% zadanej amplitudy przy 3 Hz (0,8 · 4,0 ≈ 3,2 rad/s wobec zmierzonych 2,7-3,0); przy dt = 1/480 odcięcie przesuwa się do ok. 8 Hz, stąd szybszy baseline w kroku 5;
- „wibracja” zwycięzcy ma amplitudę stawu ok. 0,136 · (4/π · 0,525) ≈ 0,09 rad (ok. 5°), co daje 2π · 30 · 0,09 ≈ 17 rad/s, zgodnie z pomiarem. **Robot jechał 2 m/s na drganiach stawów o amplitudzie ok. 5°.**

## 7. Poprawka napędu: filtr EMA na akcjach

**Wybór:** zamiast obniżać kp (przesuwa odcięcie istniejącego filtru, ale przy kp = 0,05 chód przy 3 Hz traci prawie połowę amplitudy) dodaliśmy drugi filtr pierwszego rzędu na akcjach (60 Hz), szeregowo ze stawem. Dwa filtry pierwszego rzędu dają filtr drugiego rzędu, który lepiej oddziela częstotliwość chodu od drgań.

**Teoria:** y[n] = α·u[n] + (1 − α)·y[n−1], y[−1] = 0 (poza stania).
- Odpowiedź na skok: 1 − (1 − α)^(n+1); stała czasowa τ = −Ts / ln(1 − α).
- Przerzut ±1 co krok ustala się na ±α/(2 − α).
- Wzmocnienie dla sinusoidy: |H| = α / |1 − (1 − α)·e^(−jω)|, ω = 2πf/60.
- Sieć może skompensować filtr w paśmie chodu (opóźnienie fazowe przez wagi przy wejściach sin i cos, spadek amplitudy przez większą komendę), ale nie przy 30 Hz, bo komenda nie przekroczy ±1. Filtr nakłada więc fizyczny limit amplitudy drgań.

| α | τ | f_c | 3 Hz: wzmocnienie, opóźnienie | 30 Hz |
|---|---|---|---|---|
| 0,7 | 14 ms | 11,5 Hz | 0,97, 7 ms | 0,54 |
| 0,5 | 24 ms | 6,6 Hz | 0,91, 15 ms | 0,33 |
| 0,3 | 47 ms | 3,4 Hz | 0,75, 31 ms | 0,18 |

Razem ze stawem (kp = 0,1): przy α = 0,5 wzmocnienie 0,73 przy 3 Hz i 0,045 przy 30 Hz (stosunek 16 wobec 5,9 dla samego stawu).

**Implementacja:** `RobotSpec.action_filter_alpha` (domyślnie 1,0, czyli brak filtru), stan filtru w `Robot`, zerowany w `reset()`, akcja obcinana do [−1, 1] przed filtrem.

**Błędy znalezione testami:**
- filtr zmieniał akcję wywołującego w miejscu i trzymał ją jako swój stan (metryka |Δa| w `run_episode` widziałaby akcję przefiltrowaną, a nie surową);
- wzór prev + α·(u − prev) przy α = 1 nie zawsze daje dokładnie u (0,3 + (−0,9 − 0,3) = −0,8999999999999999), więc stare przebiegi nie odtwarzałyby się co do bitu; poprawna postać α·u + (1 − α)·y.

**Wybór α (krok B):**

| | α = 0,5 | α = 0,3 |
|---|---|---|
| baseline | meta po 7,88 s (było 7,38), prosto | meta po 9,52 s, większe zbaczanie |
| stary zwycięzca | stoi: kontakty z ok. 26 do 1,8 na sekundę, koniec na limicie czasu | — |

Przewidywania teorii się zgadzają: przy α = 0,5 amplituda chodu spada o ok. 9%, a czas wydłuża się o ok. 7%; przy α = 0,3 amplituda spada o ok. 25%, a czas o 29%. Drganie starego zwycięzcy po filtrze to ok. 0,03 rad (< 2°) i już nie napędza robota.

**Decyzja: α = 0,5** (najsilniejsze tłumienie, przy którym baseline nadal chodzi prawie tak samo). Fitness v1 baseline'u z filtrem: 4,41.

## 8. Poprawka funkcji celu: fitness_v2

**Słabości v1:**
- premia czasowa nie ma sensownej górnej granicy: wśród robotów, które doszły do mety, różni je tylko czas, więc ewolucja optymalizuje wyłącznie szybkość, niezależnie od sposobu ruchu;
- nic nie opisuje sposobu ruchu: średnia |Δa| baseline'u ≈ 0,05 (teoretycznie 4·A·f/f_s = 4 · 0,4 · 3/60 = 0,08 dla uda), zwycięzcy 1,17, a fitness tego nie widzi.

**Projekt v2** (sposób łączenia wybrany przez użytkownika: iloczyn):
- postęp jak w v1,
- premia za szybkość z prędkością docelową v_t i ciągłością na granicy mety: przy mecie po czasie t, v = postęp / t, v_min = D/T = 0,4 m/s, premia = w · clip((v − v_min)/(v_t − v_min), 0, 1), w = 2, v_t = 1,0 m/s,
- czynnik gładkości s = min(1, m₀/m), m = średnia |Δa| surowej akcji sieci (po stawach i krokach, pierwsza akcja względem 0), m₀ = 0,2,
- fitness = (postęp + premia) · s − kara; kara odejmowana po iloczynie.

**Sam limit prędkości nie usuwa exploitu:** zwycięzca dostałby premię 2,0 (v1: 1,6). Exploit usuwa dopiero czynnik gładkości:

| | v1 | v2 |
|---|---|---|
| baseline | 4,52 | (3,99 + 0,47) · 1 = **4,46** |
| zwycięzca pierwszego przebiegu | 5,6 | (4,0 + 2,0) · 0,17 = **1,03** |

**Błędy znalezione przy review i testach:**
- `np.min(1, x)` (drugi argument `np.min` to oś),
- dzielenie przez zero przy m = 0 (sieć ze stałym wyjściem),
- kara wewnątrz iloczynu: (0 − 1) · 0,17 = −0,17 wobec −1 przy gładkim upadku, czyli drganie przy upadku by się opłacało,
- v_t = D/T daje dzielenie przez zero: walidacja v_t > D/T w `TrainingConfig` (od razu wykryła dwie konfiguracje testowe ze skróconym epizodem),
- metryka |Δa| w `run_episode` liczona ze znakiem (drganie ±1 dawało średnią ok. 0, czyli metryka była ślepa na exploit),
- dodatkowe wywołanie polityki przed pętlą epizodu (przesuwało polityki ze stanem, nagrywanie i podgląd w czasie rzeczywistym).

**Infrastruktura:** `FitnessFunction` (enum z wartościami tekstowymi) i rejestr `FITNESS` zamiast enuma z funkcjami (funkcja w ciele `Enum` jest deskryptorem i nie staje się członkiem; nazwa w enumie jest też stabilna w pickle i JSON), pole `TrainingConfig.fitness_function` (domyślnie v1, więc stare przebiegi wczytują się jako v1), argumenty `train.py`: `--fitness` i `--filter_alpha`. Kontrola po zmianach: `replay.py` starego zwycięzcy nadal daje dokładnie 5,6.

## 9. Przebiegi porównawcze (25.11)

Trzy przebiegi z ziarnem 1, 100 × 100, na tym samym komputerze, porównywane z pierwszym przebiegiem. Fitness v1 i v2 przeliczone z tych samych metryk epizodu.

| przebieg | czas do mety | max \|y\| | max \|yaw\| | m | v1 | v2 | kontakty/s | pik widma |
|---|---|---|---|---|---|---|---|---|
| 1st (v1, bez filtru) | 2,00 s | 0,20 | 0,98 | 1,165 | 5,60 | 1,03 | ok. 26 | 30 Hz |
| v1 + filtr | 4,78 s | 0,89 | 0,44 | 0,208 | 5,04 | 5,23 | 11-14 | 6,3 Hz |
| v2 bez filtru | 4,48 s | 0,70 | 0,84 | 0,114 | 5,10 | 5,63 | 5-18 | 2,9 Hz i 30 Hz (RL_lower) |
| v2 + filtr | 4,80 s | 1,18 | 1,04 | 0,157 | 5,03 | 5,42 | 5-9 | 2,9 Hz |
| baseline (α = 0,5) | 7,88 s | 0,20 | 0,11 | 0,047 | 4,42 | 4,35 | 6 | 3,0 Hz |

Test odporności (ustawienie z kroku 5b: dt = 1/480, k = 8, kp = 0,05): zwycięzca v1 + filtr uderza w ścianę po 3,1 s; zwycięzcy v2 i v2 + filtr oraz baseline dochodzą do mety.

**Obserwacje w `replay.py`:**
- v1 + filtr: ruch najbardziej przypomina chód, ale jest ok. 2 razy bardziej gwałtowny niż baseline (dwa razy więcej kontaktów);
- v2 bez filtru: nogi FL, RR i RL drgają albo stoją, napęd daje głównie noga FR, której ruch najbardziej przypomina chód;
- v2 + filtr: ruch najmniej gwałtowny, ale przypomina krótkie skoki zająca z zygzakiem wokół osi x.

**Wnioski:**
1. **Filtr usunął drgania 30 Hz.** Oba przebiegi z filtrem mają pik w okolicach zegara fazy: 2,9 Hz (v2 + filtr, czyli sieć korzysta z wejść sin i cos) i 6,3 Hz (v1 + filtr, druga harmoniczna 3 Hz, krok dwa razy na cykl).
2. **v2 bez filtru nie usunęła drgań z powodu luki w metryce:** m to średnia po 12 stawach. Zwycięzca drga jednym stawem (RL_lower, |Δa| = 0,88, 30 Hz) i kilka trzyma stale, więc m = 0,114 < m₀ i czynnik gładkości wynosi 1. W przebiegu v2 + filtr też s = 1 (m = 0,157), więc z części v2 zadziałał tam tylko limit prędkości.
3. **Wszyscy nowi zwycięzcy idą szybko (ok. 0,85 m/s, blisko v_t), ale krzywo:** max |y| do 1,18 m przy ścianach na ±1,5 m, yaw do 60°. Żadna wersja funkcji celu nie opisuje kierunku, a ściany kończą epizod dopiero przy kontakcie.
4. Jedno ziarno na wariant: wnioski jakościowe (widmo, drgania pojedynczego stawu, zbaczanie) są mocne, różnice w fitness między przebiegami są w granicach przypadku.
5. Wszystkie przebiegi utknęły w stagnacji z przyczyn leżących w konfiguracji NEAT (osobny dokument: `neat_config_analysis.md`).

## 10. Co zostaje otwarte w funkcji celu

- **Gładkość per staw**, żeby jeden drgający staw nie chował się w średniej po 12 stawach (np. maksimum po stawach ze średniej |Δa|: baseline ok. 0,08, zwycięzca v2 0,88).
- **Składowa kierunku lub odchylenia bocznego** (yaw, |y|).
- Potem: konfiguracja NEAT, a na końcu powtórzenia na kilku ziarnach (blok 5).

## 11. Lekcje metodyczne

- Jedna zmiana na przebieg albo układ czynnikowy (2 × 2), żeby każdy efekt dało się przypisać przyczynie.
- Baseline jako kontrola przy każdej zmianie warunków symulacji.
- Narzędzie pomiarowe sprawdzać najpierw na sygnale o znanej odpowiedzi.
- Nowe pola konfiguracji z wartością domyślną odtwarzającą dotychczasowe zachowanie, sprawdzone testem co do bitu, żeby stare przebiegi odtwarzały się bez zmian.
- Fitness z różnych funkcji celu porównywać przez metryki fizyczne, a nie wprost.
- Parametry regulatora (kp) i krok fizyki (dt) opisywać razem: kp/dt wyznacza szybkość stawu.
