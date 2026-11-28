# Kalibracja tarcia podłoża: eksperyment z równią pochyłą

Skrypt: `physics/friction_incline.py`, wyniki: `physics/friction_results.json`.
Uruchomienie z katalogu głównego repozytorium:

```
uv run python physics/friction_incline.py
```

Całość trwa kilka sekund (tryb `DIRECT`). Ustawienia symulacji (`dt = 1/240 s`, 50 iteracji solvera,
`g = 9.81 m/s²`) pochodzą z `SimulationData`, więc są takie same jak w treningu.

## 1. Cel

Na nagraniach w zwolnionym tempie robot się ślizga. Trzeba sprawdzić, jaki współczynnik tarcia
Bullet faktycznie stosuje między stopą A1 a podłożem areny. Potem należy dobrać parametr `friction`
podłoża tak, żeby efektywne tarcie odpowiadało rzeczywistej parze materiałów: gumowej stopie
na betonie.

## 2. Z czego jest stopa A1 i co o niej mówi URDF

**Rzeczywisty robot.** Unitree nie podaje w specyfikacji A1 materiału stopy. Pisze tylko, że końcówka
nogi zawiera czujnik siły, jest wodo- i pyłoszczelna oraz łatwo wymienna
([unitree.com/a1](https://www.unitree.com/mobile/A1)). Zestawienie specyfikacji A1 w serwisie
Origin of Bots podaje materiały „aluminum, rubber foot pads”
([originofbots.com](https://www.originofbots.com/robot-dog/unitree-a1-by-unitree-robotics-robot-dog-specifications)).
Zamienne nakładki stóp nowszych robotów Unitree (Go2, B2) są z kauczuku naturalnego
([Go2 foot pad](https://robotshop.com/products/unitree-go2-foot-pad-pair),
[B2 foot pad](https://www.robotshop.com/products/unitree-foot-pad-b2-robotic-dog)).
**Przyjmuję, że stopa A1 jest gumowa.** To założenie oparte na źródłach pośrednich, nie na
oficjalnej specyfikacji A1, i tak warto je opisać w pracy.

**Model w symulacji** (`pybullet_data/a1/a1.urdf`, odczytane przez `getDynamicsInfo`):

| Parametr linku `*_toe` | Wartość | Skąd |
|---|---|---|
| kształt kolizyjny | kula, r = 0,02 m | URDF |
| `lateral_friction` | 0,5 | domyślna PyBullet: URDF nie ma sekcji `<contact>` |
| `rolling_friction`, `spinning_friction` | 0, 0 | domyślne |
| `restitution` | 0 | domyślna |
| `contact_stiffness`, `contact_damping` | −1 (brak) | kontakt sztywny |

Wszystkie linki A1 (korpus, uda, łydki, stopy) mają `lateral_friction = 0,5`. Taką samą wartość
domyślną ma pudełko zbudowane jak podłoże areny (`createMultiBody` bez `changeDynamics`).

## 3. Tarcie guma–beton w źródłach

| Źródło | Para | μ statyczny | μ kinetyczny |
|---|---|---|---|
| Engineering ToolBox ([link](https://engineeringtoolbox.com/friction-coefficients-d_778.html)) | guma / suchy beton | — | 0,6–0,85 |
| Engineering ToolBox | guma / mokry beton | — | 0,45–0,75 |
| Serway, Beichner, *Physics for Scientists and Engineers*, tab. 5.2 (za [hypertextbook.com](https://hypertextbook.com/facts/2006/MatthewMichaels.shtml)) | guma / beton (sucho) | 1,0 | 0,8 |
| SIU Center for Advanced Friction Studies (za hypertextbook.com) | guma / beton | 1,0–4,0 | 0,8 |
| University of Florida, Design Lab ([link](https://web.mae.ufl.edu/designlab/Class%20Projects/Background%20Information/Friction%20coefficients.htm)) | miękka / twarda guma na suchym betonie | 0,85 / 0,6 | — |
| TeachEngineering, tabela współczynników tarcia ([PDF](https://www.teachengineering.org/content/nyu_/activities/nyu_heavy/nyu_heavy_activity1_coftable.pdf)) | guma / beton (sucho) | 0,6–0,85 | 0,6–0,85 |

Ostatnie źródło zaznacza, że guma nie ma jednego stałego współczynnika tarcia, bo zależy on m.in.
od temperatury. Zależy też od nacisku, prędkości poślizgu, mieszanki gumy i wykończenia betonu.

**Wartość docelowa: μ = 0,8, przedział 0,6–1,0.**
- 0,8 mieści się w każdym z powyższych zakresów dla suchego betonu.
- To kinetyczny współczynnik u Serwaya i środek przedziału statycznego 0,6–1,0.
- Bullet ma **jeden** współczynnik Coulomba, bez rozróżnienia na statyczny i kinetyczny (patrz p. 4).
  Wartość z pogranicza obu zakresów jest więc rozsądnym kompromisem.
- Przedział 0,6–1,0 przyda się później, np. do losowania tarcia (domain randomization) albo do
  analizy wrażliwości.

## 4. Jak Bullet liczy tarcie (teoria do sprawdzenia)

- **Prawo Coulomba z jednym współczynnikiem.** Ciało stoi, dopóki siła styczna ≤ μN. Gdy się ślizga,
  tarcie wynosi μN przeciwnie do prędkości. Tarcie nie zależy od masy, prędkości ani rozmiaru
  powierzchni.
- **Łączenie współczynników dwóch ciał.** Każde ciało ma własne `lateralFriction`, a kontakt
  dostaje jedną liczbę. Bullet łączy je w `btManifoldResult::calculateCombinedFriction`.
  Eksperyment sprawdza, którą regułą: iloczyn, min, max, średnia arytmetyczna czy geometryczna.
- **Dwa kierunki tarcia (piramida zamiast stożka).** Solver nakłada tarcie w dwóch prostopadłych
  kierunkach stycznych. Gdy ciało się ślizga, pierwszy kierunek jest ustawiany wzdłuż prędkości.
  Gdy stoi, kierunki zależą od geometrii kontaktu. To może dawać anizotropię tarcia statycznego.

## 5. Metoda

Na nieruchomej równi (pudło 12 × 2 × 0,1 m, kąt θ, opcjonalnie obrócone o kąt kursu ψ) puszczamy
z zerową prędkością jeden z dwóch suwaków:

- **klocek**: płaski prostopadłościan 0,2 × 0,2 × 0,04 m, 1 kg. Jest płaski, żeby się zsuwał,
  a nie przewracał;
- **„sanki ze stóp”**: sztywne ciało złożone z czterech kul o promieniu stopy A1 (0,02 m)
  w narożnikach kwadratu 0,2 × 0,2 m. Sprawdza, czy kontakt punktowy kuli zachowuje się tak samo
  jak płaska ściana klocka.

Oba suwaki dostają dynamikę kontaktu stopy A1 (tarcie boczne, toczne, wiertne, restytucję).
Tarcie równi jest zmienną eksperymentu.

Dwa niezależne pomiary μ_eff:

1. **Kąt krytyczny (tarcie statyczne).** Bisekcją po kącie szukam najmniejszego θ, przy którym
   suwak po 2 s ma prędkość > 1 cm/s. Teoria: tg θ_c = μ_eff.
2. **Przyspieszenie (tarcie kinetyczne).** Przy θ = 60° (tg 60° = 1,73, więc każde badane μ_eff
   daje poślizg) dopasowuję parabolę s(t) do drogi wzdłuż równi (od 0,1 s do 1 s). Z równania
   a = g(sin θ − μ cos θ) wynika μ = (g sin θ − a) / (g cos θ).

**Ważne: tłumienie ciał w PyBullet.** Pierwsze uruchomienie dało μ zależne od kąta
(0,50 przy 28°, 0,62 przy 60°, 0,88 przy 75°), a przyspieszenie malało z prędkością. Przyczyną jest
domyślne tłumienie liniowe i kątowe każdego ciała: `linearDamping = angularDamping = 0.04`
(PyBullet Quickstart Guide, `changeDynamics`). W multibody Bullet działa ono jak opór
F = −m·c·(1 + |v|)·v. To nie jest tarcie, więc w pomiarach tarcia suwak ma tłumienie 0.
Wpływ tłumienia jest zmierzony osobno w tabeli 6.4.

## 6. Wyniki

Pełne dane są w `friction_results.json`. Liczby poniżej pochodzą z uruchomienia z 2026-11-27.
Pomiar zrobiłem na pakiecie `pybullet 3.2.7` (Linux), a projekt używa `pybullet-arm64 3.2.8`.
Uruchom skrypt u siebie, żeby JSON zawierał wynik z Twojej wersji.

### 6.1 Reguła łączenia współczynników

| μ klocka | μ równi | μ zmierzone | iloczyn | min | śr. geom. |
|---|---|---|---|---|---|
| 0,5 | 0,5 | 0,2500 | 0,25 | 0,5 | 0,5 |
| 0,5 | 1,0 | 0,5002 | 0,50 | 0,5 | 0,707 |
| 0,5 | 1,6 | 0,8003 | 0,80 | 0,5 | 0,894 |
| 1,0 | 0,6 | 0,6002 | 0,60 | 0,6 | 0,775 |
| 0,8 | 1,2 | 0,9604 | 0,96 | 0,8 | 0,980 |
| 1,0 | 1,0 | 1,0004 | 1,00 | 1,0 | 1,000 |
| 0,3 | 2,0 | 0,6002 | 0,60 | 0,3 | 0,775 |

Błąd RMS: iloczyn 0,0003, średnia geometryczna 0,158, min 0,196, średnia arytmetyczna 0,275,
max 0,669. **Bullet mnoży współczynniki: μ_eff = μ_stopy · μ_podłoża.**

### 6.2 Obecne ustawienie projektu (stopa 0,5 · podłoże 1,0)

| Pomiar | μ_eff |
|---|---|
| klocek, przyspieszenie | 0,5002 |
| sanki ze stóp, przyspieszenie | 0,5000 |
| klocek, kąt krytyczny 26,58° | 0,5003 |
| sanki ze stóp, kąt krytyczny 26,59° | 0,5005 |

**Obecnie μ_eff = 0,5.** To poniżej dolnej granicy dla gumy na suchym betonie (0,6) i o 37 % mniej
niż wartość docelowa 0,8. Pełzanie 1° poniżej kąta krytycznego wynosi 0,06 mm dla klocka i 0,3 mm
dla sanek w ciągu 2 s, więc tarcie statyczne trzyma.

### 6.3 Odporność pomiaru

- **Krok czasu:** μ = 0,5002 przy `dt = 1/240 s` i 0,5000 przy `1/960 s`.
- **Masa** 0,25 / 1 / 3 kg: μ = 0,5002 za każdym razem. Tarcie Coulomba nie zależy od nacisku.
- **Kierunek poślizgu** (kurs 0°, 30°, 45°): μ = 0,5002 za każdym razem. Tarcie kinetyczne jest
  izotropowe.
- **Kierunek przy tarciu statycznym:** to jest anizotropia z p. 4.

| kurs ψ | 0° | 15° | 30° | 45° | 60° | 90° |
|---|---|---|---|---|---|---|
| θ_c | 26,58° | 26,56° | 25,60° | 24,02° | 25,77° | 26,58° |
| tg θ_c | 0,500 | 0,500 | 0,479 | 0,446 | 0,483 | 0,500 |

Tarcie statyczne zależy od kierunku względem osi świata i na przekątnej (45°) jest o ok. 11 %
mniejsze. To artefakt przybliżenia stożka tarcia dwoma kierunkami w solverze Bullet.
Tarcie kinetyczne tego problemu nie ma.

### 6.4 Wpływ domyślnego tłumienia 0,04 (pozorne μ)

| kąt | tłumienie 0 | tłumienie 0,04 |
|---|---|---|
| 35° | 0,5001 | 0,5085 |
| 60° | 0,5002 | 0,6162 |
| 75° | 0,5000 | 0,8810 |

### 6.5 Kalibracja tarcia podłoża (stopa A1: 0,5)

| μ docelowe | `friction` podłoża (bisekcja) | analitycznie μ/0,5 | μ klocek | μ sanki | θ_c zmierzony / oczekiwany |
|---|---|---|---|---|---|
| 0,6 | 1,1996 | 1,2 | 0,6000 | 0,5998 | 30,98° / 30,96° |
| **0,8** | **1,5994** | **1,6** | **0,8000** | **0,7997** | **38,66° / 38,66°** |
| 1,0 | 1,9991 | 2,0 | 1,0000 | 0,9995 | 45,00° / 45,00° |

## 7. Wnioski i rekomendacje

1. **Ustawić `friction = 1.6` podłoża** (`BaseEnvironmentData.friction`). Wtedy μ_eff stopy A1
   wynosi 0,8, czyli tyle, co guma na suchym betonie. Obecne 1,0 daje tylko 0,5, co częściowo
   tłumaczy ślizganie się robota. Przedział 1,2–2,0 odpowiada μ_eff 0,6–1,0.
2. **Skutek uboczny reguły iloczynu:** wszystkie linki A1 mają tarcie 0,5. Przy podłożu 1,6 również
   korpus i łydki dostają μ_eff = 0,8, jak guma, choć korpus jest aluminiowy. Alternatywa bliższa
   fizyce: ustawić stopom `lateralFriction = 1.0` (`changeDynamics` na linkach `*_toe` w klasie
   `Robot`) i podłożu 0,8. Wtedy stopa–beton daje 0,8, a reszta robota 0,4. Ta decyzja należy
   do Ciebie i warto ją uzasadnić w pracy.
3. **Domyślne tłumienie 0,04 działa też na linki robota** jako opór zależny od prędkości, bez
   odpowiednika fizycznego. Przy prędkościach chodu (~1 m/s) jest małe, ale warto świadomie
   zdecydować: zostawić i opisać albo wyzerować `linearDamping` i `angularDamping` robota.
4. **Anizotropia tarcia statycznego (do 11 % na przekątnej)** to ograniczenie symulatora, które
   warto opisać. Ma znaczenie teraz, gdy robot może iść w dowolnym kierunku: ruch po przekątnej
   osi świata ma nieco mniejszy zapas przyczepności.
5. Bullet nie rozróżnia tarcia statycznego i kinetycznego (μ_s = μ_k). Prawdziwa guma ma μ_s > μ_k.
   To kolejne uproszczenie modelu do opisania.

## 8. Ograniczenia eksperymentu

- Materiał stopy A1 jest ustalony pośrednio (p. 2). Mieszanka gumy i jej twardość nie są znane.
- Tablicowe współczynniki guma–beton mają duży rozrzut (0,6–1,0 na sucho). Wartość 0,8 jest
  rozsądnym wyborem, a nie pomiarem.
- Eksperyment mierzy tarcie samego kontaktu. Nie bada podatności kontaktu (sztywny kontakt
  w URDF) ani tego, jak ślizganie zależy od sposobu sterowania nogą. To zadanie na test z robotem.
