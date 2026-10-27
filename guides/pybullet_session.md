## Nauka silnika PyBullet

### Jak naprawdę działa PyBullet

PyBullet to nie "biblioteka fizyki w Pythonie". To cienki klient, który wysyła polecenia do silnika Bullet
napisanego w C++. Silnik przechowuje cały stan świata: ciała, stawy, kontakty.
Z poziomu pythona wysyłamy tylko zapytania, na które dostajemy odpowiedzi.

- `p.connect(...)` uruchamia serwer fizyki i zwraca jego identyfikator (`physicsClientId`). Każde kolejne wywołanie
  trafia domyślnie do odtatniego podłączonego serwera.
- `p.GUI` uruchamia serwer razem z oknem OpenGL. Może być tylko jedno takie połączenie na proces. Renderowanie kosztuje
  czas.
- `p.DIRECT` uruchamia serwer w tym samym procesie, bez okna. Jest dużo szybszy i można mieć wiele takich połączeń.
- Stan żyje w silniku, nie w Pythonie. Zmienna robot_id to tylko liczba (uchwyt). Gdy rozłączysz się albo zresetujesz
  symulację, ten uchwyt traci sens.

NEAT ocenia setki genomów na pokolenie. Każda ocena to osobny „epizod” chodu. Robisz to w `DIRECT` i równolegle
(multiprocessing). Na Windows każdy proces startuje od zera (spawn), więc każdy proces potrzebuje własnego p.connect
(`p.DIRECT`). Nie przekażesz połączenia do innego procesu, bo to tylko liczba wskazująca na silnik w innym procesie.
`GUI`
zostawiasz do oglądania najlepszego genomu.

kod budujący środowisko powinien przyjmować `physicsClientId` jako parametr (albo trzymać go w obiekcie), zamiast
polegać
na „pierwszym podłączonym” serwerze. Inaczej przy wielu połączeniach polecenia trafią nie tam, gdzie chcesz.

### Czas w symulacji

Symulacja nie płynie sama. Czas w świecie Bulleta przesuwa się tylko wtedy, gdy wywołasz `p.stepSimulation ()`. Każde
wywołanie przesuwa świat o stały odcinek `dt` (krok czasowy). Domyślnie `dt = 1/240 s`, więc 240 wywołań to dokładnie 1
sekunda czasu symulowanego, niezależnie od tego, czy na Twoim komputerze zajęło to 0,04 s, czy 10 s.

Trzeba więc rozróżnić dwa zegary:

- **Czas symulowany**: liczba kroków × `dt`. Tylko on ma znaczenie dla fizyki i dla funkcji celu.
- **Czas rzeczywisty** (zegar ścienny): ile trwały obliczenia. Ma znaczenie tylko dla wydajności i dla oglądania w
  `GUI`.

Jeśli chcesz oglądać ruch w naturalnym tempie, dodajesz w pętli `time.sleep(dt)`. Robisz to tylko w `GUI` i tylko do
podglądu. W uczeniu nigdy.

**Dlaczego `dt` jest decyzją projektową**. Fizyka jest liczona numerycznie, krokami. Mniejszy `dt` daje dokładniejszą i
stabilniejszą symulację (szczególnie kontakty stóp z podłożem), ale każda sekunda ruchu kosztuje więcej kroków. Większy
`dt` przyspiesza, ale robot może przenikać w podłoże albo drgać. Dlatego:F

- ustalasz `dt` raz, jawnie (`p.setTimeStep (...)`), i nie zmieniasz go w trakcie eksperymentów,
- zapisujesz go w opisie metodyki w pracy, bo zmiana dt zmienia dynamikę, a więc i wyniki.

Częstotliwość fizyki a częstotliwość sterowania. Fizyka może działać z 240 Hz, a
sieć NEAT nie musi podejmować decyzji co krok. Typowo kontroler działa np. z 30 albo 60 Hz, a między decyzjami
wykonujesz kilka kroków fizyki z tą samą akcją. Na razie tylko zapamiętaj, że to są dwie różne rzeczy. Wybór zrobimy
później.

Bullet bazowo do obiektów aplikuje `linearDamping` oraz, `angularDamping` o domyślnej wartości `0.04`. Dotyczy on też
każdego ogniwa robota wczytanego z URDF. Spowalnia ruch nóg niezależnie od tego, co zrobi sieć. Nie musisz go wyłączać,
ale musisz o nim wiedzieć, świadomie zdecydować w etapie 2 i opisać tę decyzję w metodyce. Nie myl go też z tłumieniem w
stawach (<dynamics damping> w URDF), to osobny mechanizm.

### Parametry świata

**Tarcie**. Chód to w gruncie rzeczy odpychanie się stopami od podłoża, więc tarcie decyduje o tym, czy to w ogóle
działa. Trzy rzeczy do zapamiętania:

- Tarcie jest właściwością każdego ciała (a nawet każdego ogniwa), a nie świata. Ustawiasz je przez `changeDynamics`,
  odczytujesz przez `getDynamicsInfo`.
- W kontakcie dwóch ciał Bullet łączy ich współczynniki w jeden efektywny. Domyślnie jest to iloczyn obu współczynników.
  Kontakt stopy z podłożem zależy więc od obu stron.
- Oprócz zwykłego tarcia poślizgowego (`lateralFriction`) istnieją tarcie toczne i obrotowe. Na razie wystarczy Ci
  poślizgowe.

Dlaczego to ważne dla NEAT: ewolucja wykorzysta każdą właściwość środowiska. Za małe tarcie i robot się ślizga, ewolucja
nic sensownego nie znajdzie. Za duże, a robot nauczy się „chodu” opartego na nierealistycznym przyczepianiu się do
podłoża. To też naturalne miejsce na losowanie parametrów przy teście uogólniania

**Solver**. Kontakty i stawy Bullet rozwiązuje iteracyjnie. Więcej iteracji daje dokładniejsze kontakty, ale wolniejszy
krok. Parametry ustawiasz przez `setPhysicsEngineParameter`, a obecne wartości odczytasz przez
`getPhysicsEngineParameters`.
Nie podaję Ci domyślnych liczb, sprawdź je sam tak jak damping.

**UWAGI**:

- Domyślna ilość iteracji solvera to `numSolverIterations=50`
- Współczynnik tarcia posiada każde ogniwo robota, przy sprawdzaniu nieprawidłowości trzeba wskazać części robota, które
  stykają się z podłożem.
- Bullet nie ma rozróżnienia na tarcie statyczne i kinetyczne, zawsze korzysta z jednego współczynnika
- Przy badaniu ruchu warto uwzględnić małe drgania numeryczne
- Ewolucja nie "wie", co jest realistyczne. Optymalizuje funkcję celu i wykorzysta każdą lukę dlatego tak ważne jest
  odpowiednie dobranie parametrów świata

### Tworzenie obiektów

Do tej pory wszystko ładowałeś z gotowych plików URDF. Środowisko (ściany, rampa, teren) wygodniej budować bezpośrednio
z kodu. Służą do tego trzy funkcje:

- `createCollisionShape` tworzy kształt, który bierze udział w fizyce i kontaktach.
- `createVisualShape` tworzy to, co widać w GUI. Na fizykę nie ma żadnego wpływu.
- `createMultiBody` łączy kształty w ciało: dodaje masę, pozycję i orientację.

Kluczowe idee:
- Kolizja i wygląd są niezależne. Ciało bez kształtu kolizji jest „duchem”: widzisz je, ale przelatuje przez wszystko.
  Zajrzyj w tym kontekście do swojego test_robot.urdf w repo. Ma tylko visual. Co się z nim stanie po wczytaniu?
- **Masa 0 oznacza ciało statyczne**. Nie podlega grawitacji ani siłom, ale inne ciała się z nim zderzają. Tak budujesz
  podłoże, ściany i rampy.
- Rozmiary podajesz jako połowy. Prostopadłościan (`GEOM_BOX`) definiujesz przez `halfExtents`. Ściana o grubości 0,1 m ma
  połowę grubości 0,05. To częste źródło błędów.
- **Orientacja to kwaternion** w kolejności `[x, y, z, w]`. Nie wpisuj go ręcznie, tylko konwertuj z kątów przez
  `p.getQuaternionFromEuler`.
- **Inercję liczy silnik**. Dla ciała dynamicznego Bullet wyznacza tensor bezwładności z kształtu kolizji i masy. W URDF
  podajesz ją sam (wrócimy do tego w etapie 2).
- **Jeden kształt można użyć wielokrotnie**. `createCollisionShape` zwraca identyfikator, którego możesz użyć w wielu ciałach,
  np. dla czterech identycznych ścian.


