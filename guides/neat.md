## NEAT i neat-python: ściągawka

Opracowane na podstawie: Stanley, Miikkulainen 2002, „Efficient evolution of neural network topologies” (materiały projektu) oraz kodu biblioteki neat-python 2.0.0 (sprawdzone na przykładzie XOR w `scratch/etap4/`).

---

### 1. Neuroewolucja w jednym akapicie

Zamiast uczyć sieć gradientem, **hodujemy populację sieci**. W każdym pokoleniu: każdą sieć oceniamy (fitness), najlepsze rozmnażają się z mutacjami i krzyżowaniem, słabe wymierają. Fitness może być dowolną liczbą, np. dystansem przebytym w symulacji. Nie trzeba liczyć pochodnych, więc symulator może być „czarną skrzynką”.

### 2. Co wyróżnia NEAT

NEAT (NeuroEvolution of Augmenting Topologies) ewoluuje **wagi i strukturę** sieci jednocześnie. Trzy pomysły:

1. **Numery innowacji (historyczne oznaczenia genów).** Każde nowe połączenie dostaje numer. Przy krzyżowaniu geny o tym samym numerze sobie odpowiadają, a pozostałe są nadmiarowe lub rozłączne. W neat-python 2.0 obsługuje to `InnovationTracker`.
2. **Specjacja.** Populacja dzieli się na gatunki według podobieństwa genomów. Nowa struktura ma zwykle na początku gorszy fitness (wagi nie są dostrojone), więc bez ochrony zginęłaby od razu. W gatunku genomy konkurują głównie między sobą (dzielony fitness), co daje innowacjom czas na dojrzenie.
3. **Komplikowanie od minimum.** Start bez neuronów ukrytych. Struktura rośnie tylko wtedy, gdy to się opłaca.

### 3. Genom

- **Geny neuronów** (`DefaultNodeGene`): identyfikator, bias, funkcja aktywacji, funkcja agregacji, `response` (mnożnik sumy wejść). Wejścia mają ujemne identyfikatory (-1, -2, ...), wyjścia 0, 1, ..., a ukryte kolejne liczby.
- **Geny połączeń** (`DefaultConnectionGene`): skąd i dokąd, waga, czy włączone.

Neuron liczy: wyjście = aktywacja(bias + response · agregacja(waga_i · wejście_i)).

### 4. Mutacje

| Mutacja | Co robi |
|---|---|
| zmiana wagi | do wagi dodawany szum o skali `weight_mutate_power`; rzadziej (`weight_replace_rate`) waga losowana od nowa |
| dodanie połączenia | nowe połączenie między dotąd niepołączonymi neuronami |
| dodanie neuronu | połączenie A→B (waga w) rozcinane na A→N (waga 1) i N→B (waga w), stare wyłączone; sieć tuż po mutacji liczy prawie to samo |
| usunięcie połączenia / neuronu | odwrotność powyższych |
| włączenie / wyłączenie połączenia | `enabled_mutate_rate` |

### 5. Specjacja: odległość genomów

  δ = c_disjoint · (liczba niepasujących genów) / N + c_weight · (średnia różnica wag pasujących genów)

N to liczba genów większego genomu. Genomy z δ < `compatibility_threshold` należą do jednego gatunku.
- Mały próg: dużo małych gatunków (silna ochrona innowacji, ale mało osobników w każdym).
- Duży próg (np. 100): jeden gatunek, czyli brak ochrony. W XOR czasem szybciej, czasem wolniej (ćwiczenie: ziarna 15, 26, 42, 43 dały 27, 105, 25, 41 pokoleń zamiast 20, 64, 84, 111). Przy 4 przebiegach różnica nie jest rozstrzygająca; zysk ze specjacji widać zwłaszcza w trudniejszych problemach.

### 6. Cykl jednego pokolenia w neat-python

1. `eval_genomes(genomes, config)` nadaje każdemu genomowi `genome.fitness`.
2. Gatunki: przypisanie genomów, aktualizacja reprezentantów.
3. Stagnacja: gatunki bez poprawy przez `max_stagnation` pokoleń są usuwane (oprócz `species_elitism` najlepszych).
4. Reprodukcja: każdy gatunek dostaje liczbę potomków proporcjonalną do swojego (dzielonego) fitness; `elitism` najlepszych przechodzi bez zmian; rozmnaża się tylko `survival_threshold` najlepszych z gatunku.
5. Koniec, gdy fitness najlepszego (wg `fitness_criterion`) przekroczy `fitness_threshold` albo po `n` pokoleniach.

---

### 7. Parametry pliku konfiguracyjnego

W wersji 2.0 **każdy parametr musi być wpisany**; brak daje `RuntimeError` z sugerowaną wartością.

#### [NEAT]
| Parametr | Znaczenie | XOR |
|---|---|---|
| `fitness_criterion` | jak łączyć fitness populacji przy sprawdzaniu progu: `max`, `min`, `mean` | max |
| `fitness_threshold` | próg zakończenia | 3.9 |
| `no_fitness_termination` | `True`: ignoruj próg, licz zawsze `n` pokoleń (przydatne w eksperymentach o stałej długości) | False |
| `pop_size` | liczba genomów w pokoleniu | 150 |
| `reset_on_extinction` | gdy wszystkie gatunki wymrą: `True` nowa losowa populacja, `False` wyjątek `CompleteExtinctionException` | False |
| `seed` | ziarno (opcjonalne); ustawia globalny moduł `random`, nie NumPy | 42 |

#### [DefaultGenome]: struktura
| Parametr | Znaczenie |
|---|---|
| `num_inputs`, `num_outputs` | liczba wejść i wyjść (stała przez całą ewolucję) |
| `num_hidden` | neurony ukryte na starcie (zwykle 0) |
| `feed_forward` | `True`: sieć jednokierunkowa, bez pamięci; `False`: dopuszcza pętle (sieć rekurencyjna) |
| `initial_connection` | połączenia na starcie: `unconnected`, `full_direct` (każde wejście z każdym wyjściem), `full_nodirect`, `partial_direct 0.5` (losowa część) i inne |

#### [DefaultGenome]: mutacje struktury
| Parametr | Znaczenie |
|---|---|
| `conn_add_prob`, `conn_delete_prob` | prawdopodobieństwo dodania / usunięcia połączenia na genom na pokolenie |
| `node_add_prob`, `node_delete_prob` | to samo dla neuronów |
| `single_structural_mutation` | `True`: najwyżej jedna mutacja struktury na raz |
| `enabled_default`, `enabled_mutate_rate` | czy nowe połączenia są włączone; szansa przełączenia |

Bez `node_add_prob` (ćwiczenie: `node_add_prob = 0`) XOR nie zostaje rozwiązany: bez neuronu ukrytego sieć liczy tylko funkcję liniowo separowalną, a XOR nią nie jest.

#### [DefaultGenome]: wagi, bias, response
Wspólny schemat dla `weight_*`, `bias_*`, `response_*`:
| Przyrostek | Znaczenie |
|---|---|
| `_init_mean`, `_init_stdev` | rozkład normalny wartości początkowej |
| `_min_value`, `_max_value` | obcięcie wartości |
| `_mutate_rate` | szansa mutacji w pokoleniu |
| `_mutate_power` | odchylenie standardowe szumu przy mutacji |
| `_replace_rate` | szansa wylosowania wartości od nowa |

`response` w XOR jest wyłączony (stałe 1.0, `mutate_rate = 0`).

#### [DefaultGenome]: aktywacja i agregacja
| Parametr | Znaczenie |
|---|---|
| `activation_default`, `activation_options`, `activation_mutate_rate` | funkcja aktywacji nowych neuronów, lista dozwolonych, szansa zmiany |
| `aggregation_default`, `aggregation_options`, `aggregation_mutate_rate` | jak łączyć wejścia neuronu (`sum`, `product`, `max`, ...) |

**Uwaga:** w neat-python aktywacje mają wbudowane wzmocnienie: `tanh` to tanh(2,5·z), `sigmoid` to 1/(1 + e^(−5·z)). Neurony są „strome”, więc wejścia sieci warto skalować do rzędu 1.

#### [DefaultGenome]: odległość genomów
| Parametr | Znaczenie |
|---|---|
| `compatibility_disjoint_coefficient` | c_disjoint we wzorze δ |
| `compatibility_weight_coefficient` | c_weight we wzorze δ |

#### [DefaultSpeciesSet]
| Parametr | Znaczenie |
|---|---|
| `compatibility_threshold` | próg δ dla jednego gatunku |

#### [DefaultStagnation]
| Parametr | Znaczenie |
|---|---|
| `species_fitness_func` | jak liczyć fitness gatunku do oceny stagnacji (`max`, `mean`, ...) |
| `max_stagnation` | po ilu pokoleniach bez poprawy gatunek jest usuwany |
| `species_elitism` | ile najlepszych gatunków chronić przed usunięciem |

#### [DefaultReproduction]
| Parametr | Znaczenie |
|---|---|
| `elitism` | ile najlepszych genomów każdego gatunku przechodzi bez zmian |
| `survival_threshold` | jaka część gatunku może się rozmnażać |
| `min_species_size` | minimalna liczba potomków gatunku |

---

### 8. API neat-python 2.0

```python
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, "config.ini")
pop = neat.Population(config)               # or Population(config, seed=...)
pop.add_reporter(neat.StdOutReporter(True)) # progress in the console
stats = neat.StatisticsReporter()           # fitness and species per generation
pop.add_reporter(stats)
pop.add_reporter(neat.Checkpointer(10))     # save every 10 generations
winner = pop.run(eval_genomes, 100)         # eval_genomes(genomes, config)

net = neat.nn.FeedForwardNetwork.create(winner, config)   # or neat.nn.RecurrentNetwork
outputs = net.activate(inputs)              # list in, list out
```

**Ewaluacja równoległa:**
```python
pe = neat.ParallelEvaluator(num_workers, eval_genome, initializer=init_worker, seed=...)
winner = pop.run(pe.evaluate, 100)
```
- `eval_genome(genome, config)` ocenia **jeden** genom i **zwraca** fitness (inaczej niż `eval_genomes`).
- `initializer` uruchamia się raz w każdym procesie: miejsce na własne połączenie PyBullet (`DIRECT`) i budowę środowiska z robotem.
- `seed`: każdy genom dostaje ziarno `seed + genome.key` dla modułu `random` (NumPy trzeba ziarnić osobno).
- Na Windows procesy startują metodą spawn: punkt wejścia programu w `if __name__ == "__main__":`.

### 9. Powtarzalność i statystyka

- Ten sam `seed` daje identyczny przebieg (sprawdzone na XOR: dwa razy 84 pokolenia, ta sama sieć).
- Różne ziarna dają bardzo różne wyniki (XOR: 6, 20, 64, 84, 111 pokoleń). Jeden przebieg nic nie dowodzi: potrzebne są wielokrotne niezależne przebiegi i porównanie rozkładów (np. test Manna-Whitneya, jak w Reyes, Escobar 2019).
