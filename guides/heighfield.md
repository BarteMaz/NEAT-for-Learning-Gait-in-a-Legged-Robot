## `p.GEOM_HEIGHTFIELD`

`p.GEOM_HEIGHTFIELD` w PyBullecie to jeden z najwygodniejszych sposobów na nierówny teren.

### Czym jest heightfield

To siatka punktów rozłożonych w regularnych odstępach w płaszczyźnie XY. Każdy punkt ma przypisaną wysokość Z. Bullet
łączy sąsiednie punkty w trójkąty i powstaje powierzchnia terenu. W porównaniu z dowolną siatką trójkątów (`GEOM_MESH`)
ma
dwie zalety. Kolizje są szybsze, bo silnik wie, że to regularna siatka. Da się go też podmieniać w trakcie symulacji.

Ograniczenie: heightfield może być tylko statyczny, więc zawsze tworzysz go z `baseMass=0`.

**Parametry** `p.createCollisdionShape`

| **Parametr**                   | **Znaczenie**                                                                                         |
|--------------------------------|-------------------------------------------------------------------------------------------------------|
| `shapeType=p.GEOM_HEIGHTFIELD` | typ kształtu                                                                                          |
| `meshScale=[sx, sy, sz]`       | `sx`, `sy` to odstęp między punktami siatki w metrach, a `sz` to mnożnik wysokości                    |
| `heightfieldData`              | płaska lista floatów o długości `rows * cols`                                                         |
| `numHeightfieldRows`           | liczba punktów wzdłuż osi X                                                                           |
| `numHeightfieldColumns`        | liczba punktów wzdłuż osi Y                                                                           |
| `heightfieldTextureScaling`    | ile razy tekstura powtarza się na terenie (wpływa tylko na wygląd, nie na fizykę)                     |
| `fileName`                     | alternatywa dla `heightfieldData`: plik `.png` (skala szarości) albo `.txt` (CSV z wysokościami)      |
| `replaceHeightfieldIndex`      | ID istniejącego kształtu do podmiany danych                                                           |
| `flags`                        | np. `p.GEOM_CONCAVE_INTERNAL_EDGE`, który ogranicza „podskakiwanie” obiektów na krawędziach trójkątów |

Rzeczywisty rozmiar terenu to `(rows-1)*sx` na `(cols-1)*sy`. Teren 256×256 z `sx=0.05` ma więc około 12,75 m × 12,75 m.




