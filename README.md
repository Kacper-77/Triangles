## Wymagania
- C++17
- CMake 3.20+
- Vulkan SDK z `glslc`
- GLFW 3

## Budowanie
```bash
cmake -S . -B build
cmake --build build
./build/Triangles 10000
```
Argument to liczba trójkątów: 1–100000 (domyślnie 1000).

## Sterowanie
- Przytrzymaj lewy przycisk myszy i przeciągnij — obrót kamery.
- Kółko myszy — zoom.
- `C` — włącz/wyłącz CPU frustum culling.
- `B` — rozpocznij/zatrzymaj zapis próbek do `benchmark.csv` w bieżącym katalogu roboczym.

CSV zawiera: czas od rozpoczęcia zapisu, liczbę trójkątów w scenie, liczbę widocznych trójkątów, stan cullingu (0/1), FPS i średni czas klatki CPU w ms. Próbka jest zapisywana co około 0,5 s. Naciśnij `B` ponownie, aby zamknąć plik. Ponowne rozpoczęcie zapisu nadpisuje poprzedni `benchmark.csv`.
