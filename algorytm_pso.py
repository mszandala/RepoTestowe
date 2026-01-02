import numpy as np
import pandas as pd
from typing import List, Tuple, Callable

# ==============================================================
# 1. FUNKCJE TESTOWE
# ==============================================================

def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function. Minimum (0, 0, ..., 0) = 0. Domain: -5.12 <= xi <= 5.12."""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function. Minimum (1, 1, ..., 1) = 0."""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def sphere(x: np.ndarray) -> float:
    """Sphere function. Minimum (0, 0, ..., 0) = 0."""
    return np.sum(x**2)

def beale(x: np.ndarray) -> float:
    """Beale function. Minimum (3, 0.5) = 0. Domain: -4.5 <= x, y <= 4.5."""
    x1, x2 = x
    return (1.5 - x1 + x1 * x2)**2 + (2.25 - x1 + x1 * x2**2)**2 + (2.625 - x1 + x1 * x2**3)**2

def bukin_n6(x: np.ndarray) -> float:
    """Bukin function N.6. Minimum (-10, 1) = 0. Domain: -15 <= x <= -5, -3 <= y <= 3."""
    x1, x2 = x
    return 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2)) + 0.01 * np.abs(x1 + 10)

# ==============================================================
# 2. KLASA CZASTKI (Particle)
# ==============================================================

class Particle:
    def __init__(self, dim: int, bounds: List[Tuple[float, float]]):
        self.dim = dim
        # Pozycja i prędkość inicjowane losowo w ramach granic
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array([np.random.uniform(-(high - low), (high - low)) for low, high in bounds]) # Inicjalizacja prędkości
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

# ==============================================================
# 3. INTERFEJS I IMPLEMENTACJA ALGO (PSO)
# ==============================================================

# Definicja interfejsu
class OptimizationAlgorithm:
    Name: str
    XBest: np.ndarray
    FBest: float
    NumberOfEvaluationFitnessFunction: int

    def Solve(self) -> float: #fbest zwraca to
        raise NotImplementedError

class PSO(OptimizationAlgorithm):

    # Parametry wewnętrzne (P_1, P_k) to w, c1, c2

    def __init__(self, func: Callable[[np.ndarray], float], dim: int, bounds: List[Tuple[float, float]],
                 num_particles: int = 20, max_iter: int = 30, w: float = 0.7, c1: float = 1.5, c2: float = 1.5):

        self.Name = "PSO"
        self.func = func
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        self.NumberOfEvaluationFitnessFunction = 0

        self.swarm = [Particle(dim, self.bounds) for _ in range(num_particles)]
        self.global_best_position = self.swarm[0].position
        self.global_best_value = float('inf')

        self.XBest = np.array([])
        self.FBest = float('inf')

    def Solve(self) -> float:
        self.NumberOfEvaluationFitnessFunction = 0

        # Resetowanie w przypadku wielokrotnego uruchomienia
        self.swarm = [Particle(self.dim, self.bounds) for _ in range(self.num_particles)]
        self.global_best_position = self.swarm[0].position
        self.global_best_value = float('inf')

        for t in range(self.max_iter):
            for particle in self.swarm:
                fitness = self.func(particle.position)
                self.NumberOfEvaluationFitnessFunction += 1

                # Aktualizacja najlepszej lokalnej pozycji cząstki
                if fitness < particle.best_value:
                    particle.best_value = fitness
                    particle.best_position = np.copy(particle.position)

                # Aktualizacja najlepszego globalnego rozwiązania
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = np.copy(particle.position)

            # Aktualizacja prędkości i pozycji
            for particle in self.swarm:
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)

                # Zgodnie z implementacja: w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
                particle.velocity = self.w * particle.velocity + cognitive + social
                particle.position += particle.velocity

                # ogr. do zakresów
                for i in range(self.dim):
                    low, high = self.bounds[i]
                    particle.position[i] = np.clip(particle.position[i], low, high)

        self.XBest = self.global_best_position
        self.FBest = self.global_best_value
        return self.FBest

# ==============================================================
# 4. FUNKCJE STATYSTYCZNE I RAPORTUJĄCE
# ==============================================================

def get_bounds(func_name: str, dim: int) -> List[Tuple[float, float]]:
    """Zwraca granice dla danej funkcji testowej."""
    if func_name == "Rastrigin":
        return [(-5.12, 5.12)] * dim
    elif func_name == "Rosenbrock":
        return [(-2, 2)] * dim
    elif func_name == "Sphere":
        return [(-2, 2)] * dim
    elif func_name == "Beale":
        return [(-4.5, 4.5)] * dim
    elif func_name == "Bukin N.6":
        return [(-15, -5), (-3, 3)]
    return [(-10, 10)] * dim

def variability_measure(values: np.ndarray, expected_min: float) -> Tuple[float, str]:
    mean = np.mean(values)
    std = np.std(values)

    if np.isclose(expected_min, 0.0, atol=1e-5) and np.isclose(mean, 0.0, atol=1e-3):
        return std, "Odchylenie Standardowe"
    elif mean == 0:
        # zabezpieczeniee
        return std, "Odchylenie Standardowe (Mean=0)"
    else:

        return std / mean * 100, "Współczynnik Zmienności (%)"

def run_tests(algorithm_class: type, func_data: List[Tuple[str, Callable, int, float]],
              num_runs: int = 10, num_particles: int = 20, max_iter: int = 30,
              p_params: Tuple[float, float, float] = (0.7, 1.5, 1.5)) -> List[dict]:

    results = []

    w, c1, c2 = p_params

    # Ustawienie seed dla powtarzalności w ramach sesji
    np.random.seed(42)

    for func_name, func, dim, expected_min in func_data:
        bounds = get_bounds(func_name, dim)

        # tablice do zbierania wyników z n powtórzeń
        f_best_runs = []
        x_best_runs = []

        for _ in range(num_runs):
#stale parametryy
            algo = algorithm_class(
                func=func, dim=dim, bounds=bounds,
                num_particles=num_particles, max_iter=max_iter,
                w=w, c1=c1, c2=c2
            )
            algo.Solve()
            f_best_runs.append(algo.FBest)
            x_best_runs.append(algo.XBest)

        f_best_runs_np = np.array(f_best_runs)
        x_best_runs_np = np.array(x_best_runs)

        # 1. Najlepsze rozwiązanie -- najmniejsza wartość funkcji celu
        best_run_index = np.argmin(f_best_runs_np)
        best_f_value = f_best_runs_np[best_run_index]
        best_x_coords = x_best_runs_np[best_run_index]

        # 2. Odchylenie standardowe poszukiwanych par
        #  współrzędne dla n najlepszych osobników (z każdego powtórzenia)
        std_x = np.std(x_best_runs_np, axis=0)

        # 3. Odchylenie standardowe/Współczynnik zmienności wartości funkcji celu
        std_or_v_f, measure_f_name = variability_measure(f_best_runs_np, expected_min)

        # Dodanie Najgorszego Wyniku z 10 prób
        worst_f_value = np.max(f_best_runs_np)

        # --- UJEDNOLICONE FORMATOWANIE WYNIKÓW ---
        # Formatowanie Znalezionego minimum do 4 miejsc po przecinku
        formatted_x_best = str(tuple(f"{val:.4f}" for val in best_x_coords))
        # Formatowanie Odchylenia standardowego do 3 miejsc po przecinku
        formatted_std_x = str(tuple(f"{val:.3f}" for val in std_x))

        results.append({
            "Algorytm": algo.Name,
            "Funkcja testowa": func_name,
            "Liczba szukanych parametrów": dim,
            "P_1 (w)": w,
            "P_k (c1)": c1,
            "Liczba iteracji": max_iter,
            "Rozmiar populacji": num_particles,
            "Znalezione minimum": formatted_x_best,
            "Odchylenie standardowe poszukiwanych parametrów": formatted_std_x,
            "Wartość funkcji celu": f"{best_f_value:.4e}", # 4 miejsca znaczące w notacji wykładniczej
            "Odchylenie standardowe wartości funkcji celu": f"{std_or_v_f:.4e} ({measure_f_name.split('(')[0].strip()})",
            "Najgorszy wynik z 10 prób": f"{worst_f_value:.4e}",
        })

    return results

def export_to_csv(results: List[dict], filename: str = "raport_pso_wyniki.csv"):
    """Zapisuje wyniki do pliku CSV."""
    df = pd.DataFrame(results)
    df = df.rename(columns={
        "P_1 (w)": "P_1",
        "P_k (c1)": "P_k",
        "Liczba szukanych parametrów": "Liczba szukanych parametrów (dim)",
    })

    df = df.sort_values(
        by=["Liczba szukanych parametrów (dim)", "Funkcja testowa", "Rozmiar populacji", "Liczba iteracji"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    df.to_csv(filename, index=False, sep=';', encoding='utf-8')
    print(f"\n Wyniki zostały zapisane do pliku: {filename}")

def format_results_table(results: List[dict]): #formatowanie tabelki cn
    if not results:
        print("Brak wyników.")
        return

    df = pd.DataFrame(results)

    # Dostosowanie nagłówków do wyświetlania w konsoli
    df = df.rename(columns={
        "P_1 (w)": "P_1",
        "P_k (c1)": "P_k",
        "Liczba szukanych parametrów": "Liczba szukanych parametrów (dim)",
    })

    print("\n Tabela z wynikami (PSO)")
    print("---")
    # Użycie to_markdown z pandas dla gwarancji poprawnych nagłówków i formatu tabeli
    print(df.to_markdown(index=False, numalign="left", stralign="left"))
    print("---")
    print("> Uwaga: Kolumna 'Odchylenie standardowe wartości funkcji celu' zawiera odchylenie standardowe (Std) lub współczynnik zmienności (V) w zależności od bliskości wyniku do zera. Statystyki obliczono na podstawie n=10 powtórzeń.")


# ==============================================================
# 5. Uruchomienie
# ==============================================================

if __name__ == '__main__':
    # === STAŁE DLA TESTÓW ===
    NUM_RUNS = 10 # Liczba powt
    # par  wewnętrzne (P_1, P_k)
    P_PARAMS = (0.5, 1.5, 1.5) # Przykładowe w, c1, c2

    # Zakresy N (Rozmiar populacji) i I (Liczba iteracji)
    N_VALUES = [10, 20, 40, 80]
    I_VALUES = [5, 10, 20, 40, 60, 80]

    # lista funkcji do testowania
    
    # test_functions_data = [
    #     ("Rosenbrock", rosenbrock, 2, 0.0), # Minimum (1, 1) = 0 [cite: 27]
    #     ("Rastrigin", rastrigin, 2, 0.0), # Minimum (0, 0) = 0 [cite: 5]
    #     ("Sphere", sphere, 2, 0.0), # Minimum (0, 0) = 0 [cite: 48]
    #     ("Beale", beale, 2, 0.0), # Minimum (3, 0.5) = 0 [cite: 49]
    #     ("Bukin N.6", bukin_n6, 2, 0.0), # Minimum (-10, 1) = 0 [cite: 50]
    # ]


    test_functions_data = [
    # --- Funkcje wielowymiarowe  ---
    ("Rosenbrock", rosenbrock, 2, 0.0),
    ("Rosenbrock", rosenbrock, 3, 0.0),
    ("Rosenbrock", rosenbrock, 5, 0.0),

    ("Rastrigin", rastrigin, 2, 0.0),
    ("Rastrigin", rastrigin, 3, 0.0),
    ("Rastrigin", rastrigin, 5, 0.0),

    ("Sphere", sphere, 2, 0.0),
    ("Sphere", sphere, 3, 0.0),
    ("Sphere", sphere, 5, 0.0),

    # --- Funkcje 2D ---
    ("Beale", beale, 2, 0.0),
    ("Bukin N.6", bukin_n6, 2, 0.0),
]

    all_results = []

    # === GŁÓWNA PĘTLA TESTUJĄCA WSZYSTKIE KOMBINACJE N i I ===

    for num_particles in N_VALUES:
        for max_iter in I_VALUES:
            print(f"\n=== Testowanie kombinacji: N={num_particles}, I={max_iter} ===")

            for func_name, func, dim, expected_min in test_functions_data:
                print(f"-> Funkcja: {func_name:12s} | Wymiar (dim)={dim} | N={num_particles}, I={max_iter}")

                current_results = run_tests(
                    algorithm_class=PSO,
                    func_data=[(func_name, func, dim, expected_min)],
                    num_runs=NUM_RUNS,
                    num_particles=num_particles,
                    max_iter=max_iter,
                    p_params=P_PARAMS
                )
                all_results.extend(current_results)


    # tabelkaaa <3
    format_results_table(all_results)
    export_to_csv(all_results, filename="raport_pso_wyniki.csv")

