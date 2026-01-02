import numpy as np
import time
from abc import ABC, abstractmethod
from scipy.special import gamma

# Interfejs dla algorytmów
class IOptimizationAlgorithm(ABC):
    def __init__(self):
        self.name = "IOptimizationAlgorithm"
        self.xbest = None
        self.fbest = float('inf')
        self._eval_count = 0

    @property
    def Name(self) -> str:
        return self.name
    
    @Name.setter
    def Name(self, value: str):
        self.name = value

    @property
    def XBest(self) -> np.ndarray:
        return self.xbest
    
    @XBest.setter
    def XBest(self, value: np.ndarray):
        self.xbest = value

    @property
    def FBest(self) -> float:
        return self.fbest
    
    @FBest.setter
    def FBest(self, value: float):
        self.fbest = value

    @property
    def EvalCount(self) -> int:
        return self._eval_count
    
    @EvalCount.setter
    def EvalCount(self, value: int):
        self._eval_count = value

    @abstractmethod
    def Solve(self) -> float:
        pass


class AquilaMath:
    """Klasa pomocnicza implementująca strategie matematyczne dla Aquila Optimizer"""
    
    def __init__(self, dim: int, s: float = 0.01, alpha: float = 0.1, 
                 beta: float = 1.5, delta: float = 0.1, 
                 big_u: float = 0.00565, omega: float = 0.005, r1: float = 10):
        self.dim = dim
        self.s = s
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.big_u = big_u
        self.omega = omega
        self.r1 = r1
        
        # Oblicz sigma dla rozkładu Levy'ego (z linii 12 AquilaMath.cs)
        numerator = gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
        denominator = gamma((1.0 + beta) / 2.0) * beta * np.power(2.0, (beta - 1.0) / 2.0)
        self.sigma = numerator / denominator
        
        # d1 vector (z linii 13 AquilaMath.cs)
        self.d1 = np.arange(1, dim + 1, dtype=float)
    
    def levy(self) -> float:
        """Rozkład Levy'ego (z linii 24-29 AquilaMath.cs)"""
        u = np.random.normal(0, 1)
        v = np.random.normal(0, 1)
        return self.s * u * self.sigma / np.power(np.abs(v), 1.0 / self.beta)
    
    def expanded_exploration(self, x_best: np.ndarray, x_mean: np.ndarray, 
                            t: int, T: int) -> np.ndarray:
        """Step 1: Expanded Exploration (linie 15-22 AquilaMath.cs)"""
        a1 = 1.0 - t / T
        a2 = np.random.random()
        return x_best * a1 + (x_mean - x_best) * a2
    
    def narrowed_exploration(self, x_best: np.ndarray, x_random: np.ndarray) -> np.ndarray:
        """Step 2: Narrowed Exploration z ruchem spiralnym (linie 31-44 AquilaMath.cs)"""
        a1 = self.levy()
        a2 = np.random.random()
        
        # Oblicz ruch spiralny dla każdego wymiaru
        r = self.r1 + self.big_u * self.d1
        theta = -self.omega * self.d1 + 3.0 * np.pi / 2.0
        spiral_y = r * np.cos(theta)
        spiral_x = r * np.sin(theta)
        
        return x_best * a1 + x_random + (spiral_y - spiral_x) * a2
    
    def expanded_exploitation(self, x_best: np.ndarray, x_mean: np.ndarray,
                             upper_bounds: np.ndarray, lower_bounds: np.ndarray) -> np.ndarray:
        """Step 3: Expanded Exploitation (linie 46-54 AquilaMath.cs)"""
        a1 = np.random.random()
        a2 = np.random.random()
        return (x_best - x_mean) * self.alpha - a1 + ((upper_bounds - lower_bounds) * a2 + lower_bounds) * self.delta
    
    def narrowed_exploitation(self, x_best: np.ndarray, x_prev: np.ndarray,
                             t: int, T: int) -> np.ndarray:
        """Step 4: Narrowed Exploitation z QF (linie 56-65 AquilaMath.cs)"""
        qf = np.power(t, (2.0 * np.random.random() - 1.0) / np.power(1.0 - T, 2.0))
        g1 = 2.0 * np.random.random() - 1.0
        g2 = 2.0 * (1.0 - t / T)
        a1 = np.random.random()
        a2 = self.levy()
        a3 = np.random.random()
        return qf * x_best - g1 * x_prev * a1 - g2 * a2 + a3 * g1


class AO(IOptimizationAlgorithm):
    """Aquila Optimizer - implementacja zgodna z interfejsem IOptimizationAlgorithm"""
    
    def __init__(self, obj_func, dim: int, lb: np.ndarray, ub: np.ndarray,
                 pop_size: int, max_iter: int,
                 alpha: float = 0.1, delta: float = 0.1,
                 s: float = 0.01, beta: float = 1.5,
                 big_u: float = 0.00565, omega: float = 0.005, r1: float = 10):
        """
        Konstruktor algorytmu Aquila Optimizer
        
        Args:
            obj_func: funkcja celu do minimalizacji
            dim: liczba wymiarów problemu
            lb: dolne granice jako np.ndarray
            ub: górne granice jako np.ndarray
            pop_size: liczba orłów (search agents)
            max_iter: maksymalna liczba iteracji
            alpha: współczynnik eksploracji (domyślnie 0.1)
            delta: współczynnik eksploatacji (domyślnie 0.1)
            s: parametr dla rozkładu Levy'ego (domyślnie 0.01)
            beta: parametr dla rozkładu Levy'ego (domyślnie 1.5)
            big_u: parametr dla ruchu spiralnego (domyślnie 0.00565)
            omega: parametr dla ruchu spiralnego (domyślnie 0.005)
            r1: parametr dla ruchu spiralnego (domyślnie 10)
        """
        super().__init__()
        self.Name = "Aquila Optimizer"
        
        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        # Parametry specyficzne dla AO
        self.alpha = alpha
        self.delta = delta
        
        # Inicjalizacja pomocnika matematycznego
        self.aquila_math = AquilaMath(dim, s=s, alpha=alpha, beta=beta, 
                                      delta=delta, big_u=big_u, omega=omega, r1=r1)
        
        # Inicjalizacja populacji (linie 40-54 Aquila.cs)
        self.population = np.random.uniform(lb, ub, (pop_size, dim))
    
    def _population_best(self) -> tuple[np.ndarray, float]:
        """Znajdź najlepsze rozwiązanie w populacji (linie 57-79 Aquila.cs)"""
        fitness_values = np.array([self.obj_func(ind) for ind in self.population])
        self.EvalCount += self.pop_size
        best_idx = np.argmin(fitness_values)
        return self.population[best_idx].copy(), fitness_values[best_idx]
    
    def _mean_population(self) -> np.ndarray:
        """Oblicz średnią pozycję populacji (linie 82-93 Aquila.cs)"""
        return np.mean(self.population, axis=0)
    
    def _apply_bounds(self, vec: np.ndarray) -> np.ndarray:
        """Zastosuj granice do wektora (linie 95-106 Aquila.cs)"""
        return np.clip(vec, self.lb, self.ub)
    
    def Solve(self) -> float:
        """
        Główna metoda optymalizacji (odpowiednik Predict() z linii 108-187)
        
        Returns:
            float: najlepsza znaleziona wartość funkcji celu
        """
        self.EvalCount = 0
        
        # Reinicjalizacja populacji
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        # Znajdź początkowe najlepsze rozwiązanie
        x_best, x_best_fitness = self._population_best()
        
        T = self.max_iter
        
        # Główna pętla optymalizacji (linie 118-185)
        for t in range(1, T + 1):
            x_mean = self._mean_population()
            
            for i in range(self.pop_size):
                x_current = self.population[i].copy()
                current_fitness = self.obj_func(x_current)
                self.EvalCount += 1
                
                rand = np.random.random()
                candidate = None
                
                # Faza eksploracji vs eksploatacji (linie 134-159)
                if t <= 2.0 / 3.0 * T:
                    # EXPLORATION PHASE
                    if rand <= 0.5:
                        # Step 1: Expanded Exploration
                        candidate = self.aquila_math.expanded_exploration(x_best, x_mean, t, T)
                    else:
                        # Step 2: Narrowed Exploration
                        random_idx = np.random.randint(0, self.pop_size)
                        x_random = self.population[random_idx]
                        candidate = self.aquila_math.narrowed_exploration(x_best, x_random)
                else:
                    # EXPLOITATION PHASE
                    if rand <= 0.5:
                        # Step 3: Expanded Exploitation
                        candidate = self.aquila_math.expanded_exploitation(x_best, x_mean, self.ub, self.lb)
                    else:
                        # Step 4: Narrowed Exploitation
                        candidate = self.aquila_math.narrowed_exploitation(x_best, x_current, t, T)
                
                # Sprawdź i zaktualizuj (linie 161-181)
                if candidate is not None:
                    candidate = self._apply_bounds(candidate)
                    candidate_fitness = self.obj_func(candidate)
                    self.EvalCount += 1
                    
                    if candidate_fitness < current_fitness:
                        self.population[i] = candidate
                        
                        if candidate_fitness < x_best_fitness:
                            x_best = candidate.copy()
                            x_best_fitness = candidate_fitness
        
        # Ustaw wyniki końcowe
        self.XBest = x_best
        self.FBest = x_best_fitness
        return self.FBest


# Funkcje testowe
def get_test_functions() -> dict:
    """Zwraca słownik z funkcjami testowymi"""
    funcs = {}

    def rastrigin(x: np.ndarray) -> float:
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    funcs['Rastrigin'] = {
        'func': rastrigin,
        'dim': [2, 5, 10],
        'bounds': [-5.12, 5.12],
        'opt_f': 0.0
    }

    def rosenbrock(x: np.ndarray) -> float:
        return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (x[:-1] - 1.0)**2.0)
    funcs['Rosenbrock'] = {
        'func': rosenbrock,
        'dim': [2, 5, 10],
        'bounds': [-10, 10],
        'opt_f': 0.0
    }

    def sphere(x: np.ndarray) -> float:
        return np.sum(x**2)
    funcs['Sphere'] = {
        'func': sphere,
        'dim': [2, 5, 10, 20],
        'bounds': [-100, 100],
        'opt_f': 0.0
    }

    def beale(x: np.ndarray) -> float:
        if len(x) != 2:
            raise ValueError("Beale function is only defined for 2 dimensions.")
        x1, x2 = x[0], x[1]
        t1 = (1.5 - x1 + x1 * x2)**2
        t2 = (2.25 - x1 + x1 * x2**2)**2
        t3 = (2.625 - x1 + x1 * x2**3)**2
        return t1 + t2 + t3
    funcs['Beale'] = {
        'func': beale,
        'dim': [2],
        'bounds': [-4.5, 4.5],
        'opt_f': 0.0
    }

    def bukin_n6(x: np.ndarray) -> float:
        if len(x) != 2:
            raise ValueError("Bukin N.6 function is only defined for 2 dimensions.")
        x1, x2 = x[0], x[1]
        t1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2))
        t2 = 0.01 * np.abs(x1 + 10)
        return t1 + t2
    funcs['Bukin N.6'] = {
        'func': bukin_n6,
        'dim': [2],
        'bounds': [[-15, -5], [-3, 3]],
        'opt_f': 0.0
    }

    return funcs


def run_tests(algo_class: type[IOptimizationAlgorithm],
              func_name: str,
              test_func_data: dict,
              dim: int,
              pop_size: int,
              max_iter: int,
              n_runs: int = 10,
              ao_params: dict = None) -> dict:
    """
    Funkcja testująca algorytm AO na wybranej funkcji testowej
    
    Args:
        algo_class: klasa algorytmu (AO)
        func_name: nazwa funkcji testowej
        test_func_data: dane funkcji testowej
        dim: wymiar problemu
        pop_size: rozmiar populacji
        max_iter: maksymalna liczba iteracji
        n_runs: liczba uruchomień (domyślnie 10)
        ao_params: parametry dla AO (domyślnie None - użyje wartości domyślnych)
    
    Returns:
        dict: słownik z wynikami testów
    """
    print("--- Uruchamianie testu ---")
    print(f"Algorytm: {algo_class.__name__}")
    print(f"Funkcja: {func_name}, Wymiar: {dim}")
    print(f"Ustawienia: Populacja (N) = {pop_size}, Iteracje (I) = {max_iter}")
    print(f"Liczba uruchomień (n): {n_runs}")

    all_fbest_results = []
    all_xbest_results = []

    # Ustalenie granic (obsługa Bukin N.6)
    if func_name == 'Bukin N.6':
        lb = np.array([test_func_data['bounds'][0][0], test_func_data['bounds'][1][0]])
        ub = np.array([test_func_data['bounds'][0][1], test_func_data['bounds'][1][1]])
    else:
        lb = np.array([test_func_data['bounds'][0]] * dim)
        ub = np.array([test_func_data['bounds'][1]] * dim)

    # Walidacja wymiarów dla funkcji 2D
    if func_name == 'Beale' and dim != 2:
        raise ValueError(f"Beale function wymaga dokładnie 2 wymiarów, otrzymano {dim}")
    if func_name == 'Bukin N.6' and dim != 2:
        raise ValueError(f"Bukin N.6 function wymaga dokładnie 2 wymiarów, otrzymano {dim}")

    if ao_params is None:
        print("INFO: Używam domyślnych parametrów AO (nie podano 'ao_params')")
        ao_params = {
            'alpha': 0.1,
            'delta': 0.1,
            's': 0.01,
            'beta': 1.5,
            'big_u': 0.00565,
            'omega': 0.005,
            'r1': 10
        }

    print(f"Parametry AO: alpha={ao_params.get('alpha', 0.1)}, delta={ao_params.get('delta', 0.1)}, "
          f"s={ao_params.get('s', 0.01)}, beta={ao_params.get('beta', 1.5)}, "
          f"big_u={ao_params.get('big_u', 0.00565)}, omega={ao_params.get('omega', 0.005)}, "
          f"r1={ao_params.get('r1', 10)}")

    total_evals = 0
    start_time = time.time()

    for run in range(n_runs):
        # Nowa instancja algorytmu dla każdego uruchomienia
        algo = algo_class(
            obj_func=test_func_data['func'],
            dim=dim,
            lb=lb,
            ub=ub,
            pop_size=pop_size,
            max_iter=max_iter,
            alpha=ao_params.get('alpha', 0.1),
            delta=ao_params.get('delta', 0.1),
            s=ao_params.get('s', 0.01),
            beta=ao_params.get('beta', 1.5),
            big_u=ao_params.get('big_u', 0.00565),
            omega=ao_params.get('omega', 0.005),
            r1=ao_params.get('r1', 10)
        )

        algo.Solve()

        all_fbest_results.append(algo.FBest)
        all_xbest_results.append(algo.XBest)
        total_evals += algo.EvalCount

        print(f" Uruchomienie {run+1}/{n_runs}, Wynik: {algo.FBest:.6e}")

    all_fbest_results = np.array(all_fbest_results)
    all_xbest_results = np.array(all_xbest_results)

    # Obliczanie statystyk
    best_fitness_value = np.min(all_fbest_results)
    best_run_index = np.argmin(all_fbest_results)
    best_solution_vector = all_xbest_results[best_run_index, :]
    worst_fitness_value = np.max(all_fbest_results)
    std_dev_fitness = np.std(all_fbest_results)
    std_dev_params = np.std(all_xbest_results, axis=0)
    end_time = time.time()

    print(f"\n--- Podsumowanie wyników (n={n_runs}) ---")
    print(f"| Czas (n={n_runs}): {end_time - start_time:.4f} s")
    print(f"| Najlepszy wynik (funkcja celu): {best_fitness_value:.6e}")
    print(f"| Najgorszy wynik (funkcja celu): {worst_fitness_value:.6e}")
    print(f"| Odchylenie standardowe (funkcja celu): {std_dev_fitness:.6e}")
    print(f"| Średnia liczba ewaluacji: {total_evals / n_runs:.2f}")
    print("\n--- Najlepsze znalezione rozwiązanie ---")
    print(f"Wektor X (rozwiązanie): {np.array2string(best_solution_vector, precision=6, suppress_small=True)}")
    print(f"Odchylenie standardowe (parametry X): {np.array2string(std_dev_params, precision=6, suppress_small=True)}")
    print("------------------------------\n")

    # Funkcja zwraca słownik z wynikami
    results_dict = {
        "func_name": func_name,
        "dim": dim,
        "pop_size": pop_size,
        "max_iter": max_iter,
        "alpha": ao_params.get('alpha', 0.1),
        "delta": ao_params.get('delta', 0.1),
        "s": ao_params.get('s', 0.01),
        "beta": ao_params.get('beta', 1.5),
        "big_u": ao_params.get('big_u', 0.00565),
        "omega": ao_params.get('omega', 0.005),
        "r1": ao_params.get('r1', 10),
        "best_fitness": best_fitness_value,
        "worst_fitness": worst_fitness_value,
        "std_dev_fitness": std_dev_fitness,
        "avg_evaluations": total_evals / n_runs,
        "best_x": np.array2string(best_solution_vector, precision=6, suppress_small=True),
        "time_s": end_time - start_time
    }
    return results_dict


if __name__ == "__main__":
    print("=== Testowanie algorytmu Aquila Optimizer (AO) ===\n")
    test_functions = get_test_functions()

    POP_SIZE_N = 40
    MAX_ITER_I = 60
    N_RUNS = 10

    func_name_1 = 'Rastrigin'
    func_data_1 = test_functions[func_name_1]
    dim_1 = 5

    run_tests(
        algo_class=AO,
        func_name=func_name_1,
        test_func_data=func_data_1,
        dim=dim_1,
        pop_size=POP_SIZE_N,
        max_iter=MAX_ITER_I,
        n_runs=N_RUNS
    )
