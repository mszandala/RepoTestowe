import numpy as np
import time
from abc import ABC, abstractmethod
from scipy.special import gamma

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
        
        # Historia iteracji do śledzenia konwergencji
        self.iteration_history = []
    
    def _apply_bounds(self, vec: np.ndarray) -> np.ndarray:
        """Zastosuj granice do wektora z reflection (jak w GWO)"""
        vec_clipped = np.copy(vec)
        for i in range(self.dim):
            if vec_clipped[i] < self.lb[i]:
                vec_clipped[i] = self.lb[i] + np.random.random() * (self.ub[i] - self.lb[i]) * 0.1
            elif vec_clipped[i] > self.ub[i]:
                vec_clipped[i] = self.ub[i] - np.random.random() * (self.ub[i] - self.lb[i]) * 0.1
        return vec_clipped
    
    def Solve(self) -> float:
        """
        Główna metoda optymalizacji (odpowiednik Predict() z linii 108-187)
        
        Returns:
            float: najlepsza znaleziona wartość funkcji celu
        """
        self.EvalCount = 0
        self.iteration_history = []
        
        # Inicjalizacja populacji
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        # Inicjalizacja fitness dla każdego osobnika
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitness[i] = self.obj_func(population[i])
            self.EvalCount += 1
        
        # Znajdź początkowe najlepsze rozwiązanie
        best_idx = np.argmin(fitness)
        x_best = population[best_idx].copy()
        x_best_fitness = fitness[best_idx]
        
        T = self.max_iter
        
        # Główna pętla optymalizacji
        for t in range(1, T + 1):
            x_mean = np.mean(population, axis=0)
            
            # Aktualizacja każdego osobnika
            for i in range(self.pop_size):
                x_current = population[i].copy()
                rand = np.random.random()
                candidate = None
                
                # Eksploracja vs eksploatacja
                if t <= 2.0 / 3.0 * T:
                    # EXPLORATION PHASE
                    if rand <= 0.5:
                        # Step 1: Expanded Exploration
                        candidate = self.aquila_math.expanded_exploration(x_best, x_mean, t, T)
                    else:
                        # Step 2: Narrowed Exploration
                        random_idx = np.random.randint(0, self.pop_size)
                        x_random = population[random_idx]
                        candidate = self.aquila_math.narrowed_exploration(x_best, x_random)
                else:
                    # EXPLOITATION PHASE
                    if rand <= 0.5:
                        # Step 3: Expanded Exploitation
                        candidate = self.aquila_math.expanded_exploitation(x_best, x_mean, self.ub, self.lb)
                    else:
                        # Step 4: Narrowed Exploitation
                        candidate = self.aquila_math.narrowed_exploitation(x_best, x_current, t, T)
                
                # Aplikuj granice i oceń kandydata
                if candidate is not None:
                    candidate = self._apply_bounds(candidate)
                    candidate_fitness = self.obj_func(candidate)
                    self.EvalCount += 1
                    
                    # Aktualizacja osobnika jeśli lepszy
                    if candidate_fitness < fitness[i]:
                        population[i] = candidate
                        fitness[i] = candidate_fitness
                        
                        # Aktualizacja globalnego najlepszego
                        if candidate_fitness < x_best_fitness:
                            x_best = candidate.copy()
                            x_best_fitness = candidate_fitness
            
            # Zapisz historię iteracji
            self.iteration_history.append(x_best_fitness)
        
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
    """Funkcja testująca algorytm AO na wybranej funkcji testowej"""
    
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

    # Walidacja wymiarów
    if func_name == 'Beale' and dim != 2:
        raise ValueError(f"Beale wymaga dokładnie 2 wymiarów, otrzymano {dim}")
    if func_name == 'Bukin N.6' and dim != 2:
        raise ValueError(f"Bukin N.6 wymaga dokładnie 2 wymiarów, otrzymano {dim}")

    if ao_params is None:
        print("INFO: Używam domyślnych parametrów AO")
        ao_params = {
            'alpha': 0.1,
            'delta': 0.1,
            's': 0.01,
            'beta': 1.5,
            'big_u': 0.00565,
            'omega': 0.005,
            'r1': 10
        }

    total_evals = 0
    start_time = time.time()

    for run in range(n_runs):
        algo = algo_class(
            obj_func=test_func_data['func'],
            dim=dim,
            lb=lb,
            ub=ub,
            pop_size=pop_size,
            max_iter=max_iter,
            **ao_params
        )

        algo.Solve()

        all_fbest_results.append(algo.FBest)
        all_xbest_results.append(algo.XBest)
        total_evals += algo.EvalCount

        print(f" Uruchomienie {run+1}/{n_runs}, Wynik: {algo.FBest:.6e}")

    all_fbest_results = np.array(all_fbest_results)
    all_xbest_results = np.array(all_xbest_results)

    best_fitness_value = np.min(all_fbest_results)
    best_run_index = np.argmin(all_fbest_results)
    best_solution_vector = all_xbest_results[best_run_index, :]
    worst_fitness_value = np.max(all_fbest_results)
    std_dev_fitness = np.std(all_fbest_results)
    std_dev_params = np.std(all_xbest_results, axis=0)
    end_time = time.time()

    print(f"\n--- Podsumowanie wyników (n={n_runs}) ---")
    print(f"| Czas (n={n_runs}): {end_time - start_time:.4f} s")
    print(f"| Najlepszy wynik: {best_fitness_value:.6e}")
    print(f"| Najgorszy wynik: {worst_fitness_value:.6e}")
    print(f"| Odch. stand. (funkcja celu): {std_dev_fitness:.6e}")
    print(f"| Średnia liczba ewaluacji: {total_evals / n_runs:.2f}")
    print(f"| Wektor X: {np.array2string(best_solution_vector, precision=6, suppress_small=True)}")
    print("------------------------------\n")

    results_dict = {
        "func_name": func_name,
        "dim": dim,
        "pop_size": pop_size,
        "max_iter": max_iter,
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

    POP_SIZE_N = 80
    MAX_ITER_I = 150
    N_RUNS = 10

    # Test 1: Sphere
    print("TEST 1: Sphere (dim=5)")
    func_name_1 = 'Sphere'
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

    # Test 2: Rastrigin
    print("TEST 2: Rastrigin (dim=5)")
    func_name_2 = 'Rastrigin'
    func_data_2 = test_functions[func_name_2]
    dim_2 = 5
    run_tests(
        algo_class=AO,
        func_name=func_name_2,
        test_func_data=func_data_2,
        dim=dim_2,
        pop_size=POP_SIZE_N,
        max_iter=MAX_ITER_I,
        n_runs=N_RUNS
    )

    # Test 3: Rosenbrock
    print("TEST 3: Rosenbrock (dim=5)")
    func_name_3 = 'Rosenbrock'
    func_data_3 = test_functions[func_name_3]
    dim_3 = 5
    run_tests(
        algo_class=AO,
        func_name=func_name_3,
        test_func_data=func_data_3,
        dim=dim_3,
        pop_size=POP_SIZE_N,
        max_iter=MAX_ITER_I,
        n_runs=N_RUNS
    )

    # Test 4: Beale
    print("TEST 4: Beale (dim=2)")
    func_name_4 = 'Beale'
    func_data_4 = test_functions[func_name_4]
    dim_4 = 2
    run_tests(
        algo_class=AO,
        func_name=func_name_4,
        test_func_data=func_data_4,
        dim=dim_4,
        pop_size=POP_SIZE_N,
        max_iter=MAX_ITER_I,
        n_runs=N_RUNS
    )

    # Test 5: Bukin N.6
    print("TEST 5: Bukin N.6 (dim=2)")
    func_name_5 = 'Bukin N.6'
    func_data_5 = test_functions[func_name_5]
    dim_5 = 2
    run_tests(
        algo_class=AO,
        func_name=func_name_5,
        test_func_data=func_data_5,
        dim=dim_5,
     # Przy za małych wartościach algorytm nie znajduje minimum
        pop_size=POP_SIZE_N,
        max_iter=MAX_ITER_I,
        n_runs=N_RUNS
    )
