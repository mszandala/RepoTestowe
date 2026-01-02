import numpy as np
import time
from abc import ABC, abstractmethod

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

# Implementacja GWO
class GWO(IOptimizationAlgorithm):
    def __init__(self, obj_func, dim: int, lb: np.ndarray, ub: np.ndarray,
                 pop_size: int, max_iter: int,
                 a_coeff: float = 2.0, c_coeff: float = 2.0):
        super().__init__()
        self.Name = "Grey Wolf Optimizer"

        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.a_coeff = a_coeff
        self.c_coeff = c_coeff

        # Historia iteracji
        self.iteration_history = []

    def Solve(self) -> float:
        # Reset licznika ewaluacji
        self.EvalCount = 0

        # Inicjalizacja pozycji wilków losowo
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        # Inicjalizacja tablicy fitness
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitness[i] = self.obj_func(positions[i])
            self.EvalCount += 1

        # Inicjalizacja alpha, beta, delta (3 najlepsze wilki)
        alpha_score = float('inf')
        beta_score = float('inf')
        delta_score = float('inf')
        alpha_pos = np.zeros(self.dim)
        beta_pos = np.zeros(self.dim)
        delta_pos = np.zeros(self.dim)

        # Główna pętla optymalizacji
        for iter in range(self.max_iter):
            # Aktualizacja alpha, beta, delta
            for i in range(self.pop_size):
                if fitness[i] < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = fitness[i]
                    alpha_pos = positions[i].copy()
                elif fitness[i] < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = fitness[i]
                    beta_pos = positions[i].copy()
                elif fitness[i] < delta_score:
                    delta_score = fitness[i]
                    delta_pos = positions[i].copy()
            
            # Oblicz a (linearly decreases from a_coeff to 0)
            a = self.a_coeff - iter * (self.a_coeff / self.max_iter)
            
            # Aktualizacja pozycji wilków
            for i in range(self.pop_size):
                for j in range(self.dim):
                    # Alpha
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A1 = 2.0 * a * r1 - a
                    C1 = self.c_coeff * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    # Beta
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A2 = 2.0 * a * r1 - a
                    C2 = self.c_coeff * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    # Delta
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A3 = 2.0 * a * r1 - a
                    C3 = self.c_coeff * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    # Średnia z trzech pozycji
                    positions[i, j] = (X1 + X2 + X3) / 3.0
                    
                    # Boundary check with reflection (jak w C#)
                    if positions[i, j] < self.lb[j]:
                        positions[i, j] = self.lb[j] + np.random.random() * (self.ub[j] - self.lb[j]) * 0.1
                    if positions[i, j] > self.ub[j]:
                        positions[i, j] = self.ub[j] - np.random.random() * (self.ub[j] - self.lb[j]) * 0.1
                
                # Oblicz fitness po aktualizacji pozycji
                fitness[i] = self.obj_func(positions[i])
                self.EvalCount += 1
            
            # Zapisz historię iteracji
            self.iteration_history.append(alpha_score)
        
        # Ustaw wyniki
        self.XBest = alpha_pos
        self.FBest = alpha_score
        return self.FBest

# Funkcje testowe
def get_test_functions() -> dict:
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
        if (len(x) != 2):
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
        if (len(x) != 2):
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

# Funkcja testująca
def run_tests(algo_class: type[IOptimizationAlgorithm],
              func_name: str,
              test_func_data: dict,
              dim: int,
              pop_size: int,
              max_iter: int,
              n_runs: int = 10,
              gwo_params: dict = None) -> dict:

    print("--- Uruchamianie testu ---")
    print(f"Algorytm: {algo_class.__name__}")
    print(f"Funkcja: {func_name}, Wymiar: {dim}")
    print(f"Ustawienia: Populacja (N) = {pop_size}, Iteracje (I) = {max_iter}")
    print(f"Liczba uruchomień (n): {n_runs}")

    all_fbest_results = []
    all_xbest_results = []

    # ustalenie granic (obsługa Bukin N.6)
    if func_name == 'Bukin N.6':
        lb = np.array([test_func_data['bounds'][0][0], test_func_data['bounds'][1][0]])
        ub = np.array([test_func_data['bounds'][0][1], test_func_data['bounds'][1][1]])
    else:
        lb = np.array([test_func_data['bounds'][0]] * dim)
        ub = np.array([test_func_data['bounds'][1]] * dim)

    if gwo_params is None:
        print("INFO: Używam domyślnych parametrów GWO (nie podano 'gwo_params')")
        gwo_params = {
            'a_coeff': 2.0,
            'c_coeff': 2.0
        }

    print(f"Parametry GWO: a_coeff={gwo_params.get('a_coeff', 2.0)}, c_coeff={gwo_params.get('c_coeff', 2.0)}")

    total_evals = 0
    start_time = time.time()

    for run in range(n_runs):
        # nowa instancja algorytmu dla każdego uruchomienia
        algo = algo_class(
            obj_func=test_func_data['func'],
            dim=dim,
            lb=lb,
            ub=ub,
            pop_size=pop_size,
            max_iter=max_iter,
            a_coeff=gwo_params.get('a_coeff', 2.0),
            c_coeff=gwo_params.get('c_coeff', 2.0)
        )

        algo.Solve()

        all_fbest_results.append(algo.FBest)
        all_xbest_results.append(algo.XBest)
        total_evals += algo.EvalCount

        print(f" Uruchomienie {run+1}/{n_runs}, Wynik: {algo.FBest:.6e}")

    all_fbest_results = np.array(all_fbest_results)
    all_xbest_results = np.array(all_xbest_results)

    # obliczanie statystyk
    best_fitness_value = np.min(all_fbest_results)
    best_run_index = np.argmin(all_fbest_results)
    best_solution_vector = all_xbest_results[best_run_index, :]
    worst_fitness_value = np.max(all_fbest_results)
    # odchylenie standardowe
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

    # funkcja zwraca słownik z wynikami które możecie zapisać potem do pliku
    results_dict = {
        "func_name": func_name,
        "dim": dim,
        "pop_size": pop_size,
        "max_iter": max_iter,
        "a_coeff": gwo_params.get('a_coeff', 2.0),
        "c_coeff": gwo_params.get('c_coeff', 2.0),
        "best_fitness": best_fitness_value,
        "worst_fitness": worst_fitness_value,
        "std_dev_fitness": std_dev_fitness,
        "avg_evaluations": total_evals / n_runs,
        "best_x": np.array2string(best_solution_vector, precision=6, suppress_small=True),
        "time_s": end_time - start_time
    }
    return results_dict


if __name__ == "__main__":
    print("=== Testowanie algorytmu GWO ===\n")
    test_functions = get_test_functions()

    POP_SIZE_N = 40
    MAX_ITER_I = 60
    N_RUNS = 10

    func_name_1 = 'Rastrigin'
    func_data_1 = test_functions[func_name_1]
    dim_1 = 5

    run_tests(
        algo_class=GWO,
        func_name=func_name_1,
        test_func_data=func_data_1,
        dim=dim_1,
        pop_size=POP_SIZE_N,
        max_iter=MAX_ITER_I,
        n_runs=N_RUNS
    )
