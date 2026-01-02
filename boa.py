import numpy as np
import time
import itertools # spoko opcja do generowania kombinacji np. parametrów do strojenia
from abc import ABC, abstractmethod

# interfejs dla algorytmów
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

# implementacja BOA
class BOA(IOptimizationAlgorithm):
    def __init__(self, obj_func, dim: int, lb: np.ndarray, ub: np.ndarray,
                pop_size: int, max_iter: int,
                p: float = 0.8, c: float = 0.01,
                a_dynamic: bool = True, a_static: float = 0.1):
        super().__init__()
        self.Name = "Butterfly Optimization Algorithm"

        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.max_iter = max_iter

        # parametry wewnętrzne
        self.p = p
        self.c = c
        self.a_dynamic = a_dynamic
        self.a_static = a_static
        self.a = self.a_static

        # inicjalizacja populacji i obliczenie początkowego fitnessu
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)
        self.fragrance = np.zeros(self.pop_size)

        for i in range(self.pop_size):
            self.fitness[i] = self.obj_func(self.population[i])

        self.EvalCount += self.pop_size

        # ustawienie początkowego najlepszego rozwiązania
        best_idx = np.argmin(self.fitness)
        self.FBest = self.fitness[best_idx]
        self.XBest = self.population[best_idx, :].copy()

    def _update_fragrance(self, fitness_values: np.ndarray) -> np.ndarray:
        safe_fitness = np.fmax(fitness_values, 1e-290)
        return self.c * (safe_fitness ** self.a)
    
    def Solve(self) -> float:
        if self.max_iter <= 1:
            a_divisor = 1.0
        else:
            a_divisor = float(self.max_iter - 1)

        for t in range(self.max_iter):
            # aktualizacja a jeśli dynamiczne
            if self.a_dynamic:
                self.a = 0.1 + (0.3 - 0.1) * (t / a_divisor)
            else: 
                self.a = self.a_static

            self.fragrance = self._update_fragrance(self.fitness)

            # pętla po wszystkich motylach
            for i in range(self.pop_size):
                r_switch = np.random.random()

                if r_switch < self.p:
                    # global search
                    r_step = np.random.random()
                    step = (r_step**2 * self.XBest - self.population[i, :]) * self.fragrance[i]
                    self.population[i, :] += step
                else:
                    # local search
                    j, k = np.random.choice(self.pop_size, 2, replace=False)
                    r_step = np.random.random()
                    step = (r_step**2 * (self.population[j, :] - self.population[k, :])) * self.fragrance[i]
                    self.population[i, :] += step
                
                self.population[i, :] = np.clip(self.population[i, :], self.lb, self.ub)

                # ewaluacja nowego rozwiązania
                new_fitness = self.obj_func(self.population[i, :])
                self.EvalCount += 1

                # aktualizacja fitnessu i najlepszego rozwiązania
                self.fitness[i] = new_fitness
                if new_fitness < self.FBest:
                    self.FBest = new_fitness
                    self.XBest = self.population[i, :].copy()
        return self.FBest

# zwraca słownik z funkcjami testowymi
def get_test_functions() -> dict:
    funcs = {}

    def rastrin(x: np.ndarray) -> float:
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    funcs['Rastrigin'] = {
        'func': rastrin,
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

# szkielet testowy
def run_tests(algo_class: type[IOptimizationAlgorithm],
              func_name: str,
              test_func_data: dict,
              dim: int,
              pop_size: int,
              max_iter: int,
              n_runs: int = 10,
              boa_params: dict = None) -> dict:

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

    if boa_params is None:
        print("INFO: Używam domyślnych parametrów BOA (nie podano 'boa_params')")
        boa_params = {
            'p': 0.8,
            'c': 0.01,
            'a_dynamic': True,
            'a_static': 0.1
        }

    print(f"Parametry BOA: p={boa_params['p']}, c={boa_params['c']}, "
          f"a_dynamic={boa_params['a_dynamic']}, a_static={boa_params.get('a_static', 0.1)}")

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
            **boa_params # przekazanie parametrów BOA
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
    print(f"| Czas (n=10): {end_time - start_time:.4f} s")
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
        "p": boa_params['p'],
        "c": boa_params['c'],
        "a_dynamic": boa_params['a_dynamic'],
        "a_static": boa_params.get('a_static', 0.1),
        "best_fitness": best_fitness_value,
        "worst_fitness": worst_fitness_value,
        "std_dev_fitness": std_dev_fitness,
        "avg_evaluations": total_evals / n_runs,
        "best_x": np.array2string(best_solution_vector, precision=6, suppress_small=True),
        "time_s": end_time - start_time
    }
    return results_dict


if __name__ == "__main__":
    print("=== Testowanie algorytmu BOA ===\n")
    test_functions = get_test_functions()

    POP_SIZE_N = 40
    MAX_ITER_I = 60
    N_RUNS = 10

    func_name_1 = 'Rastrigin'
    func_data_1 = test_functions[func_name_1]
    dim_1 = 5

    # przykładowe uruchomienie testu
    run_tests(
        algo_class=BOA,
        func_name=func_name_1,
        test_func_data=func_data_1,
        dim=dim_1,
        pop_size=POP_SIZE_N,
        max_iter=MAX_ITER_I,
        n_runs=N_RUNS
    )