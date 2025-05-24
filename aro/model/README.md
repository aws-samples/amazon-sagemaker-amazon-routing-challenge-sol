## `aro/model/` Directory Summary

The `aro/model/` directory houses a suite of tools and algorithms designed for route optimization and sequence prediction. These are particularly relevant for applications in logistics, scheduling deliveries or services by zone, and any domain requiring efficient pathfinding that also considers learned sequential patterns.

The primary modules within this directory work in concert to achieve this:

*   **`ppm.py` (Prediction by Partial Matching)**: This module is responsible for learning from historical data to predict the probability of zone sequences. It implements the PPM algorithm, which builds a statistical model of symbol (zone) occurrences based on varying lengths of preceding contexts. A notable feature is its capability to handle hierarchical zone naming (e.g., interpreting "C-17.3D" as components "C", "17", "3D" for more granular pattern recognition) and the use of zone clusters to influence predictions.

*   **`ortools_helper.py`**: This module acts as a dedicated interface to Google's OR-Tools library, specifically for solving Traveling Salesperson Problems (TSP). Given a distance matrix between locations, it provides the functions to model the problem, find an optimal route, and extract the solution. This forms the backbone of the raw route optimization capability.

*   **`zone_utils.py`**: This module serves as the integrator, bridging the probabilistic sequence modeling of `ppm.py` with the route optimization capabilities of `ortools_helper.py`. Its core function is to first determine a statistically probable or "natural" ordering of a given set of zones using the PPM model (via functions like `sort_zones` and `ppm_rollout`). Once this preferred sequence of logical zones is established, `zone_utils.py` maps these zones to their corresponding physical stops/nodes. It then uses the `ortools_helper.py` module to solve a TSP for these physical stops, ensuring the final route is not only short but also respects the learned sequential likelihoods.

**Key Combined Capability:**

The synergistic action of these modules provides a powerful capability: finding routes that are optimized for minimal travel distance/cost (via TSP solving in `ortools_helper.py`) while also adhering to learned, established, or contextually likely sequences (via PPM in `ppm.py`). This dual consideration is orchestrated by `zone_utils.py`, leading to solutions that are both efficient and practical in real-world scenarios where operational patterns matter.


## `aro/model/ortools_helper.py`

This script serves as a helper module for solving Traveling Salesperson Problems (TSP) using Google's OR-Tools library. It provides the necessary functions to model and solve a TSP instance, given a distance matrix.

The core functionality is adapted from a Google OR-Tools code snippet and operates under the Apache 2.0 License, as indicated in the source file's comments.

### Key Functions:

*   **`create_data_model(dmatrix)`**:
    *   **Purpose**: Prepares the data required by the OR-Tools solver.
    *   **Description**: This function takes a distance matrix (`dmatrix`) as input and stores it along with other problem parameters. For the TSP solved here, the number of vehicles is always set to 1, and the depot (the starting and ending point of the tour) is always at index 0.
    *   **Arguments**:
        *   `dmatrix` (list of lists of integers): Represents the distances between locations. `dmatrix[i][j]` is the distance from location `i` to location `j`.
    *   **Returns**: A dictionary containing the distance matrix, number of vehicles, and depot index.

*   **`print_solution(manager, routing, solution)`**:
    *   **Purpose**: Extracts the calculated route from the OR-Tools solution object.
    *   **Description**: After the solver finds a solution, this function processes the solution object to determine the sequence of nodes (locations) in the optimal tour.
    *   **Arguments**:
        *   `manager` (ortools.constraint_solver.pywrapcp.RoutingIndexManager): The OR-Tools index manager.
        *   `routing` (ortools.constraint_solver.pywrapcp.RoutingModel): The OR-Tools routing model.
        *   `solution` (ortools.constraint_solver.pywrapcp.Assignment): The solution found by the solver.
    *   **Returns**: A list of integers representing the sequence of nodes in the calculated route, starting and ending at the depot.

*   **`run_ortools(matrix)`**:
    *   **Purpose**: The main entry point for solving a TSP.
    *   **Description**: This function orchestrates the TSP solving process. It takes a distance matrix, sets up the OR-Tools routing model and search parameters, invokes the solver, and then uses `print_solution` to extract the route. If the solver fails to find a solution, it raises an exception.
    *   **Arguments**:
        *   `matrix` (list of lists of integers): The distance matrix for the TSP.
    *   **Returns**: A list of integers representing the optimal route found by OR-Tools.
    *   **Raises**: Exception if no solution is found.


## `aro/model/ppm.py`

This file implements the Prediction by Partial Matching (PPM) algorithm, a statistical data compression and sequence prediction method. It builds a model based on observed sequences to predict the probability of subsequent symbols.

### `PPM` Class

The `PPM` class is the core of the PPM algorithm implementation. It constructs a probabilistic model by analyzing contexts of varying orders (lengths of preceding symbol sequences) to predict the next symbol.

*   **`__init__(self, nb_order, vocab_size)`**:
    *   **Purpose**: Initializes the PPM model.
    *   **Description**: Sets up the model with the maximum context order (`nb_order`) it will consider and the size of the vocabulary (`vocab_size`) of possible symbols.
    *   **Arguments**:
        *   `nb_order` (int): The maximum length of the context (sequence of preceding symbols) to be used for prediction.
        *   `vocab_size` (int): The total number of unique symbols in the input alphabet.

*   **`add_sequence(self, zone_list)`**:
    *   **Purpose**: Trains the PPM model by incorporating an observed sequence.
    *   **Description**: This method processes the input `zone_list` (a sequence of symbols) to update the model's statistics. It iterates through the sequence, and for each symbol, it updates the counts in contexts of different orders (from `nb_order` down to 0). For example, if `nb_order` is 3 and the sequence is `[A, B, C, D]`, it will record the occurrence of `D` following `[A, B, C]`, `D` following `[B, C]`, `D` following `[C]`, and `D` in the zero-order context.
    *   **Arguments**:
        *   `zone_list` (list): A list of symbols representing an observed sequence.

*   **`query(self, preceding_zone_list, following_zone, cluster_dict=None, cluster_weights=None, use_clusters=False, default_value=None, hierarchical_mode=False)`**:
    *   **Purpose**: Calculates the probability of a specific `following_zone` occurring after a given `preceding_zone_list`.
    *   **Description**: This is the primary method for making predictions. It leverages the `_query` internal mechanism. A key feature is its ability to handle hierarchical zone structures. For instance, if `hierarchical_mode` is true, a zone like "C-17.3D" can be broken down into its constituent parts ("C", "17", "3D"), and predictions can be made based on these components. Additionally, `cluster_weights` can be applied if `use_clusters` is true, allowing predictions to be influenced by predefined zone groupings (clusters).
    *   **Arguments**:
        *   `preceding_zone_list` (list): The sequence of symbols that precedes the symbol to be predicted.
        *   `following_zone` (any): The symbol whose probability of occurrence is being queried.
        *   `cluster_dict` (dict, optional): A dictionary mapping zones to cluster IDs.
        *   `cluster_weights` (dict, optional): Weights associated with clusters, influencing probability calculations.
        *   `use_clusters` (bool, optional): If true, cluster information is used in probability calculation. Defaults to `False`.
        *   `default_value` (float, optional): A default probability to return if the context/symbol is not found.
        *   `hierarchical_mode` (bool, optional): If true, zones are treated hierarchically. Defaults to `False`.
    *   **Returns**: The calculated probability (float).

*   **`_query(self, preceding_zone_list, following_zone, default_value=None, alpha_penalty=1.0, beta_penalty=1.0)`**:
    *   **Purpose**: The internal engine for probability calculation.
    *   **Description**: This method recursively searches for the `following_zone` within contexts of decreasing order, starting from the longest possible match with `preceding_zone_list`. It applies `alpha_penalty` and `beta_penalty` to the probabilities when "escaping" to lower-order contexts if the symbol is not found in the current context. This penalization helps balance the influence of longer, more specific contexts versus shorter, more general ones.
    *   **Arguments**:
        *   `preceding_zone_list` (list): The current context being evaluated.
        *   `following_zone` (any): The symbol being queried.
        *   `default_value` (float, optional): Default probability if not found.
        *   `alpha_penalty` (float, optional): Penalty applied to escape probabilities.
        *   `beta_penalty` (float, optional): Penalty applied to symbol probabilities in lower-order contexts.
    *   **Returns**: The calculated probability (float).

### `Ctx` Class

The `Ctx` class represents a single context within the PPM model. Each context keeps track of the symbols that have followed it and their frequencies.

*   **Core Role**: To store and manage statistical information for a specific sequence of preceding symbols (a context).
*   **Attributes**:
    *   `entries` (dict): A dictionary where keys are the symbols that have followed this context, and values are their respective counts.
    *   `n` (int): The total number of times this context has been observed (i.e., the sum of counts in `entries`).
*   **Key Calculated Properties (implicitly via methods or internal logic)**:
    *   `escape_prob` (float): The probability of encountering a novel symbol not yet seen in this context. This is crucial for the PPM algorithm's ability to handle unseen events by "escaping" to a lower-order (less specific) context.
    *   `pc_prob` (dict): A dictionary storing the probability of each symbol occurring within this specific context. `pc_prob[symbol]` would give P(symbol | context).

### `build_ppm_model` Function

*   **`build_ppm_model(zdf, nb_orders, zone_col="zone", visit_id_col="visit_id", sequence_id_col="sequence_id", model=None, strictly_setbased_sequences=False, vocab_size=None)`**:
    *   **Purpose**: A utility function to conveniently create, train, and return a PPM model from zone sequence data stored in a pandas DataFrame.
    *   **Description**: This function processes a DataFrame (`zdf`) where each row can represent a zone visit. It groups zones by `visit_id_col` and `sequence_id_col` to form sequences. These sequences are then fed into a PPM model. If an existing `model` is provided, it will be further trained; otherwise, a new model is instantiated. The `strictly_setbased_sequences` option allows treating sequences as sets, where the order of elements within a sequence doesn't matter for that specific training instance (though the PPM model itself is order-aware). `vocab_size` must be provided if a new model is created.
    *   **Arguments**:
        *   `zdf` (pandas.DataFrame): DataFrame containing sequence data.
        *   `nb_orders` (int): Maximum context order for the PPM model.
        *   `zone_col` (str, optional): Name of the column containing zone identifiers. Defaults to `"zone"`.
        *   `visit_id_col` (str, optional): Name of the column for visit identifiers. Defaults to `"visit_id"`.
        *   `sequence_id_col` (str, optional): Name of the column for sequence identifiers within a visit. Defaults to `"sequence_id"`.
        *   `model` (PPM, optional): An existing PPM model to continue training. If `None`, a new model is created.
        *   `strictly_setbased_sequences` (bool, optional): If `True`, processes sequences as sets for training. Defaults to `False`.
        *   `vocab_size` (int, optional): Required if `model` is `None`. Specifies the vocabulary size for the new PPM model.
    *   **Returns**: A trained `PPM` model instance.


## `aro/model/zone_utils.py`

This module provides utilities for optimizing sequences of zones, such as those encountered in delivery or service routes. It uniquely combines sequence probability information derived from a Prediction by Partial Matching (PPM) model with Traveling Salesperson Problem (TSP) optimization to determine efficient and contextually likely zone orderings.

### Key Functions:

*   **`sort_zones(orig_zone_list, prob_model, route_id, zone_col="zone", visit_id_col="visit_id", sequence_id_col="sequence_id", no_context_panelty=0.01, hierarchical_mode=False, use_clusters=False, cluster_dict=None, cluster_weights=None, max_nb_rollout=100)`**:
    *   **Purpose**: Sorts a given list of zones (`orig_zone_list`) to find the most "probable" sequence according to a provided PPM model (`prob_model`).
    *   **Description**: This function aims to order zones in a way that is statistically likely based on historical data captured by the PPM model. It primarily utilizes the `ppm_rollout` function internally to achieve this ordering. The parameters allow for fine-tuning the rollout behavior, including penalties for missing context, use of hierarchical zone structures, and incorporation of zone clusters.
    *   **Arguments**:
        *   `orig_zone_list` (list): The initial list of zones to be sorted.
        *   `prob_model` (PPM): The trained PPM model used for probability scoring.
        *   `route_id` (any): An identifier for the current route, often used for logging or context.
        *   `zone_col`, `visit_id_col`, `sequence_id_col` (str, optional): Column names relevant if data is conceptualized from a DataFrame structure (though not directly used by this function for DataFrame manipulation, they inform the conceptual model).
        *   `no_context_panelty` (float, optional): Penalty applied during PPM queries when context is sparse.
        *   `hierarchical_mode` (bool, optional): Enables hierarchical interpretation of zone names in the PPM model.
        *   `use_clusters` (bool, optional): Enables the use of zone clusters in probability calculations.
        *   `cluster_dict` (dict, optional): Dictionary mapping zones to cluster IDs.
        *   `cluster_weights` (dict, optional): Weights for different clusters.
        *   `max_nb_rollout` (int, optional): Maximum number of rollouts/explorations in `ppm_rollout`.
    *   **Returns**: A list of zones sorted according to the PPM model's probabilities.

*   **`zone_based_tsp(matrix, zone_list, prob_model, route_id, stop_zone_map, zone_col="zone", visit_id_col="visit_id", sequence_id_col="sequence_id", no_context_panelty=0.01, hierarchical_mode=False, use_clusters=False, cluster_dict=None, cluster_weights=None, max_nb_rollout=100)`**:
    *   **Purpose**: Solves a Traveling Salesperson Problem where the entities to be visited are zones, which are first ordered by probability.
    *   **Description**: This function orchestrates a multi-step process:
        1.  It first calls `sort_zones` to get a probabilistically optimal ordering of the input `zone_list` based on the `prob_model`.
        2.  It then maps these sorted zones to actual physical nodes or stops using the `stop_zone_map`. This mapping is crucial as a single logical zone might correspond to one or more physical locations, or a representative point for that zone needs to be chosen.
        3.  Finally, it constructs a distance matrix for these physical nodes and uses `ortools_helper.run_ortools` to find the optimal tour.
    *   **Arguments**:
        *   `matrix` (list of lists): The distance matrix between all physical stops/nodes.
        *   `zone_list` (list): The list of logical zones to be included in the tour.
        *   `prob_model` (PPM): The trained PPM model.
        *   `route_id` (any): Identifier for the route.
        *   `stop_zone_map` (dict): A mapping from zone identifiers to their corresponding physical stop indices in the `matrix`.
        *   Other arguments are similar to `sort_zones` and are passed down to it.
    *   **Returns**: A list of integers representing the sequence of physical stop indices in the optimized tour.

*   **`ppm_rollout(zone_list, ppm_model, no_context_panelty, budget_exploration=10, hierarchical_mode=False, use_clusters=False, cluster_dict=None, cluster_weights=None, route_id=None)`**:
    *   **Purpose**: Implements a rollout heuristic to iteratively construct a high-probability sequence of zones.
    *   **Description**: This function builds a sequence one zone at a time. In each step, it considers adding each of the remaining unvisited zones to the current partial sequence. The "best" next zone is chosen based on the probability score obtained from `ppm_base`. The `budget_exploration` parameter allows the algorithm to explore a certain number of alternatives at each step, rather than being purely greedy, which can lead to better overall sequences.
    *   **Arguments**:
        *   `zone_list` (list): The set of available zones to form a sequence from.
        *   `ppm_model` (PPM): The PPM model.
        *   `no_context_panelty` (float): Penalty for sparse context.
        *   `budget_exploration` (int, optional): The number of top candidates to explore at each step.
        *   `hierarchical_mode`, `use_clusters`, `cluster_dict`, `cluster_weights` (optional): Parameters for PPM querying.
        *   `route_id` (any, optional): Route identifier for logging/context.
    *   **Returns**: A list representing the constructed zone sequence.

*   **`ppm_base(zone_list, start_zone, ppm_model, no_context_panelty, cycle=False, hierarchical_mode=False, use_clusters=False, cluster_dict=None, cluster_weights=None, route_id=None)`**:
    *   **Purpose**: A base heuristic function that scores the "next best" zone to append to a current sequence, based on a PPM model.
    *   **Description**: Given an existing partial sequence (`start_zone` effectively forms the end of this current sequence, or is the starting point if the sequence is empty), and a list of remaining `zone_list` to choose from, this function evaluates each zone in `zone_list` as a potential next zone. It calculates a score (typically a probability or log-probability) using the `ppm_model` for appending each candidate zone. If `cycle` is true, it also considers the probability of returning to the very first zone of the sequence from the candidate, optimizing for cyclical routes.
    *   **Arguments**:
        *   `zone_list` (list): List of available zones to choose the next one from.
        *   `start_zone` (list): The current sequence of zones already selected.
        *   `ppm_model` (PPM): The PPM model.
        *   `no_context_panelty` (float): Penalty for sparse context.
        *   `cycle` (bool, optional): If `True`, optimizes for cyclical routes by considering the probability of transitioning from a candidate back to the initial zone. Defaults to `False`.
        *   `hierarchical_mode`, `use_clusters`, `cluster_dict`, `cluster_weights` (optional): Parameters for PPM querying.
        *   `route_id` (any, optional): Route identifier for logging/context.
    *   **Returns**: A list of tuples, where each tuple contains a candidate zone and its associated score, sorted by score in descending order (best first).
