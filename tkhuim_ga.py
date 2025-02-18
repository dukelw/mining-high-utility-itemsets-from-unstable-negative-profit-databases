import random


def get_dataset() -> list:
    """
    This function reads a dataset from a text file and returns a list of transactions.

    Parameters:
    None

    Returns:
    data (list): A list of transactions. Each transaction is represented as a list containing:
        - Transaction ID (string)
        - List of items (list of strings)
        - List of quantities (list of integers)
        - List of profits (list of integers)
    """
    data = []
    with open("3rddataset.txt", "r", encoding="utf-8") as dataset:
        next(dataset)  # Skip the header line
        for line in dataset:
            parts = line.split()
            transaction_id = parts[0]
            items = [str(u) for u in parts[1].split(",")]
            quantity = [int(u) for u in parts[2].split(",")]
            profit = [int(u) for u in parts[3].split(",")]
            data.append([[transaction_id], items, quantity, profit])
    return data


def get_utility(data: list) -> list:
    """
    This function calculates the utility of each unique item in the dataset.

    Parameters:
    data (list): A list of transactions. Each transaction is represented as a list containing:
        - Transaction ID (string)
        - List of items (list of strings)
        - List of quantities (list of integers)
        - List of profits (list of integers)

    Returns:
    utility (list): A list of tuples, where each tuple contains an item and its utility.
        The utility of an item is initially set to 0.
    """
    unit = set()
    for trans in data:
        for item in trans[1]:
            unit.add(item)
    utility = []
    for item in unit:
        utility.append([item, 0])
    return utility


def calculate_utility(data: list) -> list:
    """
    Calculates the utility of each item in the dataset.

    This function takes a list of transactions and a list of utility values for each unique item,
    and returns a new list of transactions with the utility values of each item in the transaction.

    Parameters:
    data (list): A list of transactions. Each transaction is represented as a list containing:
        - Transaction ID (string)
        - List of items (list of strings)
        - List of quantities (list of integers)
        - List of profits (list of integers)

    Returns:
    dataset (list): A new list of transactions. Each transaction is represented as a list containing:
        - Transaction ID (string)
        - List of items (list of strings)
        - List of quantities (list of integers) multiplied by their respective profits (list of integers)
    """
    dataset = []
    for i in range(len(data)):
        dataset.append([data[i][0], data[i][1], data[i][2]])
        for j in range(len(data[i][1])):
            dataset[i][2][j] = data[i][2][j] * data[i][3][j]
    return dataset


data = get_dataset()
utility = list(get_utility(data))
dataset = calculate_utility(data)


def utility_itemset(dataset: list, utility: list) -> list:
    """
    Calculates the utility of each item in the dataset.

    This function takes a list of transactions and a list of utility values for each unique item,
    and returns a new list of tuples representing the utility of each item.

    Parameters:
    dataset (list): A list of transactions. Each transaction is represented as a list containing:
        - Transaction ID (string)
        - List of items (list of strings)
        - List of quantities (list of integers) multiplied by their respective profits (list of integers)
    utility (list): A list of tuples, where each tuple contains an item and its utility.
        The utility of an item is initially set to 0.

    Returns:
    utility_trans (list): A new list of tuples, where each tuple contains an item and its utility.
        The utility of an item is calculated as the sum of the quantities of that item in all transactions.
    """
    utility_trans = utility
    for k in range(len(utility_trans)):
        utility_trans[k][1] = 0
    for i in range(len(dataset)):
        for j in range(len(dataset[i][1])):
            for k in range(len(utility_trans)):
                if dataset[i][1][j] == utility_trans[k][0]:
                    utility_trans[k][1] += dataset[i][2][j]
    return utility_trans


u = utility_itemset(dataset, utility)


def TU(transaction: list) -> int:
    """
    Calculates the total utility of a transaction.

    This function takes a transaction (represented as a list) and returns the sum of the quantities
    of items in the transaction. Each quantity is multiplied by its respective profit, as given in the
    transaction.

    Parameters:
    transaction (list): A list representing a transaction. The list contains three elements:
        - Transaction ID (string)
        - List of items (list of strings)
        - List of quantities (list of integers) multiplied by their respective profits (list of integers)

    Returns:
    int: The total utility of the transaction, calculated as the sum of the quantities of items.
    """
    return sum(transaction[2])


def get_top_m_items(transaction: list, m: int) -> list:
    """
    Extracts the top 'm' items with the highest utilities from a given transaction.

    Parameters:
    transaction (list): A list representing a transaction. The list contains three elements:
        - Transaction ID (string)
        - List of items (list of strings)
        - List of quantities (list of integers) multiplied by their respective profits (list of integers)

    m (int): The number of top items to extract.

    Returns:
    list: A list containing the transaction ID, the top 'm' items, and their corresponding utilities.
    """
    items_with_utilities = list(zip(transaction[1], transaction[2]))
    items_with_utilities.sort(key=lambda x: x[1], reverse=True)
    top_m_items = items_with_utilities[:m]
    top_items = [item[0] for item in top_m_items]
    top_utilities = [item[1] for item in top_m_items]
    return [transaction[0], top_items, top_utilities]


def initial_solutions(dataset: list, n: int, m: int) -> list:
    """
    This function generates initial solutions for the Top-k High Utility Itemset Mining (TKHUIM) problem.

    Parameters:
    dataset (list): A list of transactions. Each transaction is represented as a list containing:
        - Transaction ID (string)
        - List of items (list of strings)
        - List of quantities (list of integers) multiplied by their respective profits (list of integers)

    n (int): The number of initial solutions to generate.

    m (int): The number of top items to extract from each transaction.

    Returns:
    P (list): A list of initial solutions, where each solution is represented as a list of item names.
    """
    trans_P = []
    P = []
    for Ty in dataset:
        u = TU(Ty)
        X = get_top_m_items(Ty, m)
        if len(P) < n:
            trans_P.append(X)
            P.append(X[1])
        else:
            min_utility = min(trans_P, key=TU)
            if u > TU(min_utility):
                trans_P.remove(min_utility)
                P.remove(min_utility[1])
                trans_P.append(X)
                P.append(X[1])
    return P


def F(X: list) -> int:
    """
    Calculates the total utility of a given itemset in a transaction dataset.

    Parameters:
    X (list): A list of item names. The itemset is considered to be present in each transaction
        if all items in the itemset are present in the transaction.

    Returns:
    int: The total utility of the itemset in the transaction dataset. The utility of an itemset
        is calculated as the sum of the quantities of items in the itemset for all transactions
        where the itemset is present. If the input itemset is None, the function returns 0.
    """
    sum = 0
    if X is None:
        return sum
    for i in range(len(dataset)):
        if set(X).issubset(set(dataset[i][1])):
            for j in range(len(dataset[i][1])):
                if dataset[i][1][j] in X:
                    sum += dataset[i][2][j]
    return sum


def roullete_wheel(utility_itemset: list) -> list:
    """
    Performs a roulette wheel selection based on the utility values of items.

    Parameters:
    utility_itemset (list): A list of tuples, where each tuple contains an item and its utility.
        The utility of an item is initially set to 0.

    Returns:
    list: A list containing a single item selected based on the roulette wheel selection.
    """
    elements = []
    weights = []
    sum = 0
    for item in utility_itemset:
        elements.append(item[0])
        weights.append(item[1])
        sum += item[1]
    for i in range(len(weights)):
        weights[i] = weights[i] / sum
    return random.choices(elements, weights, k=1)


def genetic_operators(S: list, a: float, b: float) -> set:
    """
    Performs genetic operators (crossover and mutation) on a set of itemsets (S) based on the given probabilities (a and b).

    Parameters:
    S (set): A set of itemsets. Each itemset is represented as a string of item names.
    a (float): The probability of performing crossover between two itemsets.
    b (float): The probability of performing mutation on an itemset.

    Returns:
    set: A new set of itemsets resulting from the genetic operators.
    """
    P = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            Xi = S[i]
            Xj = S[j]
            if a > random.uniform(0, 1):
                x = ""
                y = ""
                if F(Xi) > F(Xj):
                    minXi = 0
                    maxXj = 0
                    for item in utility_itemset(dataset, utility):
                        if Xi is not None:
                            if item[0] in Xi:
                                minXi = item[1]
                                x = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xi is not None:
                            if item[0] in Xi:
                                if item[1] < minXi:
                                    minXi = item[1]
                                    x = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xj is not None:
                            if item[0] in Xj:
                                maxXj = item[1]
                                y = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xj is not None:
                            if item[0] in Xj:
                                if item[1] > maxXj:
                                    maxXj = item[1]
                                    y = item[0]
                else:
                    minXj = 0
                    maxXi = 0
                    for item in utility_itemset(dataset, utility):
                        if Xj is not None:
                            if item[0] in Xj:
                                minXj = item[1]
                                y = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xj is not None:
                            if item[0] in Xj:
                                if item[1] < minXj:
                                    minXj = item[1]
                                    y = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xi is not None:
                            if item[0] in Xi:
                                maxXi = item[1]
                                x = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xi is not None:
                            if item[0] in Xi:
                                if item[1] > maxXi:
                                    maxXi = item[1]
                                    x = item[0]

                if Xi is None:
                    Xi = set()
                else:
                    Xi = set(Xi)

                Xi = Xi - {x}
                Xi = Xi | {y}

                if Xj is None:
                    Xj = set()
                else:
                    Xj = set(Xj)

                Xj = Xj - {y}
                Xj = Xj | {x}

            for X in [Xi, Xj]:
                if b > random.uniform(0, 1):
                    x = ""
                    if 0.5 > random.uniform(0, 1):
                        minX = 0
                        for item in utility_itemset(dataset, utility):
                            if X is not None:
                                if item[0] in X:
                                    minX = item[1]
                                    x = item[0]
                        for item in utility_itemset(dataset, utility):
                            if X is not None:
                                if item[0] in X:
                                    if item[1] < minX:
                                        minX = item[1]
                                        x = item[0]
                        if X is None:
                            X = set()
                        else:
                            X = set(X)
                        X = set(X) - {x}
                    else:
                        x = roullete_wheel(utility_itemset(dataset, utility))
                        if X is None:
                            X = set()
                        else:
                            X = set(X)
                        if x is None:
                            x = set()
                        else:
                            x = set(x)
                        X = set(X) | set(x)
                if X is not None and len(X) != 0:
                    P.add("".join(X))
    return P


def contains_same_characters(E: list, target: str) -> bool:
    """
    Checks if a list of strings (E) contains a specific string (target) with the same characters.

    Parameters:
    E (list): A list of strings.
    target (str): The string to check for in the list.

    Returns:
    bool: True if the list contains the target string with the same characters, False otherwise.
    """
    target_set = set(target)

    for item in E:
        if set(item) == target_set:
            return True

    return False


def TKHUIM_GA(dataset: list, n: int, m: int, e: int) -> dict:
    """
    Performs the Top-k High Utility Itemset Mining (TKHUIM) problem using a Genetic Algorithm (GA).

    Parameters:
    dataset (list): A list of transactions. Each transaction is represented as a list containing:
        - Transaction ID (string)
        - List of items (list of strings)
        - List of quantities (list of integers) multiplied by their respective profits (list of integers)

    n (int): The number of initial solutions to generate.

    m (int): The number of top items to extract from each transaction.

    e (int): The number of top-k high utility itemsets to find.

    Returns:
    dict: A dictionary containing the top-k high utility itemsets and their corresponding utilities.
    """
    HUP = {}
    P = []
    E = []
    u = utility_itemset(dataset, utility)
    u.sort(key=lambda x: x[1], reverse=True)

    top_m_items = u[:e]
    top_items = [item[0] for item in top_m_items]
    E = top_items

    exit = False
    P = initial_solutions(dataset, n, m)
    a = 0.5
    b = 0.5
    while True:
        S = tournament_selection(P, len(P) - 1, n)

        P = genetic_operators(S, a, b)

        new_E = []
        for item in P:
            if not contains_same_characters(new_E, item):
                new_E.append(item)
            if not contains_same_characters(HUP.keys(), item):
                HUP[item] = F(list(item))
        for item in E:
            if not contains_same_characters(new_E, item):
                new_E.append(item)
            if not contains_same_characters(HUP.keys(), item):
                HUP[item] = F(list(item))

        new_E.sort(key=F, reverse=True)
        new_E = list(set(new_E))
        HUP = dict(sorted(HUP.items(), key=lambda item: item[1], reverse=True))

        if new_E != E:
            a = a + 0.05
            b = b - 0.05
            E = new_E
        else:
            a = a - 0.05
            b = b + 0.05

        if round(b, 2) == 1.00:
            exit = True

        if exit:
            break
    print("HUP", HUP)
    return dict(list(HUP.items())[:e])


def tournament(T: list, k: int) -> list:
    """
    Performs a tournament selection among a list of individuals.

    Parameters:
    T (list): A list of individuals randomly selected from a population.
    k (int): The tournament size, i.e., the number of elements in T.

    Returns:
    The fittest individual from the tournament.

    """
    if k == 0:
        return
    best = T[0]
    for i in range(1, k):
        next = T[i]
        if F(next) > F(best):
            best = next
    return best


def tournament_selection(P: set, k: int, n: int) -> list:
    """
    Performs a tournament selection among a list of individuals.

    Parameters:
    P (set): The population as a set of individuals.
    k (int): The tournament size, such that 1 ≤ k ≤ the number of individuals in P.
    n (int): The total number of individuals we wish to select.

    Returns:
    list: The pool of individuals selected in the tournaments.

    """
    P_list = list(P)
    B = [None] * n

    for i in range(n):
        T = random.choices(P_list, k=k)  # Picks with replacement
        B[i] = tournament(T, len(T))

    return B


E = TKHUIM_GA(dataset, 4, 5, 4)
for item in E:
    print(item, "-", E[item])
