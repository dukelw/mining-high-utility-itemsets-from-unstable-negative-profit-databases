# Demo

import time
import tracemalloc

start_time = time.time()
tracemalloc.start()


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


def read_data(file_name="dataset.txt"):
    """
    Read and parse the dataset from the given file.

    Parameters:
    file_name (str): The name of the file containing the dataset. The default is "ehmintable.txt".

    Returns:
    list: A list of dictionaries representing the transactions. Each dictionary contains:
        - 'TID': Transaction ID.
        - 'items': List of item names in the transaction.
        - 'profit': List of profit per item in the transaction.
        - 'quantities': List of quantities of each item in the transaction.

    Example:
    >>> dataset = read_data()
    >>> print(dataset)
    [
        {'TID': 'T1', 'items': ['apple', 'banana'], 'profit': [5, 3], 'quantities': [2, 3]},
        {'TID': 'T2', 'items': ['apple', 'date'], 'profit': [8, 7], 'quantities': [1, 5]},
        ...
    ]
    """
    with open(file_name, "r") as file:
        data = file.read()

    dataset = eval(data)
    return dataset


dataset = read_data()


class TransactionInfo:
    """
    A class to represent a transaction info <TID, U, PRU>.

    Attributes:
        tid (str): The unique identifier of the transaction (TID).
        utility (int): The total utility (U) of the transaction.
        pru (int): The positive remaining utility (PRU) of the transaction.

    Methods:
        __repr__(): Provides a string representation of the transaction info.
    """

    def __init__(self, tid: str, utility: int, pru: int):
        """
        Initialize a Transaction Information (TI) object with a transaction ID (TID),
        total utility (U), and positive remaining utility (PRU).

        Parameters:
        - tid (str): The unique identifier of the transaction.
        - utility (int): The total utility of the transaction.
        - pru (int): The positive remaining utility of the transaction.
        """
        self.tid = tid
        self.utility = utility
        self.pru = pru

    def __repr__(self):
        return f"<TID: {self.tid}, U: {self.utility}, PRU: {self.pru}>"


class TIVector:
    """
    A class to represent the TI-vector, a container for <TID, U, PRU> structures.

    Attributes:
        transactions (list[TransactionInfo]): A list of TransactionInfo objects.

    Methods:
        add_transaction(tid, utility, pru):
            Adds a new TransactionInfo object to the transactions list.

        __repr__():
            Provides a string representation of the TI-vector.
    """

    def __init__(self):
        self.transactions = []

    def add_transaction(self, tid: str, utility: int, pru: int):
        """
        Add new transaction information

        Attributes:
            tid (str): The unique identifier of the transaction (TID).
            utility (int): The total utility (U) of the transaction.
            pru (int): The positive remaining utility (PRU) of the transaction.
        """
        self.transactions.append(TransactionInfo(tid, utility, pru))

    def __repr__(self):
        return f"TI-Vector({self.transactions})"


class EHMINItem:
    """
    A class to represent an item in the EHMIN-list with its utility, PRU, and TI-vector.

    Attributes:
        item_name (str): The name of the item.
        utility (int): The total utility (U) associated with the item.
        pru (int): The positive remaining utility (PRU) of the item.
        ti_vector (TIVector): The TI-vector containing transaction details for the item.

    Methods:
        add_transaction_info(tid, utility, pru):
            Adds a transaction info to the item's TI-vector.

        set_ti_vector(ti_vector):
            Sets a pre-created TI-vector for the item.

        __repr__():
            Provides a string representation of the EHMINItem object.
    """

    def __init__(self, item_name=None, utility=0, pru=0):
        """
        Initialize an instance of the EHMINItem class.

        Parameters:
        - item_name (str): The name of the item.
        - utility (int): The total utility of the item.
        - pru (int): The potential remaining utility of the item.

        The class also initializes a Transaction Information Vector (TIVector) for the item.
        If the item_name is None, the TIVector is set to None.
        """
        self.item_name = item_name
        self.utility = utility
        self.pru = pru
        self.ti_vector = TIVector() if item_name is not None else None

    def add_transaction_info(self, tid: str, utility: int, pru: int):
        """
        Sets a pre-created TI-vector for the item.

        :param tid: The unique identifier of the transaction (TID).
        :type tid: str

        :param utility: The total utility (U) of the transaction.
        :type utility: int

        :param pru: The positive remaining utility (PRU) of the transaction.
        :type pru: int

        :return: None
        """
        self.ti_vector.add_transaction(tid, utility, pru)

    def set_ti_vector(self, ti_vector: TIVector) -> None:
        """
        Sets a pre-created TI-vector for the item.

        This method assigns the provided TI-vector to the current item. The TI-vector
        contains transaction-specific information such as transaction IDs and their
        corresponding utilities.

        Parameters:
        - ti_vector (TIVector): The pre-created TI-vector to be assigned to the item.

        Returns:
        - None: This method does not return any value.
        """
        self.ti_vector = ti_vector

    def __repr__(self):
        """
        Return a string representation of the EHMINItem object.

        The string representation includes the item name, utility, potential remaining utility (PRU),
        and the transaction information vector (TI Vector).

        Parameters:
        None

        Returns:
        str: A string representation of the EHMINItem object.
        """
        return (
            f"EHMINItem(Item: {self.item_name}, U: {self.utility}, PRU: {self.pru}, "
            f"{self.ti_vector})"
        )


class EHMINList:
    """
    A class to manage a global EHMIN-list.

    Attributes:
        items (dict[str, EHMINItem]): A dictionary of items where keys are item names
            and values are EHMINItem objects.

    Methods:
        find_or_create(item_name, utility=0, pru=0):
            Finds an item by name or creates a new one if it doesn't exist.

        increase_pru(item_name, pru):
            Increases the PRU of an item by a given amount.

        __repr__():
            Provides a string representation of the EHMINList.
    """

    def __init__(self):
        """
        Initialize an empty EHMINList.

        The EHMINList is a data structure used in the EHMIN algorithm for mining frequent itemsets.
        It contains a dictionary of items, where each item is represented by an EHMINItem object.

        Attributes:
        - items (dict): A dictionary of items, where keys are item names (strings) and values are EHMINItem objects.
        """
        self.items = {}

    def find_or_create(self, item_name: str, utility=0, pru=0):
        """
        Finds an item by name or creates a new one if it doesn't exist.

        :param item_name (str): The name of item.
        :param utility (int): The total utility (U) of the transaction.
        :param pru (int): The positive remaining utility (PRU) of the transaction.

        :return: The EHMINItem object corresponding to the given item_name.
        """
        if item_name not in self.items:
            self.items[item_name] = EHMINItem(item_name, utility, pru)
        return self.items[item_name]

    def increase_pru(self, item_name: str, pru: int):
        """
        Increase pru value of an item.

        This function updates the positive remaining utility (PRU) of a specific item by adding the given pru value.

        Parameters:
        - item_name (str): The name of the item whose PRU needs to be updated.
        - pru (int): The positive remaining utility (PRU) of the transaction.

        The function does not return any value. It directly updates the PRU value of the specified item.
        """
        self.items[item_name].pru += pru

    def __repr__(self):
        return f"EHMINList({list(self.items.values())})"


def utility_of_itemset(itemset: list[str], transaction: dict) -> int:
    """
    Calculate the utility of an itemset in a given transaction.

    Parameters:
    itemset (list[str]): List of items in the itemset.
    transaction (dict): A dictionary with transaction data. It should contain the following keys:
        - 'items' (list[str]): List of item names in the transaction.
        - 'quantities' (list[int]): Quantities of each item in the transaction, corresponding to 'items'.
        - 'profit' (list[int]): Profit values for each item in the transaction, corresponding to 'items'.

    Returns:
    int: Total utility of the itemset in the transaction.
    """
    utility = 0
    for item in itemset:
        if item in transaction["items"]:
            idx = transaction["items"].index(item)
            utility += transaction["quantities"][idx] * transaction["profit"][idx]
    return utility


def transaction_utility(transaction: dict) -> int:
    """
    Calculate the Transaction Utility (TU) for a given transaction.

    :param transaction: A dictionary with transaction data. It should contain keys:
        - "items" (list of str): List of item names in the transaction.
        - "profit" (list of int): Profit values for each item in the transaction.
        - "quantities" (list of int): Quantities of each item in the transaction.
    :return: Total utility of the transaction.
    """
    utility = 0
    for i in range(len(transaction["items"])):
        utility += transaction["quantities"][i] * transaction["profit"][i]
    return utility


def redefine_transaction_utility(transaction: dict) -> int:
    """
    Calculate the PTU (Positive Transaction Utility)/Redefined Transaction Utility (RTU) for a given transaction.

    :param transaction: A dictionary with transaction data. It should contain keys:
        - "items" (list of str): List of item names in the transaction.
        - "profit" (list of int): Profit values for each item in the transaction.
        - "quantities" (list of int): Quantities of each item in the transaction.
    :return: Total Reduced Utility of the transaction.
    """
    RTU = 0
    for i in range(len(transaction["items"])):
        if transaction["profit"][i] > 0:
            RTU += transaction["quantities"][i] * transaction["profit"][i]
    return RTU


def calculate_rtwu(itemset: list[str], dataset: list[dict]) -> int:
    """
    Calculate the Redefined Transactional Weighted Utility (RTWU)/Positive Transactional Weighted Utility (PTWU) for a given itemset across the dataset.
    :param itemset: A list of items defining the base itemset (e.g., ['a', 'b']).
    :param dataset: A list of transactions.
    :return: PTWU/RTWU value for the itemset across the dataset.
    """
    RTWU = 0
    for transaction in dataset:
        # Check if all items in the itemset are in the transaction
        if all(item in transaction["items"] for item in itemset):
            # Calculate RTU for this transaction and add it to RTWU
            RTWU += redefine_transaction_utility(transaction)
    return RTWU


def redefined_remaining_utility(itemset: list[str], transaction: dict) -> int:
    """
    Calculate the positive remaining utility (pru)/redefined remaining utility (rru) of an itemset in a transaction.

    :param itemset: List of items in the itemset.
    :param transaction: A dictionary with transaction data. It contains keys:
        - 'items': List of item names in the transaction.
        - 'quantities': Quantities of each item in the transaction, corresponding to 'items'.
        - 'profit': Profit values for each item in the transaction, corresponding to 'items'.
    :return: Positive/Redefined remaining utility of the itemset in the transaction.
    """
    rru = 0

    valid_items = [item for item in itemset if item in transaction["items"]]

    # In case item set is an empty set, rru is an empty set
    if len(itemset) == 0:
        for i in range(0, len(transaction["items"])):
            quantity = transaction["quantities"][i]
            profit = transaction["profit"][i]
            if profit > 0:  # Only consider items with positive profit
                rru += quantity * profit
        return rru

    if not valid_items:
        return rru

    # Get the last index in the sorted transaction where itemset items appear
    max_idx = max(transaction["items"].index(item) for item in valid_items)
    # Sum the utility of items appearing after the itemset with positive profit
    for i in range(max_idx + 1, len(transaction["items"])):
        quantity = transaction["quantities"][i]
        profit = transaction["profit"][i]
        if profit > 0:  # Only consider items with positive profit
            rru += quantity * profit

    return rru


def categorize_items(dataset: list[dict]) -> tuple[set[str], set[str]]:
    """
    Calculate item utilities and classify items into positive and negative utility sets.
    :param dataset: List of transactions.
    :return: Tuple of (positive_items, negative_items)
    """
    item_utilities = {}
    for transaction in dataset:
        for idx, item in enumerate(transaction["items"]):
            utility = transaction["quantities"][idx] * transaction["profit"][idx]
            if item not in item_utilities:
                item_utilities[item] = []
            item_utilities[item].append(utility)

    positive_items = set()
    negative_items = set()

    for item, utilities in item_utilities.items():
        if all(u > 0 for u in utilities):
            positive_items.add(item)
        elif all(u < 0 for u in utilities):
            negative_items.add(item)

    return positive_items, negative_items


def calculate_ptwus(dataset: list[dict]) -> tuple[dict, dict]:
    """
    Calculate the Positive Transactional Weighted Utility (PTWU) and support for each item across the dataset.

    PTWU measure the utility contribution and frequency of each item in the dataset.

    Parameters:
    dataset (list): A list of dictionaries, where each dictionary represents a transaction with:
        - 'items' (list of str): List of item names in the transaction.

    Returns:
    tuple: A tuple containing:
        - ptwus (dict): A dictionary where keys are items and values are their RTWU across transactions.
        - supports (dict): A dictionary where keys are items and values are their occurrence counts.
    """
    rtwus = {}
    supports = {}

    for transaction in dataset:
        for item in transaction["items"]:
            if item in supports:
                supports[item] += 1
            else:
                supports[item] = 1

            if item not in rtwus:
                itemset = [item]
                rtwus[item] = calculate_rtwu(itemset, dataset)

    return rtwus, supports


def get_items_order(
    itemset: list[str],
    positive_items: list[str],
    negative_items: list[str],
    rtwus: dict,
    supports: dict,
) -> list[str]:
    """
    Sort items in a transaction according to the processing order:
    (i) PI items are sorted by RTWU (ascending), (ii) NI items are sorted by support (ascending).
    :param items: List of items to sort
    :param positive_items: List of positive items
    :param negative_items: List of negative items
    :param rtwus/ptwus: Dictionary of rtwus/ptwus
    :supports: Dictionary of support for sorting
    """
    itemset = dict(sorted(itemset.items()))

    def sort_key(item):
        # If item is in positive_items (PI), sort by RTWU (ascending)
        if item in positive_items:
            return (1, rtwus.get(item, float("inf")))
        # If item is in negative_items (NI), sort by support (ascending)
        elif item in negative_items:
            return (2, supports.get(item, float("inf")))
        # Default priority for items not in PI or NI
        return (3, float("inf"))

    # Sort the items in itemset using the custom sort_key
    sorted_items = sorted(itemset, key=sort_key)
    return sorted_items


def calculate_utility(itemset: list[str], dataset: list[dict]) -> int:
    """
    Calculate the utility of the given itemset.

    :param itemset: The itemset for which utility is to be calculated. This is a list of strings representing the items.
    :param dataset: The dataset containing transactions. This is a list of dictionaries, where each dictionary represents a transaction. Each transaction dictionary contains keys 'items', 'profit', and 'quantities'.
    :return: Total utility. This is an integer representing the sum of the utilities of the items in the itemset across all transactions in the dataset.
    """

    utility = 0

    for transaction in dataset:
        # Convert the itemset to a set for easier subset checking
        itemset_set = set(itemset)

        # Check if the itemset is a subset of the transaction's items
        if itemset_set.issubset(set(transaction["items"])):
            # Calculate utility for this transaction
            transaction_utility = 0

            # Calculate utility based on profit and quantities
            for item, quantity in zip(transaction["items"], transaction["quantities"]):
                if item in itemset_set:
                    index = transaction["items"].index(item)
                    profit = transaction["profit"][index]
                    transaction_utility += profit * quantity

            utility += transaction_utility

    return utility


def calculate_pu(
    pattern: set[str], transaction: dict, positive_items: list[str]
) -> int:
    """
    Calculate the Positive Utility (PU) of a given pattern in a transaction.

    The positive utility of a pattern, X, in a transaction, Tk, is the sum of the utilities
    of items in the pattern that are also in the positive items list (PI) in the transaction.

    Parameters:
    - pattern: Set of items (list of item names) representing the pattern X.
    - transaction: Dictionary representing a single transaction with keys:
        - 'TID': Transaction ID.
        - 'items': List of item names in the transaction.
        - 'quantities': List of item quantities in the transaction, corresponding to 'items'.
        - 'profit': List of profit per item in the transaction, corresponding to 'items'.
    - positive_items: Set of items considered as positive (PI).

    Returns:
    - Positive utility (PU) of the pattern in the given transaction (integer).

    Example:
    - For pattern {D, E} in transaction T5 from the dataset, if only D is in PI,
    PU({D, E}, T5) = U(D, T5).
    """
    pu = 0  # Initialize positive utility

    # Loop through items in the transaction and calculate utility for items in both pattern and positive_items
    for item, quantity, profit in zip(
        transaction["items"], transaction["quantities"], transaction["profit"]
    ):
        if item in pattern and item in positive_items:
            utility = quantity * profit
            pu += (
                utility  # Add to positive utility if item is in both the pattern and PI
            )

    return pu


def build_eucs(order: list[str]) -> list[list[any]]:
    """
    Build an Estimated Utility Co-occurrence Structure (EUCS).

    This function creates a 2D matrix representing the relationships between
    items in the input `order`. The matrix is filled with RTWU (Revised
    Transaction Weighted Utility) values computed for each pair of items.

    Args:
        order (list[str]): A list of item names representing the pattern X.

    Returns:
        list[list[Any]]: A 2D matrix (EUCS) where:
            - `eucs[0][i]` contains item names from the input order.
            - Other cells (eucs[i][j]) contain RTWU values for pairs of items.

    Notes:
        This function uses a global variable `dataset` for transaction data and
        a helper function `calculate_rtwu` to compute utility values. Ensure these
        are defined and accessible.
    """
    eucs = [[0 for _ in range(len(order))] for _ in range(len(order))]

    for i in range(1, len(order)):
        eucs[0][i] = order[i - 1]
        eucs[i][0] = order[i]

    for i in range(0, len(order)):
        row_item = order[i]
        for j in range(i + 1, len(order)):
            col_item = order[j]
            tmp_set = {row_item, col_item}
            eucs[j][i + 1] = calculate_rtwu(tmp_set, dataset)

    return eucs


def calculate_pru(itemset: list[str], dataset: list[dict]) -> int:
    """
    Calculate the Potential Remaining Utility (PRU) of an itemset in a dataset.

    The PRU of an itemset quantifies the total utility of items in transactions
    that are not part of the given itemset but co-occur with it.

    Args:
        itemset (list[str]): A list of item names representing the itemset X.
        dataset (list[dict]): A list of transactions. Each transaction is a dictionary with keys:
            - "items" (list[str]): List of items in the transaction.
            - "profit" (list[int]): Profit values for each item in the transaction.
            - "quantities" (list[int]): Quantities of each item in the transaction.

    Returns:
        int: The total PRU value for the given itemset across all transactions in the dataset.

    Notes:
        - PRU(X, T_k) is calculated for items in a transaction that:
            1. Are not part of the given itemset.
            2. Have a positive profit value.
        - This function assumes the dataset is pre-validated to ensure all fields are present.

    Examples:
        dataset = [
            {
                "items": ["apple", "banana", "cherry"],
                "profit": [5, 3, 10],
                "quantities": [2, 3, 1],
            },
            {
                "items": ["apple", "date"],
                "profit": [8, 7],
                "quantities": [1, 5],
            },
        ]

        itemset = ["apple"]

        result = calculate_pru(itemset, dataset)
        print(result)  # Output will depend on the dataset and itemset.
    """
    # Initialize PRU value
    pru = 0

    # Iterate through each transaction in the dataset
    for transaction in dataset:
        items = transaction["items"]
        profits = transaction["profit"]
        quantities = transaction["quantities"]

        # Check if itemset is a subset of the transaction's items
        if set(itemset).issubset(set(items)):
            # Calculate PRU(X, T_k) for items after the last item in the itemset
            pru_x_tk = sum(
                profits[i] * quantities[i]
                for i in range(0, len(items))
                if profits[i] > 0 and not set(items[i]).issubset(set(itemset))
            )

            # Add PRU(X, T_k) to the total PRU(X)
            pru += pru_x_tk

    return pru


def ehmin_combine(Uk: EHMINItem, Ul: EHMINItem, pfutils: dict, minU: int) -> EHMINItem:
    """
    Combine two EHMIN-lists (Uk, Ul) and create a new conditional EHMIN-list.

    Args:
        Uk: EHMIN-list for item Uk
        Ul: EHMIN-list for item Ul
        pfutils: Prefix utility map containing transaction IDs and their utilities
        minU: Minimum utility threshold for pruning

    Returns:
        A new EHMIN-list C if conditions are met, otherwise None
    """
    # Initialize the conditional utility pattern list C
    C = EHMINItem(Ul.item_name, 0, Ul.pru)
    # C.set_ti_vector(Ul.ti_vector)  # Start with Ul's transaction info
    x = Uk.utility + Uk.pru
    y = Uk.utility

    # Initialize iterators for the utility vectors of Uk and Ul
    current_k = 0
    current_l = 0

    # Get the lengths of utility vectors
    length_k = len(Uk.ti_vector.transactions)
    length_l = len(Ul.ti_vector.transactions)

    while current_k < length_k and current_l < length_l:
        sk = Uk.ti_vector.transactions[current_k]
        sl = Ul.ti_vector.transactions[current_l]

        if sk.tid == sl.tid:
            # Retrieve the prefix utility for the shared transaction ID
            pfutil = pfutils.get(sk.tid, 0)

            # Calculate the combined utility and remaining utility
            util = sk.utility + sl.utility - pfutil
            rutil = min(sk.pru, sl.pru)

            # Add the combined transaction to C
            C.add_transaction_info(sk.tid, utility=util, pru=rutil)
            C.utility += util

            # Update the `y` value for pruning
            y += sl.utility - pfutil

            # N-Prune condition
            if sk.pru == 0 and y < minU:
                return None

            # Move both iterators forward
            current_k += 1
            current_l += 1
        elif sk.tid > sl.tid:
            # Move iterator for Ul forward
            current_l += 1
        else:
            # LA-Prune condition: check if further processing is beneficial
            x -= sk.utility + sk.pru
            if x < minU:
                return None

            # Move iterator for Uk forward
            current_k += 1

    if len(C.ti_vector.transactions) == 0:
        return None

    return C


def ehmin_mine(
    P: EHMINItem,
    UL: EHMINList,
    pref: set[str],
    eucs: list[list[any]],
    minU: int,
    sorted_item: list[str],
):
    """
    Recursive function to mine data.

    Parameters:
    - P (EHMINItem): The prefix pattern.
    - UL (EHMINList): The list of unprocessed items.
    - pref (set[str]): The prefix pattern.
    - eucs (list[list[any]]): The estimated utility co-occurrence structure.
    - minU (int): The minimum utility threshold for pruning.
    - sorted_item (list[str]): The sorted list of items.

    Returns:
    - None. The function performs the EHMIN algorithm recursively.
    """
    # Initialize the prefix utility map
    pfutils = {}
    if P.item_name != None:
        for s in P.ti_vector.transactions:
            pfutils[s.tid] = s.utility

    # Iterate over each Uk in UL
    for Uk in UL.items.values():
        # First pruning condition (U ≥ minUtil)
        tmp = pref.union({Uk.item_name})

        if Uk.utility >= minU:
            HUP["".join(tmp)] = Uk.utility
            check_list.append(tmp)

        # Second pruning condition (U + PRU ≥ minUtil)
        if Uk.utility + Uk.pru >= minU:
            # Initialize the conditional EHMIN-lists, CL
            CL = EHMINList()

            # Iterate over each Ul in UL where l > k
            for Ul in UL.items.values():
                k = sorted_item.index(Uk.item_name) + 1
                l = sorted_item.index(Ul.item_name)
                if k <= l:  # Ensure l > k
                    # EUCS pruning condition
                    if eucs[l][k] >= minU:
                        C = ehmin_combine(Uk, Ul, pfutils, minU)
                        if C:
                            CL.items[C.item_name] = C

            # Recursive call to EHMIN_Mine if CL is non-empty
            if len(CL.items) > 0:
                ehmin_mine(Uk, CL, pref | {Uk.item_name}, eucs, minU, sorted_item)


def ehmin(k, δ: float):
    """Using for execute EHMIN Algorithm finding top K high utility

    Args:
        k: The number of patterns to find
        δ: Minimum utility threshold for pruning

    Returns:
        Print all the top K high utility item that greater than or equal threshold
    """
    # Step 1: 1st Database Scan
    # Calculate PTWU (RTWU)
    global HUP
    ptwus, supports = calculate_ptwus(dataset)
    ptus = {}
    print("ptwus", ptwus)
    print("supports", supports)
    # Calculate PTU (RTU) of each transaction
    for transaction in dataset:
        ptu = redefine_transaction_utility(transaction)
        ptus[transaction["TID"]] = ptu
    print("ptus", ptus)
    # Calculate minU
    minU = sum(value * δ for value in ptus.values())
    print("minU", minU)
    # Get EHMN-list for 1-itemsets
    positive_items, negative_items = categorize_items(dataset)
    list_item = {item: ptwu for item, ptwu in ptwus.items() if ptwu >= minU}
    sorted_item = get_items_order(
        list_item, positive_items, negative_items, ptwus, supports
    )
    print("sorted_item", sorted_item)
    # Index EHMIN-list sorted item
    # Calculate utility
    for item in sorted_item:
        utility = calculate_utility(item, dataset)
        # Index EHMIN-list with utility and pru = 0
        ehmin_item = ehmin_list.find_or_create(item, utility)
    print("Ehmin", ehmin_list)

    # Step 2: 2nd Database Scan
    for transaction in dataset:
        ptu_k = 0  # Recompute PTU(Tk) and initialize it to 0

        # Step 1: Calculate PTU for each transaction
        for item in transaction["items"]:
            # Check PTWU(i) condition for pruning
            if ptwus[item] > minU:
                ptu_k += calculate_pu(set(item), transaction, positive_items)

        # Initialize a temporary map
        tmp = {}

        # # Step 2: Insert items into tmp and calculate PTWU if necessary
        for item, quantity, profit in zip(
            transaction["items"], transaction["quantities"], transaction["profit"]
        ):
            tmp[item] = quantity * profit  # Store internal utility and external utility

            # PTWU condition to recompute PTWU
            if ptwus[item] > minU:
                new_PTWIU = calculate_rtwu(set(item), dataset)
                ptwus[item] = new_PTWIU + ptu_k

        rutil = 0  # Initialize rutil
        # Sort to calculate PRU (inportant)
        tmp_list_item = {item: ptwu for item, ptwu in tmp.items()}
        tmp = get_items_order(
            tmp_list_item, positive_items, negative_items, ptwus, supports
        )
        tmp = {item: tmp_list_item[item] for item in tmp}
        # # Process each item in reverse order
        for item, utility in reversed(list(tmp.items())):
            # Find or create the item in the EHMIN-list
            ehmin_item = ehmin_list.find_or_create(item, utility, pru=0)
            # Insert values into Ui.Tk vector
            ehmin_item.add_transaction_info(
                transaction["TID"], utility=utility, pru=rutil
            )

            # Update rutil if U(i) > 0
            if utility > 0:
                ehmin_list.increase_pru(item, rutil)
                rutil += utility

    # Calculate EUCS[v_ik, v_jk] with PTU_k
    eucs = build_eucs(sorted_item)

    print("After 2nd scan", ehmin_list)

    # Step 3: Mining
    ehmin_mine(EHMINItem(), ehmin_list, set(), eucs, minU, sorted_item)
    HUP = dict(sorted(HUP.items(), key=lambda item: item[1], reverse=True)[:k])


# Create an empty EHMINList
HUP = {}
ehmin_list = EHMINList()
check_list = []
ehmin(10, 0.1)

end_time = time.time()

execution_time = end_time - start_time

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

with open("ehmin_result.txt", "w") as outputfile:
    for item in HUP:
        outputfile.write(f"{item} - {HUP[item]}\n")
    outputfile.write(f"Execution time: {execution_time:.6f} seconds\n")
    outputfile.write(f"Current memory usage: {current / 1024:.2f} KB\n")
    outputfile.write(f"Peak memory usage: {peak / 1024:.2f} KB\n")
