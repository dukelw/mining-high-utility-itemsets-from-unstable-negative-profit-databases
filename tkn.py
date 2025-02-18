import heapq
from collections import defaultdict

# Define dataset and profits
TKHQ = []  # Sử dụng danh sách để làm hàng đợi ưu tiên thủ công
# transactions = {
#     "t1": {"a": 1, "b": 2, "c": 2, "d": 1},
#     "t2": {"a": 1, "b": 3, "c": 3, "d": 2, "e": 2},
#     "t3": {"a": 1, "c": 6, "e": 3},
#     "t4": {"b": 3, "d": 5, "e": 2},
#     "t5": {"b": 1, "c": 5, "d": 1, "e": 4},
#     "t6": {"c": 2, "d": 1, "e": 1},
# }

# profits = {"a": 4, "b": 3, "c": 1, "d": -1, "e": 2}

transactions = {
    "t1": {"a": 2, "b": 2, "d": 1, "e": 3, "f": 2, "g": 1},
    "t2": {"b": 1, "c": 5},
    "t3": {"b": 2, "c": 1, "d": 3, "e": 2, "f": 1},
    "t4": {"c": 2, "d": 1, "e": 3},
    "t5": {"a": 2, "f": 3},
    "t6": {"a": 2, "b": 1, "c": 4, "d": 2, "e": 1, "f": 3, "g": 1},
    "t7": {"b": 3, "c": 2, "e": 2},
}
profits = {"a": 4, "b": 3, "c": 1, "d": -1, "e": 2, "f": -1, "g": -2}

# transactions = {
#     "t1": {"a": 2, "b": 3, "d": 1, "h": 1},
#     "t2": {"a": 2, "c": 4, "e": 2, "h": 3},
#     "t3": {"b": 6, "c": 3, "d": 1, "e": 3, "f": 2},
#     "t4": {"a": 4, "b": 3, "c": 3, "g": 2},
#     "t5": {"b": 4, "d": 4, "e": 1, "g": 2, "h": 1},
# }

# profits = {"a": 2, "b": 1, "d": 3, "h": -1, "c": 1, "e": -1, "f": 5, "g": -1}


def calculate_priu(transactions, profits):
    priu_dict = {}
    for transaction in transactions.values():
        for item, quantity in transaction.items():
            utility = quantity * profits.get(
                item, 0
            )  # calculate utility for each item in transaction
            if item in priu_dict:
                priu_dict[item] += utility
            else:
                priu_dict[item] = utility
    return priu_dict


def LIUS_Build_contiguous(sorted_database, priu_list):
    """
    Tạo ma trận tam giác với utility tích lũy của các chuỗi liên tiếp của item dương, chỉ chứa chuỗi có độ dài >= 2.

    Parameters:
    - sorted_database: Danh sách các giao dịch, mỗi giao dịch là danh sách các item.
    - priu_list: Một dictionary chứa utility của từng item dương.

    Returns:
    - lius_matrix: Ma trận tam giác chứa utility tích lũy của các chuỗi liên tiếp của item dương có độ dài >= 2.
    """
    # Sắp xếp các item dương theo thứ tự định sẵn
    items = list(priu_list.keys())
    n = len(items)

    # Khởi tạo ma trận tam giác với None
    lius_matrix = [[None for _ in range(n)] for _ in range(n)]

    # Điền utility tích lũy của các chuỗi liên tiếp vào ma trận tam giác
    for transaction in sorted_database:
        for i in range(n):
            cumulative_utility = 0
            for j in range(
                i + 1, n
            ):  # Bắt đầu từ i + 1 để đảm bảo chuỗi có độ dài >= 2
                # Chuỗi liên tiếp từ items[i] đến items[j]
                contiguous_sequence = items[i : j + 1]

                # Tính utility của chuỗi liên tiếp trong giao dịch hiện tại
                utility = calculate_utility_of_itemset(transaction, contiguous_sequence)
                cumulative_utility += utility

                # Nếu ô ma trận chưa có giá trị, khởi tạo với chuỗi liên tiếp và utility ban đầu
                if lius_matrix[i][j] is None:
                    lius_matrix[i][j] = {"sequence": contiguous_sequence, "utility": 0}

                # Cộng dồn vào utility của chuỗi liên tiếp
                lius_matrix[i][j]["utility"] += utility

    return lius_matrix


def priu_strategy(priu_dict, k):
    priu_values = sorted(priu_dict.values(), reverse=True)
    min_util = priu_values[k - 1] if k <= len(priu_values) else priu_values[-1]
    return min_util


def PLIU_E_strategy(lius_matrix, minUtil, k):
    """
    Áp dụng chiến lược PLIU_E để tăng giá trị minUtil dựa trên các giá trị trong ma trận LIUS.

    Parameters:
    - lius_matrix: Ma trận LIUS chứa utility của các chuỗi liên tiếp từ hàm LIUS_Build_contiguous.
    - minUtil: Giá trị minUtil ban đầu.
    - k: Số lượng itemsets có utility cao nhất cần lưu trữ trong hàng đợi ưu tiên PIQULIU.

    Returns:
    - minUtil: Giá trị minUtil được cập nhật.
    """
    # Khởi tạo hàng đợi ưu tiên PIQULIU với kích thước tối đa là k
    PIQULIU = []

    # Duyệt qua ma trận LIUS để thêm các giá trị utility vào PIQULIU
    for i in range(len(lius_matrix)):
        for j in range(i + 1, len(lius_matrix)):
            if lius_matrix[i][j] is not None:  # Kiểm tra nếu ô ma trận có chứa giá trị
                utility = lius_matrix[i][j]["utility"]

                # Chỉ thêm vào PIQULIU nếu utility >= minUtil
                if utility >= minUtil:
                    if len(PIQULIU) < k:
                        # Nếu hàng đợi chưa đạt kích thước k, thêm trực tiếp
                        heapq.heappush(PIQULIU, utility)
                    else:
                        # Nếu hàng đợi đã đạt kích thước k, thay thế giá trị nhỏ nhất nếu utility lớn hơn
                        heapq.heappushpop(PIQULIU, utility)

    # Nếu PIQULIU có đủ k phần tử, cập nhật minUtil thành giá trị nhỏ nhất trong PIQULIU
    if len(PIQULIU) == k:
        # Giá trị nhỏ nhất trong PIQULIU là giá trị thứ k lớn nhất do sử dụng hàng đợi tối thiểu
        new_minUtil = PIQULIU[0]
        if new_minUtil > minUtil:
            minUtil = new_minUtil

    return minUtil, PIQULIU


def PLIU_LB_strategy(lius_matrix, priu_list, piqu_liu, k, minUtil):
    """
    Chiến lược PLIU_LB để tăng giá trị minUtil dựa trên ma trận LIUS.

    Parameters:
    - lius_matrix: Ma trận LIUS chứa utility của các chuỗi liên tiếp.
    - priu_list: Bảng PRIU chứa utility của các item dưới dạng dictionary.
    - piqu_liu: Priority Queue ban đầu chứa k giá trị utility lớn nhất từ LIUS.
    - k: Số lượng itemsets có utility cao nhất cần lưu trữ trong hàng đợi.
    - minUtil: Giá trị minUtil ban đầu.

    Returns:
    - minUtil: Giá trị minUtil được cập nhật.
    """
    # Tạo một hàng đợi ưu tiên mới để lưu các giá trị lower bound utility
    piqu_lb_liu = []

    # Duyệt qua tất cả các cặp trong LIUS
    for i in range(len(lius_matrix)):
        for j in range(i + 1, len(lius_matrix)):
            if lius_matrix[i][j] is not None:
                util_base = lius_matrix[i][j]["utility"]

                # Các item từ đầu đến cuối của chuỗi
                pos_start_item = i + 1
                pos_end_item = j - 1

                # Duyệt qua các item giữa pos_start_item và pos_end_item
                for x in range(pos_start_item, pos_end_item + 1):
                    # Truy cập utility của item x qua tên của nó
                    item_x = list(priu_list.keys())[x]
                    util_lb = util_base - priu_list[item_x]

                    # Nếu utility lớn hơn minUtil, thêm vào hàng đợi
                    if util_lb > minUtil:
                        heapq.heappush(piqu_lb_liu, util_lb)

                    # Tiếp tục loại bỏ thêm item y
                    for y in range(x + 1, pos_end_item + 1):
                        item_y = list(priu_list.keys())[y]
                        util_lb = util_base - priu_list[item_x] - priu_list[item_y]

                        if util_lb > minUtil:
                            heapq.heappush(piqu_lb_liu, util_lb)

                        # Tiếp tục loại bỏ thêm item z
                        for z in range(y + 1, pos_end_item + 1):
                            item_z = list(priu_list.keys())[z]
                            util_lb = (
                                util_base
                                - priu_list[item_x]
                                - priu_list[item_y]
                                - priu_list[item_z]
                            )

                            if util_lb > minUtil:
                                heapq.heappush(piqu_lb_liu, util_lb)

                            # Loại bỏ thêm item w
                            for w in range(z + 1, pos_end_item + 1):
                                item_w = list(priu_list.keys())[w]
                                util_lb = (
                                    util_base
                                    - priu_list[item_x]
                                    - priu_list[item_y]
                                    - priu_list[item_z]
                                    - priu_list[item_w]
                                )

                                if util_lb > minUtil:
                                    heapq.heappush(piqu_lb_liu, util_lb)

    # Tạo PIQU_ALL là hợp của piqu_liu và piqu_lb_liu
    piqu_all = piqu_liu + piqu_lb_liu
    piqu_all = heapq.nlargest(k, piqu_all)

    # Nếu có đủ k giá trị trong piqu_all và giá trị k-th lớn nhất lớn hơn minUtil, cập nhật minUtil
    if len(piqu_all) >= k:
        kth_largest_value = piqu_all[k - 1]
        if kth_largest_value > minUtil:
            minUtil = kth_largest_value

    return minUtil


def calculate_utility(quantity, profit):
    """Calculate utility as quantity * profit."""
    return quantity * profit


def calculate_ptwu(transactions, profit):
    PTWU = {}
    for transaction_id, items in transactions.items():
        for item, quantity in items.items():
            if item not in PTWU:
                PTWU[item] = 0
    for item in PTWU:
        for transaction_id, items in transactions.items():
            if any(i[0] == item for i, quantity in items.items()):
                for i, quantity in items.items():
                    profit = profits.get(i)
                    if profit > 0:
                        PTWU[item] += calculate_utility(quantity, profit)
    return {item: ptwu for item, ptwu in PTWU.items()}


def compute_priu_list(transactions, profits):
    """
    Compute the PRIU values for all positive items x R g and store them in PRIU_list.

    Input:
        transactions: Dictionary of transactions, each containing items and their quantities.
        profits: Dictionary of profits for each item.

    Output:
        priu_list: Dictionary of PRIU values for all positive items x R g.
    """
    priu_dict = calculate_priu(transactions, profits)  # Get PRIU values for all items
    g = {
        item for item, utility in priu_dict.items() if utility < 0
    }  # Set of promising negative items

    # Filter out items in g, keeping only those with positive PRIU values (PRIU >= 0)
    priu_list = {
        item: priu for item, priu in priu_dict.items() if priu >= 0 and item not in g
    }

    return priu_list


def sort_secondary_a_by_total_order(secondary_a, ptwu, profits):
    # Tách thành các nhóm item dương và item âm dựa trên profit
    positive_items = {
        item: ptwu[item] for item in secondary_a if profits.get(item, 0) > 0
    }
    negative_items = {
        item: ptwu[item] for item in secondary_a if profits.get(item, 0) < 0
    }

    # Sắp xếp các item dương theo thứ tự Ptwu tăng dần
    sorted_positive = sorted(positive_items, key=lambda x: positive_items[x])

    # Sắp xếp các item âm theo thứ tự Ptwu tăng dần
    sorted_negative = sorted(negative_items, key=lambda x: negative_items[x])

    # Kết hợp các item dương và âm theo thứ tự tổng quát
    sorted_secondary_a = sorted_positive + sorted_negative
    return sorted_secondary_a


def filter_transactions_by_secondary_a(D, sorted_secondary_a):
    # Chuyển Secondary(a) từ danh sách thành tập hợp để tăng hiệu quả kiểm tra thuộc tính
    secondary_set = set(sorted_secondary_a)

    # Tạo một danh sách giao dịch đã lọc
    filtered_transactions = {}
    for transaction_id, items in D.items():
        # Lọc các item trong transaction hiện tại để chỉ giữ lại các item nằm trong secondary_set
        filtered_items = {
            item: quantity for item, quantity in items.items() if item in secondary_set
        }

        # Chỉ thêm giao dịch vào danh sách kết quả nếu nó không rỗng sau khi lọc
        if filtered_items:
            filtered_transactions[transaction_id] = filtered_items

    return filtered_transactions


def sort_transaction_items_by_total_order(transactions, sorted_secondary_a):
    # Tạo một từ điển thứ tự của các item dựa trên sorted_secondary_a
    item_order = {item: index for index, item in enumerate(sorted_secondary_a)}

    # Sắp xếp các item trong mỗi giao dịch theo thứ tự đã xác định
    sorted_transactions = {}
    for transaction_id, items in transactions.items():
        sorted_items = dict(
            sorted(
                items.items(), key=lambda item: item_order.get(item[0], float("inf"))
            )
        )
        sorted_transactions[transaction_id] = sorted_items

    return sorted_transactions


def transform_to_utilities(transactions, profits):
    """Chuyển cơ sở dữ liệu giao dịch sang dạng tiện ích (profit * quantity) cho từng mục."""
    transformed_db = []

    for transactionID, items in transactions.items():
        transformed_transaction = {}
        for itemID, quantity in items.items():
            # Tính tiện ích cho từng mục item trong giao dịch
            profit = profits.get(itemID, 0)
            utility = profit * quantity
            transformed_transaction[itemID] = utility
        transformed_db.append(transformed_transaction)

    return transformed_db


def calculate_utility_of_itemset(transaction, itemset):
    total_utility = 0

    if itemset == []:
        return 0

    for item in itemset:
        # Kiểm tra nếu item có trong giao dịch và tính utility của nó
        if item in transaction:
            utility = transaction[item]
            total_utility += utility  # Cộng utility của item vào tổng
        else:
            return 0
    return total_utility


def calculate_utility_all(database, itemset):
    utility = 0
    for transaction in database:
        utility += calculate_utility_of_itemset(transaction, itemset)
    return utility


def calculate_rsu(sorted_database, X, target_item):
    """Calculate the RSU value for a set X and a target item in a sorted transaction database."""
    rsu = 0

    # Duyệt qua từng giao dịch trong cơ sở dữ liệu
    for transaction in sorted_database:
        if target_item in transaction:  # Nếu giao dịch chứa mục target_item
            start_counting = False
            item_utility = 0
            rru = 0

            # Duyệt qua tất cả các mục trong giao dịch
            for i in transaction:

                # Tính tiện ích tổng hợp của itemset X trong giao dịch này
                if (all(item in transaction for item in X)) or X == []:
                    # Bắt đầu tính tiện ích sau khi gặp mục target_item
                    if i == target_item:
                        start_counting = True
                        item_utility = transaction[i] + calculate_utility_of_itemset(
                            transaction, X
                        )
                    elif start_counting and transaction[i] > 0:
                        rru += transaction[i]  # Tiện ích còn lại sau mục target_item

            # Cộng tiện ích của RSU
            rsu += item_utility + rru

    return rsu


def calculate_rlu(sorted_database, X, target_item, ptwu):
    rlu = 0
    if X == []:
        return ptwu[target_item]
    for transaction in sorted_database:
        if all(item in transaction for item in X):
            if target_item in transaction:
                start_counting = False
                item_utility = 0
                rru = 0
                utility_of_X = calculate_utility_of_itemset(transaction, X)
                for i in transaction:
                    # Start counting utilities after reaching the target item
                    if i == X[-1]:
                        start_counting = True
                    elif start_counting and transaction[i] > 0:
                        rru += transaction[
                            i
                        ]  # Remaining positive utility after target item
                rlu += (
                    utility_of_X + rru
                )  # Sum of target item utility and remaining utilities
    return rlu


def transaction_projection(transaction, itemset):
    """
    Project the given transaction using the specified itemset.

    :param transaction: A single transaction containing items and their quantities/profits.
    :param itemset: The itemset used for the projection.
    :return: A dictionary representing the projected items and their utilities (quantities/profits), or an empty dictionary if not all items are present.
    """
    projected_transaction = {}
    # Chuyển itemset thành set để tìm kiếm nhanh hơn
    itemset_items = set(itemset)
    # Kiểm tra xem tất cả các item trong itemset có mặt trong giao dịch không
    if itemset_items.issubset(transaction.keys()):
        # Lấy danh sách các item trong giao dịch
        items = list(transaction.keys())
        quantities = list(transaction.values())

        # Tìm vị trí của các item trong itemset và lấy các item sau nó
        last_index = -1
        for i, item in enumerate(items):
            if item in itemset_items:
                last_index = i

        # Nếu tìm thấy các item trong itemset, lấy các item và giá trị sau chúng
        if last_index != -1:
            # Các item và quantities sau itemset
            for i in range(last_index + 1, len(items)):
                projected_transaction[items[i]] = quantities[
                    i
                ]  # Lưu vào projected_transaction

    return projected_transaction


def database_projection(dataset, itemset):
    """
    Project the entire dataset using the specified itemset.

    :param dataset: The dataset containing all transactions.
    :param itemset: The itemset used for projecting the database.
    :return: A list of projected transactions with the same format as input dataset.
    """
    projected_dataset = []

    for transaction in dataset:
        # Gọi hàm transaction_projection cho mỗi giao dịch trong dataset
        projected_transaction = transaction_projection(transaction, itemset)
        # Nếu giao dịch projection không rỗng, thêm vào dataset kết quả
        if projected_transaction:
            projected_dataset.append(projected_transaction)

    return projected_dataset


def merge_transactions(projected_dataset):
    """
    Merge similar transactions by summing their utility values,
    while preserving the original order of transactions.

    :param projected_dataset: List of projected transactions, where each transaction is a dictionary.
    :return: A list of merged transactions in the same format as the input dataset, keeping original order.
    """
    merged_dataset = []

    # Duyệt qua từng giao dịch trong projected_dataset
    for transaction in projected_dataset:
        # Chuyển transaction thành tuple (sorted) để dễ dàng so sánh
        sorted_items = tuple(sorted(transaction.keys()))
        existing_transaction = None

        # Kiểm tra xem có giao dịch nào đã có với cùng itemset không
        for merged_transaction in merged_dataset:
            if tuple(sorted(merged_transaction.keys())) == sorted_items:
                existing_transaction = merged_transaction
                break

        if existing_transaction:
            # Nếu tìm thấy giao dịch tương tự, cộng dồn các giá trị utility
            for item in sorted_items:
                existing_transaction[item] += transaction[item]
        else:
            # Nếu không tìm thấy, thêm giao dịch mới vào merged_dataset
            merged_dataset.append(transaction)

    return merged_dataset


def search(
    alpha,
    prefix_utility,
    eta,
    projected_dataset,
    primary_items,
    secondary_items,
    minUtil,
    ptwu,
    k,
):
    # Duyệt qua từng item z trong primary_items của alpha
    global TKHQ  # Sử dụng biến toàn cục TKHQ
    for z in primary_items:
        # Khởi tạo beta là sự mở rộng của alpha với z
        beta = alpha + [z]
        # Tính utility của beta và tạo dataset beta - D
        beta_utility = calculate_utility_all(projected_dataset, beta)

        # Kiểm tra nếu utility của beta >= minUtil
        if beta_utility >= minUtil:
            # Thêm beta vào hàng đợi ưu tiên TKHQ
            TKHQ.append((beta, beta_utility))
            TKHQ = sorted(
                TKHQ, key=lambda x: x[1], reverse=True
            )  # Sắp xếp TKHQ theo utility giảm dần
            # Nếu TKHQ có k phần tử, nâng minUtil lên utility của phần tử cao nhất
            if len(TKHQ) > k:
                TKHQ.pop()  # Xóa phần tử có utility thấp nhất
            if len(TKHQ) == k:
                minUtil = TKHQ[-1][
                    1
                ]  # minUtil là utility của phần tử nhỏ nhất trong TKHQ
        w = None
        # Tìm item tiếp theo `w` trong `secondary_items` (không dùng tìm kiếm nhị phân)
        zindex = secondary_items.index(z)
        windex = zindex + 1
        if windex > len(secondary_items):
            continue
        # Nếu utility của beta < minUtil và w là item tiêu cực
        if beta_utility < minUtil and w in eta:
            continue  # Bỏ qua và tiếp tục vòng lặp tiếp theo
        remaining_secondary_items = secondary_items[windex::]
        # Xác định primary(beta) và secondary(beta)
        # Tính PSU và PLU của beta và các item còn lại trong secondary_items
        psu_beta_w = {
            w: calculate_rsu(projected_dataset, beta, w)
            for w in remaining_secondary_items
        }
        plu_beta_w = {
            w: calculate_rlu(projected_dataset, beta, w, ptwu)
            for w in remaining_secondary_items
        }
        # Xác định primary(beta) và secondary(beta)
        primary_beta = [
            w for w in remaining_secondary_items if psu_beta_w[w] >= minUtil
        ]
        secondary_beta = [
            w for w in remaining_secondary_items if plu_beta_w[w] >= minUtil
        ]
        # Đệ quy gọi hàm search để mở rộng beta với các item trong primary_items theo DFS
        search(
            beta,
            beta_utility,
            eta,
            projected_dataset,
            primary_beta,
            secondary_beta,
            minUtil,
            ptwu,
            k,
        )


def TKN_algorithm(D, k):
    #    Input: D: A transation dataset, k: the required number of HUIs.
    #    Output: topkHUIs.
    Alpha = []
    minUtil = 0
    # Step 1: Calculate PRIU for each item
    priu_dict = calculate_priu(D, profits)

    # Step 2: Let g be the set of promising negative items in D
    # Define threshold to classify promising negative items
    g = {item for item, utility in priu_dict.items() if utility < 0}

    ptwu = calculate_ptwu(transactions, profits)

    # Bước 5: Tính toán PRIU cho tất cả các items có giá trị PRIU dương và lưu vào PRIU_list
    priu_dict = calculate_priu(D, profits)
    g = {
        item for item, utility in priu_dict.items() if utility < 0
    }  # Tập hợp các item có PRIU âm

    # Bước 6
    priu_list = {
        item: priu for item, priu in priu_dict.items() if priu >= 0 and item not in g
    }

    # Bước 7: Áp dụng chiến lược PRIU để nâng minUtil lên giá trị cao thứ k trong PRIU_list
    minUtil = priu_strategy(priu_list, k)

    # Bước 9: Tính secondary(a) cho các item thỏa mãn điều kiện Ptwu >= minUtil
    secondary_a = {item for item, ptwu_value in ptwu.items() if ptwu_value >= minUtil}

    # Bước 9: Sắp xếp secondary(a) theo thứ tự tổng quát
    sorted_secondary_a = sort_secondary_a_by_total_order(secondary_a, ptwu, profits)

    # Bước 10: Quét D và loại bỏ các item không thuộc Secondary(a)
    filtered_transactions = filter_transactions_by_secondary_a(D, sorted_secondary_a)

    # Step 11: Sort transaction
    sorted_transactions = sort_transaction_items_by_total_order(
        filtered_transactions, sorted_secondary_a
    )

    # Chuyển db nhân cho profit luôn
    db_transform = transform_to_utilities(sorted_transactions, profits)

    # Step 12: Caculate PSU
    PSU_table = {}
    for item in sorted_secondary_a:
        PSU_table[item] = calculate_rsu(db_transform, Alpha, item)

    # Step 13: Build LIUS
    LIUS = LIUS_Build_contiguous(db_transform, priu_list)

    # Step 14:
    PLIU_E, PIQU_LIU = PLIU_E_strategy(LIUS, minUtil, k)

    # Step 15:
    PLIU_LB = PLIU_LB_strategy(LIUS, priu_list, PIQU_LIU, k, PLIU_E)

    # Step 16: Identify Primary items based on PSU
    primary_items = [item for item, psu in PSU_table.items() if psu >= PLIU_LB]

    topkHUIs = search(
        Alpha, 0, g, db_transform, primary_items, sorted_secondary_a, PLIU_LB, ptwu, k
    )
    print("Result:", TKHQ[0:k])


k = 4
result = TKN_algorithm(transactions, k)
