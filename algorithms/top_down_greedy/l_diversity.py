# Count number of distict sensitive attribute values in a partition
# partition: A top-down greedy partition object
# sensitive_index: int, index of the sensitive attribute in the reordered record
# returns: int, number of distinct sensitive values in the partition

def distinct_sensitive_values(partition, sensitive_index):
    return len({
        record[sensitive_index] for record in partition.member
    })


# Check if a partition satisfies l-diversity
# partition is l-diverse if it contains at least l distinct values of the sensitive attribute
# partition: A top-down greedy partition object
# l: int, the l value for l-diversity
# sensitive_index: int, index of the sensitive attribute in the reordered record
# returns: bool, True if the partition is l-diverse, False otherwise

def is_l_diverse(partition, l, sensitive_index):
    if l < 1:
        raise ValueError("l must be at least 1")

    return distinct_sensitive_values(
        partition,
        sensitive_index
    ) >= l