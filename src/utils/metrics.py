def reciprocal_rank(output_relevances):
    for i, x in enumerate(output_relevances):
        if x == 1:
            break
    else:
        return 0
    return 1 / (i + 1)


def relevant_in_k(output_relevances, k):
    return max(output_relevances[:k])
