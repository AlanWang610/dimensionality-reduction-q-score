from sklearn.metrics.pairwise import pairwise_distances
def calculate_q_score(data, X_transformed):
    n = data.shape[0]
    # initialize permutations
    Pi = np.identity(n)
    P = np.identity(n)
    # calculate pairwise distances
    Delta = pairwise_distances(data)
    D = pairwise_distances(X_transformed)
    # sort distances
    for j in range(n):
        Pi[j] = np.argsort(Delta[j])
        P[j] = np.argsort(D[j])
        Delta[j] = np.sort(Delta[j])
        D[j] = np.sort(D[j])
    # initialize and populate ranks in low dimension
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            i_coordinate = int(P[i, j])
            R[i_coordinate, j] = i
    # initialize and increment co-ranking matrix
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            i_coordinate = int(Pi[i, j])
            k = int(R[i_coordinate, j])
            Q[i, k] += 1
    # remove first row and column of Q
    Q = Q[1:, 1:]
    # function to calculate Q_nx
    def calculate_Q_nx(Q, k, n):
        return np.sum(Q[:k, :k]) / (k * n)
    # calculate Q_avg
    sum = 0
    for k in range(1, n):
        sum += calculate_Q_nx(Q, k, n)
    Q_avg = sum / (n - 1)
    return Q_avg