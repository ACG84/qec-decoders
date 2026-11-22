"""Numba-compiled versions of the two graph decoders, with shots decoded in parallel threads.

Same algorithms and the same tie-breaking as `mwpm.py` and `unionfind.py`, so the outputs are
identical shot for shot (checked in the tests). The pure-Python modules stay as the readable
reference. Each shot is independent, so the batch loop is a `prange`.
"""
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads


# ---------------------------------------------------------------- exact matching (subset DP)

@njit(cache=True)
def _find(comp, x):
    while comp[x] != x:
        comp[x] = comp[comp[x]]
        x = comp[x]
    return x


@njit(cache=True)
def _greedy_group(idx, m, D, P, B, BP):
    """Greedy matching of one group (the fallback for groups above max_defects)."""
    npairs = m * (m - 1) // 2
    cost = np.empty(npairs); pi = np.empty(npairs, np.int64); pj = np.empty(npairs, np.int64)
    po = np.empty(npairs, np.uint8)
    c = 0
    for a in range(m):
        i = idx[a]
        for b in range(a + 1, m):
            j = idx[b]
            direct, via = D[i, j], B[i] + B[j]
            if direct <= via:
                cost[c] = direct; po[c] = P[i, j]
            else:
                cost[c] = via; po[c] = BP[i] ^ BP[j]
            pi[c] = a; pj[c] = b; c += 1
    order = np.argsort(cost, kind="mergesort")           # stable, like Python's sort
    free = np.ones(m, np.bool_)
    obs = 0
    for t in order:
        if free[pi[t]] and free[pj[t]]:
            free[pi[t]] = False; free[pj[t]] = False; obs ^= po[t]
    for a in range(m):
        if free[a]:
            obs ^= BP[idx[a]]
    return obs


@njit(cache=True)
def _dp_subset(idx, m, D, P, B, BP, mate):
    """Exact matching of the m defects idx[:m] among themselves and the boundary.
    Writes partners into mate (defect index, or -1 for the boundary)."""
    full = (1 << m) - 1
    cost = np.zeros(full + 1); choice = np.full(full + 1, -1, np.int64)
    for mask in range(1, full + 1):
        i = 0
        while not (mask >> i) & 1:
            i += 1
        rest = mask & ~(1 << i)
        best = B[idx[i]] + cost[rest]; bc = -1
        j = i + 1
        while rest >> j:
            if (rest >> j) & 1:
                c = D[idx[i], idx[j]] + cost[rest & ~(1 << j)]
                if c < best:
                    best = c; bc = j
            j += 1
        cost[mask] = best; choice[mask] = bc
    mask = full
    while mask:
        i = 0
        while not (mask >> i) & 1:
            i += 1
        j = choice[mask]
        if j < 0:
            mate[idx[i]] = -1; mask &= ~(1 << i)
        else:
            mate[idx[i]] = idx[j]; mate[idx[j]] = idx[i]; mask &= ~(1 << i); mask &= ~(1 << j)
    return cost[full]


@njit(cache=True)
def _matching_cost(idx, m, D, B, mate):
    c = 0.0
    for a in range(m):
        i = idx[a]
        if mate[i] < 0:
            c += B[i]
        elif mate[i] > i:
            c += D[i, mate[i]]
    return c


@njit(cache=True)
def _lns_group(idx, m, D, P, B, BP, window):
    """Large groups: start from greedy, then repeatedly re-solve exactly the `window` defects
    around each matched unit (whole units only) until a full pass brings no improvement."""
    k = D.shape[0]
    mate = np.full(k, -2)
    # greedy start (same order as _greedy_group)
    npairs = m * (m - 1) // 2
    cost = np.empty(npairs); pi = np.empty(npairs, np.int64); pj = np.empty(npairs, np.int64)
    direct_ok = np.empty(npairs, np.bool_)
    c = 0
    for a in range(m):
        for b in range(a + 1, m):
            i = idx[a]; j = idx[b]
            direct, via = D[i, j], B[i] + B[j]
            direct_ok[c] = direct <= via
            cost[c] = direct if direct <= via else via
            pi[c] = i; pj[c] = j; c += 1
    for t in np.argsort(cost, kind="mergesort"):
        i = pi[t]; j = pj[t]
        if mate[i] == -2 and mate[j] == -2:
            if direct_ok[t]:
                mate[i] = j; mate[j] = i
            else:
                mate[i] = -1; mate[j] = -1
    for a in range(m):
        if mate[idx[a]] == -2:
            mate[idx[a]] = -1
    sub = np.empty(window + 1, np.int64)
    near = np.empty(m, np.int64); nd = np.empty(m)
    for _ in range(6):
        improved = False
        for a in range(m):
            seed = idx[a]
            for b in range(m):
                nd[b] = D[seed, idx[b]] if idx[b] != seed else -1.0
                near[b] = idx[b]
            order = np.argsort(nd, kind="mergesort")
            used = np.zeros(k, np.bool_)
            w = 0
            for t in order:
                i = near[t]
                if used[i]:
                    continue
                need = 1 if mate[i] < 0 else 2
                if w + need > window:
                    continue
                sub[w] = i; used[i] = True; w += 1
                if mate[i] >= 0:
                    sub[w] = mate[i]; used[mate[i]] = True; w += 1
                if w >= window - 1:
                    break
            before = _matching_cost(sub, w, D, B, mate)
            old = np.empty(w, np.int64)
            for t in range(w):
                old[t] = mate[sub[t]]
            ssub = np.sort(sub[:w])
            after = _dp_subset(ssub, w, D, P, B, BP, mate)
            if after < before - 1e-9:
                improved = True
            else:
                for t in range(w):
                    mate[sub[t]] = old[t]
        if not improved:
            break
    obs = 0
    for a in range(m):
        i = idx[a]
        if mate[i] < 0:
            obs ^= BP[i]
        elif mate[i] > i:
            obs ^= P[i, mate[i]]
    return obs


@njit(cache=True)
def _solve_group(idx, m, D, P, B, BP, max_defects, window):
    """Returns (observable parity, 1 if the greedy fallback was used)."""
    if m == 1:
        return BP[idx[0]], 0
    if m > max_defects:
        if window > 0:
            return _lns_group(idx, m, D, P, B, BP, window), 1
        return _greedy_group(idx, m, D, P, B, BP), 1
    full = (1 << m) - 1
    cost = np.zeros(full + 1); obs = np.zeros(full + 1, np.uint8)
    for mask in range(1, full + 1):
        i = 0
        while not (mask >> i) & 1:
            i += 1
        rest = mask & ~(1 << i)
        best = B[idx[i]] + cost[rest]; bo = BP[idx[i]] ^ obs[rest]
        mm = rest
        j = i + 1
        while mm >> j:
            if (mm >> j) & 1:
                r2 = rest & ~(1 << j)
                c = D[idx[i], idx[j]] + cost[r2]
                if c < best:
                    best = c; bo = P[idx[i], idx[j]] ^ obs[r2]
            j += 1
        cost[mask] = best; obs[mask] = bo
    return obs[full], 0


@njit(cache=True)
def _mwpm_one(syn, n, dist, par, bdist, bpar, max_defects, window):
    k = 0
    for i in range(n):
        if syn[i]:
            k += 1
    if k == 0:
        return 0, 0
    defects = np.empty(k, np.int64)
    k = 0
    for i in range(n):
        if syn[i]:
            defects[k] = i; k += 1
    D = np.empty((k, k)); P = np.empty((k, k), np.uint8); B = np.empty(k); BP = np.empty(k, np.uint8)
    for a in range(k):
        B[a] = bdist[defects[a]]; BP[a] = bpar[defects[a]]
        for b in range(k):
            D[a, b] = dist[defects[a], defects[b]]; P[a, b] = par[defects[a], defects[b]]
    comp = np.arange(k)
    for a in range(k):
        for b in range(a + 1, k):
            if D[a, b] < B[a] + B[b]:
                comp[_find(comp, a)] = _find(comp, b)
    root = np.empty(k, np.int64)
    for a in range(k):
        root[a] = _find(comp, a)
    obs = 0; fb = 0
    done = np.zeros(k, np.bool_)
    idx = np.empty(k, np.int64)
    for a in range(k):
        if done[a]:
            continue
        m = 0
        for b in range(a, k):
            if root[b] == root[a]:
                idx[m] = b; m += 1; done[b] = True
        o, f = _solve_group(idx, m, D, P, B, BP, max_defects, window)
        obs ^= o; fb += f
    return obs, fb


@njit(cache=True, parallel=True)
def _mwpm_batch(S, n, dist, par, bdist, bpar, max_defects, window):
    out = np.zeros(S.shape[0], np.uint8); fb = np.zeros(S.shape[0], np.int64)
    for s in prange(S.shape[0]):
        o, f = _mwpm_one(S[s], n, dist, par, bdist, bpar, max_defects, window)
        out[s] = o; fb[s] = f
    return out, fb


class FastMWPMDecoder:
    """large="lns" (default): groups above max_defects start from greedy and are improved by
    exact re-matching of 12-defect windows until nothing improves. large="greedy" reproduces
    the pure-Python reference exactly."""

    def __init__(self, graph, max_defects=14, large="lns", window=12):
        self.g, self.max_defects = graph, max_defects
        self.window = window if large == "lns" else 0
        self.name = "MWPM (subset DP + window search, numba)" if large == "lns" else "MWPM (exact, subset DP, numba)"
        self.dist, self.par, self.bdist, self.bpar = graph.shortest_paths()
        self.fallbacks = 0

    def decode_batch(self, syndromes):
        S = np.ascontiguousarray(np.atleast_2d(syndromes), dtype=np.uint8)
        out, fb = _mwpm_batch(S, self.g.n, self.dist, self.par, self.bdist, self.bpar, self.max_defects, self.window)
        self.fallbacks += int(fb.sum())
        return out

    def decode(self, syndrome):
        return int(self.decode_batch(np.asarray(syndrome)[None, :])[0])


# ---------------------------------------------------------------- Union-Find

@njit(cache=True)
def _uf_one(syn, n, N, eu, ev, eobs):
    E = eu.shape[0]
    nd = 0
    for i in range(n):
        if syn[i]:
            nd += 1
    if nd == 0:
        return 0
    parent = np.arange(N)
    parity = np.zeros(N, np.uint8); boundary = np.zeros(N, np.bool_); member = np.zeros(N, np.bool_)
    defects = np.empty(nd, np.int64)
    nd = 0
    for i in range(n):
        if syn[i]:
            defects[nd] = i; nd += 1; parity[i] = 1; member[i] = True
    growth = np.zeros(E, np.int8); full = np.zeros(E, np.bool_)
    active = np.zeros(N, np.bool_)
    grew = np.empty(E, np.int64)
    while True:
        active[:] = False
        any_active = False
        for t in range(nd):
            r = _find(parent, defects[t])
            if parity[r] and not boundary[r]:
                active[r] = True; any_active = True
        if not any_active:
            break
        ng = 0
        for e in range(E):
            if full[e]:
                continue
            u = eu[e]; v = ev[e]
            inc = 0
            if member[u] and active[_find(parent, u)]:
                inc += 1
            if member[v] and active[_find(parent, v)]:
                inc += 1
            if inc:
                growth[e] += inc
                if growth[e] >= 2:
                    grew[ng] = e; ng += 1
        for t in range(ng):
            e = grew[t]
            full[e] = True
            u = eu[e]; v = ev[e]
            if not member[u]:
                member[u] = True
                if u >= n:
                    boundary[u] = True
            if not member[v]:
                member[v] = True
                if v >= n:
                    boundary[v] = True
            ru = _find(parent, u); rv = _find(parent, v)
            if ru != rv:
                parent[ru] = rv
                parity[rv] ^= parity[ru]
                boundary[rv] = boundary[rv] or boundary[ru]
    # peeling: adjacency of fully grown edges, per node in increasing edge order
    deg = np.zeros(N + 1, np.int64)
    first = np.full(N, -1)                 # order of first appearance, as the reference's dict
    nseen = 0
    for e in range(E):
        if full[e]:
            for x in (eu[e], ev[e]):
                deg[x + 1] += 1
                if first[x] < 0:
                    first[x] = nseen; nseen += 1
    start = np.cumsum(deg)
    fill = start[:-1].copy()
    nbr = np.empty(start[N], np.int64); nbe = np.empty(start[N], np.int64)
    for e in range(E):
        if full[e]:
            u = eu[e]; v = ev[e]
            nbr[fill[u]] = v; nbe[fill[u]] = e; fill[u] += 1
            nbr[fill[v]] = u; nbe[fill[v]] = e; fill[v] += 1
    nodes = np.empty(nseen, np.int64)
    for x in range(N):
        if first[x] >= 0:
            nodes[first[x]] = x
    roots = np.empty(nseen, np.int64)
    r = 0
    for t in range(nseen):                 # boundary vertices first, each in appearance order
        if nodes[t] >= n:
            roots[r] = nodes[t]; r += 1
    for t in range(nseen):
        if nodes[t] < n:
            roots[r] = nodes[t]; r += 1
    synd = np.zeros(N, np.uint8)
    for t in range(nd):
        synd[defects[t]] = 1
    seen = np.zeros(N, np.bool_)
    order = np.empty(nseen, np.int64); pnode = np.full(N, -1); pedge = np.full(N, -1)
    obs = 0
    for t in range(nseen):
        root = roots[t]
        if seen[root]:
            continue
        head = 0; tail = 0
        order[tail] = root; tail += 1; seen[root] = True
        while head < tail:
            u = order[head]; head += 1
            for a in range(start[u], start[u + 1]):
                v = nbr[a]
                if not seen[v]:
                    seen[v] = True; pnode[v] = u; pedge[v] = nbe[a]
                    order[tail] = v; tail += 1
        for a in range(tail - 1, -1, -1):
            u = order[a]
            if pnode[u] >= 0 and synd[u]:
                obs ^= eobs[pedge[u]]
                synd[u] = 0
                if pnode[u] < n:
                    synd[pnode[u]] ^= 1
    return obs


@njit(cache=True, parallel=True)
def _uf_batch(S, n, N, eu, ev, eobs):
    out = np.zeros(S.shape[0], np.uint8)
    for s in prange(S.shape[0]):
        out[s] = _uf_one(S[s], n, N, eu, ev, eobs)
    return out


class FastUnionFindDecoder:
    name = "Union-Find (numba)"

    def __init__(self, graph):
        self.g = graph
        self.eu = np.ascontiguousarray(graph.eu, dtype=np.int64)
        self.ev = np.ascontiguousarray(graph.ev, dtype=np.int64)
        self.eobs = np.ascontiguousarray(graph.eobs, dtype=np.uint8)

    def decode_batch(self, syndromes):
        S = np.ascontiguousarray(np.atleast_2d(syndromes), dtype=np.uint8)
        return _uf_batch(S, self.g.n, self.g.num_nodes, self.eu, self.ev, self.eobs)

    def decode(self, syndrome):
        return int(self.decode_batch(np.asarray(syndrome)[None, :])[0])


def threads(k=None):
    """Get or set the number of threads the batch decoders use."""
    if k is not None:
        set_num_threads(k)
    return get_num_threads()
