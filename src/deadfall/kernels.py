"""Numba kernels: exact evaluation of tree ensembles in cell space."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _leaf(row, root, feat, tcell, left, right, default_left):
    n = root
    while feat[n] >= 0:
        c = row[feat[n]]
        if c < 0:
            n = left[n] if default_left[n] else right[n]
        elif c < tcell[n]:
            n = left[n]
        else:
            n = right[n]
    return n


@njit(cache=True)
def predict_cells(Xc, tree_offsets, feat, tcell, left, right, default_left, value, base):
    n = Xc.shape[0]
    out = np.full(n, base)
    for i in range(n):
        row = Xc[i]
        s = 0.0
        for t in range(len(tree_offsets) - 1):
            s += value[_leaf(row, tree_offsets[t], feat, tcell, left, right, default_left)]
        out[i] += s
    return out


@njit(cache=True)
def sweep(Xc, tree_offsets, feat, tcell, left, right, default_left, value, base,
          cell_offsets, tree_uses, margin_floor):
    """For every (feature, cell): force that feature into that cell for every
    reference row and record what the ensemble does.

    Returns (n_to_pos, n_to_neg, sum_margin, n_pos_strong, n_neg_strong, margins)
    where n_to_pos[fc] counts rows originally class 0 that become class 1, and
    *_strong counts rows pushed beyond +-margin_floor.
    """
    n, d = Xc.shape
    T = len(tree_offsets) - 1
    total = cell_offsets[d]
    n_to_pos = np.zeros(total, dtype=np.int64)
    n_to_neg = np.zeros(total, dtype=np.int64)
    n_pos_strong = np.zeros(total, dtype=np.int64)
    n_neg_strong = np.zeros(total, dtype=np.int64)
    sum_margin = np.zeros(total)
    margins = np.empty(n)
    lv = np.empty(T)
    tl = np.empty(T, dtype=np.int64)
    row = np.empty(d, dtype=np.int32)
    for i in range(n):
        for j in range(d):
            row[j] = Xc[i, j]
        m0 = base
        for t in range(T):
            lv[t] = value[_leaf(row, tree_offsets[t], feat, tcell, left, right, default_left)]
            m0 += lv[t]
        margins[i] = m0
        for j in range(d):
            rest = m0
            nt = 0
            for t in range(T):
                if tree_uses[t, j]:
                    rest -= lv[t]
                    tl[nt] = t
                    nt += 1
            orig = row[j]
            for c in range(cell_offsets[j + 1] - cell_offsets[j]):
                m = rest
                row[j] = c
                for k in range(nt):
                    m += value[_leaf(row, tree_offsets[tl[k]], feat, tcell, left, right,
                                     default_left)]
                fc = cell_offsets[j] + c
                sum_margin[fc] += m
                if m > 0:
                    if m0 <= 0:
                        n_to_pos[fc] += 1
                    if m > margin_floor:
                        n_pos_strong[fc] += 1
                else:
                    if m0 > 0:
                        n_to_neg[fc] += 1
                    if m < -margin_floor:
                        n_neg_strong[fc] += 1
            row[j] = orig
    return n_to_pos, n_to_neg, sum_margin, n_pos_strong, n_neg_strong, margins


@njit(cache=True)
def tree_feature_usage(tree_offsets, feat, n_features):
    T = len(tree_offsets) - 1
    uses = np.zeros((T, n_features), dtype=np.bool_)
    for t in range(T):
        for n in range(tree_offsets[t], tree_offsets[t + 1]):
            if feat[n] >= 0:
                uses[t, feat[n]] = True
    return uses


@njit(cache=True)
def node_parents(tree_offsets, feat, left, right):
    n = len(feat)
    parent = np.full(n, -1, dtype=np.int32)
    is_left = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if feat[i] >= 0:
            parent[left[i]] = i
            is_left[left[i]] = True
            parent[right[i]] = i
    return parent, is_left


@njit(cache=True)
def split_bounds(feat, tcell, parent, is_left, n_cells):
    """For each internal node: the cell range [lo, hi) of its own split feature
    that is still reachable given same-feature splits among its ancestors."""
    n = len(feat)
    lo = np.zeros(n, dtype=np.int32)
    hi = np.zeros(n, dtype=np.int32)
    for i in range(n):
        j = feat[i]
        if j < 0:
            continue
        l, h = 0, n_cells[j]
        a = i
        while parent[a] >= 0:
            p = parent[a]
            if feat[p] == j:
                if is_left[a]:
                    if tcell[p] < h:
                        h = tcell[p]
                elif tcell[p] > l:
                    l = tcell[p]
            a = p
        lo[i] = l
        hi[i] = h
    return lo, hi


@njit(cache=True)
def climb(Xc, direction, target, max_steps, seed, tree_offsets, feat, tcell, left, right,
          default_left, value, base, n_cells):
    """Coordinate descent on rows until direction * margin < -target: produces
    reference rows that the model confidently assigns to the non-target class."""
    np.random.seed(seed)
    n, d = Xc.shape
    T = len(tree_offsets) - 1
    for i in range(n):
        row = Xc[i]
        for _ in range(max_steps):
            m = base
            for t in range(T):
                m += value[_leaf(row, tree_offsets[t], feat, tcell, left, right, default_left)]
            if direction * m < -target:
                break
            j = np.random.randint(d)
            best_c, best_m = row[j], direction * m
            for c in range(n_cells[j]):
                row[j] = c
                mm = base
                for t in range(T):
                    mm += value[_leaf(row, tree_offsets[t], feat, tcell, left, right,
                                      default_left)]
                if direction * mm < best_m:
                    best_m = direction * mm
                    best_c = c
            row[j] = best_c
    return Xc


@njit(cache=True)
def leaf_boxes(tree_offsets, feat, tcell, left, right, value, cover, parent, is_left,
               n_cells, boff, cmass, cell_offsets, mean_tab, direction, narrow, max_k):
    """For every leaf voting for ``direction``: the up-to-``max_k`` narrowest
    constraints on its root-to-leaf path (constraints whose box holds less than
    ``narrow`` of the feature's training mass). These are candidate triggers
    read directly out of the model structure."""
    n = len(feat)
    T = len(tree_offsets) - 1
    out_f = np.full((n, max_k), -1, dtype=np.int32)
    out_c = np.zeros((n, max_k), dtype=np.int32)
    keep = np.zeros(n, dtype=np.bool_)
    weight = np.zeros(n)
    pf = np.empty(256, dtype=np.int32)
    plo = np.empty(256, dtype=np.int32)
    phi = np.empty(256, dtype=np.int32)
    pm = np.empty(256)
    for t in range(T):
        a, b = tree_offsets[t], tree_offsets[t + 1]
        num = 0.0
        den = 0.0
        for i in range(a, b):
            if feat[i] < 0:
                num += value[i] * cover[i]
                den += cover[i]
        neutral = num / den if den > 0 else 0.0
        for i in range(a, b):
            if feat[i] >= 0 or direction * (value[i] - neutral) <= 0:
                continue
            k = 0
            x = i
            while parent[x] >= 0:
                p = parent[x]
                j = feat[p]
                q = -1
                for z in range(k):
                    if pf[z] == j:
                        q = z
                        break
                if q < 0 and k < 256:
                    q = k
                    pf[q] = j
                    plo[q] = 0
                    phi[q] = n_cells[j]
                    k += 1
                if q >= 0:
                    if is_left[x]:
                        if tcell[p] < phi[q]:
                            phi[q] = tcell[p]
                    elif tcell[p] > plo[q]:
                        plo[q] = tcell[p]
                x = p
            reachable = True
            for z in range(k):
                pm[z] = cmass[boff[pf[z]] + phi[z]] - cmass[boff[pf[z]] + plo[z]]
                if phi[z] <= plo[z]:
                    reachable = False   # empty box: no input can reach this leaf
            if not reachable:
                continue
            got = 0
            for _ in range(max_k):
                best = -1
                for z in range(k):
                    if pm[z] <= narrow and (best < 0 or pm[z] < pm[best]):
                        best = z
                if best < 0:
                    break
                j = pf[best]
                bc = plo[best]
                for c in range(plo[best], phi[best]):
                    if mean_tab[cell_offsets[j] + c] > mean_tab[cell_offsets[j] + bc]:
                        bc = c
                out_f[i, got] = j
                out_c[i, got] = bc
                got += 1
                pm[best] = 2.0
            keep[i] = got > 0
            weight[i] = direction * (value[i] - neutral)
    return out_f, out_c, keep, weight


@njit(cache=True)
def eval_seeds(V, seed_f, seed_c, tree_offsets, feat, tcell, left, right, default_left,
               value, base):
    """Stamp each seed on V: number of rows with positive margin, and margin sum."""
    S, K = seed_f.shape
    n, d = V.shape
    T = len(tree_offsets) - 1
    out = np.zeros(S, dtype=np.int64)
    tot = np.zeros(S)
    row = np.empty(d, dtype=np.int32)
    for s in range(S):
        for i in range(n):
            for j in range(d):
                row[j] = V[i, j]
            for k in range(K):
                if seed_f[s, k] >= 0:
                    row[seed_f[s, k]] = seed_c[s, k]
            m = base
            for t in range(T):
                m += value[_leaf(row, tree_offsets[t], feat, tcell, left, right, default_left)]
            tot[s] += m
            if m > 0:
                out[s] += 1
    return out, tot


@njit(cache=True)
def feature_curve(V, j, n_cells_j, tree_offsets, feat, tcell, left, right, default_left,
                  value, base):
    """Number of rows of V with positive margin for every cell of feature j."""
    n, d = V.shape
    T = len(tree_offsets) - 1
    out = np.zeros(n_cells_j, dtype=np.int64)
    row = np.empty(d, dtype=np.int32)
    for i in range(n):
        for q in range(d):
            row[q] = V[i, q]
        for c in range(n_cells_j):
            row[j] = c
            m = base
            for t in range(T):
                m += value[_leaf(row, tree_offsets[t], feat, tcell, left, right, default_left)]
            if m > 0:
                out[c] += 1
    return out


@njit(cache=True)
def tree_deltas(V, W, tree_offsets, feat, tcell, left, right, default_left, value):
    """Mean per-tree change in output between row sets V (before) and W (after)."""
    T = len(tree_offsets) - 1
    out = np.zeros(T)
    for i in range(V.shape[0]):
        for t in range(T):
            out[t] += (value[_leaf(W[i], tree_offsets[t], feat, tcell, left, right, default_left)]
                       - value[_leaf(V[i], tree_offsets[t], feat, tcell, left, right,
                                     default_left)])
    return out / max(V.shape[0], 1)


@njit(cache=True)
def leaf_ids(Xc, tree_offsets, feat, tcell, left, right, default_left):
    """Node index of the leaf each row reaches in each tree."""
    n = Xc.shape[0]
    T = len(tree_offsets) - 1
    out = np.empty((n, T), dtype=np.int32)
    for i in range(n):
        for t in range(T):
            out[i, t] = _leaf(Xc[i], tree_offsets[t], feat, tcell, left, right, default_left)
    return out


@njit(cache=True)
def leaf_box_logmass(feat, tcell, parent, is_left, n_cells, boff, cmass):
    """Per leaf: log10 of the product of marginal masses of its path constraints,
    i.e. the share of training data the leaf would hold if features were
    independent."""
    n = len(feat)
    out = np.zeros(n)
    pf = np.empty(256, dtype=np.int32)
    plo = np.empty(256, dtype=np.int32)
    phi = np.empty(256, dtype=np.int32)
    for i in range(n):
        if feat[i] >= 0:
            continue
        k = 0
        x = i
        while parent[x] >= 0:
            p = parent[x]
            j = feat[p]
            q = -1
            for z in range(k):
                if pf[z] == j:
                    q = z
                    break
            if q < 0 and k < 256:
                q = k
                pf[q] = j
                plo[q] = 0
                phi[q] = n_cells[j]
                k += 1
            if q >= 0:
                if is_left[x]:
                    if tcell[p] < phi[q]:
                        phi[q] = tcell[p]
                elif tcell[p] > plo[q]:
                    plo[q] = tcell[p]
            x = p
        s = 0.0
        for z in range(k):
            m = cmass[boff[pf[z]] + phi[z]] - cmass[boff[pf[z]] + plo[z]]
            s += np.log10(max(m, 1e-9))
        out[i] = s
    return out


@njit(cache=True)
def pav(y, w):
    """Weighted isotonic regression (pool adjacent violators), non-decreasing."""
    n = len(y)
    val = np.empty(n)
    wt = np.empty(n)
    size = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(n):
        val[k] = y[i]
        wt[k] = w[i]
        size[k] = 1
        k += 1
        while k > 1 and val[k - 2] > val[k - 1]:
            tw = wt[k - 2] + wt[k - 1]
            val[k - 2] = (val[k - 2] * wt[k - 2] + val[k - 1] * wt[k - 1]) / tw
            wt[k - 2] = tw
            size[k - 2] += size[k - 1]
            k -= 1
    out = np.empty(n)
    pos = 0
    for b in range(k):
        for _ in range(size[b]):
            out[pos] = val[b]
            pos += 1
    return out
