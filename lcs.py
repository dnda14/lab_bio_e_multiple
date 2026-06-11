"""
Alineación de Múltiples Secuencias (MSA)
Implementación con:
  - LCS + clasificación (lógica manual del usuario)
  - Árbol guía UPGMA
  - Fusión progresiva de secuencias
"""

import numpy as np

def leer(filename):
    with open(filename, mode='r') as f:
        lines = f.readlines()
    if not lines: return ""
    if lines[0].startswith('>'):
        return "".join(line.strip() for line in lines[1:])
    else:
        return lines[0].strip()


# ─────────────────────────────────────────────
# 1. LCS + CLASIFICACIÓN (lógica del usuario)
# ─────────────────────────────────────────────

def lcs_matrix(seq1: list, seq2: list) -> np.ndarray:
    """
    Construye la matriz LCS (Longest Common Subsequence).
    Ignora X al comparar (X hace match con todo, costo 0).
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = seq1[i-1].upper(), seq2[j-1].upper()
            if a == b or a == 'X' or b == 'X':
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp


def lcs_traceback(seq1: list, seq2: list, dp: np.ndarray) -> tuple:
    """
    Traceback del LCS.
    Retorna (matched1, matched2) — índices de los caracteres que forman el LCS.
    """
    matched1, matched2 = [], []
    i, j = len(seq1), len(seq2)
    while i > 0 and j > 0:
        a, b = seq1[i-1].upper(), seq2[j-1].upper()
        if a == b or a == 'X' or b == 'X':
            matched1.append(i - 1)
            matched2.append(j - 1)
            i -= 1; j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    matched1.reverse()
    matched2.reverse()
    return matched1, matched2


def align_lcs(seq1: list, seq2: list) -> tuple:
    """
    Alinea dos secuencias usando LCS + clasificación:
      - Posiciones que coinciden en el LCS → match (costo 0)
      - Lo que queda 'al aire':
          * solo seq1 tiene algo  → inserción en seq1 / deleción en seq2  → X en seq2
          * solo seq2 tiene algo  → inserción en seq2 / deleción en seq1  → X en seq1
          * ambas tienen algo diferente → sustitución → X en ambas (costo 0)

    Retorna (distancia, a1_alineada, a2_alineada)
      distancia = número de sustituciones reales (cuando ambas tienen algo distinto)
    """
    dp = lcs_matrix(seq1, seq2)
    matched1, matched2 = lcs_traceback(seq1, seq2, dp)

    # Conjuntos de índices que participan en el LCS
    set1 = set(matched1)
    set2 = set(matched2)

    # Índices que quedaron fuera del LCS
    unmatched1 = [i for i in range(len(seq1)) if i not in set1]
    unmatched2 = [j for j in range(len(seq2)) if j not in set2]

    # Construir la alineación intercalando:
    # segmentos no-LCS entre cada par de posiciones LCS
    a1, a2 = [], []
    substitutions = 0

    prev_m1, prev_m2 = -1, -1

    for k in range(len(matched1)):
        m1, m2 = matched1[k], matched2[k]

        # Segmento no-LCS antes de este match
        gap1 = [i for i in range(prev_m1 + 1, m1)]  # índices en seq1 sin match
        gap2 = [j for j in range(prev_m2 + 1, m2)]  # índices en seq2 sin match

        # Clasificar lo que quedó al aire
        while gap1 or gap2:
            if gap1 and gap2:
                # Ambas tienen algo → sustitución
                a1.append(seq1[gap1.pop(0)])
                a2.append(seq2[gap2.pop(0)])
                substitutions += 1
            elif gap1:
                # Solo seq1 tiene → deleción en seq2 (inserción en seq1)
                a1.append(seq1[gap1.pop(0)])
                a2.append('X')
            else:
                # Solo seq2 tiene → deleción en seq1 (inserción en seq2)
                a1.append('X')
                a2.append(seq2[gap2.pop(0)])

        # Agregar el match
        a1.append(seq1[m1])
        a2.append(seq2[m2])
        prev_m1, prev_m2 = m1, m2

    # Segmento final después del último match
    gap1 = list(range(prev_m1 + 1, len(seq1)))
    gap2 = list(range(prev_m2 + 1, len(seq2)))
    while gap1 or gap2:
        if gap1 and gap2:
            a1.append(seq1[gap1.pop(0)])
            a2.append(seq2[gap2.pop(0)])
            substitutions += 1
        elif gap1:
            a1.append(seq1[gap1.pop(0)])
            a2.append('X')
        else:
            a1.append('X')
            a2.append(seq2[gap2.pop(0)])

    # Distancia = solo sustituciones reales (indels no cuentan)
    distance = substitutions
    return distance, a1, a2


# ─────────────────────────────────────────────
# 2. DISTANCIA ENTRE SECUENCIAS / GRUPOS
# ─────────────────────────────────────────────

def seq_distance(s1: list, s2: list) -> float:
    """Distancia LCS entre dos secuencias (solo sustituciones reales)."""
    score, _, _ = align_lcs(s1, s2)
    return score


def group_distance(g1: list, g2: list) -> float:
    """Distancia UPGMA entre dos grupos."""
    total, count = 0, 0
    for s1 in g1:
        for s2 in g2:
            total += seq_distance(s1, s2)
            count += 1
    return total / count if count else 0


# ─────────────────────────────────────────────
# 3. CONSTRUCCIÓN DEL ÁRBOL GUÍA (UPGMA)
# ─────────────────────────────────────────────

def build_guide_tree(sequences: list, names: list) -> list:
    clusters      = [[seq] for seq in sequences]
    cluster_names = list(names)
    merge_steps   = []

    print("\n" + "="*60)
    print("  CONSTRUCCIÓN DEL ÁRBOL GUÍA (UPGMA + LCS)")
    print("="*60)

    step = 0
    while len(clusters) > 1:
        step += 1
        k = len(clusters)

        dist_matrix = np.full((k, k), np.inf)
        for i in range(k):
            for j in range(i+1, k):
                d = group_distance(clusters[i], clusters[j])
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d

        print(f"\n  Paso {step} — Matriz de distancias:")
        header = "         " + "".join(f"{cluster_names[j]:>10}" for j in range(k))
        print(header)
        for i in range(k):
            row = f"  {cluster_names[i]:>7}"
            for j in range(k):
                if i == j:
                    row += f"{'---':>10}"
                else:
                    row += f"{dist_matrix[i][j]:>10.3f}"
            print(row)

        # Par con menor distancia
        min_d = np.inf
        best_i, best_j = 0, 1
        for i in range(k):
            for j in range(i+1, k):
                if dist_matrix[i][j] < min_d:
                    min_d = dist_matrix[i][j]
                    best_i, best_j = i, j

        new_name = f"({cluster_names[best_i]},{cluster_names[best_j]})"
        print(f"\n  → Fusionando: {cluster_names[best_i]} + {cluster_names[best_j]}  "
              f"[distancia = {min_d:.3f}]  →  {new_name}")

        merge_steps.append({
            'name':       new_name,
            'dist':       min_d,
            'left_name':  cluster_names[best_i],
            'right_name': cluster_names[best_j],
        })

        new_cluster  = clusters[best_i] + clusters[best_j]
        new_clusters = [clusters[i] for i in range(k) if i not in (best_i, best_j)]
        new_names    = [cluster_names[i] for i in range(k) if i not in (best_i, best_j)]
        new_clusters.append(new_cluster)
        new_names.append(new_name)

        clusters      = new_clusters
        cluster_names = new_names

    return merge_steps


# ─────────────────────────────────────────────
# 4. ALINEACIÓN PROGRESIVA
# ─────────────────────────────────────────────

def align_two_groups(group1: list, group2: list) -> tuple:
    """
    Alinea dos grupos usando LCS.
    Representante de cada grupo = primera secuencia (X actúa como comodín).
    Propaga los mismos gaps a todas las secuencias del grupo.
    """
    rep1 = list(group1[0])
    rep2 = list(group2[0])

    _, a1, a2 = align_lcs(rep1, rep2)

    gaps1 = _gap_positions(rep1, a1)
    gaps2 = _gap_positions(rep2, a2)

    new_group1 = [_insert_gaps(seq, gaps1) for seq in group1]
    new_group2 = [_insert_gaps(seq, gaps2) for seq in group2]

    return new_group1, new_group2


def _gap_positions(original: list, aligned: list) -> list:
    """Posiciones donde se insertaron X/gaps nuevos en el alineado."""
    positions = []
    orig_idx = 0
    for pos, ch in enumerate(aligned):
        if orig_idx >= len(original) or ch != original[orig_idx]:
            positions.append(pos)
        else:
            orig_idx += 1
    return positions


def _insert_gaps(seq: list, gap_positions: list) -> list:
    result = list(seq)
    for pos in gap_positions:
        result.insert(pos, 'X')
    return result


def progressive_alignment(sequences: list, names: list, merge_steps: list) -> list:
    print("\n" + "="*60)
    print("  ALINEACIÓN PROGRESIVA")
    print("="*60)

    groups = {name: [list(seq)] for name, seq in zip(names, sequences)}

    for step in merge_steps:
        left  = step['left_name']
        right = step['right_name']
        new   = step['name']

        g1 = groups[left]
        g2 = groups[right]

        print(f"\n  Fusionando: {left}  +  {right}  →  {new}")
        g1_aligned, g2_aligned = align_two_groups(g1, g2)

        max_len = max(len(s) for s in g1_aligned + g2_aligned)
        for s in g1_aligned + g2_aligned:
            while len(s) < max_len:
                s.append('X')

        groups[new] = g1_aligned + g2_aligned

        print(f"  Resultado ({len(groups[new])} secuencias, longitud {max_len}):")
        merged_names = (_get_leaf_names(left,  merge_steps, names) +
                        _get_leaf_names(right, merge_steps, names))
        for mname, seq in zip(merged_names, groups[new]):
            print(f"    {mname:>6}: {''.join(seq)}")

    return groups[merge_steps[-1]['name']]


def _get_leaf_names(node: str, merge_steps: list, original_names: list) -> list:
    if node in original_names:
        return [node]
    for step in merge_steps:
        if step['name'] == node:
            left  = _get_leaf_names(step['left_name'],  merge_steps, original_names)
            right = _get_leaf_names(step['right_name'], merge_steps, original_names)
            return left + right
    return [node]


# ─────────────────────────────────────────────
# 5. PIPELINE COMPLETO
# ─────────────────────────────────────────────

def multiple_sequence_alignment(sequences: list, names: list = None) -> list:
    if names is None:
        names = [f"S{i+1}" for i in range(len(sequences))]

    seq_lists = [list(s.upper()) for s in sequences]

    print("\n" + "="*60)
    print("  SECUENCIAS DE ENTRADA")
    print("="*60)
    for name, seq in zip(names, sequences):
        print(f"  {name:>6}: {seq.upper()}")

    merge_steps    = build_guide_tree(seq_lists, names)
    aligned_groups = progressive_alignment(seq_lists, names, merge_steps)

    leaf_order      = _get_leaf_names(merge_steps[-1]['name'], merge_steps, names)
    name_to_aligned = {n: s for n, s in zip(leaf_order, aligned_groups)}

    max_len = max(len(s) for s in aligned_groups)

    print("\n" + "="*60)
    print("  ALINEACIÓN FINAL (con X = indel/hueco)")
    print("="*60)
    for name in names:
        seq = name_to_aligned.get(name, [])
        while len(seq) < max_len:
            seq.append('X')
        print(f"  {name:>6}: {''.join(seq)}")

    print("\n" + "="*60)
    print("  ALINEACIÓN FINAL (X reemplazados por -)")
    print("="*60)
    clean = []
    for name in names:
        seq = name_to_aligned.get(name, [])
        while len(seq) < max_len:
            seq.append('X')
        cleaned = ''.join(c if c != 'X' else '-' for c in seq)
        clean.append(cleaned)
        print(f"  {name:>6}: {cleaned}")

    return clean


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    OUTPUT_FILE = "msa_resultados2.txt"

    seqs  = ["ATTGGCACCA", "ATTTGGACCA", "TGGTTCCA", "ATTCCACCAC"]
    names = ["S1", "S2", "S3", "S4"]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        sys.stdout = f

        print("#"*60)
        print("  MSA con LCS + clasificación (indel vs sustitución)")
        print("#"*60)
        result = multiple_sequence_alignment(seqs, names)

    sys.stdout = sys.__stdout__
    print(f"✓ Resultados guardados en: {OUTPUT_FILE}")