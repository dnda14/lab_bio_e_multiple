"""
Alineación de Múltiples Secuencias (MSA)
Implementación con:
  - Needleman-Wunsch modificado (soporte para X)
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
# 1. NEEDLEMAN-WUNSCH MODIFICADO (con X)
# ─────────────────────────────────────────────

def cost(a: str, b: str) -> int:
    """
    Modelo de costo unitario con soporte para X (gap profile):
      s(a, a) = 0
      s(a, b) = 1  (a != b, ninguno es X)
      s(X, ·) = 0  (X no añade penalización)
    """
    a, b = a.upper(), b.upper()
    if a == 'X' or b == 'X':
        return 0
    return 0 if a == b else 1


def needleman_wunsch(seq1: list, seq2: list, gap_penalty: int = 1) -> tuple:
    """
    Needleman-Wunsch que acepta listas de caracteres (soporta 'X').
    Retorna (score_distancia, seq1_alineada, seq2_alineada).
    """
    n, m = len(seq1), len(seq2)

    # Matriz de programación dinámica
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[0, :] = [j * gap_penalty for j in range(m + 1)]
    dp[:, 0] = [i * gap_penalty for i in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match   = dp[i-1][j-1] + cost(seq1[i-1], seq2[j-1])
            delete  = dp[i-1][j]   + gap_penalty
            insert  = dp[i][j-1]   + gap_penalty
            dp[i][j] = min(match, delete, insert)

    # Traceback
    a1, a2 = [], []
    i, j = n, m
    while i > 0 and j > 0:
        sc = dp[i][j]
        if sc == dp[i-1][j-1] + cost(seq1[i-1], seq2[j-1]):
            a1.append(seq1[i-1])
            a2.append(seq2[j-1])
            i -= 1; j -= 1
        elif sc == dp[i-1][j] + gap_penalty:
            a1.append(seq1[i-1])
            a2.append('-')
            i -= 1
        else:
            a1.append('-')
            a2.append(seq2[j-1])
            j -= 1
    while i > 0:
        a1.append(seq1[i-1]); a2.append('-'); i -= 1
    while j > 0:
        a1.append('-'); a2.append(seq2[j-1]); j -= 1

    a1.reverse(); a2.reverse()
    return dp[n][m], a1, a2


# ─────────────────────────────────────────────
# 2. DISTANCIA ENTRE SECUENCIAS / GRUPOS
# ─────────────────────────────────────────────

def seq_distance(s1: list, s2: list) -> float:
    """Distancia NW normalizada entre dos secuencias."""
    score, _, _ = needleman_wunsch(s1, s2)
    return score


def group_distance_to_seq(group: list[list], seq: list) -> float:
    """
    Distancia de una secuencia a un grupo (media de distancias individuales).
    Usa NW modificado con soporte X.
    """
    dists = [seq_distance(member, seq) for member in group]
    return sum(dists) / len(dists)


def group_distance(g1: list[list], g2: list[list]) -> float:
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

def build_guide_tree(sequences: list[list], names: list[str]) -> list:
    """
    Construye el árbol guía usando UPGMA con NW.
    Retorna lista de pasos de fusión: [(nombre_grupo, dist, idx_i, idx_j), ...]
    """
    n = len(sequences)

    # Cada cluster inicial contiene una sola secuencia
    clusters     = [[seq] for seq in sequences]
    cluster_names = list(names)
    merge_steps   = []

    print("\n" + "="*60)
    print("  CONSTRUCCIÓN DEL ÁRBOL GUÍA (UPGMA + NW)")
    print("="*60)

    step = 0
    while len(clusters) > 1:
        step += 1
        k = len(clusters)

        # Calcular matriz de distancias
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

        # Encontrar el par con menor distancia
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
            'name':  new_name,
            'dist':  min_d,
            'left':  best_i,
            'right': best_j,
            'left_name':  cluster_names[best_i],
            'right_name': cluster_names[best_j],
        })

        # Fusionar clusters
        new_cluster = clusters[best_i] + clusters[best_j]
        new_clusters = [clusters[i] for i in range(k) if i not in (best_i, best_j)]
        new_names    = [cluster_names[i] for i in range(k) if i not in (best_i, best_j)]
        new_clusters.append(new_cluster)
        new_names.append(new_name)

        clusters      = new_clusters
        cluster_names = new_names

    return merge_steps

import matplotlib.pyplot as plt
import networkx as nx

def draw_tree(merge_steps: list, names: list[str]):
    """
    Dibuja el árbol guía de manera robusta usando NetworkX basándose 
    estrictamente en los nombres de las fusiones. Evita errores de SciPy.
    """
    G = nx.DiGraph()
    
    # 1. Construir las relaciones del árbol de arriba hacia abajo
    for step in merge_steps:
        padre = step['name']
        hijo_izq = step['left_name']
        hijo_der = step['right_name']
        distancia = step['dist']
        
        # Añadir aristas (etiquetadas con la distancia si se desea)
        G.add_edge(padre, hijo_izq, weight=distancia)
        G.add_edge(padre, hijo_der, weight=distancia)

    # 2. Algoritmo para posicionar los nodos jerárquicamente de forma automática
    def asignar_posiciones(grafo, nodo, pos={}, x=0, y=0, layer_width=1.0):
        pos[nodo] = (x, y)
        hijos = list(grafo.successors(nodo))
        if len(hijos) == 2:
            # El hijo izquierdo va un poco a la izquierda, el derecho a la derecha
            asignar_posiciones(grafo, hijos[0], pos, x - layer_width/2, y - 1, layer_width/2)
            asignar_posiciones(grafo, hijos[1], pos, x + layer_width/2, y - 1, layer_width/2)
        return pos

    # La raíz es el último nodo creado en las fusiones
    raiz = merge_steps[-1]['name']
    posiciones = asignar_posiciones(G, raiz)

    # 3. Clasificar nodos para darles un aspecto limpio y estilizado
    nodos_hoja = [n for n in G.nodes if n in names]
    nodos_internos = [n for n in G.nodes if n not in names]

    plt.figure(figsize=(10, 6))
    
    # Dibujar las ramas (líneas conectoras)
    nx.draw_networkx_edges(G, posiciones, edge_color="gray", width=2)

    # Dibujar nodos internos (puntos de bifurcación evolutiva)
    nx.draw_networkx_nodes(G, posiciones, nodelist=nodos_internos, 
                           node_color="skyblue", node_size=300)

    # Dibujar hojas (tus secuencias biológicas originales S1, S2...)
    nx.draw_networkx_nodes(G, posiciones, nodelist=nodos_hoja, 
                           node_color="lightgreen", node_size=500)

    # Añadir los nombres de las secuencias y nodos
    # Para las hojas mostramos el nombre limpio; para los internos, una etiqueta corta opcional
    etiquetas = {n: n if n in names else "" for n in G.nodes}
    nx.draw_networkx_labels(G, posiciones, labels=etiquetas, font_size=10, font_weight="bold")

    # Añadir distancias en el gráfico (opcional, adorna las ramas)
    # etiquetas_aristas = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, posiciones, edge_labels=etiquetas_aristas, font_size=8)

    plt.title("Árbol Guía de Alineación Múltiple (UPGMA)", fontsize=12, fontweight='bold')
    plt.axis('off') # Ocultar los ejes cartesianos para que parezca un árbol limpio
    
    # Exportar de forma segura
    plt.savefig("arbol_guia.png", bbox_inches='tight', dpi=300)
    print("\n✓ ¡Árbol exportado con éxito mediante NetworkX a 'arbol_guia.png'!")
    plt.show()

# ─────────────────────────────────────────────
# 4. ALINEACIÓN PROGRESIVA
# ─────────────────────────────────────────────

def align_two_groups(group1: list[list], group2: list[list]) -> tuple[list[list], list[list]]:
    """
    Alinea dos grupos de secuencias (ya alineadas internamente).
    Estrategia: usa los perfiles (sustituye gaps interiores por X) para guiar la alineación,
    luego aplica los mismos gaps a todas las secuencias del grupo.
    """
    # Representante de cada grupo: primera secuencia (las X actúan como comodines)
    rep1 = [('X' if c == '-' else c) for c in group1[0]]
    rep2 = [('X' if c == '-' else c) for c in group2[0]]

    _, a1, a2 = needleman_wunsch(rep1, rep2)

    # Reconstruir la posición de gaps introducidos en rep1
    gaps1 = _gap_positions(rep1, a1)
    gaps2 = _gap_positions(rep2, a2)

    new_group1 = []
    for seq in group1:
        seq_con_x = [('X' if c == '-' else c) for c in seq]
        new_group1.append(_insert_gaps(seq_con_x, gaps1))
        
    new_group2 = []
    for seq in group2:
        seq_con_x = [('X' if c == '-' else c) for c in seq]
        new_group2.append(_insert_gaps(seq_con_x, gaps2))

    return new_group1, new_group2


def _gap_positions(original: list, aligned: list) -> list[int]:
    """Devuelve las posiciones (en el alineado) donde se insertaron gaps."""
    gap_positions = []
    orig_idx = 0
    for pos, ch in enumerate(aligned):
        if ch == '-':
            gap_positions.append(pos)
        else:
            orig_idx += 1
    return gap_positions


def _insert_gaps(seq: list, gap_positions: list[int]) -> list:
    """Inserta gaps ('-') en las posiciones indicadas de la secuencia."""
    result = list(seq)
    for pos in gap_positions:
        result.insert(pos, '-')
    return result


def progressive_alignment(sequences: list[list], names: list[str],
                           merge_steps: list) -> list[list]:
    """
    Fusiona secuencias siguiendo el árbol guía.
    """
    print("\n" + "="*60)
    print("  ALINEACIÓN PROGRESIVA")
    print("="*60)

    # Grupos iniciales: cada secuencia en su propio grupo
    groups = {name: [list(seq)] for name, seq in zip(names, sequences)}

    for step in merge_steps:
        left  = step['left_name']
        right = step['right_name']
        new   = step['name']

        g1 = groups[left]
        g2 = groups[right]

        print(f"\n  Fusionando: {left}  +  {right}  →  {new}")
        g1_aligned, g2_aligned = align_two_groups(g1, g2)

        # Unificar longitud (por si queda diferencia de 1 por traceback)
        max_len = max(len(s) for s in g1_aligned + g2_aligned)
        for s in g1_aligned + g2_aligned:
            while len(s) < max_len:
                s.append('-')

        groups[new] = g1_aligned + g2_aligned

        # Mostrar alineación parcial
        print(f"  Resultado ({len(groups[new])} secuencias, longitud {max_len}):")
        merged_names = _get_leaf_names(left, merge_steps, names) + \
                       _get_leaf_names(right, merge_steps, names)
        for mname, seq in zip(merged_names, groups[new]):
            print(f"    {mname:>6}: {''.join(seq)}")

    # Retornar el grupo final
    final_key = merge_steps[-1]['name']
    return groups[final_key]


def _get_leaf_names(node: str, merge_steps: list, original_names: list) -> list[str]:
    """Obtiene los nombres hoja de un nodo del árbol."""
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

def multiple_sequence_alignment(sequences: list[str],
                                 names: list[str] = None) -> list[str]:
    """
    Alineación de múltiples secuencias completa.
    Retorna lista de secuencias alineadas (sin X).
    """
    if names is None or len(names) != len(sequences):
        names = [f"S{i+1}" for i in range(len(sequences))]

    seq_lists = [list(s.upper()) for s in sequences]

    print("\n" + "="*60)
    print("  SECUENCIAS DE ENTRADA")
    print("="*60)
    for name, seq in zip(names, sequences):
        print(f"  {name:>6}: {seq.upper()}")

    # 1. Árbol guía
    merge_steps = build_guide_tree(seq_lists, names)

    # 2. Alineación progresiva
    aligned_groups = progressive_alignment(seq_lists, names, merge_steps)

    # 3. Obtener orden original de secuencias
    leaf_order = _get_leaf_names(merge_steps[-1]['name'], merge_steps, names)
    name_to_aligned = {name: seq for name, seq in zip(leaf_order, aligned_groups)}

    # 4. Mostrar resultados con X
    print("\n" + "="*60)
    print("  ALINEACIÓN FINAL (con X internos)")
    print("="*60)
    max_len = max(len(s) for s in aligned_groups)
    for name in names:
        seq = name_to_aligned.get(name, [])
        while len(seq) < max_len:
            seq.append('-')
        seq =''.join(seq).replace('-','X')
        print(f"  {name:>6}: {seq}")

    # 5. Remover X y mostrar resultado limpio
    clean = []
    for name in names:
        seq = name_to_aligned.get(name, [])
        while len(seq) < max_len:
            seq.append('-')
        cleaned = ''.join(c if c != 'X' else '-' for c in seq)
        clean.append(cleaned)

    print("\n" + "="*60)
    print("  ALINEACIÓN FINAL (X reemplazados por -)")
    print("="*60)
    for name, seq in zip(names, clean):
        print(f"  {name:>6}: {seq}")

    return clean


# ─────────────────────────────────────────────
# 6. PRUEBA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    INPUT_FOLDER = "muestras/"
    OUTPUT_FILE = "msa_resultados.txt"
    seqs  = []
    archivos = os.listdir(INPUT_FOLDER)
    for i in archivos:
        seqs.append(leer(INPUT_FOLDER+i))
    
    seqs2 = ["ATTGGCACCA","ATTTGGACCA","TGGTTCCA","ATTCCACCAC"]
    # Redirigir toda la salida estándar al archivo .txt
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        sys.stdout = f

        print("#"*60)
        print(" ALINEAMIENTO MULTIPLE")
        print("#"*60)
        names1 = ["S1", "S2", "S3", "S4","S5"]
        steps_tree = build_guide_tree(seqs, names1)
        result1 = multiple_sequence_alignment(seqs, names1)


    # Restaurar stdout y confirmar
    sys.stdout = sys.__stdout__
    
    # Fuera del "with open" (para que no se rompa la interfaz gráfica de matplotlib):
    try:
        draw_tree(steps_tree, names1)
    except ImportError:
        print("Error al generar el gráfico en PNG.")
    print(f"Resultados guardados en: {OUTPUT_FILE}")