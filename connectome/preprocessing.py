"""
Preprocessing of MaleCNS v1.0 for fly-connectome-ml.

This module transforms the released MaleCNS data into a reproducible
computational representation of the connectome.

Important distinction:

    MaleCNS data
        ↓
    preprocessing
        ↓
    structural graph
        ↓
    neural simulation
        ↓
    learning / experiments

The preprocessing stage does not define neural dynamics or learning rules.
"""
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


SOURCE_FILES = {
    "annotations": RAW_DIR / "annotations.feather",
    "neurotransmitters": RAW_DIR / "neurotransmitters.feather",
    "edges": RAW_DIR / "edges.feather",
}


def check_source_files() -> None:
    """Check that all required MaleCNS files are available."""
    for name, path in SOURCE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required {name} file not found: {path}"
            )

        print(f"[OK] {name}: {path}")

def inspect_sources() -> None:
    """Print basic schema and size information for each source file."""
    import pyarrow.feather as feather

    for name, path in SOURCE_FILES.items():
        table = feather.read_table(path, memory_map=True)

        print()
        print(f"=== {name} ===")
        print(f"Rows: {table.num_rows:,}")
        print(f"Columns: {table.num_columns}")
        print("Schema:")
        print(table.schema)

def inspect_annotations(ann) -> None:
    """Inspect annotation categories before defining the node policy."""

    print()
    print("=== Annotation inspection ===")

    print("\nStatus:")
    print(ann["status"].value_counts(dropna=False).to_string())

    print("\nStatusLabel:")
    print(ann["statusLabel"].value_counts(dropna=False).to_string())

    print("\nSuperclass:")
    print(ann["superclass"].value_counts(dropna=False).to_string())

    missing_superclass = (
        ann["superclass"].isna()
        | ann["superclass"].astype(str).eq("")
    )

    print("\nSuperclass:")
    print(f"  With superclass:    {(~missing_superclass).sum():,}")
    print(f"  Without superclass: {missing_superclass.sum():,}")

    print("\nBody IDs:")
    print(f"  Total:  {ann['bodyId'].size:,}")
    print(f"  Unique: {ann['bodyId'].nunique():,}")

    if ann["bodyId"].nunique() != len(ann):
        print("  WARNING: duplicate bodyId values detected.")
    else:
        print("  All bodyId values are unique.")

def filter_nodes(ann):
    """
    Select neuronal simulation nodes from MaleCNS annotations.

    Inclusion policy:
      - assigned superclass
      - not explicitly annotated as Glia

    This is a DATA-SELECTION decision, not a neural-dynamics model.
    """

    has_superclass = (
        ann["superclass"].notna()
        & ann["superclass"].astype(str).ne("")
    )

    not_glia = ~ann["status"].eq("Glia")

    retain = has_superclass & not_glia

    nodes = ann.loc[retain].copy()

    nodes = nodes.sort_values(
        "bodyId",
        ignore_index=True,
    )

    return nodes

def inspect_neurotransmitters() -> None:
    """Inspect the granularity and coverage of the neurotransmitter data."""
    import pyarrow.feather as feather

    path = SOURCE_FILES["neurotransmitters"]

    nt = feather.read_table(path, memory_map=True).to_pandas()

    print("\n=== Neurotransmitter inspection ===")
    print(f"Rows: {len(nt):,}")
    print(f"Unique bodies: {nt['body'].nunique():,}")

    # How many rows does each body have?
    rows_per_body = nt["body"].value_counts()

    print("\nRows per body:")
    print(rows_per_body.describe())

    print("\nDistribution of rows per body:")
    print(rows_per_body.value_counts().sort_index().head(20))

    # Bodies appearing more than once
    duplicated_bodies = (rows_per_body > 1).sum()

    print(f"\nBodies with multiple rows: {duplicated_bodies:,}")

    # Neurotransmitter labels
    print("\nConsensus neurotransmitters:")
    print(nt["consensus_nt"].value_counts(dropna=False).to_string())

    print("\nPredicted neurotransmitters:")
    print(nt["predicted_nt"].value_counts(dropna=False).to_string())

    print("\nGround truth:")
    print(nt["ground_truth"].value_counts(dropna=False).to_string())

def analyze_neurotransmitter_agreement() -> None:
    """Analyze agreement between neurotransmitter annotation sources."""
    import pyarrow.feather as feather

    nt = feather.read_table(
        SOURCE_FILES["neurotransmitters"],
        memory_map=True,
    ).to_pandas()

    print("\n=== Neurotransmitter agreement ===")

    # Consensus vs predicted
    agreement = (
        nt["consensus_nt"] == nt["predicted_nt"]
    ).mean()

    print(f"Consensus == predicted: {agreement:.2%}")

    print("\nConsensus vs predicted:")
    print(
        nt.groupby(
            ["consensus_nt", "predicted_nt"],
            dropna=False,
        )
        .size()
        .sort_values(ascending=False)
        .head(30)
        .to_string()
    )

    # Ground truth coverage
    ground_truth_known = nt["ground_truth"].notna().sum()

    print(
        f"\nGround truth available: "
        f"{ground_truth_known:,} / {len(nt):,} "
        f"({ground_truth_known / len(nt):.2%})"
    )

    # Consensus vs ground truth where ground truth exists
    known_gt = nt["ground_truth"].notna()

    if known_gt.any():
        gt_agreement = (
            nt.loc[known_gt, "consensus_nt"]
            == nt.loc[known_gt, "ground_truth"]
        ).mean()

        print(
            f"Consensus == ground truth "
            f"(where available): {gt_agreement:.2%}"
        )

def analyze_retained_neurotransmitters(ann) -> None:
    """Analyze neurotransmitter coverage for retained simulation nodes."""
    import pyarrow.feather as feather

    nodes = filter_nodes(ann)

    nt = feather.read_table(
        SOURCE_FILES["neurotransmitters"],
        memory_map=True,
    ).to_pandas()

    # Map body ID → consensus neurotransmitter
    nt_by_body = nt.set_index("body")["consensus_nt"]

    nodes["consensus_nt"] = nodes["bodyId"].map(nt_by_body)

    known = (
        nodes["consensus_nt"].notna()
        & nodes["consensus_nt"].ne("unclear")
    )

    print("\n=== Retained node neurotransmitters ===")

    print(f"Retained nodes: {len(nodes):,}")
    print(f"Known neurotransmitter: {known.sum():,}")
    print(f"Unclear neurotransmitter: {(~known).sum():,}")

    print("\nCoverage:")
    print(f"Known:   {known.mean():.2%}")
    print(f"Unclear: {(~known).mean():.2%}")

    print("\nDistribution:")
    print(
        nodes["consensus_nt"]
        .fillna("missing")
        .value_counts()
        .to_string()
    )

def investigate_missing_neurotransmitters(ann) -> None:
    """Investigate retained nodes without a neurotransmitter record."""
    import pyarrow.feather as feather

    nodes = filter_nodes(ann)

    nt = feather.read_table(
        SOURCE_FILES["neurotransmitters"],
        memory_map=True,
    ).to_pandas()

    nt_bodies = set(nt["body"].astype("int64"))

    missing_mask = ~nodes["bodyId"].isin(nt_bodies)
    missing = nodes.loc[missing_mask].copy()

    print("\n=== Missing neurotransmitter records ===")

    print(f"Retained nodes: {len(nodes):,}")
    print(f"Missing NT records: {len(missing):,}")

    if len(missing) == 0:
        print("\nNo missing records found.")
        return

    print("\nMissing body IDs:")
    print(missing["bodyId"].to_list())

    print("\nMissing nodes by superclass:")
    print(
        missing["superclass"]
        .fillna("missing")
        .value_counts()
        .to_string()
    )

    print("\nMissing nodes by status:")
    print(
        missing["status"]
        .fillna("missing")
        .value_counts()
        .to_string()
    )

    print("\nMissing nodes:")
    print(
        missing[
            ["bodyId", "superclass", "type", "status", "statusLabel"]
        ].to_string(index=False)
    )

def investigate_missing_nt_connectivity(ann) -> None:
    """Check graph connectivity of nodes without NT records."""
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc

    nodes = filter_nodes(ann)

    # Load neurotransmitter IDs
    nt = feather.read_table(
        SOURCE_FILES["neurotransmitters"],
        memory_map=True,
    ).to_pandas()

    nt_bodies = set(nt["body"].astype("int64"))

    missing_mask = ~nodes["bodyId"].isin(nt_bodies)
    missing_ids = set(
        nodes.loc[missing_mask, "bodyId"].astype("int64")
    )

    print("\n=== Connectivity of missing NT nodes ===")
    print(f"Nodes without NT record: {len(missing_ids):,}")

    # Track whether each missing node appears as pre/post
    has_pre = set()
    has_post = set()

    reader = ipc.open_file(
        pa.memory_map(str(SOURCE_FILES["edges"]), "r")
    )

    for number in range(reader.num_record_batches):
        batch = reader.get_batch(number)

        pre = batch.column(
            batch.schema.get_field_index("body_pre")
        ).to_numpy(zero_copy_only=False)

        post = batch.column(
            batch.schema.get_field_index("body_post")
        ).to_numpy(zero_copy_only=False)

        # Only compare against our 178 target IDs
        pre_mask = np.isin(pre, list(missing_ids))
        post_mask = np.isin(post, list(missing_ids))

        if pre_mask.any():
            has_pre.update(pre[pre_mask].astype("int64"))

        if post_mask.any():
            has_post.update(post[post_mask].astype("int64"))

    both = has_pre & has_post
    only_pre = has_pre - has_post
    only_post = has_post - has_pre
    isolated = missing_ids - has_pre - has_post

    print(f"\nWith outgoing edges: {len(has_pre):,}")
    print(f"With incoming edges: {len(has_post):,}")
    print(f"With both: {len(both):,}")
    print(f"Only outgoing: {len(only_pre):,}")
    print(f"Only incoming: {len(only_post):,}")
    print(f"With no edges: {len(isolated):,}")

def analyze_missing_nt_nodes(ann) -> None:
    """Characterize nodes without neurotransmitter records in the graph."""
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc

    nodes = filter_nodes(ann).copy()

    # ---------------------------------------------------------
    # 1. Identify nodes without a neurotransmitter record
    # ---------------------------------------------------------
    nt = feather.read_table(
        SOURCE_FILES["neurotransmitters"],
        memory_map=True,
    ).to_pandas()

    nt_bodies = set(nt["body"].astype("int64"))

    missing_mask = ~nodes["bodyId"].isin(nt_bodies)
    missing_nodes = nodes.loc[missing_mask].copy()

    missing_ids = missing_nodes["bodyId"].astype("int64").to_numpy()

    print("\n=== Detailed analysis of missing NT nodes ===")
    print(f"Missing NT nodes: {len(missing_nodes):,}")

    # ---------------------------------------------------------
    # 2. Counters indexed by bodyId
    # ---------------------------------------------------------
    incoming_rows = dict.fromkeys(missing_ids, 0)
    incoming_contacts = dict.fromkeys(missing_ids, 0)

    # ---------------------------------------------------------
    # 3. Stream the 151M edge rows
    # ---------------------------------------------------------
    reader = ipc.open_file(
        pa.memory_map(str(SOURCE_FILES["edges"]), "r")
    )

    missing_id_set = set(missing_ids.tolist())

    for number in range(reader.num_record_batches):
        batch = reader.get_batch(number)

        pre = batch.column(
            batch.schema.get_field_index("body_pre")
        ).to_numpy(zero_copy_only=False)

        post = batch.column(
            batch.schema.get_field_index("body_post")
        ).to_numpy(zero_copy_only=False)

        weight = batch.column(
            batch.schema.get_field_index("weight")
        ).to_numpy(zero_copy_only=False)

        # Only post-synaptic occurrences matter here because
        # previous analysis showed that none of the missing
        # NT nodes appears as body_pre.
        post_mask = np.isin(post, missing_ids)

        if not post_mask.any():
            continue

        matching_post = post[post_mask]
        matching_weight = weight[post_mask]

        unique_post, counts = np.unique(
            matching_post,
            return_counts=True,
        )

        for body_id, count in zip(unique_post, counts):
            incoming_rows[int(body_id)] += int(count)

        # Sum synaptic contacts, not merely edge rows
        for body_id, syn_count in zip(
            matching_post,
            matching_weight,
        ):
            incoming_contacts[int(body_id)] += int(syn_count)

    # ---------------------------------------------------------
    # 4. Build analysis table
    # ---------------------------------------------------------
    missing_nodes["incoming_edge_rows"] = (
        missing_nodes["bodyId"]
        .astype("int64")
        .map(incoming_rows)
        .fillna(0)
        .astype("int64")
    )

    missing_nodes["incoming_synaptic_contacts"] = (
        missing_nodes["bodyId"]
        .astype("int64")
        .map(incoming_contacts)
        .fillna(0)
        .astype("int64")
    )

    missing_nodes["graph_role"] = np.where(
        missing_nodes["incoming_edge_rows"] > 0,
        "post_only",
        "isolated",
    )

    # ---------------------------------------------------------
    # 5. Summary
    # ---------------------------------------------------------
    print("\nGraph role:")
    print(
        missing_nodes["graph_role"]
        .value_counts()
        .to_string()
    )

    print("\nBy superclass:")
    print(
        missing_nodes.groupby(
            ["graph_role", "superclass"],
            dropna=False,
        )
        .size()
        .to_string()
    )

    print("\nBy status:")
    print(
        missing_nodes.groupby(
            ["graph_role", "status"],
            dropna=False,
        )
        .size()
        .to_string()
    )

    # ---------------------------------------------------------
    # 6. Detailed table
    # ---------------------------------------------------------
    columns = [
        "bodyId",
        "superclass",
        "type",
        "status",
        "statusLabel",
        "incoming_edge_rows",
        "incoming_synaptic_contacts",
        "graph_role",
    ]

    print("\nDetailed nodes:")
    print(
        missing_nodes[columns]
        .sort_values(
            ["graph_role", "incoming_synaptic_contacts"],
            ascending=[True, False],
        )
        .to_string(index=False)
    )


def compute_degrees(ann) -> "pd.DataFrame":
    """
    Mede a conectividade de cada nó re em uma única passada
    no stream de arestas.

    Conta, por nó:
      in_degree / out_degree     -> nº de conexões distintas
      in_contacts / out_contacts -> soma de contatos sinápticos (weight)

    Retorna o DataFrame de nós enriquecido, ordenado por bodyId.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.ipc as ipc

    nodes = filter_nodes(ann)

    # ids ordenados -> posição = índice no array
    ids = nodes["bodyId"].astype("int64").to_numpy()
    n = len(ids)

    in_degree = np.zeros(n, dtype=np.int64)
    out_degree = np.zeros(n, dtype=np.int64)
    in_contacts = np.zeros(n, dtype=np.int64)
    out_contacts = np.zeros(n, dtype=np.int64)

    retained_edge_rows = 0

    reader = ipc.open_file(
        pa.memory_map(str(SOURCE_FILES["edges"]), "r")
    )

    for number in range(reader.num_record_batches):
        batch = reader.get_batch(number)

        pre = batch.column(
            batch.schema.get_field_index("body_pre")
        ).to_numpy(zero_copy_only=False)

        post = batch.column(
            batch.schema.get_field_index("body_post")
        ).to_numpy(zero_copy_only=False)

        weight = batch.column(
            batch.schema.get_field_index("weight")
        ).to_numpy(zero_copy_only=False)

        # Arestas com ambas as extremidades nos nós retidos
        pre_mask = np.isin(pre, ids)
        post_mask = np.isin(post, ids)
        both_mask = pre_mask & post_mask

        retained_edge_rows += int(both_mask.sum())

        # ---- nós pré-sinápticos (saída) ----
        if pre_mask.any():
            pos_pre = np.searchsorted(ids, pre[pre_mask])
            np.add.at(out_degree, pos_pre, 1)
            np.add.at(out_contacts, pos_pre, weight[pre_mask])

        # ---- nós pós-sinápticos (entrada) ----
        if post_mask.any():
            pos_post = np.searchsorted(ids, post[post_mask])
            np.add.at(in_degree, pos_post, 1)
            np.add.at(in_contacts, pos_post, weight[post_mask])

    nodes = nodes.copy()
    nodes["in_degree"] = in_degree
    nodes["out_degree"] = out_degree
    nodes["total_degree"] = in_degree + out_degree
    nodes["in_contacts"] = in_contacts
    nodes["out_contacts"] = out_contacts

    print("\n=== Degrees ===")
    print(f"Retained nodes:        {n:,}")
    print(f"Retained edge rows:    {retained_edge_rows:,}")
    print(f"Isolated nodes:        {(nodes['total_degree'] == 0).sum():,}")
    print(f"Density (directed):    {retained_edge_rows / (n * (n - 1)):.3e}")

    return nodes

def select_node_subset(
    nodes: "pd.DataFrame",
    n: int = 10_000,
    strategy: str = "stratified",
    seed: int = 42,
) -> "pd.DataFrame":
    """
    Seleciona n nós do grafo retido.

    Estratégias:
      stratified   -> amostra aleatória proporcional a `superclass`
                      (representa a composição do connectome)
      top_degree   -> os n nós de maior total_degree
                      (rede densa, centrada em hubs — viés grande)
      seed_bfs     -> (adiada) depende de build_adjacency

    Sempre retorna ordenado por bodyId, reproduzível via seed.
    """
    import pandas as pd

    if n <= 0:
        raise ValueError("n deve ser maior que 0")

    if n > len(nodes):
        raise ValueError(
            f"n={n:,} é maior que o número de nós disponíveis "
            f"({len(nodes):,})"
        )

    if strategy == "top_degree":
        subset = nodes.nlargest(n, "total_degree")
        return subset.sort_values("bodyId", ignore_index=True)

    if strategy == "stratified":
        counts = nodes["superclass"].value_counts()
        quotas = counts / counts.sum() * n

        base = np.floor(quotas).astype(int)

        # Distribui o restante pelas classes com maior fração
        remainder = int(n - base.sum())

        if remainder > 0:
            frac = (quotas - base).sort_values(ascending=False)
            base[frac.head(remainder).index] += 1

        parts = []

        for superclass, k in base.items():
            if k <= 0:
                continue

            group = nodes[nodes["superclass"] == superclass]

            parts.append(
                group.sample(
                    n=int(k),
                    random_state=seed
                )
            )

        subset = pd.concat(parts)

        return subset.sort_values(
            "bodyId",
            ignore_index=True
        )

    raise ValueError(f"Estratégia desconhecida: {strategy}")

def build_induced_subgraph(
    nodes: "pd.DataFrame",
    edges_path,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Constrói o subgrafo induzido pelos nós fornecidos.

    Mantém somente arestas em que:
        body_pre  ∈ nodes["bodyId"]
        body_post ∈ nodes["bodyId"]

    Recalcula, exclusivamente dentro do subgrafo:
        in_degree / out_degree
        in_contacts / out_contacts

    Retorna:
        nodes_subgraph -> nós enriquecidos com graus do subgrafo
        edges_subgraph -> arestas internas ao subconjunto
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.ipc as ipc

    ids = np.sort(
        nodes["bodyId"].astype("int64").to_numpy()
    )

    n = len(ids)

    in_degree = np.zeros(n, dtype=np.int64)
    out_degree = np.zeros(n, dtype=np.int64)

    in_contacts = np.zeros(n, dtype=np.int64)
    out_contacts = np.zeros(n, dtype=np.int64)

    retained_batches = []
    retained_edge_rows = 0

    reader = ipc.open_file(
        pa.memory_map(str(edges_path), "r")
    )

    for number in range(reader.num_record_batches):
        batch = reader.get_batch(number)

        pre = batch.column(
            batch.schema.get_field_index("body_pre")
        ).to_numpy(zero_copy_only=False)

        post = batch.column(
            batch.schema.get_field_index("body_post")
        ).to_numpy(zero_copy_only=False)

        weight = batch.column(
            batch.schema.get_field_index("weight")
        ).to_numpy(zero_copy_only=False)

        pre_mask = np.isin(pre, ids)
        post_mask = np.isin(post, ids)

        both_mask = pre_mask & post_mask

        if not both_mask.any():
            continue

        retained_edge_rows += int(both_mask.sum())

        # Guardamos somente as arestas internas ao subgrafo
        retained_batches.append(
            pd.DataFrame(
                {
                    "body_pre": pre[both_mask].astype("int64"),
                    "body_post": post[both_mask].astype("int64"),
                    "weight": weight[both_mask].astype("int64"),
                }
            )
        )

        # Posições dos nós dentro do array ordenado de IDs
        pos_pre = np.searchsorted(
            ids,
            pre[both_mask]
        )

        pos_post = np.searchsorted(
            ids,
            post[both_mask]
        )

        # Graus de saída
        np.add.at(
            out_degree,
            pos_pre,
            1
        )

        np.add.at(
            out_contacts,
            pos_pre,
            weight[both_mask]
        )

        # Graus de entrada
        np.add.at(
            in_degree,
            pos_post,
            1
        )

        np.add.at(
            in_contacts,
            pos_post,
            weight[both_mask]
        )

    if retained_batches:
        edges_subgraph = pd.concat(
            retained_batches,
            ignore_index=True
        )
    else:
        edges_subgraph = pd.DataFrame(
            columns=[
                "body_pre",
                "body_post",
                "weight",
            ]
        )

    nodes_subgraph = nodes.copy()

    nodes_subgraph["in_degree"] = in_degree
    nodes_subgraph["out_degree"] = out_degree

    nodes_subgraph["total_degree"] = (
        in_degree + out_degree
    )

    nodes_subgraph["in_contacts"] = in_contacts
    nodes_subgraph["out_contacts"] = out_contacts

    print("\n=== Induced subgraph ===")
    print(f"Retained nodes:     {n:,}")
    print(f"Retained edge rows: {retained_edge_rows:,}")
    print(
        "Isolated nodes:     "
        f"{(nodes_subgraph['total_degree'] == 0).sum():,}"
    )

    if n > 1:
        density = (
            retained_edge_rows /
            (n * (n - 1))
        )
    else:
        density = 0.0

    print(
        f"Density (directed): {density:.3e}"
    )

    return nodes_subgraph, edges_subgraph


def validate_induced_subgraph(
    nodes: "pd.DataFrame",
    edges: "pd.DataFrame",
) -> None:
    """
    Valida a consistência interna do subgrafo induzido.

    Verifica:
      - IDs dos nós são únicos;
      - todas as extremidades das arestas pertencem aos nós;
      - in_degree/out_degree batem com as arestas;
      - in_contacts/out_contacts batem com a soma dos pesos;
      - número de nós isolados é consistente.

    Lança AssertionError se alguma verificação falhar.
    """
    import pandas as pd

    print("\n=== Subgraph validation ===")

    # ---------------------------------------------------------
    # 1. Estrutura básica dos nós
    # ---------------------------------------------------------

    node_ids = nodes["bodyId"].astype("int64")

    assert node_ids.is_unique, (
        "bodyId duplicado nos nós do subgrafo"
    )

    node_id_set = set(node_ids)

    print(f"Nodes: {len(nodes):,}")
    print(f"Edges: {len(edges):,}")

    # ---------------------------------------------------------
    # 2. Todas as arestas devem estar dentro do subgrafo
    # ---------------------------------------------------------

    invalid_pre = ~edges["body_pre"].isin(node_id_set)
    invalid_post = ~edges["body_post"].isin(node_id_set)

    n_invalid_pre = int(invalid_pre.sum())
    n_invalid_post = int(invalid_post.sum())

    assert n_invalid_pre == 0, (
        f"{n_invalid_pre:,} arestas possuem body_pre "
        "fora do subconjunto"
    )

    assert n_invalid_post == 0, (
        f"{n_invalid_post:,} arestas possuem body_post "
        "fora do subconjunto"
    )

    print("Edge endpoints:      OK")

    # ---------------------------------------------------------
    # 3. Recalcula graus diretamente das arestas
    # ---------------------------------------------------------

    calculated_out_degree = (
        edges.groupby("body_pre")
        .size()
    )

    calculated_in_degree = (
        edges.groupby("body_post")
        .size()
    )

    calculated_out_contacts = (
        edges.groupby("body_pre")["weight"]
        .sum()
    )

    calculated_in_contacts = (
        edges.groupby("body_post")["weight"]
        .sum()
    )

    # Alinha pelos bodyIds dos nós.
    calculated_out_degree = (
        calculated_out_degree
        .reindex(node_ids, fill_value=0)
        .to_numpy()
    )

    calculated_in_degree = (
        calculated_in_degree
        .reindex(node_ids, fill_value=0)
        .to_numpy()
    )

    calculated_out_contacts = (
        calculated_out_contacts
        .reindex(node_ids, fill_value=0)
        .to_numpy()
    )

    calculated_in_contacts = (
        calculated_in_contacts
        .reindex(node_ids, fill_value=0)
        .to_numpy()
    )

    # ---------------------------------------------------------
    # 4. Compara graus
    # ---------------------------------------------------------

    assert np.array_equal(
        nodes["out_degree"].to_numpy(),
        calculated_out_degree,
    ), "out_degree não corresponde às arestas"

    assert np.array_equal(
        nodes["in_degree"].to_numpy(),
        calculated_in_degree,
    ), "in_degree não corresponde às arestas"

    assert np.array_equal(
        nodes["out_contacts"].to_numpy(),
        calculated_out_contacts,
    ), "out_contacts não corresponde às arestas"

    assert np.array_equal(
        nodes["in_contacts"].to_numpy(),
        calculated_in_contacts,
    ), "in_contacts não corresponde às arestas"

    print("in_degree:           OK")
    print("out_degree:          OK")
    print("in_contacts:         OK")
    print("out_contacts:        OK")

    # ---------------------------------------------------------
    # 5. Verifica total_degree
    # ---------------------------------------------------------

    expected_total_degree = (
        nodes["in_degree"] +
        nodes["out_degree"]
    )

    assert np.array_equal(
        nodes["total_degree"].to_numpy(),
        expected_total_degree.to_numpy(),
    ), "total_degree inconsistente"

    print("total_degree:        OK")

    # ---------------------------------------------------------
    # 6. Isolados
    # ---------------------------------------------------------

    isolated = (
        nodes["in_degree"].eq(0)
        & nodes["out_degree"].eq(0)
    )

    print(
        f"Isolated nodes:      {isolated.sum():,}"
    )

    # ---------------------------------------------------------
    # 7. Consistência global
    # ---------------------------------------------------------

    assert (
        int(nodes["out_degree"].sum())
        == len(edges)
    ), "Soma de out_degree != número de arestas"

    assert (
        int(nodes["in_degree"].sum())
        == len(edges)
    ), "Soma de in_degree != número de arestas"

    total_weight = int(edges["weight"].sum())

    assert (
        int(nodes["out_contacts"].sum())
        == total_weight
    ), "Soma de out_contacts != soma dos pesos"

    assert (
        int(nodes["in_contacts"].sum())
        == total_weight
    ), "Soma de in_contacts != soma dos pesos"

    print("Global edge count:   OK")
    print("Global contact sum:  OK")

    print("\n[OK] Subgraph validation passed.")

if __name__ == "__main__":
    check_source_files()

    import pyarrow.feather as feather
    ann = feather.read_table(
        SOURCE_FILES["annotations"], memory_map=True
    ).to_pandas()

    nodes = compute_degrees(ann)

    print("\nDegree distribution:")
    print(nodes["total_degree"].describe())

    print("\nTop 10 hubs:")
    print(
        nodes.nlargest(10, "total_degree")[
            ["bodyId", "superclass", "type",
             "in_degree", "out_degree", "in_contacts", "out_contacts"]
        ].to_string(index=False)
    )

    # ---- CROSS-CHECK com análise anterior ----
    # Esperado: 178 nós sem NT -> 124 isolados, 54 com entrada e sem saída
    nt = feather.read_table(
        SOURCE_FILES["neurotransmitters"], memory_map=True
    ).to_pandas()
    nt_bodies = set(nt["body"].astype("int64"))
    missing = nodes[~nodes["bodyId"].isin(nt_bodies)]

    print("\nCross-check (178 nós sem NT):")
    print(f"  Isolated:            {(missing['total_degree'] == 0).sum()}  (esperado 124)")
    print(f"  Só entrada:          {((missing['in_degree'] > 0) & (missing['out_degree'] == 0)).sum()}  (esperado 54)")
    print(f"  Só saída:            {((missing['out_degree'] > 0) & (missing['in_degree'] == 0)).sum()}  (esperado 0)")
    print(f"  Com entrada e saída: {((missing['in_degree'] > 0) & (missing['out_degree'] > 0)).sum()}  (esperado 0)")

    # Cache para as próximas funções não relerem o stream
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(PROCESSED_DIR / "nodes_with_degrees.parquet")
    print("\n[OK] Salvo: data/processed/nodes_with_degrees.parquet")

    subset = select_node_subset(
        nodes,
        n=10_000,
        strategy="stratified",
        seed=42,
    )

    print("\n=== Node subset ===")
    print(f"Selected nodes: {len(subset):,}")

    print("\nSuperclass distribution:")
    print(
        subset["superclass"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    subset.to_parquet(
        PROCESSED_DIR / "nodes_subset_10000_stratified.parquet"
    )

    nodes_subgraph, edges_subgraph = build_induced_subgraph(
        subset,
        SOURCE_FILES["edges"],
    )

    validate_induced_subgraph(
        nodes_subgraph,
        edges_subgraph,
    )

    nodes_subgraph.to_parquet(
        PROCESSED_DIR / "nodes_subset_10000_stratified_with_degrees.parquet"
    )

    edges_subgraph.to_parquet(
        PROCESSED_DIR / "edges_subset_10000_stratified.parquet"
    )

    print(
        "\n[OK] Salvo: "
        "data/processed/nodes_subset_10000_stratified_with_degrees.parquet"
    )

    print(
        "\n[OK] Salvo: "
        "data/processed/nodes_subset_10000_stratified.parquet"
    )
