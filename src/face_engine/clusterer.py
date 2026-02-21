"""
Face clustering – group unassigned face embeddings into candidate clusters
that can then be reviewed and named by the user.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def get_threshold() -> float:
    """Return an appropriate cosine distance threshold based on the active backend."""
    from .detector import backend_name
    backend = backend_name()

    # InsightFace (buffalo_l) embeddings are ArcFace-trained and very discriminative.
    # 0.40 is tight but correct for same-identity matching.
    if backend == "insightface":
        return 0.40

    # face_recognition (dlib ResNet) – standard threshold is 0.6; be slightly looser
    if backend == "face_recognition":
        return 0.55

    # DeepFace Facenet512 – cosine distance is slightly noisier
    return 0.60


def cluster_embeddings(embeddings: List[np.ndarray],
                       threshold: float = None) -> List[List[int]]:
    """
    Vectorized greedy clustering using cosine distance.
    Uses a leader-based approach: O(N*C) where C = number of clusters.

    ``threshold`` defaults to the backend-appropriate value from get_threshold().
    """
    if threshold is None:
        threshold = get_threshold()

    if not embeddings:
        return []

    embs = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embs / norms

    clusters: List[List[int]] = []
    leaders: List[np.ndarray] = []

    for idx in range(len(normed)):
        current = normed[idx]

        if not leaders:
            leaders.append(current)
            clusters.append([idx])
            continue

        dots = np.dot(leaders, current)      # shape (C,)
        dists = 1.0 - dots

        best_cluster_idx = int(np.argmin(dists))
        if dists[best_cluster_idx] < threshold:
            clusters[best_cluster_idx].append(idx)
            # Update leader to be the centroid of the cluster (running mean)
            n = len(clusters[best_cluster_idx])
            leaders[best_cluster_idx] = (
                leaders[best_cluster_idx] * (n - 1) / n + current / n
            )
            # Re-normalise leader
            lnorm = np.linalg.norm(leaders[best_cluster_idx])
            if lnorm > 0:
                leaders[best_cluster_idx] /= lnorm
        else:
            leaders.append(current)
            clusters.append([idx])

    return clusters


def find_best_match(
    query_emb: np.ndarray,
    known_embeddings: List[Tuple[int, np.ndarray]],
    threshold: float = None,
) -> Optional[int]:
    """
    Match a query face embedding against all known person embeddings.

    Strategy: **per-person voting + minimum-distance**
    ──────────────────────────────────────────────────
    The naive "pick the single closest sample" fails when one person has many
    reference embeddings and another has few – the many-sample person wins by
    sheer probability even when the real distance is similar.

    Instead we:
      1. Compute cosine distance to EVERY known sample.
      2. Group results by person_id.
      3. For each person, record their BEST (minimum) distance.
      4. Also require at least 50% of their samples to be within 2× threshold
         (soft majority vote) to avoid rogue aliased samples triggering a match.
      5. Return the person with the overall best minimum distance, provided it
         is below `threshold`.

    Returns the matched person_id, or None if no confident match.
    """
    if threshold is None:
        threshold = get_threshold()

    if not known_embeddings or query_emb is None:
        return None

    # Normalise query
    qn = query_emb.astype(np.float32)
    qnorm = np.linalg.norm(qn)
    if qnorm == 0:
        return None
    qn /= qnorm

    # Build arrays
    person_ids = np.array([pid for pid, _ in known_embeddings], dtype=np.int64)
    emb_matrix = np.array([e for _, e in known_embeddings], dtype=np.float32)

    # Normalise stored embeddings row-wise
    row_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    emb_matrix /= row_norms

    # Cosine distances (1 − dot product)
    dists = 1.0 - np.dot(emb_matrix, qn)   # shape (N,)

    # Per-person aggregation
    unique_persons = np.unique(person_ids)
    best_person = None
    best_min_dist = threshold  # anything above this is rejected

    for pid in unique_persons:
        mask = person_ids == pid
        pid_dists = dists[mask]

        min_dist = float(np.min(pid_dists))

        # Majority vote: at least 30% of this person's samples must be "close"
        # (within 2× threshold). This prevents a single lucky match from aliasing.
        vote_threshold = min(threshold * 2.0, 0.80)
        close_count = int(np.sum(pid_dists < vote_threshold))
        total_count = len(pid_dists)
        if total_count >= 3 and close_count / total_count < 0.30:
            continue   # Sparse match – likely not the same person

        if min_dist < best_min_dist:
            best_min_dist = min_dist
            best_person = int(pid)

    return best_person
