"""
Face clustering – group unassigned face embeddings into candidate clusters
that can then be reviewed and named by the user.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def get_threshold() -> float:
    """Return an appropriate cosine distance threshold based on the face engine backend."""
    from .detector import backend_name
    backend = backend_name()
    
    # InsightFace (buffalo_l) embeddings are highly discriminative. 
    # 0.4 - 0.5 is usually the sweet spot for identity.
    if backend == "insightface":
        return 0.45
    
    # face_recognition (dlib) / DeepFace (Facenet) usually use 0.6 standard.
    # We use 0.65 as a slightly more lenient default for auto-matching.
    return 0.65

SIMILARITY_THRESHOLD = get_threshold()


def cluster_embeddings(embeddings: List[np.ndarray],
                       threshold: float = SIMILARITY_THRESHOLD
                       ) -> List[List[int]]:
    """
    Vectorized greedy clustering using NumPy for speed.
    Uses a leader-based approach which is O(N) comparisons against cluster leaders.
    """
    if not embeddings:
        return []

    # Convert to matrix and normalize rows in one go
    embs = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    normed = embs / norms

    clusters: List[List[int]] = []
    # Leaders will store the normalized embedding of the first element in each cluster
    leaders: List[np.ndarray] = []
    
    for idx in range(len(normed)):
        current = normed[idx]
        
        if not leaders:
            leaders.append(current)
            clusters.append([idx])
            continue
            
        # Compute cosine distances to all current cluster leaders at once
        # dist = 1 - dot_product
        dots = np.dot(leaders, current)
        dists = 1.0 - dots
        
        best_cluster_idx = np.argmin(dists)
        if dists[best_cluster_idx] < threshold:
            clusters[best_cluster_idx].append(idx)
        else:
            leaders.append(current)
            clusters.append([idx])

    return clusters


def find_best_match(query_emb: np.ndarray,
                     known_embeddings: List[Tuple[int, np.ndarray]],
                     threshold: float = SIMILARITY_THRESHOLD
                     ) -> Optional[int]:
    """
    Vectorized matching against known persons.
    """
    if not known_embeddings:
        return None

    # Normalise query
    qn = query_emb / (np.linalg.norm(query_emb) or 1.0)
    
    # Extract IDs and normalized embeddings
    person_ids = []
    embs = []
    for pid, emb in known_embeddings:
        person_ids.append(pid)
        # Normalize if not already
        n = np.linalg.norm(emb)
        embs.append(emb / n if n > 0 else emb)
        
    embs_matrix = np.array(embs, dtype=np.float32)
    
    # Compute all distances at once
    dots = np.dot(embs_matrix, qn)
    dists = 1.0 - dots
    
    best_idx = np.argmin(dists)
    if dists[best_idx] < threshold:
        return person_ids[best_idx]
        
    return None
