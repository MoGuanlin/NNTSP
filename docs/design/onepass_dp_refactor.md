# 1-Pass DP Refactoring Design Document

**Date**: 2026-03-25
**Status**: Active

## 1. Goal

Move the engineering implementation closer to the paper's main algorithm (Section 3,
Figure 2) by introducing a **1-pass bottom-up DP** that materializes cost tables at
each quadtree node, uses a **sigma-conditioned decoder** to predict child state
4-tuples, and employs **PARSE + VERIFYTUPLE + exact fallback** to guarantee feasible
cost table entries.

## 2. Adjusted Plan (vs. original paper)

| Aspect | Paper (Section 3) | Our Plan |
|--------|-------------------|----------|
| **Encoder input** | Geom(B) + C_B (cost table) | Geom(B) + child latents only (no cost table) |
| **Decoder** | sigma-conditioned, predicts child 4-tuple | Same |
| **PARSE** | Appendix A.1 design | Our own implementation (see Section 5) |
| **VERIFYTUPLE** | Algorithm 1 (Appendix E) | Our own implementation (see Section 4) |
| **Leaf solver** | Exact brute-force | Exact solver on small subproblems |
| **Fallback** | Exact per-entry enumeration | Same |
| **Training** | 2-pass schedule with BCE (unchanged) | Phase 1: same; Phase 2: per-sigma supervision |

**Rationale for keeping encoder without cost table**: The encoder's job is to compress
the child subproblem into a fixed-dimensional latent. Adding the cost table as input
would increase complexity without clear empirical benefit at this stage, since the
decoder + PARSE + exact lookup already provide the formal guarantee. We can add cost
table conditioning later if experiments show it helps.

## 3. Architecture Overview

### 3.1 Bottom-up 1-pass (inference)

```
for depth d = max_depth down to 0:
  for each box B at depth d:
    if B is leaf:
      C_B = leaf_exact_solve(B, points, spanner)   # exact cost table
      h_B = LeafEncoder(Geom(B), points)            # latent (unchanged)
    else:
      h_B = MergeEncoder(Geom(B), {h_{B_i}})        # latent (unchanged)
      C_B = {}
      parent_memory = build_memory(h_B, {h_{B_i}}, Geom(B))  # built ONCE
      for each sigma in valid_states(B):
        tau_tilde = Decoder(sigma, parent_memory)    # continuous child 4-tuple
        tau = PARSE(tau_tilde, B, sigma)             # discretize
        if VERIFYTUPLE(B, sigma, tau):
          C_B[sigma] = sum(C_{B_i}[tau[i]] for i in 1..4)  # exact lookup
        else:
          for tau_k in PARSE_TOPK(tau_tilde, K=5):
            if VERIFYTUPLE(B, sigma, tau_k):
              C_B[sigma] = sum(C_{B_i}[tau_k[i]] for i in 1..4)
              break
          else:
            C_B[sigma] = exact_fallback(B, sigma, {C_{B_i}})
```

### 3.2 Training (initial: unchanged 2-pass)

Training stays as-is initially (2-pass with BCE on crossing/iface logits). In a later
phase, we add per-sigma teacher labels derived from exact DP on teacher tours.

### 3.3 sigma-conditioned decoder

The existing `TopDownDecoderModule` is designed for the 2-pass top-down propagation.
For the 1-pass DP, we need a new **MergeDecoder** that:

- Input: sigma embedding + parent memory (self-attended tokens + child latents)
- Output: continuous child state 4-tuple (soft activation scores per child)

The parent memory is built once per box and shared across all sigma queries.

```
sigma_emb = StateEmbedding(sigma_idx)     # [d_model]
   or
sigma_emb = MLP(concat(a_vec, mate_vec))  # [d_model]

parent_memory = SelfAttn(CLS + IFACE + CROSS + CHILD_LATENT tokens)  # [T, d]

for sigma in valid_states:
  sigma_out = CrossAttn(query=sigma_emb, kv=parent_memory)
  child_activations = ChildHead(sigma_out)  # [4, num_iface_slots] soft scores
```

## 4. VERIFYTUPLE Implementation

### 4.1 Correspondence Maps

Built at runtime from packed token data using edge IDs (eid):

**phi_out (parent iface -> child iface on outer boundary):**
- For each parent interface token (node_id=P, eid=e, boundary_dir=d):
  - Scan children of P for an interface token with the same eid=e
  - The matching child interface is the corresponding slot
- Stored as: `phi_out[parent_iface_idx] -> (child_quad, child_iface_idx)`

**phi_sh (child <-> child on shared internal boundary):**
- For each parent crossing token (node_id=P, eid=e, child_pair=(i,j)):
  - Child i has an interface with eid=e (on side facing j)
  - Child j has an interface with eid=e (on side facing i)
- Stored as: `phi_sh[child_i_iface_idx] -> child_j_iface_idx` (and vice versa)

**Key insight**: The `eid` field uniquely identifies each spanner edge across all tree
nodes. The same edge appears as a crossing at the LCA and as interfaces along the
path from LCA to leaves. So eid-based matching is exact and efficient.

### 4.2 VERIFYTUPLE Algorithm

Given parent state sigma=(a, mate) and child states sigma_{B_i}=(a^(i), mate^(i)):

```
def verify_tuple(sigma, child_sigmas, phi_out, phi_sh):
    # C1: Outer-boundary activation agreement
    for each parent slot p where phi_out[p] = (child_i, child_slot_q):
        if a[p] != a^(i)[q]:
            return False

    # C2: Shared-boundary activation agreement
    for each adjacent pair (i,j) and shared slot p in phi_sh:
        q = phi_sh[(i,p)]  # corresponding slot in child j
        if a^(i)[p] != a^(j)[q]:
            return False

    # C3+C4: Connectivity composition + no internal cycles
    # Build the gluing graph G_glue and trace
    G_glue = build_glue_graph(child_sigmas, phi_sh)
    induced_mate = trace_all_paths(G_glue, phi_out, sigma)
    if induced_mate is None:  # cycle detected or unreachable
        return False
    if induced_mate != sigma.mate:
        return False

    return True
```

The gluing graph is built on child-side active interface slots:
- **Mate edges** (inside children): (i,p) -- (i, mate^(i)(p))
- **Glue edges** (across shared boundaries): (i,p) -- (j, phi_sh(p))

Tracing from each outer-boundary active parent slot through alternating mate/glue
edges must reach another outer-boundary slot without forming a closed cycle.

## 5. PARSE Implementation

PARSE converts continuous child state predictions to discrete states.

### 5.1 Overview

Input: `tau_tilde = (sigma_tilde_{B_1}, ..., sigma_tilde_{B_4})`
where each `sigma_tilde_{B_i}` is a vector of soft activation scores over child i's
interface slots (output of sigmoid).

Output: `tau = (sigma_{B_1}, ..., sigma_{B_4})` where each sigma_{B_i} is a valid
discrete boundary state in Omega(B_i).

### 5.2 Two-step process

**Step 1: Portal selection (activation rounding)**

For each child i, round the continuous activations to binary:
1. Per-side budget: on each side s of child i, keep at most r activations
   (take top-r by score, threshold at 0.5 for the rest)
2. Parity enforcement: ensure even number of active slots per child
   (flip the lowest-confidence active slot if odd)
3. Consistency enforcement: for shared boundaries between siblings,
   activations must agree. Take the average score and round jointly.

**Step 2: Pairing inference (connectivity decoding)**

Given the rounded activations, determine the noncrossing pairing for each child:
1. For each child i, collect active slots in clockwise order
2. Find the minimum-cost noncrossing perfect matching on these active slots
   - Cost derived from the continuous scores: lower score = less preferred pairing
   - This is a small DP (O(k^3) where k = number of active slots, k <= 4r)
3. Assemble the full discrete state sigma_{B_i} = (a^(i), mate^(i))

### 5.3 Top-K variant

PARSE can also emit top-K candidate tuples by:
- Keeping multiple activation roundings (vary threshold)
- For each, emit the best pairing
- Return K distinct feasible tuples sorted by estimated cost

## 6. Leaf Exact Solver

For leaf boxes containing <= max_points_per_leaf points (typically 4):

1. Enumerate all valid boundary states sigma in Omega(B_leaf)
2. For each sigma:
   - The active crossings define path endpoints on dB
   - The pairing defines which endpoints are connected
   - Find the shortest collection of paths inside B that:
     (a) visits all points in B
     (b) has endpoints matching sigma's active sites
     (c) paths are paired according to sigma's mate map
   - This is a small constrained TSP variant, solvable exactly for <=4 points
3. Store C_leaf[sigma] = cost (or +inf if infeasible)

For very small subproblems (<=4 points), we enumerate all permutations.

## 7. Phase Plan

### Phase 1: Infrastructure (current)
- [x] Design document
- [ ] `src/models/dp_core.py`: correspondence maps, VERIFYTUPLE, PARSE
- [ ] Leaf exact solver
- [ ] Unit tests

### Phase 2: sigma-conditioned decoder
- [ ] `src/models/merge_decoder.py`: new decoder module
- [ ] Integration with BottomUpTreeRunner for 1-pass inference
- [ ] Cost table data structures

### Phase 3: Full DP pipeline
- [ ] 1-pass runner with cost table materialization
- [ ] Traceback for tour reconstruction
- [ ] Fallback mechanism

### Phase 4: Training enhancement
- [ ] Per-sigma teacher labels from exact DP on teacher tours
- [ ] Training loop with per-sigma supervision
- [ ] Experiments comparing 1-pass vs 2-pass

## 8. Key Data Structures

```python
@dataclass
class CostTable:
    """Cost table for a single quadtree box."""
    costs: Tensor       # [S] float, +inf for infeasible states
    backptr: Tensor     # [S, 4] long, child state indices (-1 if N/A)
    valid: Tensor       # [S] bool

@dataclass
class CorrespondenceMaps:
    """Parent-child interface correspondence for one internal node."""
    # phi_out: parent outer iface -> (child_quad, child_iface_idx)
    phi_out_child: Tensor    # [Ti] long, which child (0-3, or -1)
    phi_out_slot: Tensor     # [Ti] long, child's iface slot idx (-1 if N/A)
    # phi_sh: for each (child_i, child_j) pair sharing a boundary,
    #   maps child_i's iface slot -> child_j's iface slot
    phi_sh_peer_child: Tensor   # [4, Ti] long, peer child index (-1 if N/A)
    phi_sh_peer_slot: Tensor    # [4, Ti] long, peer's iface slot (-1 if N/A)
```
