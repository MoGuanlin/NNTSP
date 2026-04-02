# Official Reviews of Submission 32503

**Paper:** Size-Independent Neural Dynamic Programming for Euclidean TSP with $(1+\varepsilon)$-Approximation Guarantee

**Authors:** Guanlin Mo, Lei Chen, Shaofeng H.-C. Jiang, Hu Ding

---

## Review 1 — Reviewer H8Zv

**Date:** 13 Mar 2026, 12:26 (modified: 24 Mar 2026, 22:51)

### Summary

This paper studies the Euclidean TSP problem, one of the fundamental and classical problems in combinatorial optimization. Because of the nice geometric structure of the problem, it admits a PTAS, i.e., a $(1+\varepsilon)$-approximation algorithm that runs in $O(n(\log n)^{O(1/\varepsilon)})$ time. Note that the time complexity is exponential in $1/\varepsilon$.

The underlying algorithm is a dynamic programming procedure that recursively partitions the space into small subparts and constructs a solution in a bottom-up manner. The subproblems of the dynamic program can be solved and merged in time exponential in $1/\varepsilon$.

The core idea of this paper is to learn the merge primitive in the dynamic program rather than calling an expensive algorithm. The PTAS backbone is preserved to ensure end-to-end certification, while only the merge primitive is learned in order to accelerate the algorithm. The fact that the neural merge primitive is independent of $n$ is not surprising, since in the dynamic programming algorithm the merge routine deals with constant-size subinstances (see Lemma 2.2). Therefore, it suffices to learn a merge primitive for fixed-size subinstances.

### Strengths

The idea of infusing a neural network (NN) module into classical algorithms with worst-case guarantees in order to accelerate computation is interesting. To the best of my knowledge, the only prior works that achieve this in the PTAS setting are the NN-Baker paper for the MIS problem and NN-Steiner for the Steiner tree problem. This work represents another step toward establishing NN-infused algorithms with theoretical guarantees. Thus, the problem is well motivated and of significant interest.

Their experimental results appear very strong; however, I have some reservations and clarification questions regarding the assumptions. Another positive aspect is the worst-case approximation guarantee, although it relies on a rather strong assumption.

### Weaknesses

Some parts of the writing could be improved. For instance, it would be helpful to provide a short description of the spanner construction. The experimental section also requires clarification on several points, some of which are listed below. Moreover, the assumptions required for the theoretical results appear rather strong (see below), and the proofs do not seem to involve substantial mathematical novelty. In terms of algorithmic novelty and pipeline design, I would not rank the paper among the strongest, as it is very similar to the NN-Steiner NN-Infused algorithmic framework.

**Soundness:** 2: fair
**Presentation:** 3: good
**Significance:** 3: good
**Originality:** 2: fair

### Key Questions For Authors

- Could you please comment on and justify the assumption $\delta = \varepsilon/n$?
- In Remark 3.4 it is mentioned that "... error does not grow noticeably with n ...". However, $\varepsilon/n$ suggests that the error should become smaller and go to zero as $n$ increases. Could you comment on this?
- Could you please further comment on and elaborate on Lines 349–351?
- Please explain how the NN is used to help LKH.
- For the correctness of Lemma 3.2, please explain why it makes sense to define the error in that way. What is unclear to me is that the right-hand side counts the number of side nodes, and it is not clear how this relates to the error.
- There are two unclear points for me regarding training:
  1. Could you explain this part of the training in more detail: "We aggregate these scores into sparse edge weights on the graph $S'$, and use them as guidance for a downstream heuristic solver to construct a warm-start tour and to bias the candidate edges explored by LKH"?
  2. "We use high-quality teacher tours on small instances: for each visited box we label which boundary crossings are realized by the teacher, and minimize a masked binary cross-entropy loss under free-running conditioning to match the inference schedule."
- Given the assumption on $\delta$ and the definition of the error in Lemma 3.2, the proofs seem rather straightforward. Could you comment on the theoretical novelty or on any specific obstacle in the proof that had to be overcome?
- Could you please clarify the connection between training and inference? Is it correct to say that during training LKH is used to label small instances for supervised learning? Is it also correct that during inference the NN candidates are used to guide the alpha-values in LKH, as mentioned in the appendix? I find the connection unclear and would appreciate clarification as to why the training procedure should help inference.
- In the appendix it is mentioned that the training is done on synthetic instances of size 50. How large is the dataset?

### Limitations

yes

### Overall Recommendation

3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Compliance With LLM Reviewing Policy:** Affirmed.
**Code Of Conduct Acknowledgement:** Affirmed.

---

## Review 2 — Reviewer qTM9

**Date:** 13 Mar 2026, 00:20 (modified: 24 Mar 2026, 22:51)

### Summary

This paper proposes a size-independent neural dynamic programming framework for TSP. The key idea is to learn a neural module that approximates the dynamic programming merge operation used in the PTAS algorithm based on quadtree decomposition. Because the DP state space depends only on the approximation parameter rather than the problem size, the learned module can generalize across different instance sizes. The authors show that if the learned module achieves sufficient accuracy, the overall algorithm still maintains a $(1+\varepsilon)$ approximation guarantee, while significantly accelerating the DP computation.

### Strengths And Weaknesses

The proposed method is well designed and integrates a learned module into the classical PTAS dynamic programming framework to approximate the merge operation, enabling computational acceleration while preserving theoretical approximation guarantees. The overall approach provides an interesting way to combine traditional algorithmic structures with neural models. Empirically, the method demonstrates promising performance, achieving notable speedups in dynamic programming while maintaining good solution quality.

However, the scale of the experimental instances is still relatively limited, leaving open the question of how well the approach would perform on truly large-scale TSP instances. Also, the paper mainly focuses on inference-time acceleration, while providing limited discussion of the training cost and overall computational overhead of the learning component.

**Soundness:** 3: good
**Presentation:** 2: fair
**Significance:** 2: fair
**Originality:** 3: good

### Key Questions For Authors

1. The paper mainly focuses on accelerating the dynamic programming merge step during inference. Could the authors provide more details about the training cost of the proposed neural module, such as training time, data generation cost, and how the training overhead compares with the inference-time speedup?
2. The method is demonstrated within the PTAS framework for the TSP. To what extent can the proposed approach generalize to other dynamic programming problems?

### Limitations

The paper mainly focuses on the algorithmic and theoretical aspects of the proposed method and does not explicitly discuss potential limitations or broader societal impacts. It would be helpful if the authors briefly discussed the limitations of the approach.

### Overall Recommendation

3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.

**Confidence:** 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Compliance With LLM Reviewing Policy:** Affirmed.
**Code Of Conduct Acknowledgement:** Affirmed.

---

## Review 3 — Reviewer x5Nh

**Date:** 11 Mar 2026, 17:10 (modified: 24 Mar 2026, 22:51)

### Summary

This paper focuses on size-independent TSP solver. In specifics, authors adopt the encoder-decoder framework in which cases encoder is utilized to process child solutions and decoder is utilized for merging child solutions to formalize the parent solutions iteratively. The authors also give theoretical analysis to show that the merged solutions achieve near-optimal solutions. The experimental parts also validate the utility of proposed method for large-scale TSP in both synthetic and real-world scenarios.

### Strengths And Weaknesses

**Strengths:**

1. Paper is written in a clear manner. Motivation is well formulated.
2. The architecture is simple but it is very effective and powerful. Previous neural solvers work for TSP struggle at size generalization due to overfitting of size seen during training or can't even test on large-scale TSP instances due to the complexity of self-attention module. This paper trains model on 50 and can generalize to TSP with 89k nodes. This is very impressive.
3. Experiments focus on real-world TSP instances like TSP-LIB. Performances are good and can even surpass some strong baselines. This further supports paper's claimed conclusion/contribution.

**Weaknesses:**

It seems that authors ignore some highly relevant papers in references. For instances, [1,2,3] also adopt divide-and-conquer framework to generalize neural solver from small instances to large-scale instances (even with 100k nodes [2]). I suggest authors add these references into discussions. They are very relevant.

[1] Li, Ke, et al. "Destroy and repair using hyper-graphs for routing." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 17. 2025.
[2] Luo, Fu, et al. "Boosting neural combinatorial optimization for large-scale vehicle routing problems." The Thirteenth International Conference on Learning Representations. 2025.
[3] Zheng, Zhi, et al. "UDC: A unified neural divide-and-conquer framework for large-scale combinatorial optimization problems." Advances in Neural Information Processing Systems 37 (2024): 6081-6125.

**Soundness:** 3: good
**Presentation:** 3: good
**Significance:** 3: good
**Originality:** 3: good

### Key Questions For Authors

1. As said before, authors may neglect some relevant references. I hope authors can add them into discussions and clarify the fundamental differences with them.
2. I am curious about whether your proposed framework can be extended into somewhat complex tasks like CVRP [1]? If it has some potential challenges in theory or practice, can you briefly discuss them?
3. Since [1] also tests neural solvers on large-scale TSP instances. Besides, a more related training paradigm is in [2] where solvers are trained on small-size instances (with 100 nodes) and generalize into very large problems (with 100k nodes). I am curious about whether you can compare your methods' performances with [2] on TSP under the same setting (I notice that your model is trained on instances with n=50, how about training it on n=100 and then compare generalization abilities with [2] on TSP)?
4. It seems that authors apply self-attention in encoder module, but we all know that self-attention blocks are really heavy. Could you add more descriptions and clarifications for this part? How does your light-weight attention module relieve the burden concretely?

[1] Luo, Fu, et al. "Boosting neural combinatorial optimization for large-scale vehicle routing problems." The Thirteenth International Conference on Learning Representations. 2025.
[2] Luo, Fu, et al. "Learning to Insert for Constructive Neural Vehicle Routing Solver." The Thirty-ninth Annual Conference on Neural Information Processing Systems. 2025.

### Limitations

This paper focuses on establishing solvers for TSP, but nowadays people care more about more complex tasks like CVRP, which doesn't appear in this paper. I hope authors can extend their frameworks into more complex settings in the future.

### Overall Recommendation

4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Compliance With LLM Reviewing Policy:** Affirmed.
**Code Of Conduct Acknowledgement:** Affirmed.

---

## Review 4 — Reviewer si9p

**Date:** 08 Mar 2026, 13:50 (modified: 24 Mar 2026, 22:51)

### Summary

This paper proposes a Euclidean TSP solver that combines neural networks with the classical Arora/Rao-style PTAS framework. The core idea is to retain the PTAS backbone to provide end-to-end approximation guarantees, while using a neural network to learn only the computationally expensive "merge primitive" in the dynamic programming (DP) routine. Because the r-light restriction bounds the boundary state space size to depend only on $\varepsilon$ and not on the instance size n, the merge module achieves size independence and can be reused across different scales. Experiments demonstrate that models trained solely on n=50 instances can generalize to TSPLIB instances up to n=85900, achieving a 25x to 270x speedup when integrated with LKH-3.

### Strengths

- The authors accurately identify the combinatorial explosion in the DP merge step as the primary bottleneck preventing PTAS methods from being practical. The principled approach of "learning only the bottleneck subroutine while preserving the algorithmic backbone" is methodologically significant. This concept is not only applicable to TSP but also holds potential for other problems relying on hierarchical DP-based PTAS frameworks.
- By leveraging the r-light property, the boundary interaction for each box is limited to $O(r)$ crossing sites, rendering the DP state space size $r^{O(r)}$ strictly independent of n. Furthermore, using scale-normalized geometric features (Geom(B)) allows reasonable parameter sharing across different hierarchical levels. This design fundamentally enables strong size extrapolation.

### Weaknesses

- The paper simultaneously describes a certified DP variant that provides theoretical guarantees (relying on PARSE, feasibility checking, exact table lookups, and fallbacks) and a practical two-pass guidance variant used for large-scale experiments, which explicitly bypasses exact DP evaluation, exact tables, and fallbacks. However, almost all experimental results evaluate the performance of the heuristic guidance combined with LKH-3, rather than the end-to-end performance of the certified DP solver. This creates a significant gap: the paper promises a provable $(1+\varepsilon)$-approximation algorithm implementation, but practically evaluates a strong heuristic module.
- Theorem 3.1 relies on the assumption that every internal merge satisfies a worst-case additive error bound $\delta$. Remark 3.4 directly sets $\delta = \varepsilon/n$. This raises several critical issues. First, for n=85900, $\delta \approx 10^{-5}\varepsilon$, making it highly unrealistic to expect the normalized DP value for every box to reach such strict precision. Second, the network is trained using a masked binary cross-entropy (BCE) loss on boundary crossing decisions, which does not directly translate to the entrywise $\delta$-accuracy required by the theory. Third, while the fallback mechanism guarantees correctness in theory, the practical variant bypasses it. Even in the certified variant, a high fallback frequency would negate any speedup. Finally, Lemma 3.3 bounds the root error as $error(B_{root}) \le 2\varepsilon L$. To conclude a $(1+O(\varepsilon))OPT$ guarantee, a strict relation like $L < c \cdot OPT$ must be established. The paper briefly mentions a "standard preprocessing relation between L and OPT" but lacks a formal statement. If points are clustered in a large bounding square, L could be significantly larger than OPT, breaking the multiplicative bound.
- Stripped of the PTAS theoretical framework, the practical variant's core pipeline is constructing a quadtree and sparse graph, predicting edge scores via NN, generating candidate sets, and calling LKH-3. This is fundamentally similar to existing hybrid methods like NeuroLKH. The uniqueness here is the use of the quadtree hierarchy for predictions. However, the paper lacks clear ablation studies showing how much of the final speedup is due to this hierarchical structure versus the inherent power of the LKH-3 search.
- While the neural merge module is size-independent, other steps in the PTAS backbone (spanner construction, quadtree building, patching) incur costs that grow with n. The paper does not report the time breakdown for these preprocessing steps, leaving it unclear what percentage of the total pipeline time is consumed by preprocessing for an n=85900 instance.
- The experiments use a server with 8x RTX 6000 Ada GPUs. The proposed method clearly leverages GPU acceleration for inference, whereas LKH-3 is primarily a CPU-bound algorithm. The reported "end-to-end wall-clock time" for "Ours+LKH" includes GPU inference plus LKH time, compared against a CPU-only LKH-3 baseline. The 25x-270x speedup should be contextualized under an equal resource budget. Additionally, the TSPLIB reference solutions use LKH objectives reported by DualOpt. Since LKH is heuristic, negative gaps might simply mean the reference LKH wasn't run long enough, complicating the interpretation of the gap metric.

**Soundness:** 2: fair
**Presentation:** 3: good
**Significance:** 2: fair
**Originality:** 2: fair

### Key Questions For Authors

1. Was the certified DP variant actually implemented and tested? If so, please report the scale of instances it can practically run on, the percentage of fallbacks triggered, and the empirical speedup compared to exact merge enumeration. If not, please tone down the engineering claims of "guarantees" in the title and abstract, clarifying that the paper primarily proposes a theoretically viable framework.
2. What is the formal relationship between the $\delta$-accuracy requirement and the training objective? The model is trained using BCE on boundary crossings, while the theory requires entrywise $\delta$-accuracy on the DP cost. Is there a provable link between these two?
3. Could you provide an explicit lemma or citation in the main text explaining exactly how the additive bound $error \le O(\varepsilon)L$ strictly converts to a multiplicative bound of $(1+O(\varepsilon))OPT$?
4. The core approach relies heavily on the Rao-Smith patched spanner $S'$ and its r-light properties. In the practical engineering deployment, what specific type of spanner is used, and what are the implementation details, parameter choices, and complexities for the patching procedure? Please provide reproducible statistics like $|E|$ and the degree distribution for TSPLIB instances to clarify if the speedup stems from the neural guidance or the spanner itself.

### Limitations

- The $(1+\varepsilon)$-guarantee promised in the title only applies to a certified variant that is not experimentally validated.
- The critical theoretical assumption $\delta = \varepsilon/n$ lacks empirical validation.
- There is a lack of ablation on the hierarchical structure's distinct contribution and comparisons with other scalable hybrid NCO methods.
- There is a lack of transparency regarding the scaling overhead of PTAS preprocessing and the resource fairness of GPU vs. CPU wall-clock comparisons.

### Overall Recommendation

2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.

**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Compliance With LLM Reviewing Policy:** Affirmed.
**Code Of Conduct Acknowledgement:** Affirmed.
