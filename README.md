# GGSGG
Glocal Graph Convolutional Network for Scene Graph Generation 

| Ablation(Relational Filtering) |motif R@20| R@50 | R@100 | imp R@20 | R@50 | R@100 |
|:------------------------------:|:--------:|:----:|:-----:|:--------:|:----:|:-----:|
|Baseline(nxn)                   |          |      |       |          |      |       |
|Baseline + vis                  |          |      |       |          |      |       |
|Baseline + vis + word           |          |      |       |          |      |       |
|Baseline + vis + word + pos     |          |      |       |          |      |       |
|Baseline + vis + word + pos     |          |      |       |          |      |       |

| Ablation(Graph Mask Attention) |motif R@20| R@50 | R@100 | imp R@20 | R@50 | R@100 |
|:------------------------------:|:--------:|:----:|:-----:|:--------:|:----:|:-----:|
|Baseline                        |          |      |       |          |      |       |
|Baseline + Attention            |          |      |       |          |      |       |
|Baseline + Object Mask Attention|          |      |       |          |      |       |
|Baseline + Graph Mask Attention |          |      |       |          |      |       |

- Baseline + Attention(Learn attention on Union box)
- Baseline + Object Mask Attention(Make biased to attend on objects)
- Baseline + Graph Mask Attention(Multi-head attention on each instanes(subj, obj, background))

| Ablation(Bottom-Up Context Aggregation)     |motif R@20| R@50 | R@100 | imp R@20 | R@50 | R@100 |
|:-------------------------------------------:|:--------:|:----:|:-----:|:--------:|:----:|:-----:|
|Baseline                                     |          |      |       |          |      |       |
|Baseline + avgpool sep -> embed sep -> BUCA  |          |      |       |          |      |       |
|Baseline + maxpool sep -> embed sep -> BUCA  |          |      |       |          |      |       |
|Baseline + conv sep -> embed sep -> BUCA     |          |      |       |          |      |       |

- Baseline: GMA + sum together + avgpool -> embed together
- Baseline + GMA + avgpool separately -> embed separately -> BUCA
- Baseline + GMA + maxpool separately -> embed separately -> BUCA
- Baseline + GMA + conv separately -> embed separately -> BUCA
