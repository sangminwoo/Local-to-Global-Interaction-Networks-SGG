# Cut, Split and Interact
Learning Predicate as Interaction for Scene Graph Generation 

| Ablation(Relation Pruning)        |const R@20| R@50 | R@100 |uncon R@20| R@50 | R@100 |
|:---------------------------------:|:--------:|:----:|:-----:|:--------:|:----:|:-----:|
|Baseline(nxn)                      |  0.149   | 0.191| 0.216 |  0.143   | 0.184| 0.208 |
|Baseline + vis                     |          |      |       |          |      |       |
|Baseline + vis + semantic          |          |      |       |          |      |       |
|Baseline + vis + semantic + spatial|          |      |       |          |      |       |
|Baseline + vis + semantic + spatial|          |      |       |          |      |       |

| Ablation(Instance Mask Attention)           | const R@20 | R@50 | R@100 |uncon R@20| R@50 | R@100 |
|:-------------------------------------------:|:----------:|:----:|:-----:|:--------:|:----:|:-----:|
|Separate Box                                 |    0.117   | 0.143| 0.157 |     -    |  -   |   -   |
|Separate Box + Att                           |            |      |       |          |      |       |
|UBox                                         |    0.179   | 0.229| 0.259 |     -    |  -   |   -   |
|UBox, Whole Att                              |    0.187   | 0.236| 0.264 |     -    |  -   |   -   |
|UBox, Object M + Att                         |    0.179   | 0.228| 0.256 |     -    |  -   |   -   |
|UBox, Instance M + Att, Union                |    0.192 	 | 0.239| 0.263 |   0.197  | 0.263| 0.308 |
|UBox, Instance M + Att, Interact, GT Box     |    0.191   | 0.234| 0.255 |     -    |  -   |   -   |
|UBox, Instance M + Att, Interact, No GT Box  |    0.190   | 0.235| 0.256 |   0.199  | 0.265| 0.311 |

- Baseline + Attention(Learn attention on Union box)
- Baseline + Object Mask Attention(Make biased to attend on objects)
- Baseline + Instance Mask Attention(Multi-head attention on each instanes(subj, obj, background))

| Ablation(Bottom-Up Context Aggregation)     |const R@20| R@50 | R@100 |uncon R@20| R@50 | R@100 |
|:-------------------------------------------:|:--------:|:----:|:-----:|:--------:|:----:|:-----:|
|Baseline                                     |          |      |       |          |      |       |
|Baseline + non-local                         |  0.184	 | 0.232|	0.256 |   0.194  | 0.261|  0.308|
|Baseline + avgpool sep -> embed sep -> BUCA  |          |      |       |          |      |       |
|Baseline + maxpool sep -> embed sep -> BUCA  |          |      |       |          |      |       |
|Baseline + conv sep -> embed sep -> BUCA     |          |      |       |          |      |       |

- Baseline: IMA + sum together + avgpool -> embed together
- Baseline + IMA + avgpool separately -> embed separately -> BUCA
- Baseline + IMA + maxpool separately -> embed separately -> BUCA
- Baseline + IMA + conv separately -> embed separately -> BUCA
