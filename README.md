# Cut, Attend and Connect
Learning Predicate as Interaction for Scene Graph Generation 

| Ablation(Pair Filtering)          |const R@20| R@50 | R@100 |uncon R@20| R@50 | R@100 |
|:---------------------------------:|:--------:|:----:|:-----:|:--------:|:----:|:-----:|
|Baseline(nxn)                      |  0.149   | 0.191| 0.216 |  0.143   | 0.184| 0.208 |
|Baseline + vis                     |          |      |       |          |      |       |
|Baseline + vis + word              |          |      |       |          |      |       |
|Baseline + vis + word + pos        |          |      |       |          |      |       |
|Baseline + vis + word + pos + dist |          |      |       |          |      |       |

| Ablation(Triple Mask Attention)  |const R@20| R@50 | R@100 |uncon R@20| R@50 | R@100 |
|:--------------------------------:|:--------:|:----:|:-----:|:--------:|:----:|:-----:|
|Separate box                      |  0.117   | 0.143| 0.157 |     -    |  -   |   -   |
|Separate box + Attention          |          |      |       |          |      |       |
|Union box                         |  0.179   | 0.229| 0.259 |     -    |  -   |   -   |
|Union box + Attention             |  0.187   | 0.236| 0.264 |     -    |  -   |   -   |
|Union box + Object Mask Attention |  0.179   | 0.228| 0.256 |     -    |  -   |   -   |
|Union box + Triple Mask Attention |  0.192	  | 0.239| 0.263 |   0.197  | 0.263| 0.308 |

- Baseline + Attention(Learn attention on Union box)
- Baseline + Object Mask Attention(Make biased to attend on objects)
- Baseline + Triple Mask Attention(Multi-head attention on each instanes(subj, obj, background))

| Ablation(Bottom-Up Context Aggregation)     |const R@20| R@50 | R@100 |uncon R@20| R@50 | R@100 |
|:-------------------------------------------:|:--------:|:----:|:-----:|:--------:|:----:|:-----:|
|Baseline                                     |          |      |       |          |      |       |
|Baseline + non-local                         |  0.184	 | 0.232|	0.256 |   0.194  | 0.261|  0.308|
|Baseline + avgpool sep -> embed sep -> BUCA  |          |      |       |          |      |       |
|Baseline + maxpool sep -> embed sep -> BUCA  |          |      |       |          |      |       |
|Baseline + conv sep -> embed sep -> BUCA     |          |      |       |          |      |       |

- Baseline: TMA + sum together + avgpool -> embed together
- Baseline + TMA + avgpool separately -> embed separately -> BUCA
- Baseline + TMA + maxpool separately -> embed separately -> BUCA
- Baseline + TMA + conv separately -> embed separately -> BUCA
