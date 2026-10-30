# Adjacent MID configuration; no GPU admission

The unchanged whole-row norm device body compiles under retained MID flags:NW8,TINY1,WS1,SPLIT1 and FAST1=2. Parent LUT/PRMT use148/150registers;norm uses154/152. All four compile without stack/spills. No Python payload dispatch was extended and no GPU job used this configuration.

The tiny-only check34a6e66a subsequently loses all paired comparisons. That negative measurement supersedes an inference from clean compiler output. Do not spend another check merely moving the same row-scan mechanism to M9..16. Routing/job ownership,resource capacity and a distinct exposed bottleneck would need evidence before reopening this extension.
