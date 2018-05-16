
# Priority List

## Memory Efficient Training

    1. Checkpoints are automatically inserted on the SSA formed IR to fit the current system's
       memory capacity.

    2. Operations are maximally split without introducing excessive overhead to expose
       opportunities for checkpoints.

    3. Loops are unrolled using runtime shape information to allow checkpoint analysis.

    4. Minibatches can be computed in parallel or serially to decouple the algorithm from
       memory uage.

    5. Memory is allocated statically using lifetime analysis, buffers are reused when possible.

## First Class Model Parallelism

    1. Leverage distributed GEMM and distributed convolution model parallelism when mapped
       to a multi-processor system.

    2. Leverage layer-wise model parallelism when RNN cells are mapped to a multi-processor
       system.





