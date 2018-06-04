
# Priority List

## Support arbitrary programs

    1. The lucius IR is a turing complete program representation that combines bulk operations
       over multidimensional tensors with a RISC virtual instruction set.

    2. Support for high level programming features including functions and control flow.

    2. All programs are fully differentiable.

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

## Serialization

    1. Support whole training experiment serialization/deserialization.

    2. Leverage ONNX for serializing as much of the compute graph as possible.

## Benchmarks/Applications

    1. Focus on a conforming implementation of MLPerf closed division.

## Licensing

    1. We use the Apache 2.0 License for all source code in Lucius.  No support is provided.

## Hiring

    1. If want to hire the develoment team for Lucius for support or continued development, you
       can acquire the development team with unrestricted ownership of all generated IP for
       $6.5 million USD for a 3 year contract.



