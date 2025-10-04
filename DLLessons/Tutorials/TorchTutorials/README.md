# ![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## Pytorch Foundations & Internal build system::
- core concepts: Matrics, Tensors, Variables
- Numpy <-> Pytorch Interoperability, Shapes, Axes, Rank
- Data Handling: Dataset and Dataloaders
- Basic Training Loop: Forward Method, Training Pipeline
- Detailed Overview of Pytorch's front-end (py) and backend (c++) layers
- understanding *ATen*, the dispatcher, and the role of Torchscript
- Navigating the pytorch source repo and build system
- How pytorch integrated with third-party libraries (e.g., MKL, cuBLAS, cuDNN)


## Intermediate Concepts & Internal Graph Representation:
- Pytorch classes, Modules and Containers
- Built-in Layers, Activation Functions and Custom Layers
- Under the Hood:: Pytorch Internals
- Distance Metrics, Basic Loss Functionsm Utility Functions
- Profiling Layers, MACs/FLOPs Calculations and Memory Usage
- Deep dive into Pytorch's autograd engine internals
- How computation graph are build and executed
- custom autograd functions: writing and registering backward passes
- debugging gradients, hooks, and in-place operations under the hood


## Advanced Mechanics & Extending Pytorch with custom operators & kernels
- CNN Algo Implementation and Internals
- Autograd Mechanics and Custom Gradients
- Dynamic Computation **Graph Construction**
- Intro to memory Management best practices
- creating and registering custom C++/CUDA ops in pytorch
- understanding the dispatch mechanism and code generation
- writing high-performance kernels, memory-efficient operation and briding python/C++
- testing, benchmarking ad profiling custom ops


## Advanced Tools & Techniques, performance, Memory and Distributed Internals
- Optimizers ( SGD, Adam )
- Custom Dataloaders for complex scenarios
- tensorboard Integration and advanced logging
- Half precision Training and Other performance enchancements
- Analyzing memory management and format (e.g., channels_last) in pytorch
- internals of distributed training: RPC, process groups and parameter server architecture
- Advanced JIT and FX transformations, graph optimizations and quantization internals
- Using internal tools and APIs to diagnose bottlenecks and improve performance

## Specialized Loss Function  & beyond
- Advanced Loss functions: GAN, KL Divergence, Focal, IoU
- Embedding, perceptual Losses, CTC , Triplet and Dice loss
- Integrating Specialized Loss function into your Models


## Contributing to Pytorch & Maintaining Quality
- understanding pytorch's dev workflow and contributor guidelines
- How to write unit tests, docs and follows coding standards
- reviewing ongoing proposals, handling pull requests and long-term maintenance practise
- building sustainable, maintainable features that align with pytorch roadmap


