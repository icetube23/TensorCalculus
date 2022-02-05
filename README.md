# TensorCalculus

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://icetube23.github.io/TensorCalculus.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://icetube23.github.io/TensorCalculus.jl/dev)
[![Build Status](https://github.com/icetube23/TensorCalculus.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/icetube23/TensorCalculus.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/icetube23/TensorCalculus.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/icetube23/TensorCalculus.jl)
[![Concept – Minimal or no implementation has been done yet, or the repository is only intended to be a limited example, demo, or proof-of-concept.](https://www.repostatus.org/badges/latest/concept.svg)](https://www.repostatus.org/#concept)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

TensorCalculus is a Julia package which provides a `Tensor` wrapper around native Julia arrays. It allows for many common algebraic tensor operations (e.g., summation, different tensor products). Additionally, TensorCalculus also supports tensor fields (i.e., tensor-valued functions of space) and well-known differential operations (e.g., gradients, divergence, rotation, etc.) from mathematics and physics.
<p align="center">
<img width="400px" src="https://user-images.githubusercontent.com/34234056/150592559-52d797ce-dd6e-4f2c-8b57-ea273e3285b5.svg"/><br>
<sub><sub>(Illustration by Arian Kriesch, corrections made by Xmaster1123 and Luxo - Creative work, CC BY 2.5, https://commons.wikimedia.org/w/index.php?curid=651286)</sub></sub>
</p>

## Installation

This package is still in an provisional state and thus not yet installable via the Julia package manager. As soon as it reaches a useable state, it will be added to the general registry. If, for whatever reason, you still wish to install this package at its current state, you can run the following in the Pkg prompt:
```julia
pkg> add git@github.com:icetube23/TensorCalculus.jl.git
```
or depending on your favorite GitHub protocol, you can also use:
```julia
pkg> add https://github.com/icetube23/TensorCalculus.jl.git
```

## References

The inspiration for this package comes from the German textbook "Tensoranalysis" written by H. Schade and K. Neemann. As such, most definitions in this package are either directly taken from or heavily influenced by said textbook.
