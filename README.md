# High-Performance Computing & Data Engineering Project Portfolio

This repository contains a series of technical reports and performance analyses focused on **High-Performance Computing (HPC)**, **Parallel Architectures**, and **Large-Scale Data Processing**. The projects span across shared and distributed memory systems, GPU acceleration, and distributed data frameworks like Dask.

---

## 🚀 Projects Overview

### 1. Distributed Data Processing with Dask & XGBoost
A deep dive into accelerating machine learning and data analysis using distributed scheduling.
* **Model Training**: Implemented `xgboost` classification on sparse datasets, achieving **~84% training accuracy**.
* **Scaling Analysis**: Evaluated training time across 1, 2, 4, and 8 CPUs, identifying speedup trends and actual vs. ideal linear performance.
* **Dask Performance**: Comparative study of Dask's distributed scheduler against standard local execution, maintaining accuracy while managing overhead.
* **Big Data Analysis**: Processed the NYC Flights dataset (336k+ rows), handling missing values and calculating statistical benchmarks (Mean/Std Dev) with distributed partitions.

### 2. Parallel Programming Models (OpenMP)
Exploration of core HPC concepts and CPU-level parallelism using the OpenMP API.
* **Architectural Foundations**: Analysis of the von Neumann architecture, Flynn’s Taxonomy (SISD, SIMD, MISD, MIMD), and the "Stored-Program" concept.


[Image of von Neumann architecture]

* **Performance Metrics**: Practical application of **Amdahl’s Law** (strong scaling) and **Gustafson’s Law** (weak scaling) to evaluate theoretical vs. actual speedup.
* **Memory Models**: Technical breakdown of shared memory (global address space) versus distributed memory (message-passing/MPI) architectures.
* **Implementation**: Step-by-step guide for compiling C-based parallel code using the `gcc -fopenmp` flag and setting environment variables.

### 3. Accelerator Architectures: FSDP & DDP
Analysis of GPU-based acceleration and memory hierarchies in modern supercomputing clusters.
* **CPU vs. GPU**: Comparative study of latency-oriented (CPU) versus throughput-oriented (GPU) architectures.
* **GPU Memory Hierarchy**: Detailed ranking of memory speeds—from the **Register File** (Fastest) and **Shared Memory** to **Global Memory** (Slowest/Largest).
* **Hardware Profiling**: Real-time monitoring of NVIDIA H200 GPUs using `nvidia-smi`, tracking memory bandwidth, power consumption, and compute capability.
* **PyTorch Integration**: Validation of CUDA availability and device IDs within Python-based deep learning environments.

### 4. Multiprocessing & Scaling Anti-Patterns
Experimental study on the efficiency of Python's `multiprocessing` library and the impact of "Parallel Slowdown."
* **Negative Scaling Analysis**: Identified scenarios where adding more CPUs increases execution time due to communication overhead and resource contention.
* **Granularity Tuning**: Optimized `chunk_size` parameters in `imap` and `map` functions to mitigate context switching and improve performance.
* **Super-Linear Speedup**: Observed performance spikes (Efficiency > 100%) in specific DataFrame operations, attributed to distributed cache locality benefits.
* **Resource Utilization**: Technical assessment of node `c3039` on the Discovery Cluster, featuring Intel Xeon E5-2680 CPUs and 125GiB RAM.

---

## 🛠 Tech Stack & Tools
* **Languages**: Python (Dask, PyTorch, XGBoost, Pandas), C (OpenMP).
* **HPC Environments**: Northeastern University **Discovery** and **Explorer** Clusters.
* **Monitoring**: `nvidia-smi`, Dask Performance Dashboard, `lscpu`, `free -h`.
* **Compilers/Schedulers**: GCC, SLURM (`srun`), Conda.

---

## 📊 Key Performance Indicators
| Metric | Formula / Description |
| :--- | :--- |
| **Speedup (S)** | $$S = \frac{T_{serial}}{T_{parallel}}$$ |
| **Efficiency (E)** | $$E = \frac{S}{\text{Number of Cores}}$$ |
| **FLOPS** | Floating-point Operations Per Second |
| **Bandwidth** | Data movement speed between memory and cores |

---
