# enhanced-cpu-fallback-ttbuda-demo

Below is a **mini-project** README that documents the **Partial CPU Fallback** “mock” approach. It **demonstrates** the idea of selectively routing unsupported ops to CPU, which can be **relevant** to Tenstorrent’s TT-Buda or any AI/HPC compiler stack. This type of concept-level demonstration **can** help Tenstorrent engineers – especially if it’s clear on how it would translate to a **real** environment (TT-Buda) where only certain ops run on specialized hardware.

---

# Partial CPU Fallback PoC

This repository demonstrates a **mock** proof-of-concept for **partial CPU fallback** in an AI framework. The approach assigns each operator to either a **“Mock Tenstorrent Device”** (if supported) or **CPU** (if unsupported). The goal is to **only** fallback the exact unsupported ops to CPU—rather than forcing entire blocks or layers to CPU—improving overall efficiency.

Although this code is pure Python and runs in any environment (including Google Colab), the principle **mirrors** how a real compiler (e.g., **TT-Buda**) might selectively partition a computational graph between specialized AI hardware and CPU fallback.

---

## 1. Motivation

**Naive CPU fallback** often pushes **entire layers** or subgraphs to CPU when only a few ops are unsupported. This can be suboptimal if most ops in that layer are perfectly valid for the specialized device.

**Partial fallback** ensures:
- Only truly unsupported ops hit the CPU.  
- Supported ops remain on the device, reducing unnecessary fallback overhead.  
- Minimizes data transfers back-and-forth.

In a **real** environment like **Tenstorrent’s TT-Buda**, partial fallback can significantly **improve throughput** by leveraging specialized hardware for the majority of operations, even if a small subset is unsupported.

---

## 2. Project Structure

```
.
└── partial_cpu_fallback_demo.py   # Self-contained Python script (the mock)
```

Key components in `partial_cpu_fallback_demo.py`:
- **MockOp**: Represents a single operator (name, type, shape).  
- **MockGraph**: A list of ops forming a linear mini-graph.  
- **MockDevice**: “Tenstorrent-like” device that only supports certain op types.  
- **CPUDevice**: Always runs any op (fallback).  
- **partition_graph**: Automatically splits the graph into subgraphs for either the device or CPU.  
- **run_partitioned_graph**: Executes those subgraphs in order, simulating data handoff.

---

## 3. How It Works

1. **Define a mock graph** of operators:  
   ```python
   ops = [
       MockOp("MatMul1", "Matmul", (32, 64)),
       MockOp("LayerNorm1", "LayerNorm", (32, 64)),
       MockOp("Unsupported1", "WeirdOp", (32, 64)),
       ...
   ]
   ```
2. **Create devices**:  
   ```python
   supported_ops = {"Matmul", "Softmax", "LayerNorm"}
   device = MockDevice("MockTenstorrentDevice", supported_ops)
   cpu = CPUDevice()
   ```
3. **Partition the graph** with `partition_graph(graph, device, cpu)`.  
   - Assign each op to device if supported, or CPU if unsupported.  
   - Consecutive ops on the same device form a subgraph.  
4. **Run** each partition in order, simulating partial fallback.  
   - “Supported” ops remain on the device.  
   - “Unsupported” ops get a CPU fallback call.

Sample output:

```
Initial Graph:
 <Op MatMul1 type=Matmul shape=(32, 64) device=None>
 ...

Partitions (device, ops_list):
  MockTenstorrentDevice => ['MatMul1', 'LayerNorm1']
  CPU => ['Unsupported1']
  MockTenstorrentDevice => ['Softmax1']
  CPU => ['Unsupported2']
  MockTenstorrentDevice => ['MatMul2']

--- Running partition on MockTenstorrentDevice with 2 ops ---
[MockTenstorrentDevice] Running MatMul1 ...
[MockTenstorrentDevice] Running LayerNorm1 ...

--- Running partition on CPU with 1 ops ---
[CPU] Fallback for Unsupported1 ...

... and so on.
```

---

## 4. Potential Real-World Integration

A real partial fallback in **TT-Buda** (Tenstorrent’s top-down SDK) or **similar** HPC frameworks would involve:

1. **Compiler Pass**: During graph compilation, the tool identifies operators unsupported by the device’s kernel library.  
2. **Partitioning**: Splits the graph so only those ops are mapped to CPU fallback, while everything else stays on the AI hardware.  
3. **Data Transfer**: The system automatically moves tensors between AI hardware memory and CPU memory as needed.  
4. **Concurrency**: On specialized hardware (like Tenstorrent’s Tensix cores), the supported ops run concurrently; fallback ops run on CPU. 

This mock is purely **conceptual**—no real concurrency or hardware driver calls—but illustrates the approach for partial fallback.

---

## 5. Why should an AI Hardware Team care?

- **Fills a Real Gap**: Many frameworks do coarse-grained fallback, losing performance when only a few ops are unsupported. Partial fallback addresses that limitation.  
- **Highlights HPC Thinking**: Shows an understanding of **graph partitioning** and how to keep maximum coverage on specialized hardware.  
- **Adaptable**: Could be extended for multi-device partitioning, real concurrency, or more advanced scheduling heuristics.

---

## 6. Running in Colab or Locally

1. Clone or download `partial_cpu_fallback_demo.py`.  
2. (Optional) In Colab, copy/paste the code into a single cell.  
3. Run it. The logs will show how each op is distributed between device or CPU.  

*(This is a demonstration only. No actual HPC or AI code is executed.)*

---

## 7. Future Work

1. **Multi-Core / Multi-Device**: Partition ops across multiple “MockDevices” or CPU devices.  
2. **Real HPC Framework**: Replace the mocks with an actual HPC or AI library’s concurrency calls.  
3. **Dynamic Op Support**: Possibly read an “ops.json” or “capabilities.yaml” file to decide which ops are device-supported.  
4. **Integration**: Propose a TT-Buda compiler pass that performs partial fallback in this style.

---

## 8. Contact

- [**mohitraosatya** on LinkedIn](https://www.linkedin.com/in/mohitraosatya/)  
- **Email**: [saka4331@colorado.edu](mailto:saka4331@colorado.edu)

Feel free to connect with questions or feedback regarding partial CPU fallback or any HPC/AI kernel optimization topics.

*(End of README)*
