Projects
=====


.. toctree::
   :maxdepth: 2
   :caption: Overview

This page is a showcase of OSS (open source software) and papers which have used **sc2bench** in the projects.
If your work is built on **sc2bench**, start `a "Show and tell" discussion at GitHub <https://github.com/yoshitomo-matsubara/sc2-benchmark/discussions/new?category=show-and-tell>`_.


Papers
*****
Resilience of Entropy Model in Distributed Neural Networks
----
* Author(s): Milin Zhang, Mohammad Abdi, Shahriar Rifat, and Francesco Restuccia
* Venue: ECCV 2024
* PDF: `Paper <https://arxiv.org/abs/2403.00942>`_
* Code: `GitHub <https://github.com/Restuccia-Group/EntropyR>`_

**Abstract**: Distributed deep neural networks (DNNs) have emerged as
a key technique to reduce communication overhead without sacrificing
performance in edge computing systems. Recently, entropy coding has
been introduced to further reduce the communication overhead. The key
idea is to train the distributed DNN jointly with an entropy model,
which is used as side information during inference time to adaptively
encode latent representations into bit streams with variable length. To
the best of our knowledge, the resilience of entropy models is yet to be
investigated. As such, in this paper we formulate and investigate the
resilience of entropy models to intentional interference (e.g., adversarial attacks) and unintentional interference (e.g., weather changes and
motion blur). Through an extensive experimental campaign with 3 different DNN architectures, 2 entropy models and 4 rate-distortion tradeoff factors, we demonstrate that the entropy attacks can increase the
communication overhead by up to 95%. By separating compression features in frequency and spatial domain, we propose a new defense mechanism that can reduce the transmission overhead of the attacked input
by about 9% compared to unperturbed data, with only about 2% accuracy loss. Importantly, the proposed defense mechanism is a standalone
approach which can be applied in conjunction with approaches such as
adversarial training to further improve robustness. Code is available at
https://github.com/Restuccia-Group/EntropyR.

A Multi-task Supervised Compression Model for Split Computing
----
* Author(s): Yoshitomo Matsubara, Matteo Mendula, Marco Levorato
* Venue: WACV 2025
* PDF: `Paper <https://arxiv.org/abs/2501.01420>`_
* Code: `GitHub <https://github.com/yoshitomo-matsubara/ladon-multi-task-sc2>`_

Split computing (≠ split learning) is a promising approach to deep learning models for resource-constrained
edge computing systems, where weak sensor (mobile) devices are wirelessly connected to stronger edge servers through
channels with limited communication capacity. State-of-theart work on split computing presents methods for single tasks
such as image classification, object detection, or semantic segmentation. The application of existing methods to
multitask problems degrades model accuracy and/or significantly increase runtime latency. In this study, we propose Ladon,
the first multi-task-head supervised compression model for multi-task split computing. Experimental results show that
the multi-task supervised compression model either outperformed or rivaled strong lightweight baseline models in terms
of predictive performance for ILSVRC 2012, COCO 2017, and PASCAL VOC 2012 datasets while learning compressed
representations at its early layers. Furthermore, our models reduced end-to-end latency (by up to 95.4%) and
energy consumption of mobile devices (by up to 88.2%) in multi-task split computing scenarios.


FrankenSplit: Efficient Neural Feature Compression With Shallow Variational Bottleneck Injection for Mobile Edge Computing
----
* Author(s): Alireza Furutanpey, Philipp Raith, Schahram Dustdar
* Venue: IEEE Transactions on Mobile Computing
* PDF: `Paper <https://ieeexplore.ieee.org/document/10480247/>`_
* Code: `GitHub <https://github.com/rezafuru/FrankenSplit>`_

**Abstract**: The rise of mobile AI accelerators allows latency-sensitive applications to execute lightweight Deep Neural
Networks (DNNs) on the client side. However, critical applications require powerful models that edge devices cannot host
and must therefore offload requests, where the high-dimensional data will compete for limited bandwidth.
Split Computing (SC) alleviates resource inefficiency by partitioning DNN layers across devices, but current methods are
overly specific and only marginally reduce bandwidth consumption. This work proposes shifting away from focusing on
executing shallow layers of partitioned DNNs. Instead, it advocates concentrating the local resources on variational
compression optimized for machine interpretability. We introduce a novel framework for resource-conscious compression
models and extensively evaluate our method in an environment reflecting the asymmetric resource distribution between edge
devices and servers. Our method achieves 60% lower bitrate than a state-of-the-art SC method without decreasing accuracy
and is up to 16x faster than offloading with existing codec standards.


SC2 Benchmark: Supervised Compression for Split Computing
----
* Author(s): Yoshitomo Matsubara, Ruihan Yang, Marco Levorato, Stephan Mandt
* Venue: TMLR
* PDF: `Paper + Supp <https://openreview.net/forum?id=p28wv4G65d>`_
* Code: `GitHub <https://github.com/yoshitomo-matsubara/sc2-benchmark>`_

**Abstract**: With the increasing demand for deep learning models on mobile devices, splitting neural network
computation between the device and a more powerful edge server has become an attractive solution. However, existing
split computing approaches often underperform compared to a naive baseline of remote computation on compressed data.
Recent studies propose learning compressed representations that contain more relevant information for supervised
downstream tasks, showing improved tradeoffs between compressed data size and supervised performance. However, existing
evaluation metrics only provide an incomplete picture of split computing. This study introduces supervised compression
for split computing (SC2) and proposes new evaluation criteria: minimizing computation on the mobile device, minimizing
transmitted data size, and maximizing model accuracy. We conduct a comprehensive benchmark study using 10 baseline
methods, three computer vision tasks, and over 180 trained models, and discuss various aspects of SC2. We also release
our code and sc2bench, a Python package for future research on SC2. Our proposed metrics and package will help
researchers better understand the tradeoffs of supervised compression in split computing.
