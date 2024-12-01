Projects
=====


.. toctree::
   :maxdepth: 2
   :caption: Overview

This page is a showcase of OSS (open source software) and papers which have used **sc2bench** in the projects.
If your work is built on **sc2bench**, start `a "Show and tell" discussion at GitHub <https://github.com/yoshitomo-matsubara/sc2-benchmark/discussions/new?category=show-and-tell>`_.


Papers
*****

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
