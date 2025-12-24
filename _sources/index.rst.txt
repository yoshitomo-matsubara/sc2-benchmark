.. SC2 Benchmark documentation master file, created by
   sphinx-quickstart on Tue Jul  4 17:48:09 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SC2 Benchmark documentation
=========================================

SC2: Supervised Compression for Split Computing
***********************************************


.. image:: https://raw.githubusercontent.com/yoshitomo-matsubara/sc2-benchmark/main/imgs/input_vs_supervised_compression.png
  :alt: Supervised Compression for Split Computing

This is the official documentation of `sc2bench <https://pypi.org/project/sc2bench/>`_ package and
our TMLR paper, `"SC2 Benchmark: Supervised Compression for Split Computing" <https://openreview.net/forum?id=p28wv4G65d>`_.

.. youtube:: uwwV_vAOvX4
   :align: center


As an intermediate option between local computing and edge computing (full offloading),
**split computing** has been attracting considerable attention from the research communities.

In split computing, we split a neural network model into two sequences so that
some elementary feature transformations are applied by the first sequence of the model on a weak mobile (local) device.
Then, intermediate, informative features are transmitted through a wireless communication channel to
a powerful edge server that processes the bulk part of the computation (the second sequence of the model).

Input compression is an approach to save transmitted data, but it leads to transmitting information irrelevant to
the supervised task.
To achieve better supervised rate-distortion tradeoff, we define ***supervised compression*** as
learning compressed representations for supervised downstream tasks such as classification, detection, or segmentation.
Specifically for split computing, we term the problem setting **SC2** (*Supervised Compression for Split Computing*).

Note that the training process can be done offline (i.e., on a single device without splitting),
and it is different from "split learning".


Check out the :doc:`usage` section for further information.

.. toctree::
   :maxdepth: 2
   :caption: 📚 Overview

   usage
   package

.. toctree::
   :maxdepth: 2
   :caption: 🧑🏻‍💻 Research

   projects

Reference
*********
.. code-block:: bibtex

   @article{matsubara2023sc2,
     title={{SC2 Benchmark: Supervised Compression for Split Computing}},
     author={Matsubara, Yoshitomo and Yang, Ruihan and Levorato, Marco and Mandt, Stephan},
     journal={Transactions on Machine Learning Research},
     issn={2835-8856},
     year={2023},
     url={https://openreview.net/forum?id=p28wv4G65d}
   }


Indices and Tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
