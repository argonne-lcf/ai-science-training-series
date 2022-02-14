---
title: "Hyper Parameter Optimization with DeepHyper"
theme: black
center: true
margin: 0.05
highlightTheme: github
css: 'custom/custom_theme.css'
transition: slide
revealOptions:
   transition: 'slide'
   scripts:
   - plugin/reveal.js-menu/menu.js
   - plugin/reveal.js-plugins/chalkboard/plugin.js
   - plugin/reveal.js-plugins/customcontrols/plugin.js
   - plugin.js
   menu:
     themes: true
     transitions: true
     transition: 'slide'
---

# HyperParameter Search

### with [DeepHyper](https://deephyper.readthedocs.io)

[**Prasanna Balaprakash**](https://www.mcs.anl.gov/~/pbalapra/)
<small style="float;text-align:left;">

Computer Scientist
    
Math & Computer Science Division
    
Leadership Computing Facility
    
Argonne National Laboratory
    
</small>
<a href="(https://github.com/deephyper/deephyper"><img src="./include/deephyper.png" width="200" style="align:right;"></a>

---
# Degrees of Freedom in Neural Networks Design for Scientific Data

![](./include/01_dof_sd.png)
---
# Degrees of Freedom in Neural Networks Design
![](./include/02_dof.png)
---

# BiLevel Optimization Problem
![](./include/03_biopt.png) 

---

# [DeepHyper](http://deephyper.readthedocs.io) Overview
- Documentation can be found at [deephyper.readthedocs.io](https://deephyper.readthedocs.io)
    
![](./include/04_dhoverview.png)

---
# Bayesian Optimization

![](./include/05_bayesian.png)
---
# [DeepHyper](http://deephyper.readthedocs.io) Overview
![](./include/06_dhoverview_1.png)

---
# Configuring Neural Architecture Search
![](./include/07_cfgnas.png)
---
# DeepHyper NAS-API
![](./include/08_nasapi.png)
---
# DeepHyper NAS-API
![](./include/09_nasapi_1.png)
---
# Exploring Search Space
- Regularized ageing evolution to explore the search space of possible architectures

![](./include/10_exploress.png) <!-- .element width="80%" align="center" -->
---
# Searching for a Surrogate LSTM:
## Sea Surface Temperature Forecasting
![](./include/11_lstm.png)
---
# Scaling
**Single Node, Cluster, Leadership Class**
![](./include/12_scaling.png)
---
# The DeepHyper Project

> "Automated development of machine learning algorithms to support scientific applications"

![](./include/13_dhproject.png)
---
# DeepHyper Community
![](./include/14_dhcommunity.png)
---
# References
<small>

- P. Balaprakash, M. Salim, T. Uram, V. Vishwanath, S. M. Wild. DeepHyper: Asynchronous hyperparameter search for deep neural networks. In 18 IEEE 25th international conference on high performance computing (HiPC), 2018.

- P. Balaprakash, R. Egele, M. Salim, S. Wild, V. Vishwanath, F. Xia, T. Brettin, and R. Stevens. Scalable reinforcement-learning-based neural architecture search for cancer deep learning research. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, 19.

- R. Maulik, R. Egele, B. Lusch, and P. Balaprakash. Recurrent Neural Network Architecture Search for Geophysical Emulation. In SC ’: IEEE/ACM International Conference on High Performance Computing, Networking, Storage and Analysis, 2020.

- R. Egele, P. Balaprakash, I. Guyon, V. Vishwanath, F. Xia, R. Stevens, Z. Liu.  AgEBO-Tabular: Joint neural architecture and hyperparameter search with autotuned data-parallel training for tabular data. In SC21:  International Conference for High Performance Computing, Networking, Storage and Analysis, (in press), 21.

- R. Egele, R. Maulik, K. Raghavan, P. Balaprakash, B. Lusch. AutoDEUQ: Automated Deep Ensemble with Uncertainty Quantification, (in review), 21.

</small>
---
# Acknowledgements
![](./include/15_acknowledgements.png)
