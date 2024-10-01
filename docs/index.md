---
hide:
  - navigation
  - toc
---

# <div style="text-align: center;"> <img src="assets/NeMoS_Logo_CMYK_Full.svg" width="50%" alt="NeMoS logo."> </div>

<div style="text-align: center;" markdown>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/flatironinstitute/nemos/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.10%7C3.11%7C3.12-blue.svg)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![PyPI - Version](https://img.shields.io/pypi/v/nemos)
[![codecov](https://codecov.io/gh/flatironinstitute/nemos/graph/badge.svg?token=vvtrcTFNeu)](https://codecov.io/gh/flatironinstitute/nemos)
[![Documentation Status](https://readthedocs.org/projects/nemos/badge/?version=latest)](https://nemos.readthedocs.io/en/latest/?badge=latest)
[![nemos CI](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml/badge.svg)](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml)


__Learning Resources:__ [:material-book-open-variant-outline: Neuromatch Academy's Lessons](https://compneuro.neuromatch.io/tutorials/W1D3_GeneralizedLinearModels/student/W1D3_Tutorial1.html) | [:material-youtube: Cosyne 2018 Tutorial](https://www.youtube.com/watch?v=NFeGW5ljUoI&t=424s) <br> 
__Useful Links:__ [:material-chat-question: Getting Help](getting_help) | [:material-alert-circle-outline: Issue Tracker](https://github.com/flatironinstitute/nemos/issues) | [:material-order-bool-ascending-variant: Contributing Guidelines](https://github.com/flatironinstitute/nemos/blob/main/CONTRIBUTING.md)

</div>



## __Overview__

NeMoS (Neural ModelS) is a statistical modeling framework optimized for systems neuroscience and powered by [JAX](https://jax.readthedocs.io/en/latest/). 
It streamlines the process of defining and selecting models, through a collection of easy-to-use methods for feature design.

The core of NeMoS includes GPU-accelerated, well-tested implementations of standard statistical models, currently 
focusing on the Generalized Linear Model (GLM). 

We provide a **Poisson GLM** for analyzing spike counts, and a **Gamma GLM** for calcium or voltage imaging traces.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Getting Started__

      ---

      New to NeMoS? Get the ball rolling with our quickstart.

      [:octicons-arrow-right-24: Quickstart](quickstart)

-   :material-book-open-variant-outline:{ .lg .middle } &nbsp; __Background__

    ---

    Refresh your theoretical knowledge before diving into data analysis with our notes.

    [:octicons-arrow-right-24: Background](generated/background)

-   :material-lightbulb-on-10:{ .lg .middle } &nbsp; __How-To Guide__

    ---

    Already familiar with the concepts? Learn how you to process and analyze your data with NeMoS.

    *Requires familiarity with the theory.*<br>
    [:octicons-arrow-right-24: How-To Guide](generated/how_to_guide)

-   :material-brain:{ .lg .middle} &nbsp;  __Neural Modeling__

    ---

    Explore fully worked examples to learn how to analyze neural recordings from scratch.

    *Requires familiarity with the theory.*<br>
    [:octicons-arrow-right-24: Tutorials](generated/tutorials)

-   :material-cog:{ .lg .middle } &nbsp; __API Guide__

    ---

    Access a detailed description of each module and function, including parameters and functionality. 

    *Requires familiarity with the theory.*<br>
    [:octicons-arrow-right-24: API Guide](reference/SUMMARY)

-   :material-hammer-wrench:{ .lg .middle } &nbsp; __Installation Instructions__ 

    ---
    
    Run the following `pip` command in your virtual environment.
    === "macOS/Linux"

        ```bash
        pip install nemos
        ```

    === "Windows"
    
        ```
        python -m pip install nemos
        ```
    
    *For more information see:*<br>
    [:octicons-arrow-right-24: Install](installation)

</div>

## :material-scale-balance:{ .lg } License

Open source, [licensed under MIT](https://github.com/flatironinstitute/nemos/blob/main/LICENSE).
