# Introduction to AI-driven Science on Supercomputers: A Student Training Series
## 2024 Spring Series

[Public Page for Series Schedule](https://www.alcf.anl.gov/alcf-ai-science-training-series)

[ALCF YouTube with recordings of sessions](https://www.youtube.com/@argonneleadershipcomputing8396)

[Indico registration page (CLOSED)](https://events.cels.anl.gov/event/436)

This repository is organized into one subdirectory per topic.  All content is prefixed by a two-digit index in the order of presentation in the tutorials.

<details open>
  <summary>  <b>Table of Contents</b> </summary>
  <ol start="0.">
    <li> <a href="./00_introToAlcf/">Introduction to ALCF Systems </a> </li>
    <ol>
      <li> <a href="./00_introToAlcf/00_computeSystems.md">ALCF Compute Systems Overview</a></li>
      <li> <a href="./00_introToAlcf/01_sharedResources">Shared Resources</a></li>
      <li> <a href="./00_introToAlcf/02_jupyterNotebooks.md">Introduction to Jupyter Notebooks</a></li>
      <li> <a href="./00_introToAlcf/03_githubHomework.md">How to Submit the Homeworks</a></li>
      <li> <a href="./00_introToAlcf/10_howToLogin.md">How to Login on the Command Line</a></li>
      <li> <a href="./00_introToAlcf/11_howToSetupEnvironment.md">How to Setup a Shell Enviroment</a></li>
      <li> <a href="./00_introToAlcf/12_jobQueuesSubmission.md">Submitting Jobs to a Queue</a></li>
    </ol>
    <li> <a href="./01_intro_AI_on_Supercomputer">Introduction to AI on Supercomputer </a> </li>
    <ol> 
       <li> <a href="./01_intro_AI_on_Supercomputer/evolution.md"> History of computing </a></li>
       <li> <a href="./01_intro_AI_on_Supercomputer/parallel_computing.md"> Parallel Computing </a></li>
       <li> <a href="./01_intro_AI_on_Supercomputer/01_linear_regression_sgd.ipynb"> Artificial Intelligence in a nutshell </a></li>
    </ol>
    <li> <a href="./02_intro_neural_networks"> Introduction to Neural Networks </a></li>
    <li> <a href="./03_advanced_neural_networks"> Advanced Topics in Neural Networks </a></li> 
    <li> <a href="./04_intro_to_llms"> Introduction to LLMs </a></li> 
    <li> <a href="./05_llm_part2"> LLMs -- Part II </a></li> 
    <li> <a href="./06_parallel_training"> Parallel Training Techniques</a></li> 
    <ol> 
      <li> <a href="https://saforem2.github.io/parallel-training-slides/#/">📊 Slides</a></li></ol>
    <li> <a href="./07_AITestbeds/"> AI Testbeds</a></li> 
    
</details>


*Note for contributors*: please run `git config --local include.path ../.gitconfig` once
upon cloning the repository (from anywhere in the repo) to add the	[`gitattribute`
filter](https://git-scm.com/docs/gitattributes#_filter) defintions to your local git
configuration options.[^1] Be sure that the `jupyter` command is in your `$PATH`,
otherwise the filter and git staging will fail.[^2][^3]

[^1]: https://zhauniarovich.com/post/2020/2020-10-clearing-jupyter-output-p3/
[^2]: https://stackoverflow.com/questions/28908319/how-to-clear-jupyter-notebooks-output-in-all-cells-from-the-linux-terminal
[^3]: https://bignerdranch.com/blog/git-smudge-and-clean-filters-making-changes-so-you-dont-have-to/
