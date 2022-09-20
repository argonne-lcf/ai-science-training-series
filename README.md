# Introduction to AI-driven Science on Supercomputers: A Student Training Series
## 2022 Fall Series

[Public Page for Series Schedule](https://www.alcf.anl.gov/alcf-ai-science-training-series)

[Agenda with content links](https://events.cels.anl.gov/event/337/timetable/)

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
    <li> <a href="./01_machineLearning">Introduction to Machine Learning  </a> </li>
    <ol> 
       <li> <a href="./01_machineLearning/01_linear_regression_sgd.ipynb"> Introduction to Machine Learning with Linear Regression </a></li>
       <li> <a href="./01_machineLearning/02_clustering.ipynb">Introduction to Machine Learning with k-means Clustering</a></li>
    </ol>
    <li> <a href="./02_deepLearning"> Introduction to Deep Learning </a></li>
    <li> <a href="./02_deepLearning"> Deep Learning: Using Frameworks </a></li> 
    <li> <a href="./03_dataPipelines/">Building a Data Pipeline </a></li> 
  </ol>
</details>


*Note for contributors*: please run `git config --local include.path ../.gitconfig` once
upon cloning the repository (from anywhere in the repo) to add the	[`gitattribute`
filter](https://git-scm.com/docs/gitattributes#_filter) defintions to your local git
configuration options.[^1] Be sure that the `jupyter` command is in your `$PATH`,
otherwise the filter and git staging will fail.[^2][^3]

[^1]: https://zhauniarovich.com/post/2020/2020-10-clearing-jupyter-output-p3/
[^2]: https://stackoverflow.com/questions/28908319/how-to-clear-jupyter-notebooks-output-in-all-cells-from-the-linux-terminal
[^3]: https://bignerdranch.com/blog/git-smudge-and-clean-filters-making-changes-so-you-dont-have-to/
