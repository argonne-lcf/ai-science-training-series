# AI for Science: Student Training Series

Visit the [schedule of the ALCF AI for Science Training Series](https://www.alcf.anl.gov/alcf-ai-science-training-series) with 11
sessions listed for 2021-2022!

This repository is organized into one subdirectory per session.  All content is prefixed by a two-digit index in the order of presentation in the tutorials.

<details open>
  <summary>  <b>Table of Contents</b> </summary>
  <ol start="0.">
    <li> <a href="./00_introToAlcf/">Introduction to ALCF Systems </a> </li>
    <ol>
      <li> <a href="./00_introToAlcf/00_computeSystems.md">Compute Systems Overview</a></li>
      <li> <a href="./00_introToAlcf/01_howToLogin.md">How To Login to ALCF Systems</a></li>
      <li> <a href="./00_introToAlcf/02_howToSetupEnvironment.md">Environment Setup</a></li>
      <li> <a href="./00_introToAlcf/03_jobQueuesSubmission.md">Jobs, Queues, Submissions: How To</a></li>
      <li> <a href="./00_introToAlcf/04_jupyterNotebooks.md">Jupyter Notebooks</a></li>
    </ol>
    <li> <a href="./01_machineLearning"> Machine Learning  </a> </li>
    <ol> 
       <li> <a href="./01_machineLearning/part-1_introduction-to-sklearn"> Introduction to Supervised Machine Learning with Scikit-Learn </a></li>
       <li> <a href="./01_machineLearning/part-2_ml-with-materials-data"> Machine Learning with Scientific Data </a></li>
    </ol>
    <li> <a href="./02_deepLearning"> Introduction to Deep Learning </a></li>
    <li> <a href="./03_dataPipelines"> Data Pipelines for Deep Learning </a></li> 
    <ol>
      <li> <a href="./03_dataPipelines/00_tensorflowDatasetAPI"> TensorFlow Dataset API </a></li> 
      <li> <a href="./03_dataPipelines/01_pytorchDatasetAPI"> PyTorch Dataset API </a></li> 
    </ol>
    <li> <a href="./04_images_time_series/"> Advanced AI Applications: Image and Time Series Datasets </a></li> 
    <ol>
      <li> <a href="./04_images_time_series/00_images"> Images </a></li> 
      <li> <a href="./04_images_time_series/01_time_series"> Time Series </a></li> 
    </ol>    
    <li> <a href="./05_generative_models/README.md">Generative Models: GANs + Auto Encoders</a></li>
    <ol>
      <li> <a href="./05_generative_models/GANs.ipynb">GANs Notebook</a></li>
      <li> <a href="./05_generative_models/Auto%20Encoders.ipynb">Auto Encoders Notebook</a></li>
    </ol>
    <li> <a href="./06_distributedTraining/README.md">Distributed Training</a></li>
    <ol>
          <li> <a href="./06_distributedTraining/README.md">Horovod</a></li>
          <li> <a href="./06_distributedTraining/DDP/README.md">DDP</a></li>
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
