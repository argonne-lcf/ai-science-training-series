# intro_deep_learning
Atpesc intro to deep learning
Author: Marieme Ngom (mngom@anl.gov), adapting materials from Bethany Lusch, Prasanna Balaprakash, Taylor Childers, Corey Adams, and Kyle Felker.

This is a hands-on introduction to deep learning, a machine learning technique that tends to outperform other techniques when dealing with a large amount of data. 

This is a quick overview, but the goals are:
- to introduce the fundamental concepts of deep learning through hands-on activities
- to give you the necessary background for the more advanced topics on scaling and performance that we will teach this afternoon.

Ready for more?
- Here are some of our longer training materials: https://github.com/argonne-lcf/sdl_ai_workshop
- Here's a thorough hands-on textbook: [book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) with [notebooks](https://github.com/ageron/handson-ml2).


We will work on a classification problem involving the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) that contains thousands of examples of handwritten numbers, with each digit labeled 0-9
![MNIST Task](images/mnist_task.png)

We are going to use google collab:
*  You need a Google Account to use Colaboratory
*  Goto [Google's Colaboratory Platform](https://colab.research.google.com) 
*  You should see this page
![start_page](../README_imgs/colab_start_page.png)
*  Click on the `New Python Notebook` 
*  Now you will see a new notebook where you can type in python code.
![clean_page](../README_imgs/collab_start_page1.png)
*  After you enter code, type `<shift>+<enter>` to execute the code cell.
*  A full introduction to the notebook environment is out of scope for this tutorial, but many can be found with a [simple Google search](https://www.google.com/search?q=jupyter+notebook+tutorial)
*  We will be using notebooks from this repository during the tutorial, so  you should be familiar with how to import them into Colaboratory
*  Now you can open the `File` menu at the top left and select `Open Notebook` which will open a dialogue box.
*  Select the `GitHub` tab in the dialogue box.
*  From here you can enter the url for the github repo: `https://github.com/argonne-lcf/ATPESC_MachineLearning` and hit `<enter>`.
![open_github](../README_imgs/colab_open_github.png)
*  This will show you a list of the Notebooks available in the repo.
*  Select the `introduction.ipynb` file to open and work through it.
*  As each session of the tutorial begins, you will simply select the corresponding notebook from this list and it will create a copy for you in your Colaboratory account (all `*.ipynb` files in the Colaboratory account will be stored in your Google Drive).


