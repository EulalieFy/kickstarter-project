Please click on the below link for a better display of the starting kit:

https://nbviewer.jupyter.org/github/EulalieFy/kickstarter-project/blob/master/Starting_kit.ipynb

# DataCamp Business Case 

# Kickstarter - Predicting success of fundraising

![Logo](img/kickstarter-logo.png)

###  Hamza Filali Baba, Eulalie Formery, Damien Grasset, Alice Guichenez, Hugo Perrin

Crowdfunding is the practice of funding a project or venture by raising monetary contributions from many people. Today, most of crowdfunding happens online through various websites and Kickstarter is one of the world's largest crowdfunding platforms. Kickstarter is mainly focused on creativity-related projects, in particular in art, music and design. It helps creators to find the resources and support they need to make their projects come real. Kickstarter is a huge global community; 16 million people have brought their contribution to over 150,000 successful projects all over the world.

The platform is based on an all or nothing funding model: project creators choose a deadline and a minimum funding goal, and money is collected only if the project reaches its goal by the deadline. It is a kind of insurance contract. The funding goal is chosen by the project kicker at the beginning and cannot be changed once the project has been launched. Whenever a project reaches the stated goal, the platform takes a 5% fee on the total amount of money collected. When a project fails, the platform does not gain anything. Therefore, it is to the benefit of Kickstarter that most projects are successful in reaching their funding goal by the deadline. Unlike many other platforms for fundraising and investment, Kickstarter claims no ownership over the projects and the work they produce – their profit is entirely based on the 5% fee they receive in case of success.

Our objective is therefore to increase the chances of success of the projects by finding the relevant characteristics that will make them most likely to meet the goal by the deadline. We will perform a multiclass classification to this end.


### To download the data :
 ```html
 python data_download.py 
 ```
 
 Additionnaly, you can download as well a subset of 30 000 images and descriptions using the flags --images True or --descriptions True (or both). Remark : descriptions are quick to be downloaded (just a few minutes) whereas images can take a while (>1H)
 
  ```html 
 python data_download.py --descriptions True --images True
 ```

