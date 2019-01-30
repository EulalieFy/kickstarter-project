
<h1><center> DataCamp Business Case <br> Kickstarter - Predicting success of fundraising </center></h1>

<h3>  <p style="text-align:left;">Authors : <span style="float:right;"> Professors:<br>  <br> Alexandre GRAMFORT <br> Balazs KEGL<br> <br> <br>
</span></p>

<br> Hamza Filali Baba <br> Eulalie Formery<br> 

Damien Grasset <br>  Alice Guichenez <br> Hugo Perrin</h3>



M2 DataScience - Université ParisScalay
 

***

<img src="img/kickstarter-logo.png" width="600">

## <a href="#1.-Business-case" style="color:#00925B">1.&nbsp;&nbsp;Business case </a>

- Context
- Business problem
- Data presentation
- Limits of our approach

## <a href="#2.-Exploration-of-the-data" style="color:#00925B">2.&nbsp;&nbsp;Exporation of the data </a>

- a completer

## <a href="#3.-Baseline-model" style="color:#00925B">3.&nbsp;&nbsp;Baseline model </a>

- a completer

## <a href="#4.-Local-testing-before-submission" style="color:#00925B">4.&nbsp;&nbsp;Local testing before submission  </a>


***

# <a id="#Business-case"  style="color:#00925B">1. Business case</a>

##  <span style="color:#00925B"> 1.1 Context  </span> 
 
Crowdfunding is the practice of funding a project or venture by raising monetary contributions from many people. Today, most of crowdfunding happens online through various websites and Kickstarter is one of the world's largest crowdfunding platforms. Kickstarter is mainly focused on creativity-related projects, in particular in art, music and design. It helps creators to find the resources and support they need to make their projects come real. Kickstarter is a huge global community; 16 million people have brought their contribution to over 150,000 successful projects all over the world. 

##  <span style="color:#00925B"> 1.2 Business problem  </span> 

**Funding model**
 
The platform is based on an all or nothing funding model: project creators choose a deadline and a minimum funding goal, and money is collected only if the project reaches its goal by the deadline. It is a kind of insurance contract. The funding goal is chosen by the project kicker at the beginning and cannot be changed once the project has been launched. Whenever a project reaches the stated goal, the platform takes a 5% fee on the total amount of money collected. When a project fails, the platform does not gain anything. Therefore, it is to the benefit of Kickstarter that most projects are successful in reaching their funding goal by the deadline. Unlike many other platforms for fundraising and investment, Kickstarter claims no ownership over the projects and the work they produce – their profit is entirely based on the 5% fee they receive in case of success.

**Finding**

There is currently a 36.63% overall rate of success, with some categories performing especially well (almost 62% for dance projects) and others particularly poorly (26% only for fashion).

The failure rate is 52.6% and the rest of the projects are canceled or suspended. Our goal is to increase this rate of success based on the finding that some projects are more meant to success than others. 


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

```


```python
data = pd.read_csv('data/full_data.csv')
```


```python
data.groupby('state').count()['id'].sort_values(ascending=False).plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11de1a978>




![png](output_15_1.png)


The chart below shows that it seems that the projects with highest goals have lower success rates.


```python
u = data['goal'].unique()
u.sort()

window = 40
intervals = [(u[int(len(u)*i/window)],u[int(len(u)*(i+1)/window)-1]) for i in range(window)]
successes = [len(data.loc[(data.goal <= intervals[i][1]) & (data.goal > intervals[i][0])
         & (data.state=='successful')])/len(data.loc[(data.goal <= intervals[i][1]) & (data.goal > intervals[i][0])]) 
             for i in range(window)]

plt.scatter([intervals[i][1] for i in range(len(intervals))][:-1], successes[:-1])
plt.ylabel("Gold")
plt.xlabel('Goal')
plt.ylabel('Success rate')
plt.show()
```


![png](output_17_0.png)


Based on these findings according to which the success rate may depend on the attributes of the projects, including the category and the goal among other things, our goal is to increase the success rate by predicting which attributes will make the projects get most chances to be successful.

**Business objective**

Kickstarter will make most money when most projects come to a success. Our objective is therefore to increase the chances of success of the projects by finding the relevant characteristics that will make them most likely to meet the goal by the deadline. By predicting whether a goal will be reached given a certain project, a certain funding target and a certain deadline, we should help fundraisers to pick adequate attributes and increase the rate of success, which should in turn increase Kickstarter’s profit. 

**Users of our solution**

We could sell our solution to Kickstarter and imagine that they would include a kind of recommendation tool in their app, where fundraisers could input the attributes of their project and Kickstarter would provide them guidance to pick the best characteristics given the output probability of success. Project kickers would benefit from a personalized recommendation and accompaniment and Kickstarter would increase its chances to get the 5% fee.

**Prediction task**

Our database is composed of projects along with a short description, their deadlines, their funding target and the actual funds raised. The goal is to predict among new projects which ones will have highest chances of being successful. It is therefore a **multiclass classification task** – success or not. The result would lead to an ability to provide advice for a better design of the projects, increasing their chances of meeting the target amount by the deadline. 


**Benefits of a machine learning solution**

A machine learning solution is particularly relevant in this context where we can exploit the huge amount of available data to help the business flourish. It makes it possible to emphasize which characteristics would be better suited for each project, it gives the possibility to Kickstarter to provide a personalized accompaniment for the design of fundraising projects and therefore increases the added value of the app by offering an additional service. All in all, it increases the fees gathered by Kickstarter by increasing the rate of success.


**Business metrics**

The prediction task, ie a multiclass classification with three differents output that are :
- The project hasn't been funded - Failed Fundraising
- The project has been funded (in correct proportion) - Successful Fundraising
- The project has been too much funded - Higher Success.

This is where it becomes interesting for Kickstarter to come up with a Key Perfomance Indicator (KPI). Indeed, more than just  having a good accuracy on the prediction, Kickstaster, as all businesses, wants to generate as more profit as it can with the output of the prediction.

Let's imagine a (simple) strategy of goal recommandation from Kickstarer according to the prediction : 
- if we think the goal can be increased (ie predict a Higher Sucess), Kicksarter recommand an increased goal of 20%.
- if, on the contrary, the goal is too large (Failed Fundraising), Kicksarter recommand a decreased goal of 20%.
- if it think it will be successful, the first goal is maintained.


As Kickstarter only receives 5% of the set goal of the project in case of sucess, which is the only revenue it will generate with its plateform, one can see prediction errors in the classification as potential **shortfall** in revenue :

- **(1)** if the project hasn't been 100% funded for a given goal $O_p$, but was pledged only $P_p < O_p$ but Kickstarter has predicted whether a success or a higher success, it means that Kistarted have missed a successful fundraising of $P_p$ for this project, ie :

$$\mbox{loss}_p = 5\% \times 20\% \times O_p \mbox{ if pleged at more than 80\%}$$
$$\mbox{loss}_p = 0 \mbox{ if pleged at less than 80\%}$$

- **(2)** if the project was predicted as a failed fundraising, the goal gets a 20% penalty. Hence, if the project was actually successful or with an higher success, it leads to a loss for Kickstaster : 

$$\mbox{loss}_p = 5\% \times 20\% \times 0_p \mbox{ if success}$$
$$\mbox{loss}_p = 5\% \times 40\% \times 0_p \mbox{ if higher success}$$

- **(3)** If the project was predicted as higher success but was in fact just successful, it means that Kickstarter will lose everything by recommanding an increase in the goal :

$$\mbox{loss}_p = 5\% \times 0_p$$

- **(4)** Finally, if the project was predicted as a normal success but was in fact a higher success, Kistarster loses potentially 20% of the additional amount Kickstarter could have claimed :

$$\mbox{loss}_p = 5\% \times 20\%  \times 0_p$$


Those 4 types of errors are directly connected with the 6 different cell errors in the confusion matrix as you can see below. Not that numbers inside cells are one of the error types listed above.

<img src="img/confusion_matrix.png" width="600">

**Workflow**

A COMPLETER

**Evolution of our solution in time** 

As new projects are launched, new data becomes available and our model can be improved by benefitting from this increased availability. Furthermore, it could also learn from its past mistakes and successes by evolving as the projects turn out to be failures or successes in accordance with or contrary to the predictions. Consequently, the model will have to be retrained on a regular basis – say once every two months, and this retraining could be based on online learning methods to make it as effective as possible. This would imply additional costs but also increased performance and accuracy for a thriving business.

##  <span style="color:#00925B"> 1.3 Data presentation  </span> 

rep à la question What data cleaning/tidying steps were required to obtain clean training data




To understand the story of the data collection and have all information about it, one is invited to take a look at the *data_sources* folder in the  <a href="https://github.com/EulalieFy/kickstarter-project/tree/master/data_sources">GitHub repository</a>. Here is a summary of it.

We collected the original data from the *webrobots.io* that scrapped the Kickstarter website in huge dataset containing a LOT of information. The data was a bit messy though. Indeed lots of information were hidden in string definition of dictionnary within cells of the dataframe. From this huge dataset and thanks to regex matching tools from instance, we extracted and selected features we consider interesting for the tasks.

From this lighted dataset, we reformated the data in the correct types (datetime, str, etc) and create other variables, dealing with missing value and using standard data processing methods.


**Bonus**:

In addition, we downloaded as well a subset of project images  from url founds in the original dataset, as well as another subset of descriptions - full text of presentation on the home page of a project - thanks to python scrapping packages. All this data is hosted on the same Google Drive as the datasets we will use here. This unstructed and multi-format data (png, txt) are not used in this starting kit but represent an interesting amount of data to use for the task.


##  <span style="color:#00925B"> 1.4 Limits of our approach  </span> 

A first limit of our approach comes from the fact that the data we got is probably not enough to explain the successful aspect of the fundraising all by itself. Further data including detailed project description and information about the project kicker (how many projects they already launched and how many of them were successful, host country of the project, etc) would certainly have been useful. 
Another limit is that we only know if the projects are successful in the sense that the money has been collected or not. However, this should not be the only point of focus because it does not guarantee that once the funds have been collected, the project will be properly implemented: for example, if our model recommends lowering the target amount to get higher chances to collect the money, maybe this amount will underestimate the actual requirements of the project and it will not be realizable. Our point of view was to focus on the profit aspect of the business, but the success of the project once the money has been collected might be relevant to Kickstarter’s business as well. The ideal would have been to have further data about the success of the realization of the project itself and not of the collection of the funds only.

# <a style="color:#00925B">2. Exploration of the data</a>

In our data exploration analysis, we are interested in analyzing the statistics of successful and failed projects because we would like to underline potential indicators that might differentiate successful and failed project. 


```python
data.drop('Unnamed: 0', inplace=True, axis=1)
# Vizualise the head of the dataframe
data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>short_description</th>
      <th>country</th>
      <th>city</th>
      <th>state_location</th>
      <th>main_category</th>
      <th>category</th>
      <th>created_at</th>
      <th>launched_at</th>
      <th>...</th>
      <th>project_we_love</th>
      <th>image_available</th>
      <th>description_available</th>
      <th>goal</th>
      <th>pledged</th>
      <th>achieved (%)</th>
      <th>usd_pledged</th>
      <th>static_usd_rate</th>
      <th>usd_type</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1555581815</td>
      <td>Big Top Without Borders</td>
      <td>A documentary about two circuses in remote cor...</td>
      <td>US</td>
      <td>Boston</td>
      <td>MA</td>
      <td>film &amp; video</td>
      <td>documentary</td>
      <td>2012-06-12 20:30:42</td>
      <td>2012-10-19 17:30:29</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>25000.0</td>
      <td>27455.55</td>
      <td>109.8222</td>
      <td>27455.55</td>
      <td>1.000000</td>
      <td>international</td>
      <td>successful</td>
    </tr>
    <tr>
      <th>1</th>
      <td>583419300</td>
      <td>The Story of "Pweep": From Egg - To Peacock</td>
      <td>A multi-media IPad book telling the true story...</td>
      <td>US</td>
      <td>Orlando</td>
      <td>FL</td>
      <td>publishing</td>
      <td>children's books</td>
      <td>2012-11-03 12:10:26</td>
      <td>2012-11-19 21:39:04</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>500.0</td>
      <td>535.00</td>
      <td>107.0000</td>
      <td>535.00</td>
      <td>1.000000</td>
      <td>international</td>
      <td>successful</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1745190062</td>
      <td>DC Radio</td>
      <td>We are college students that get drunk and the...</td>
      <td>CA</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>journalism</td>
      <td>audio</td>
      <td>2014-11-13 23:20:56</td>
      <td>2014-11-18 16:20:11</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>3500.0</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.886698</td>
      <td>domestic</td>
      <td>failed</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1995203117</td>
      <td>Ali Bangerz- two New Full Lenght Albums</td>
      <td>its Ali bangerz,its time to stand up for other...</td>
      <td>US</td>
      <td>Orlando</td>
      <td>FL</td>
      <td>music</td>
      <td>world music</td>
      <td>2015-11-04 20:18:23</td>
      <td>2015-11-04 22:22:47</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>20000.0</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>domestic</td>
      <td>failed</td>
    </tr>
    <tr>
      <th>4</th>
      <td>359013399</td>
      <td>Deja-Vu: Dissecting Memory on Camera</td>
      <td>A young neuroscientist attempts to reconnect w...</td>
      <td>US</td>
      <td>Brooklyn</td>
      <td>NY</td>
      <td>film &amp; video</td>
      <td>documentary</td>
      <td>2010-09-09 05:38:56</td>
      <td>2010-09-09 16:30:14</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>5000.0</td>
      <td>6705.00</td>
      <td>134.1000</td>
      <td>6705.00</td>
      <td>1.000000</td>
      <td>international</td>
      <td>successful</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
# Shape of the data
print("Data shape", data.shape)

# Column types of the data
data.dtypes
```

    Data shape (207226, 26)
    




    id                         int64
    name                      object
    short_description         object
    country                   object
    city                      object
    state_location            object
    main_category             object
    category                  object
    created_at                object
    launched_at               object
    state_changed_at          object
    deadline                  object
    currency                  object
    country.1                 object
    creator_name              object
    creator_id                 int64
    project_we_love             bool
    image_available             bool
    description_available       bool
    goal                     float64
    pledged                  float64
    achieved (%)             float64
    usd_pledged              float64
    static_usd_rate          float64
    usd_type                  object
    state                     object
    dtype: object



The data is therefore mainly composed of categorical variables, with only a few numerical variables. This is inherent to our database which must contain the names and descriptions of the projects, and we will handle some of the categorical variables with dummies variables and, when the text is relevant for prediction, we will do NLP.

The average lengths of the names and the descriptions of the projects is below.


```python
print('Average length of the names of the projets:', '%.2f'%(np.mean([len(name) for name in data['name']])))
print('Average length of the descriptions of the projects:', '%.2f'%(np.mean([len(desc) for desc in data['short_description'] if isinstance(desc, str)])))
```

    Average length of the names of the projets: 35.48
    Average length of the descriptions of the projects: 113.90
    


```python
for col in ['country', 'city', 'category', 'currency']:
    print('The number of unique values of ' + col + ' is ' + str(len(data[col].unique())) + '.')
```

    The number of unique values of country is 22
    The number of unique values of city is 12650
    The number of unique values of category is 159
    The number of unique values of currency is 14
    


```python
print('The projects are therefore based in '+ str(len(data['country'].unique())) + ' countries and are in ' + str(len(data['currency'].unique())) + ' currencies.')
```

    The projects are therefore based in 22 countries and are in 14 currencies.
    


```python
from collections import Counter
print('The main hosting country for the projects is the ' + str(max(Counter(data['country']))) + ' with over ' + '%.1f'%(100*Counter(data['country'])[max(Counter(data['country']))]/len(data)) + '% of the projects.')
print('The most represented category is ' + str(max(Counter(data['category'])))  + ' but it only represents ' + '%.1f'%(100*Counter(data['category'])[max(Counter(data['category']))]/len(data)) + '% of the projects.')
```

    The main hosting country for the projects is the US with over 78.2% of the projects.
    The most represented category is zines but it only represents 0.2% of the projects.
    

The table below is an overall exploratory data analysis to help us find trends within successes and failures.


```python
data['Target'] = np.where(data['pledged']>=data['goal'], 'success', 'failure')
data['launched_date'] = pd.to_datetime(data['launched_at'], format='%Y-%m-%d %H:%M:%S')
data['deadline_date'] = pd.to_datetime(data['deadline'], format='%Y-%m-%d %H:%M:%S')
data['year'] = [d.year for d in data['launched_date']]
data['month'] = [d.month for d in data['launched_date']]

descriptive_df = pd.DataFrame(np.zeros((11,3)))
descriptive_df.index = ['Projects', 'Proportion (%)', 'Project goal total', 'Project goal average', 'Amount pledged', 
                        'Average pledged', 'Goal vs pledged (%)', 'Average duration',
                        'Average length of the description', 'Nb of labels project_we_love', 'Nb of images']
descriptive_df.columns = ['Successful', 'Failed', 'Total']

sub_df = data[(data.state=='successful') | (data.state=='failed')] 
df_successful = data.loc[data['state']=='successful']
df_failed = data.loc[data['state']=='failed']

for i, df in enumerate([df_successful, df_failed, sub_df]):
    descriptive_df.iloc[0, i] = str(len(df))
    descriptive_df.iloc[1, i] = '%.2f'%(len(df) / len(sub_df) * 100)
    descriptive_df.iloc[2, i] = '%.2f'%(sum(df['goal']))
    descriptive_df.iloc[3, i] = '%.2f'%(df['goal'].mean())
    descriptive_df.iloc[4, i] = '%.2f'%(sum(df['pledged']))
    descriptive_df.iloc[5, i] = '%.2f'%(df['pledged'].mean())
    descriptive_df.iloc[6, i] = '%.2f'%(sum(df['pledged'])/sum(df['goal'])*100)
    descriptive_df.iloc[7, i] = '%.2f'%(df['duration'].mean())
    descriptive_df.iloc[8, i] = '%.2f'%(df['short_description'].apply(lambda x: len(x)).mean())
    descriptive_df.iloc[9, i] = str(sum(df['project_we_love']))
    descriptive_df.iloc[10, i] = str(sum(df['image_available']))
    
descriptive_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Successful</th>
      <th>Failed</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Projects</th>
      <td>115126</td>
      <td>74959</td>
      <td>190085</td>
    </tr>
    <tr>
      <th>Proportion (%)</th>
      <td>60.57</td>
      <td>39.43</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Project goal total</th>
      <td>1224191119.49</td>
      <td>6500514085.47</td>
      <td>7724705204.96</td>
    </tr>
    <tr>
      <th>Project goal average</th>
      <td>10633.49</td>
      <td>86720.93</td>
      <td>40638.16</td>
    </tr>
    <tr>
      <th>Amount pledged</th>
      <td>2619416480.88</td>
      <td>113129963.60</td>
      <td>2732546444.48</td>
    </tr>
    <tr>
      <th>Average pledged</th>
      <td>22752.61</td>
      <td>1509.22</td>
      <td>14375.39</td>
    </tr>
    <tr>
      <th>Goal vs pledged (%)</th>
      <td>213.97</td>
      <td>1.74</td>
      <td>35.37</td>
    </tr>
    <tr>
      <th>Average duration</th>
      <td>32.48</td>
      <td>34.70</td>
      <td>33.35</td>
    </tr>
    <tr>
      <th>Average length of the description</th>
      <td>114.97</td>
      <td>113.82</td>
      <td>114.52</td>
    </tr>
    <tr>
      <th>Nb of labels project_we_love</th>
      <td>24450</td>
      <td>2604</td>
      <td>27054</td>
    </tr>
    <tr>
      <th>Nb of images</th>
      <td>664</td>
      <td>401</td>
      <td>1065</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('The goal of successful projects is in average ' + str(int(df_failed['goal'].mean()/df_successful['goal'].mean())) + ' times lower than for failed projects.')
```

    The goal of successful projects is in average 8 times lower than for failed projects.
    

Therefore, our databases contains more successful projects than failed projects. The goal is in average far **lower** for successful projects. The length of the description is slightly bigger for successful projects but the difference is very low. However, the number of images is far **higher**, which means that the presence of an image with the project may increase the chances of success. As for the label *project_we_love*, it certainly is of great importance in the success of projects since the number of such labels is far higher for successful projects. This label is added by Kickstarter for the projects they want to support, and its influence is obviously non neglectable - one can imagine that people who are unsure who they want to give money to would be influenced a lot by this label.

The boxplot below shows that successful projects have **shorter** duration.


```python
data['duration'] = data['deadline_date'] - data['launched_date']
data['duration'] = data['duration'].apply(lambda x: x.days)
#sub_data = data.loc[(data.state=='successful') or (data.state=='failed')]
sub_data = data[(data.state=='successful') | (data.state=='failed')] 
sub_data.groupby('state').duration.mean().sort_index().plot(kind='box')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11af51ac8>




![png](output_42_1.png)



```python
print('Successful projects have an average duration of ' + '%.2f'%(data.groupby('state').duration.mean()['successful']) + ' days and failed projects have an average duration of ' + '%.2f'%(data.groupby('state').duration.mean()['failed'])+ ' days.')
```

    Successful projects have an average duration of 32.48 days and failed projects have an average duration of 34.70 days.
    


```python
plt.xlim((0,300))
plt.hist(data['achieved (%)'], bins=1000, range=(0,10000))
plt.show()
```


![png](output_44_0.png)


The histogram above shows that there seems to be trends in the percentage of the target gathered in the end depending on the amount. We can identify three trends: one with the amounts below 100, one with the amounts between 100 and 120 because of the thresholds at 100 and 120, and one with the amounts above 120. Hence the idea of a multiclass classification with three classes of amounts - below 100, between 100 and 120, and above 120.

The chart below shows the state of the projects.

The 15 most represented categories are represented below:


```python
data.groupby('category').count()['id'].sort_values(ascending=False)[:15].plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b3917f0>




![png](output_48_1.png)


The repartition of the projects within the years is the following:


```python
data.groupby('year').count()['id'].plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b3814e0>




![png](output_50_1.png)


In the histogram below, we see that the year with the highest failure ratio is 2015, while 2012 seems to have a high success ratio.


```python
data.groupby(['year','Target']).count()['id'].unstack().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b793da0>




![png](output_52_1.png)


The chart below shows that there is variability in the difference between goal and pledged among the categories that pledged most money. Some of them only gather a total amount just above the goal, while some others gather an amount far above the goal. 


```python
data.groupby('category')['pledged','goal'].mean().sort_values(by='pledged',ascending = False)[:15].plot(kind='bar', figsize=(15,10), title='Launched vs pledged in the categories that have pledged the more money per project')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x118a56128>




![png](output_54_1.png)


#  <a style="color:#00925B">3. Baseline model</a>

#  <a style="color:#00925B">4. Local testing before submission</a>


```python
from jyquickhelper import add_notebook_menu
add_notebook_menu(first_level=1)
```




<div id="my_id_menu_nb">run previous cell, wait for 2 seconds</div>
<script>
function repeat_indent_string(n){
    var a = "" ;
    for ( ; n > 0 ; --n)
        a += "    ";
    return a;
}
// look up into all sections and builds an automated menu //
var update_menu_string = function(begin, lfirst, llast, sformat, send, keep_item, begin_format, end_format) {
    var anchors = document.getElementsByClassName("section");
    if (anchors.length == 0) {
        anchors = document.getElementsByClassName("text_cell_render rendered_html");
    }
    var i,t;
    var text_menu = begin;
    var text_memo = "<pre>\nlength:" + anchors.length + "\n";
    var ind = "";
    var memo_level = 1;
    var href;
    var tags = [];
    var main_item = 0;
    var format_open = 0;
    for (i = 0; i <= llast; i++)
        tags.push("h" + i);

    for (i = 0; i < anchors.length; i++) {
        text_memo += "**" + anchors[i].id + "--\n";

        var child = null;
        for(t = 0; t < tags.length; t++) {
            var r = anchors[i].getElementsByTagName(tags[t]);
            if (r.length > 0) {
child = r[0];
break;
            }
        }
        if (child == null) {
            text_memo += "null\n";
            continue;
        }
        if (anchors[i].hasAttribute("id")) {
            // when converted in RST
            href = anchors[i].id;
            text_memo += "#1-" + href;
            // passer à child suivant (le chercher)
        }
        else if (child.hasAttribute("id")) {
            // in a notebook
            href = child.id;
            text_memo += "#2-" + href;
        }
        else {
            text_memo += "#3-" + "*" + "\n";
            continue;
        }
        var title = child.textContent;
        var level = parseInt(child.tagName.substring(1,2));

        text_memo += "--" + level + "?" + lfirst + "--" + title + "\n";

        if ((level < lfirst) || (level > llast)) {
            continue ;
        }
        if (title.endsWith('¶')) {
            title = title.substring(0,title.length-1).replace("<", "&lt;")
         .replace(">", "&gt;").replace("&", "&amp;");
        }
        if (title.length == 0) {
            continue;
        }

        while (level < memo_level) {
            text_menu += end_format + "</ul>\n";
            format_open -= 1;
            memo_level -= 1;
        }
        if (level == lfirst) {
            main_item += 1;
        }
        if (keep_item != -1 && main_item != keep_item + 1) {
            // alert(main_item + " - " + level + " - " + keep_item);
            continue;
        }
        while (level > memo_level) {
            text_menu += "<ul>\n";
            memo_level += 1;
        }
        text_menu += repeat_indent_string(level-2);
        text_menu += begin_format + sformat.replace("__HREF__", href).replace("__TITLE__", title);
        format_open += 1;
    }
    while (1 < memo_level) {
        text_menu += end_format + "</ul>\n";
        memo_level -= 1;
        format_open -= 1;
    }
    text_menu += send;
    //text_menu += "\n" + text_memo;

    while (format_open > 0) {
        text_menu += end_format;
        format_open -= 1;
    }
    return text_menu;
};
var update_menu = function() {
    var sbegin = "";
    var sformat = '<a href="#__HREF__">__TITLE__</a>';
    var send = "";
    var begin_format = '<li>';
    var end_format = '</li>';
    var keep_item = -1;
    var text_menu = update_menu_string(sbegin, 1, 4, sformat, send, keep_item,
       begin_format, end_format);
    var menu = document.getElementById("my_id_menu_nb");
    menu.innerHTML=text_menu;
};
window.setTimeout(update_menu,2000);
            </script>



## Modules


```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec

from gensim.models import Doc2Vec
from scipy.spatial import distance
from nltk.corpus import stopwords
from gensim.models.doc2vec import LabeledSentence

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance
from sklearn.base import BaseEstimator

from problem import get_train_data, get_test_data
from problem import metric_report
```

##  Load and clean data


```python
data = pd.read_csv('../kickstarter-bis/data/test.csv')
data = data.dropna(subset=['name'])

data.index = np.arange(0, len(data))
```


```python
## To do before in the "problem.py"

labels = data['pledged']
data.drop(['pledged', 'state', 'usd_pledged_real', 'pledged', 'usd pledged', 'backers'], 
          axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

X_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)
```

## Feature extractor


```python
# remove dashes and apostrophes from punctuation marks 
punct = string.punctuation.replace('-', '').replace("'",'')
# regex to match intra-word dashes and intra-word apostrophes
my_regex = re.compile(r"(\b[-']\b)|[\W_]")

def clean_string(string, punct=punct, my_regex=my_regex, to_lower=False):
    if to_lower:
        string = string.lower()
    # remove formatting
    str = re.sub('\s+', ' ', string)
     # remove punctuation
    str = ''.join(l for l in str if l not in punct)
    # remove dashes that are not intra-word
    str = my_regex.sub(lambda x: (x.group(1) if x.group(1) else ' '), str)
    # strip extra white space
    str = re.sub(' +',' ',str)
    # strip leading and trailing white space
    str = str.strip()
    return str
```


```python
class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        
        #### NLP BASICS ####
        names = X['name'].tolist()
        cleaned_project_names = []
        
        for idx, doc in enumerate(names):
            # clean
            doc = clean_string(doc, punct, my_regex, to_lower=True)
            # tokenize (split based on whitespace)
            tokens = doc.split(' ')
            # remove digits
            tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
            # remove tokens shorter than 3 characters in size
            tokens = [token for token in tokens if len(token) > 1]
            # remove tokens exceeding 25 characters in size
            tokens = [token for token in tokens if len(token) <= 25]
            cleaned_project_names.append(tokens)
            
        ##### Word2vect embedding
        self.model = Word2Vec(cleaned_project_names, min_count=1, size=100, workers=8)
        
        return self

    def transform(self, X):
        data = X.copy()
        
        #### SIMPLE TRANSFORMATION #### 
        
        data['launched_date'] = pd.to_datetime(data['launched'], format='%Y-%m-%d %H:%M:%S')
        data['deadline_date'] = pd.to_datetime(data['deadline'], format='%Y-%m-%d %H:%M:%S')
        
        # Length of project
        data['length'] = data['deadline_date'] - data['launched_date']
        data['length'] = [d.days for d in data['length']]
        
        # Features with month and year of launch
        data['year'] = [d.year for d in data['launched_date']]
        data['month'] = [d.month for d in data['launched_date']]
        data['day'] = [d.day for d in data['launched_date']]
        
        # Length of name
        data['name_length'] = [len(name) for name in data['name']]

        # Number of words
        data['word_number'] = [len(name.split(' ')) for name in data['name']]

        # Ponctuation
        data['question'] = (data.name.str[-1] == '?').astype(int)
        data['exclamation'] = (data.name.str[-1] == '!').astype(int)

        # Upper
        data['uppercase'] = data.name.str.isupper().astype(float)
        
        # Create dummies for categorical features
        main_category = pd.get_dummies(data['main_category'],prefix='mc')
        category = pd.get_dummies(data['category'], prefix = 'cat')
        country = pd.get_dummies(data['country'], prefix = 'country')
        currency = pd.get_dummies(data['currency'], prefix = 'currency')

        data = pd.concat([data, main_category, category, country, currency], axis=1)
        
        # Drop several features
        names = data['name'].tolist()

        features_to_drop =['main_category', 'category', 'country', 'currency', 'name',
                           'deadline', 'deadline_date', 'launched_date', 'launched',
                           'usd_goal_real', 'ID']
        data.drop(features_to_drop, axis=1, inplace=True)

        #### NLP BASICS ####
        cleaned_project_names = []
        
        for idx, doc in enumerate(names):
            # clean
            doc = clean_string(doc, punct, my_regex, to_lower=True)
            # tokenize (split based on whitespace)
            tokens = doc.split(' ')
            # remove digits
            tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
            # remove tokens shorter than 3 characters in size
            tokens = [token for token in tokens if len(token) > 1]
            # remove tokens exceeding 25 characters in size
            tokens = [token for token in tokens if len(token) <= 25]
            cleaned_project_names.append(tokens)
            
        name_matrix = np.zeros((len(cleaned_project_names), 100), dtype="float32")

        for i in range(len(cleaned_project_names)):
            try:
                name_matrix[i,]= self.model.wv[cleaned_project_names[i]].sum(0) / len(cleaned_project_names[i]) 
            except:
                pass
                
        name_embeddings = pd.DataFrame(name_matrix)
        
        data = pd.concat([data, name_embeddings], axis=1)
        
        return data
```

## Regressor


```python
class Regressor(BaseEstimator):
    
    def __init__(self):
        self.model = LGBMRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        return np.maximum(prediction, np.zeros(prediction.shape[0]))
```

## Model


```python
feature_extractor = FeatureExtractor()

feature_extractor.fit(X_train, y_train)

X_train = feature_extractor.transform(X_train)
X_test = feature_extractor.transform(X_test)
```


```python
lgb_regressor = Regressor()

lgb_regressor.fit(X_train, y_train)

y_pred = lgb_regressor.predict(X_test)
```

## Metrics


```python
metric_report(X_test, y_true=y_test, y_pred=y_pred)
```

    -------- REGRESSION METRICS --------
    
    RMSE: 78447.00
    MAE: 13146.39
    
    -------- CLASSIFICATION METRICS --------
    
    Accuracy: 0.65
    Precision: 0.49
    Recall: 0.39
    
