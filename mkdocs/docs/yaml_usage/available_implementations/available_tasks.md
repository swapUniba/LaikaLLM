# Available Tasks

Each task can have **Inference templates** and **Support templates**:

- **Inference templates** are those templates which can be used at *inference time* and are those upon which
  the evaluation is carried out
- **Support templates** are those templates which can **NOT** be used at *inference time*, since they have intrinsic
  assumption which requires the knowledge of the ground truth

Example of **inference template**:

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle">
            Recommend an item for {user_id}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>

Example of **support template**:

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle">
            Recommend an item for {user_id}, knowing that a good item to recommend is present among {candidates}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>

As you can easily see, the *support template* requires information that we **do not have** at inference time: that is,
the target item to recommend

To specify one of the following tasks the `.yaml` file, you can simply use its **name** 
(The parsing is **not** *case-sensitive*)!

---

## SequentialSideInfoTask

The SequentialSideInfoTask is built for [AmazonDataset](available_datasets.md#amazondataset): the goal is to predict
the next item of the **order history** of the user. This task has the **SideInfo** suffix because **categories** of the
items bought by the user are used additional information for the prediction.

There are two different support tasks:

- *Extractive QA*: For the specific *user_id*, given its **order history** and the **categories** of each item bought,
  select the *next element to recommend* from a list of **candidates**
- *Pair Seq*: For the specific *user_id*, given only one element from the **order history** and its **categories**,
  predict the **immediate successor** of the element

**Inference templates:**

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">0</td>
          <td style="vertical-align: middle">
            sequential recommendation - {user_id}: <br/><br/>
            Predict for the user the next element of the following sequence -> {order_history} <br/>
            The category of each element of the sequence is -> {category_history}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">1</td>
          <td style="vertical-align: middle">
            sequential recommendation - {user_id}: <br/><br/>
            Predict the next element which the user will buy given the following order history -> {order_history} <br/>
            Each item bought belongs to these categories (in order) -> {category_history}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2</td>
          <td style="vertical-align: middle">
            sequential recommendation - {user_id}: <br/><br/>
            What is the element that should be recommended to the user knowing that it has bought -> {order_history} <br/>
            Categories of the items are -> {category_history}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">3</td>
          <td style="vertical-align: middle">
            sequential recommendation - {user_id}: <br/><br/>
            Recommend to the user an item from the catalog given its order history -> {order_history}  <br/>
            Each item of the order history belongs to the following categories (in order) -> {category_history}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">4</td>
          <td style="vertical-align: middle">
            sequential recommendation - {user_id}: <br/><br/>
            This is the order history of the user -> {order_history} <br/>
            These are the categories of each item -> {category_history} <br/>
            Please recommend the next element that the user will buy
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">5</td>
          <td style="vertical-align: middle">
            sequential recommendation - {user_id}: <br/><br/>
            Please predict what item is best to recommend to the user given its order history -> {order_history} <br/>
            Categories of each item -> {category_history} <br/>
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>

**Extractive QA templates**

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">6</td>
          <td style="vertical-align: middle">
            sequential recommendation - {user_id}: <br/><br/>
            The user has the following order history -> {order_history} <br/>
            The categories of each item bought are -> {category_history} <br/>
            Which item would the user buy next? Select from the following: <br/>
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">7</td>
          <td style="vertical-align: middle">
            sequential recommendation - {user_id}: <br/><br/>
            The user has bought {order_history}, and the categories of those items are {category_history}. <br/>
            Choose an item to recommend to the user selecting from: <br/>
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>

**Pair Seq templates**:

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">8</td>
          <td style="vertical-align: middle">
            sequential recommendation - {user_id}: <br/><br/>
            The user has recently bought {precedent_item_id} which has the following categories: {categories_precedent_item} <br/>
            What is the next item to recommend? <br/>
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">9</td>
          <td style="vertical-align: middle">
            sequential recommendation - {user_id}: <br/><br/>
            The latest item bought by the user is {precedent_item_id}. <br/>
            The categories of that item are {categories_precedent_item}. <br/>
            Predict which item the user will buy next
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>

---

## DirectSideInfoTask

The DirectSideInfoTask is built for [AmazonDataset](available_datasets.md#amazondataset): the goal is to predict
a good item to recommend for the *user*. This task has the **SideInfo** suffix because **categories** of the
items bought by the user are used additional information for the prediction.

There are two different support tasks:

- *Extractive QA*: For the specific *user_id*, given **categories liked** by the user, select the *item to recommend* 
   from a list of **candidates**

**Inference templates**:

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">0</td>
          <td style="vertical-align: middle">
            direct recommendation - {user_id}: <br/><br/>
            Pick an item from the catalog knowing that these are the categories the user likes -> {unique_categories_liked} <br/>
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">1</td>
          <td style="vertical-align: middle">
            direct recommendation - {user_id}: <br/><br/>
            Recommend an item to the user. The categories of the items bought by the user are -> {unique_categories_liked} <br/>
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2</td>
          <td style="vertical-align: middle">
            direct recommendation - {user_id}: <br/><br/>
            What is the item that should be recommended to the user? It likes these categories -> {unique_categories_liked} <br/>
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">3</td>
          <td style="vertical-align: middle">
            direct recommendation - {user_id}: <br/><br/>
            Select an item to present to the user given the categories that it likes -> {unique_categories_liked} <br/>
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">4</td>
          <td style="vertical-align: middle">
            direct recommendation - {user_id}: <br/><br/>
            These are the categories of the items bought by the user -> {unique_categories_liked} <br/>
            Please recommend an item that the user will buy
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">5</td>
          <td style="vertical-align: middle">
            direct recommendation - {user_id}: <br/><br/>
            Please predict what item is best to recommend to the user. The categories that it likes are -> {unique_categories_liked} <br/>
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>

**Extractive QA templates**:

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">6</td>
          <td style="vertical-align: middle">
            direct recommendation - {user_id}: <br/><br/>
            The categories liked by the user are -> {unique_categories_liked} <br/>
            Which item can interest the user? Select one from the following: <br/>
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">7</td>
          <td style="vertical-align: middle">
            direct recommendation - {user_id}: <br/><br/>
            The user so far has bought items with these categories -> {unique_categories_liked}. <br/>
            Choose an item to recommend to the user selecting from: <br/>
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">8</td>
          <td style="vertical-align: middle">
            direct recommendation - {user_id}: <br/><br/>
            These are the categories of the items bought by the user -> {unique_categories_liked}. <br/>
            Predict an item to suggest to the user from the followings: <br/>
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>

---

## RatingPredictionTask

The *RatingPredictionTask* is built for [AmazonDataset](available_datasets.md#amazondataset): the goal is to predict
the rating that the *user* would give to an *unseen* item.

**Inference templates**

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">0</td>
          <td style="vertical-align: middle">
            rating prediction - {user_id}: <br/><br/>
            Average rating of the user -> {avg_rating} <br/>
            Continue this rating sequence for the user, predicting the rating for {item_id}: <br/>
            {order_history_w_ratings}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_rating}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">1</td>
          <td style="vertical-align: middle">
            rating prediction - {user_id}: <br/><br/>
            Average rating of the user -> {avg_rating} <br/>
            Predict the rating that the user would give to {item_id}, by considering the following previously bought item and the rating assigned: <br/>
            {order_history_w_ratings}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_rating}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2</td>
          <td style="vertical-align: middle">
            rating prediction - {user_id}: <br/><br/>
            Predict the score the user would give to {item_id} (in a 1-5 scale). <br/>
            This is the user order history with associated rating that the user previously gave: <br/>
            {order_history_w_ratings} <br/>
            Consider that the average rating of the user is {avg_rating}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_rating}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">3</td>
          <td style="vertical-align: middle">
            rating prediction - {user_id}: <br/><br/>
            This is the order history of the user with the associated rating -> <br/>
            {order_history_w_ratings} <br/>
            This is the average rating given by the user -> {avg_rating} <br/>
            Based on that, predict the score (in a 1-5 scale) the user would give to {item_id}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_rating}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">4</td>
          <td style="vertical-align: middle">
            rating prediction - {user_id}: <br/><br/>
            Please predict the user, which has an average rating of {avg_rating}, would give to {item_id} based on its order history -> <br/>
            {order_history_w_ratings} <br/>
            This is the average rating given by the user -> {avg_rating} <br/>
            The score should be in a 1-5 scale
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_rating}
          </td>
        </tr>
      </tbody>
    </table>
</center>

---

## P5 Tasks

Below there are defined the *Rating*, *Sequential*, *Direct* task prompts from the [P5 paper](https://arxiv.org/pdf/2203.13366).
Each task has its ***"Eval"*** counterpart, where there are defined the prompts used by the P5 author's for the evaluation phase.
Each *Eval* task has one *seen* prompt (a prompt used during fine-tuning) and an *unseen* one.

### P5RatingTask

The *RatingPredictionTask* is built for [AmazonDataset](available_datasets.md#amazondataset): the goal is to predict
the rating that the *user* would give to an *unseen* item.

**Inference templates**:

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">1-1</td>
          <td style="vertical-align: middle">
            Which star rating will user_{user_id} give item_{item_id} ? ( 1 being lowest and 5 being highest )
          </td>
          <td style="vertical-align: middle; text-align: center">
            {star_rating}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">1-2</td>
          <td style="vertical-align: middle">
            How will user_{user_id} rate this product : {item_title} ? ( 1 being lowest and 5 being highest )
          </td>
          <td style="vertical-align: middle; text-align: center">
            {star_rating}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">1-5</td>
          <td style="vertical-align: middle">
            Predict the user_{user_id} 's preference on item_{item_id} ( {item_title} ) <br/> 
            -1 <br/> 
            -2 <br/> 
            -3 <br/> 
            -4 <br/> 
            -5
          </td>
          <td style="vertical-align: middle; text-align: center">
            {star_rating}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">1-6</td>
          <td style="vertical-align: middle">
            What star rating do you think {user_name} will give item_{item_id} ? ( 1 being lowest and 5 being highest )
          </td>
          <td style="vertical-align: middle; text-align: center">
            {star_rating}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">1-7</td>
          <td style="vertical-align: middle">
            How will {user_name} rate this product : {item_title} ? ( 1 being lowest and 5 being highest )
          </td>
          <td style="vertical-align: middle; text-align: center">
            {star_rating}
          </td>
        </tr>
      </tbody>
    </table>
</center>

**Support templates**:

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">1-3</td>
          <td style="vertical-align: middle">
            Will user_{user_id} give item_{item_id} a {star_rating}-star rating ? ( 1 being lowest and 5 being highest )
          </td>
          <td style="vertical-align: middle; text-align: center">
            {yes_no}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">1-4</td>
          <td style="vertical-align: middle">
            Does user_{user_id} like or dislike item_{item_id} ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {like_dislike}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">1-8</td>
          <td style="vertical-align: middle">
            Will {user_name} give a {star_rating}-star rating for {item_title} ? ( 1 being lowest and 5 being highest )
          </td>
          <td style="vertical-align: middle; text-align: center">
            {yes_no}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">1-9</td>
          <td style="vertical-align: middle">
            Does {user_name} like or dislike {item_title} ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {like_dislike}
          </td>
        </tr>
      </tbody>
    </table>
</center>

### P5EvalRatingTask

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">1-6</td>
          <td style="vertical-align: middle">
            What star rating do you think {user_name} will give item_{item_id} ? ( 1 being lowest and 5 being highest )
          </td>
          <td style="vertical-align: middle; text-align: center">
            {star_rating}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">1-10</td>
          <td style="vertical-align: middle">
            Predict {user_name} 's preference towards {item_title} ( 1 being lowest and 5 being highest )
          </td>
          <td style="vertical-align: middle; text-align: center">
            {star_rating}
          </td>
        </tr>
      </tbody>
    </table>
</center>


### P5SequentialTask

The SequentialSideInfoTask is built for [AmazonDataset](available_datasets.md#amazondataset): the goal is to predict
the next item of the **order history** of the user. This task has the **SideInfo** suffix because **categories** of the
items bought by the user are used additional information for the prediction.

**Inference templates**:

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-1</td>
          <td style="vertical-align: middle">
            Given the following purchase history of user_{user_id} : <br/> 
            {order_history} <br/> 
            predict next possible item to be purchased by the user ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-2</td>
          <td style="vertical-align: middle">
            I find the purchase history list of user_{user_id} : <br/> 
            {order_history} <br/> 
            I wonder what is the next item to recommend to the user . Can you help me decide ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-3</td>
          <td style="vertical-align: middle">
            Here is the purchase history list of user_{user_id} : <br/> 
            {order_history} <br/> 
            try to recommend next item to the user
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-4</td>
          <td style="vertical-align: middle">
            Given the following purchase history of {user_name} : <br/> 
            {order_history} <br/> 
            predict next possible item for the user
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-5</td>
          <td style="vertical-align: middle">
            Based on the purchase history of {user_name} : <br/> 
            {order_history} <br/> 
            Can you decide the next item likely to be purchased by the user ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-6</td>
          <td style="vertical-align: middle">
            Here is the purchase history of {user_name} : <br/> 
            {order_history} <br/> 
            What to recommend next for the user ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>

**Extractive QAs templates:**

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-7</td>
          <td style="vertical-align: middle">
            Here is the purchase history of user_{user_id} : <br/> 
            {order_history} <br/> 
            Select the next possible item likely to be purchased by the user from the following candidates : <br/> 
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-8</td>
          <td style="vertical-align: middle">
            Given the following purchase history of {user_name} : <br/> 
            {order_history} <br/> 
            What to recommend next for the user? Select one from the following items : <br/> 
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-9</td>
          <td style="vertical-align: middle">
            Based on the purchase history of user_{user_id} : <br/> 
            {order_history} <br/> 
            Choose the next possible purchased item from the following candidates : <br/> 
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-10</td>
          <td style="vertical-align: middle">
            I find the purchase history list of {user_name} : <br/> 
            {order_history} <br/> 
            I wonder which is the next item to recommend to the user . Try to select one from the following candidates : <br/> 
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>

**Pairwise prediction templates**:

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-11</td>
          <td style="vertical-align: middle">
            user_{user_id} has the following purchase history : <br/> 
            {order_history} <br/> 
            does the user likely to buy {target_item} next ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {yes_no}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-12</td>
          <td style="vertical-align: middle">
            According to {user_name} 's purchase history list : <br/> 
            {order_history} <br/> 
            Predict whether the user will purchase {target_item} next ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {yes_no}
          </td>
        </tr>
      </tbody>
    </table>
</center>

### P5EvalSequentialTask

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-3</td>
          <td style="vertical-align: middle">
            Here is the purchase history list of user_{user_id} : <br/> 
            {order_history} <br/> 
            try to recommend next item to the user
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">2-13</td>
          <td style="vertical-align: middle">
            According to the purchase history of {user_name} : <br/> 
            {order_history} <br/> 
            Can you recommend the next possible item to the user ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>


### P5DirectTask

The P5DirectTask is built for [AmazonDataset](available_datasets.md#amazondataset): the goal is to predict
a good item to recommend for the *user*.

**Inference templates**:

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">5-5</td>
          <td style="vertical-align: middle">
            Which item of the following to recommend for {user_name} ? <br/> 
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">5-6</td>
          <td style="vertical-align: middle">
            Choose the best item from the candidates to recommend for {user_name} ? <br/> 
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">5-7</td>
          <td style="vertical-align: middle">
            Pick the most suitable item from the following list and recommend to user_{user_id} : <br/> 
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>

**Support templates**:

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">5-1</td>
          <td style="vertical-align: middle">
            Will user_{user_id} likely to interact with item_{item_id} ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {yes_no}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">5-2</td>
          <td style="vertical-align: middle">
            Shall we recommend item_{item_id} to {user_name} ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {yes_no}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">5-3</td>
          <td style="vertical-align: middle">
            For {user_name}, do you think it is good to recommend {item_title} ?
          </td>
          <td style="vertical-align: middle; text-align: center">
            {yes_no}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">5-4</td>
          <td style="vertical-align: middle">
            I would like to recommend some items for user_{user_id} . Is the following item a good choice ? <br/> 
            {item_title}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {yes_no}
          </td>
        </tr>
      </tbody>
    </table>
</center>

### P5EvalDirectTask

<center>
    <table>
      <thead>
        <tr>
          <th style="text-align: center">Template ID</th>
          <th style="text-align: center">Input Placeholder text</th>
          <th style="text-align: center">Target Placeholder text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="vertical-align: middle; text-align: center">5-5</td>
          <td style="vertical-align: middle">
            Which item of the following to recommend for {user_name} ? <br/> 
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
        <tr>
          <td style="vertical-align: middle; text-align: center">5-8</td>
          <td style="vertical-align: middle">
            We want to make recommendation for user_{user_id} .  Select the best item from these candidates : <br/> 
            {candidate_items}
          </td>
          <td style="vertical-align: middle; text-align: center">
            {target_item}
          </td>
        </tr>
      </tbody>
    </table>
</center>
