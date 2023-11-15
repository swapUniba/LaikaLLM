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
