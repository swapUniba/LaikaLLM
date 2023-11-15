# Available Metrics

To specify one of the following metrics as metric in the `.yaml` file, you can simply use its name 
(The parsing is **not** *case-sensitive*)!

For example, to use [\( Hit \)](#hit-hitk) and [\( MAP@10 \)](#map-mapk):

```yaml title="Example using Hit and MAP@10"
...
eval:
  SequentialSideInfoTask:
    - hit
    - map@10
    ...
```

## Ranking Metrics

!!! info

    Each *Ranking Metric* can evaluate recommendations produced with a **cut-off** value:
    To specify it, simply use ```METRIC_NAME@K```

### Hit <small>(Hit@K)</small>

The **Hit** metric simply check if, for each user, *at least* one **relevant item** (that is, an item present
in the ground truth of the user) has been recommended.

In math formulas, the *Hit* for the single user is computed like this:

$$
Hit_u = \begin{cases}
1, & \text{if} \;\; Preds_u \cap Truth_u \neq \emptyset \\ 
0, & \text{otherwise}
\end{cases}
$$

Where:

- \( Preds_u \) is the recommendation list for user \( u \)
- \( Truth_u \) is the ground truth for user \( u \)

And the *Hit* for the whole model is basically the *average Hit* for each user:

$$
Hit_{model} = \frac{1}{|U|} \cdot \sum_{i \in U}^{|U|} Hit_i
$$

Where:

- \( U \) is the set containing *all users*

---

### MAP <small>(MAP@K)</small>


The \( MAP \) metric (*Mean average Precision*) is a ranking metric computed by first calculating the \( AP \)
(*Average Precision*) for each user and then taking the average.

The \( AP \) is calculated as such for the single user:

$$
AP_u = \frac{1}{m_u}\sum_{i=1}^{N_u}P(i)\cdot rel_u(i)
$$

Where:

- \( m_u \) is the number of relevant items for the user \( u \)
- \( N_u \) is the number of recommended items for the user \( u \)
- \( P(i) \) is the precision computed at cutoff \( i \)
- \( rel_u(i) \) is a binary function defined as such:

$$
rel_u(i) = \begin{cases}
1, & \text{if} \;\; i \in Truth_u \\ 
0, & \text{otherwise}
\end{cases}
$$

After computing the \( AP \) for each user, the $MAP$ can be computed for the whole model:

$$
MAP_{model} = \frac{1}{|U|}\sum_{i \in |U|}^{|U|} AP_u
$$

---

### MRR <small>(MRR@K)</small>

The \( MRR \) (*Mean Reciprocal Rank*) computes, for each user, the inverse position of the first **relevant item**
(that is, an item present in the ground truth of the user), and then the *average* is computed to obtain a *model-wise*
metric.

In math formulas:

$$
MRR = \frac{1}{|U|}\cdot\sum_{i \in |U|}^{|U|}\frac{1}{rank(i)}
$$

Where:

- \( U \) is the set containing *all users*
- \( rank(i) \) is the position of the **first relevant item** in the recommendation list of the \( i-th \) user

---

### NDCG <small>(NDCG@K)</small>

The \( NDCG \) (*Normalized Discounted Cumulative Gain*) metric compares the **actual ranking** with the
**ideal one**. First, the \( DCG \) is computed for each user:

$$
DCG_{u} = \sum_{i}^{|Preds_u|}{\frac{rel_u(i)}{log_2(i + 1)}}
$$

Where:

- \( Preds_u \) is the recommendation list for user \( u \)
- \( rel_u(i) \) is a binary function defined as such:

$$
rel_u(i) = \begin{cases}
1, & \text{if} \;\; i \in Truth_u \\ 
0, & \text{otherwise}
\end{cases}
$$

Then the \( NDCG \) for a single user is calculated using the following formula:

$$
NDCG_u = \frac{DCG_{u}}{IDCG_{u}}
$$

Where:

- \( IDCG_{u} \) is the \( DCG_u \) sorted in descending order (representing the *ideal ranking*)


Finally, the \( NDCG \) of the **whole model** is calculated averaging the \( NDCG \) of each user:

$$
NDCG_{model} = \frac{1}{|U|} \cdot \sum_{i \in U}^{|U|} NDCG_i
$$

Where:

- \( U \) is the set containing *all users*


<hr style="border:2px solid gray">


## Error Metrics

Error metrics calculate the **error** the model made in predicting the rating a *particular user* would have 
given to an **unseen item**

### MAE

The \( MAE \) (*Mean Absolute Error*) computes, in absolute value, the difference between *rating predicted* and
*actual rating*:

$$
MAE_{model} = \frac{1}{|U|} \cdot \sum_{u \in U} \sum_{i \in Preds_u} |r_{u,i} - \hat{r}_{u,i}|
$$

Where:

- \( U \) is the set containing *all users*
- \( r_{u, i} \) is the *actual score* give by user \( u \) to item \( i \)
- \( \hat{r}_{u, i} \) is the *predicted score* give by user \( u \) to item \( i \)


---

### RMSE

The \( RMSE \) (*Root Mean Squared Error*) computes the difference, squared, between *rating predicted* and
*actual rating*:

$$
RMSE_{model} = \sqrt{\frac{1}{|U|} \cdot \sum_{u \in U} \sum_{i \in Preds_u} (r_{u,i} - \hat{r}_{u,i})^2}
$$

Where:

- \( U \) is the set containing *all users*
- \( r_{u, i} \) is the *actual score* give by user \( u \) to item \( i \)
- \( \hat{r}_{u, i} \) is the *predicted score* give by user \( u \) to item \( i \)
