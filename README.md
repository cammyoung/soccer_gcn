# Using a Graph Convolutional Network to Predict the Outcomes of European Soccer Matches

In this doc, we'll walk through:
1. Using `networkx` to manipulate the tabular data on European soccer matches into graph data structures based on the formations of the teams, the location of each player in the formation, and their attributes from the FIFA videogames; and
2. Using `gcn` to create a Graph Convolutional Network (GCN) to predict the outcome of the game

## Background

Looking at a number of models built off of the [publicly available data](https://www.kaggle.com/hugomathien/soccer), the majority of them used the tabular data largely as-is. As any fan of the sport who is also a machine learning expert can tell you, this does not take into account the one-on-one matchups that can be key determinants of the match outcome. This is because in a tabular format, no one player is linked to any other player in a consistent way. Nor are they associated with a position on the field. Some recent examples of key matchups that come to mind are Garreth Bale versus Maicon in the 2010 UEFA Champions League, or Lionel Messi versus Jerome Boateng in 2015's competition. If we were to use tabular data to predict the outcomes of those matches, Messi's attributes would be equally associated by the model to Jerome Boateng's as to Thomas Muller's on the other side of the pitch. Note that it would also fail to learn the famous connection between Messi, Luis Suarez, and Neymar.

Graph data structures, on the other hand, inherently connect *nodes* to each other via *edges*, and can weight these connections. A node is one point on the graph, or one player on the pitch in our case. An edge connects two nodes together. It is common to have two nodes that do not share an edge, and in this example, players who are at opposite ends of the pitch (e.g. a keeper and forward on the same team) will not share an edge. Nodes can also share an edge with themselves, called a *loop*, but we won't use these since players can't assist themselves.

Edges are defined through an *adjacency matrix* $A$, which will take a value of 1 in a particular element $A_{a,b}$ if nodes $a$ and $b$ share an edge. If the graph has weighted edges, the value of $A_{a,b}$ can be between 0 and 1, with higher values denoting a stronger connection between nodes.

We will set up the problem to create a graph based on the $(x, y)$ coordinates of players on the pitch, with weighted edges connecting them based on their distance to other players. Each node will have a number of attributes, such as age, height, left-footedness, crossing ability, etc. The GCN will perform convolution over the graph to extract features from the players in relation to other players on the pitch, and then pool those features to generate a prediction about the outcome of the match (home win, draw, or home loss).

## The Data
There are 3 tables in the dataset that we will use going forward. The `Match` table has information at the match level, including the players that started and their $(x,y)$ coordinates on the pitch. The `Player` table, which contains information on the birthday, height, and weight of the players. Lastly, we will use the `Player_Attributes` table, which contains the FIFA videogame ratings of the players' ability to perform certain skills in a game.

### Match Data
We will use the match data to create the edge weights of the graphs, by leveraging columns like `home_player_X*` and `away_player_Y*`, which tells us the starting coordinates of players in the match for both the home and away sides.

```
> match.iloc[:, :10].tail()
```

|    id |   country_id |   league_id | season    |   stage | date                |   match_api_id |   home_team_api_id |   away_team_api_id |   home_team_goal |
|------:|-------------:|------------:|:----------|--------:|:--------------------|---------------:|-------------------:|-------------------:|-----------------:|
| 25975 |        24558 |       24558 | 2015/2016 |       9 | 2015-09-22 00:00:00 |        1992091 |              10190 |              10191 |                1 |
| 25976 |        24558 |       24558 | 2015/2016 |       9 | 2015-09-23 00:00:00 |        1992092 |               9824 |              10199 |                1 |
| 25977 |        24558 |       24558 | 2015/2016 |       9 | 2015-09-23 00:00:00 |        1992093 |               9956 |              10179 |                2 |
| 25978 |        24558 |       24558 | 2015/2016 |       9 | 2015-09-22 00:00:00 |        1992094 |               7896 |              10243 |                0 |
| 25979 |        24558 |       24558 | 2015/2016 |       9 | 2015-09-23 00:00:00 |        1992095 |              10192 |               9931 |                4 |

```
> match = pd.read_sql('SELECT * FROM Match', con=con)
> match.iloc[:, 10:15].tail()
```

|   away_team_goal |   home_player_X1 |   home_player_X2 |   home_player_X3 |   home_player_X4 |
|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
|                0 |                1 |                2 |                4 |                6 |
|                2 |                1 |                3 |                5 |                7 |
|                0 |                1 |                2 |                4 |                6 |
|                0 |                1 |                2 |                4 |                6 |
|                3 |                1 |                2 |                4 |                6 |

The nuance here is that all starting lineups show the team's formation from their point of view, so we will need to flip the away team's coordinates about a horizontal axis in every match. First, however, we need to convert the data from wide to long format. We'll define some functions that help us achieve that.

```
### Functions to transform X/Y coords of certain players
# Keeper X coords are too far left
def keeper_xfrm(df):
    df['away_player_X'] = np.where(df.player == 1, 5, df.away_player_X)
    df['home_player_X'] = np.where(df.player == 1, 5, df.home_player_X)
    
    return df

# Flip away player Y coords about a horizontal axis
def away_xfrm(df):
    df['away_player_Y'] = -1*(df.away_player_Y - 11) + 3
    
    return df

### Clean data
def clean_match(match):
    # Drop betting odds columns
    bet_cols = match.columns[list(match.columns).index('B365H'):]
    match = match.drop(bet_cols, axis=1)
    
    # Create outcome column
    match['outcome'] = np.where(match.home_team_goal > match.away_team_goal,
                                'home_win',
                                np.where(match.home_team_goal == match.away_team_goal,
                                         'draw',
                                         'home_loss'))
    
    match = pd.get_dummies(match, columns = ['outcome'])
    
    # Create sin & cos date vars
    match['date'] = pd.to_datetime(match.date)
    match['sin_date'] = np.sin(2*np.pi*match.date.dt.dayofyear/365)
    match['cos_date'] = np.cos(2*np.pi*match.date.dt.dayofyear/365)
    
    # Rename some player columns to work with pd.wide_to_long
    match.columns = [re.sub('^(.*)_player_(\\d{,2})$',
                            '\\1_player_id\\2',
                            i) for i in match.columns]
    
    # Drop rows with any null values for ids or X/Y coords
    player_cols = [i for i in match.columns if 'player' in i]
    match = match.loc[~match[player_cols].isnull().any(axis=1)]
    
    return match
```

After some cleaning of the data to remove

### Player Data

Finally, we have data on the players themselves and their FIFA attributes at various points in time. I joined these together in the query from the `sqlite` file.

```
> sql = """SELECT p.birthday, p.height, p.weight, a.*
         FROM Player p, Player_Attributes a
         WHERE p.player_api_id = a.player_api_id
             AND p.player_fifa_api_id = a.player_fifa_api_id
         ORDER BY a.player_api_id, date"""
> player = pd.read_sql(sql, con)
> player.iloc[:,:10].head()
```

| birthday            |   height |   weight |     id |   player_fifa_api_id |   player_api_id | date                |   overall_rating |   potential | preferred_foot   |
|:--------------------|---------:|---------:|-------:|---------------------:|----------------:|:--------------------|-----------------:|------------:|:-----------------|
| 1981-01-27 00:00:00 |   175.26 |      154 | 139857 |               148544 |            2625 | 2007-02-22 00:00:00 |               63 |          64 | right            |
| 1981-01-27 00:00:00 |   175.26 |      154 | 139856 |               148544 |            2625 | 2007-08-30 00:00:00 |               63 |          64 | right            |
| 1981-01-27 00:00:00 |   175.26 |      154 | 139855 |               148544 |            2625 | 2008-08-30 00:00:00 |               60 |          64 | right            |
| 1981-01-27 00:00:00 |   175.26 |      154 | 139854 |               148544 |            2625 | 2010-08-30 00:00:00 |               60 |          64 | right            |
| 1981-01-27 00:00:00 |   175.26 |      154 | 139853 |               148544 |            2625 | 2011-02-22 00:00:00 |               59 |          63 | right            |

# Matrices

$$\begin {aligned}
n &= \text{number of nodes} \\
m &= \text{number of matches} \\
p &= \text{number of players} = 22 \\
d &= \text{number of features per node} \\
c &= \text{number of match outcomes} = 3 \\
\\
A &= A_{mn, mn} \ \text{adjacency matrix} \\
X &= X_{mn, md} \ \text{feature matrix} \\
P &= P_{mn, m} \ \text{pooling matrix} \\
Y &= Y_{m,c} \ \text{target matrix}
\end{aligned}$$