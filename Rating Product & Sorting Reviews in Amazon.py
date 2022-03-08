#Rating Product & Sorting Reviews in Amazon

########################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv(r"C:\Users\hp\PycharmProjects\pythonProject2\amazon_review (1).csv")
df.head(10)
df.info()

#First, we can look at our average rating.

average_rating = df["overall"].mean()


#Note:day difference is between in the analysis date and the comment date.
df["day_diff"].sort_values()
df["day_diff"].describe([0.03, 0.05, 0.01, 0.25, 0.75, 0.90, 0.95, 0.99])


df["day_diff_scaled"] = MinMaxScaler(feature_range=(1,360)).fit(df[["day_diff"]]).transform(df[["day_diff"]])
df["day_diff_scaled"] = df["day_diff_scaled"].astype(int)

df.head()

timeBased_wa = df.loc[df["day_diff_scaled"] <= 30, "overall"].mean() * 28 / 100 + \
df.loc[(df["day_diff_scaled"] > 30) & (df["day_diff_scaled"] <= 90), "overall"].mean() * 26 / 100 + \
df.loc[(df["day_diff_scaled"] > 90) & (df["day_diff_scaled"] <= 180), "overall"].mean() * 24 / 100 + \
df.loc[df["day_diff_scaled"] > 180, "overall"].mean() * 22 / 100


print("Time-Based Weighted Average Rating: " + f'{timeBased_wa:.2f}' + "\n"
      "Average Rating: " + f'{average_rating:.2f}')


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
up = df["helpful_yes"].tolist()
down = df["helpful_no"].tolist()
comments = pd.DataFrame({"up": up, "down": down})

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

comments["wlb"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)
df["wilson_lower_bound"] = comments["wlb"]


df_top_comments = df.sort_values("wilson_lower_bound",ascending=False).head(20)
df_top_comments[["overall", "summary", "helpful_yes", "helpful_no", "wilson_lower_bound"]]









