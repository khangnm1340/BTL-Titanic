# %%
import os
from joblib.parallel import ThreadingBackend
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split,
    GridSearchCV,
)

output_dir = "titanic_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

print("Generating Histograms/KDE Plots for Numerical Features...")

# %%
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# %%
train_df.head(5)
train_df.info()
train_df.describe()
train_df.describe(include=["O"])
print(train_df.isnull().sum())
# %%
train_df.groupby(["Pclass"], as_index=False)["Survived"].mean()
train_df.groupby(["Sex"], as_index=False)["Survived"].mean()
train_df.groupby(["SibSp"], as_index=False)["Survived"].mean()
train_df.groupby(["Parch"], as_index=False)["Survived"].mean()
train_df["Family_Size"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["Family_Size"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df.head(10)
train_df.groupby(["Family_Size"], as_index=False)["Survived"].mean()
# %%
family_map = {
    1: "Alone",
    2: "Small",
    3: "Small",
    4: "Small",
    5: "Medium",
    6: "Medium",
    7: "Medium",
    8: "Large",
    11: "Large",
}
train_df["Family_Size_Group"] = train_df["Family_Size"].map(family_map)  # type: ignore[reportArgumentType]
test_df["Family_Size_Group"] = train_df["Family_Size"].map(family_map)  # type: ignore[reportArgumentType]

# %%
train_df.groupby(["Family_Size_Group"], as_index=False)["Survived"].mean()

# --- Family_Size Distribution ---
plt.figure(figsize=(8, 5))
sns.histplot(
    train_df["Family_Size"],
    bins=train_df["Family_Size"].max(),
    kde=True,
    color="lightgreen",
)
plt.title("Distribution of Family Size")
plt.xlabel("Family Size")
plt.ylabel("Count")
plt.xticks(range(1, train_df["Family_Size"].max() + 1))  # Show integer ticks
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "family_size_distribution.png"))
plt.close()
# %%
print("Saved family_size_distribution.png")
train_df.groupby(["Embarked"], as_index=False)["Survived"].mean()
sns.displot(train_df, x="Age", col="Survived", binwidth=10, height=5)
plt.savefig(os.path.join(output_dir, "age_distribution_by_survival.png"))
plt.close()

# %%
train_df["Age_Cut"] = pd.qcut(train_df["Age"], 8)
test_df["Age_Cut"] = pd.qcut(test_df["Age"], 8)
train_df.groupby(["Age_Cut"], as_index=False)["Survived"].mean()

train_df.loc[train_df["Age"] <= 16, "Age"] = 0
train_df.loc[(train_df["Age"] > 16) & (train_df["Age"] <= 20.125), "Age"] = 1
train_df.loc[(train_df["Age"] > 20.125) & (train_df["Age"] <= 24.0), "Age"] = 2
train_df.loc[(train_df["Age"] > 24.0) & (train_df["Age"] <= 28.0), "Age"] = 3
train_df.loc[(train_df["Age"] > 28.0) & (train_df["Age"] <= 32.312), "Age"] = 4
train_df.loc[(train_df["Age"] > 32.312) & (train_df["Age"] <= 38.0), "Age"] = 5
train_df.loc[(train_df["Age"] > 38.0) & (train_df["Age"] <= 47.0), "Age"] = 6
train_df.loc[(train_df["Age"] > 47.0) & (train_df["Age"] <= 80.0), "Age"] = 7
train_df.loc[train_df["Age"] > 80, "Age"] = 8

test_df.loc[test_df["Age"] <= 16, "Age"] = 0
test_df.loc[(test_df["Age"] > 16) & (test_df["Age"] <= 20.125), "Age"] = 1
test_df.loc[(test_df["Age"] > 20.125) & (test_df["Age"] <= 24.0), "Age"] = 2
test_df.loc[(test_df["Age"] > 24.0) & (test_df["Age"] <= 28.0), "Age"] = 3
test_df.loc[(test_df["Age"] > 28.0) & (test_df["Age"] <= 32.312), "Age"] = 4
test_df.loc[(test_df["Age"] > 32.312) & (test_df["Age"] <= 38.0), "Age"] = 5
test_df.loc[(test_df["Age"] > 38.0) & (test_df["Age"] <= 47.0), "Age"] = 6
test_df.loc[(test_df["Age"] > 47.0) & (test_df["Age"] <= 80.0), "Age"] = 7
test_df.loc[test_df["Age"] > 80, "Age"] = 8

# %%
# --- Original Age Distribution with Cut Points ---
# To get the "actual age" before binning, we'll temporarily load the CSV again for this plot.
original_train_df = pd.read_csv("train.csv")
age_cut_bins = pd.qcut(original_train_df["Age"], 8, retbins=True)[1]

plt.figure(figsize=(10, 6))
sns.histplot(original_train_df["Age"], kde=True, color="purple", bins=30)
plt.title("Distribution of Original Age with Quantile Cut Points")
plt.xlabel("Original Age")
plt.ylabel("Count")
for cut in age_cut_bins:
    plt.axvline(cut, color="red", linestyle="--", linewidth=1, alpha=0.7)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "age_with_cuts_distribution.png"))
plt.close()
print("Saved original_age_with_cuts_distribution.png")
# %%
# --- Original Age Distribution with Cut Points ---
# To get the "actual age" before binning, we'll temporarily load the CSV again for this plot.
original_train_df = pd.read_csv("train.csv")
age_cut_bins = pd.qcut(original_train_df["Age"], 8, retbins=True)[1]

plt.figure(figsize=(10, 6))
sns.histplot(original_train_df["Age"], kde=True, color="purple", bins=30)
plt.title("Distribution of Original Age with Quantile Cut Points")
plt.xlabel("Original Age")
plt.ylabel("Count")
for cut in age_cut_bins:
    plt.axvline(cut, color="red", linestyle="--", linewidth=1, alpha=0.7)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "original_age_with_cuts_distribution.png"))
plt.close()
print("Saved original_age_with_cuts_distribution.png")

# %%
# --- Age Distribution (after binning to 0-8 categories) ---
plt.figure(figsize=(8, 5))
age_categories = sorted(train_df["Age"].dropna().unique())
# Create bin edges that center the bars on the integer categories
age_bins = [x - 0.5 for x in age_categories] + [age_categories[-1] + 0.5]
sns.histplot(train_df["Age"], bins=age_bins, kde=True, color="skyblue")
plt.title("Distribution of Age Categories (After Binning)")
plt.xlabel("Age Category")
plt.ylabel("Count")
plt.xticks(age_categories)  # Ensure all categories are shown as ticks
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "binned_age_distribution.png"))
plt.close()
print("Saved binned_age_distribution.png")
# %%
train_df.head()
sns.displot(train_df, x="Fare", col="Survived", binwidth=80, height=5)
plt.savefig(os.path.join(output_dir, "fare_distribution_by_survival.png"))
plt.close()

# %%
train_df["Fare_Cut"] = pd.qcut(train_df["Fare"], 8)
test_df["Fare_Cut"] = pd.qcut(test_df["Fare"], 8)
train_df.groupby(["Fare_Cut"], as_index=False)["Survived"].mean()

# %%
train_df.loc[train_df["Fare"] <= 7.75, "Fare"] = 0
train_df.loc[(train_df["Fare"] > 7.75) & (train_df["Fare"] <= 7.91), "Fare"] = 1
train_df.loc[(train_df["Fare"] > 7.91) & (train_df["Fare"] <= 9.841), "Fare"] = 2
train_df.loc[(train_df["Fare"] > 9.841) & (train_df["Fare"] <= 14.454), "Fare"] = 3
train_df.loc[(train_df["Fare"] > 14.454) & (train_df["Fare"] <= 24.479), "Fare"] = 4
train_df.loc[(train_df["Fare"] > 24.479) & (train_df["Fare"] <= 31.0), "Fare"] = 5
train_df.loc[(train_df["Fare"] > 31.0) & (train_df["Fare"] <= 69.488), "Fare"] = 6
train_df.loc[(train_df["Fare"] > 69.488) & (train_df["Fare"] <= 512.329), "Fare"] = 7
train_df.loc[train_df["Fare"] > 512.329, "Fare"] = 8

test_df.loc[test_df["Fare"] <= 7.75, "Fare"] = 0
test_df.loc[(test_df["Fare"] > 7.75) & (test_df["Fare"] <= 7.91), "Fare"] = 1
test_df.loc[(test_df["Fare"] > 7.91) & (test_df["Fare"] <= 9.841), "Fare"] = 2
test_df.loc[(test_df["Fare"] > 9.841) & (test_df["Fare"] <= 14.454), "Fare"] = 3
test_df.loc[(test_df["Fare"] > 14.454) & (test_df["Fare"] <= 24.479), "Fare"] = 4
test_df.loc[(test_df["Fare"] > 24.479) & (test_df["Fare"] <= 31.0), "Fare"] = 5
test_df.loc[(test_df["Fare"] > 31.0) & (test_df["Fare"] <= 69.488), "Fare"] = 6
test_df.loc[(test_df["Fare"] > 69.488) & (test_df["Fare"] <= 512.329), "Fare"] = 7
test_df.loc[test_df["Fare"] > 512.329, "Fare"] = 8
# %%

# --- Original Fare Distribution with Cut Points ---
# To get the "actual fare" before binning, we'll temporarily load the CSV again for this plot.
original_train_df = pd.read_csv("train.csv")
# Handle potential NaN values in original_train_df['Fare'] before qcut
original_train_df["Fare"].fillna(original_train_df["Fare"].mean(), inplace=True)
fare_cut_bins = pd.qcut(original_train_df["Fare"], 8, retbins=True)[1]
plt.figure(figsize=(10, 6))
sns.histplot(original_train_df["Fare"], kde=True, color="teal", bins=30)
plt.title("Distribution of Original Fare with Quantile Cut Points")
plt.xlabel("Original Fare")
plt.ylabel("Count")
for cut in fare_cut_bins:
    plt.axvline(cut, color="red", linestyle="--", linewidth=1, alpha=0.7)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "original_fare_with_cuts_distribution.png"))
plt.close()
print("Saved original_fare_with_cuts_distribution.png")
# %%
# --- Fare Distribution (after binning to 0-8 categories) ---
plt.figure(figsize=(8, 5))
sns.histplot(
    train_df["Fare"], bins=len(train_df["Fare"].unique()), kde=True, color="lightcoral"
)
plt.title("Distribution of Fare Categories (After Binning)")
plt.xlabel("Fare Category")
plt.ylabel("Count")
plt.xticks(
    sorted(train_df["Fare"].unique())
)  # Ensure all categories are shown as ticks
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "binned_fare_distribution.png"))
plt.close()
print("Saved binned_fare_distribution.png")
# %%
# String Split
train_df["Name"]

train_df["Title"] = (
    train_df["Name"]
    .str.split(pat=",", expand=True)[1]
    .str.split(pat=".", expand=True)[0]
    .apply(lambda x: x.strip())
)

test_df["Title"] = (
    test_df["Name"]
    .str.split(pat=",", expand=True)[1]
    .str.split(pat=".", expand=True)[0]
    .apply(lambda x: x.strip())
)
train_df.groupby(["Title"], as_index=False)["Survived"].mean()

# NOTE:  military - Capt,Col,Major
# Noble - Jonkheer, the Countess, Don, Lady, Sir
# Unmarried Female - Mlle, Ms, Mme
# %%

train_df["Title"] = train_df["Title"].replace(
    {
        "Capt": "Military",
        "Col": "Military",
        "Major": "Military",
        "Jonkheer": "Noble",
        "the Countess": "Noble",
        "Don": "Noble",
        "Lady": "Noble",
        "Sir": "Noble",
        "Mlle": "Miss",
        "Mme": "Mrs",
    }
)
test_df["Title"] = test_df["Title"].replace(
    {
        "Capt": "Military",
        "Col": "Military",
        "Major": "Military",
        "Jonkheer": "Noble",
        "the Countess": "Noble",
        "Don": "Noble",
        "Lady": "Noble",
        "Sir": "Noble",
        "Mlle": "Miss",
        "Mme": "Mrs",
    }
)

train_df.groupby(["Title"], as_index=False)["Survived"].agg(["count", "mean"])
plt.clf()
# %%
train_df["Name_Length"] = train_df["Name"].apply(lambda x: len(x))
test_df["Name_Length"] = test_df["Name"].apply(lambda x: len(x))
# %%
g = sns.kdeplot(
    train_df["Name_Length"][
        (train_df["Survived"] == 0) & (train_df["Name_Length"].notnull())
    ],  # pyright: ignore
    color="Red",
    fill=True,
)
g = sns.kdeplot(
    train_df["Name_Length"][
        (train_df["Survived"] == 1) & (train_df["Name_Length"].notnull())
    ],  # pyright: ignore
    ax=g,
    color="Blue",
    fill=True,
)
g.set_xlabel("Name_Length")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived", "Survived"])
plt.savefig(os.path.join(output_dir, "Name_length_Survived.png"))
plt.close()

train_df["Name_LengthGB"] = pd.qcut(train_df["Name_Length"], 8)
test_df["Name_LengthGB"] = pd.qcut(test_df["Name_Length"], 8)
train_df.groupby(["Name_LengthGB"], as_index=False)["Survived"].mean()

# %%
train_df.loc[train_df["Name_Length"] <= 18.0, "Name_Size"] = 0
train_df.loc[
    (train_df["Name_Length"] > 18) & (train_df["Name_Length"] <= 20), "Name_Size"
] = 1
train_df.loc[
    (train_df["Name_Length"] > 20) & (train_df["Name_Length"] <= 23.0), "Name_Size"
] = 2
train_df.loc[
    (train_df["Name_Length"] > 23.0) & (train_df["Name_Length"] <= 25.0), "Name_Size"
] = 3
train_df.loc[
    (train_df["Name_Length"] > 25.0) & (train_df["Name_Length"] <= 27.25), "Name_Size"
] = 4
train_df.loc[
    (train_df["Name_Length"] > 27.25) & (train_df["Name_Length"] <= 30), "Name_Size"
] = 5
train_df.loc[
    (train_df["Name_Length"] > 30) & (train_df["Name_Length"] <= 38), "Name_Size"
] = 6
train_df.loc[
    (train_df["Name_Length"] > 38) & (train_df["Name_Length"] <= 82), "Name_Size"
] = 7
train_df.loc[train_df["Name_Length"] > 82, "Name_Size"] = 8

test_df.loc[test_df["Name_Length"] <= 18.0, "Name_Size"] = 0
test_df.loc[
    (test_df["Name_Length"] > 18) & (test_df["Name_Length"] <= 20), "Name_Size"
] = 1
test_df.loc[
    (test_df["Name_Length"] > 20) & (test_df["Name_Length"] <= 23.0), "Name_Size"
] = 2
test_df.loc[
    (test_df["Name_Length"] > 23.0) & (test_df["Name_Length"] <= 25.0), "Name_Size"
] = 3
test_df.loc[
    (test_df["Name_Length"] > 25.0) & (test_df["Name_Length"] <= 27.25), "Name_Size"
] = 4
test_df.loc[
    (test_df["Name_Length"] > 27.25) & (test_df["Name_Length"] <= 30), "Name_Size"
] = 5
test_df.loc[
    (test_df["Name_Length"] > 30) & (test_df["Name_Length"] <= 38), "Name_Size"
] = 6
test_df.loc[
    (test_df["Name_Length"] > 38) & (test_df["Name_Length"] <= 82), "Name_Size"
] = 7
test_df.loc[test_df["Name_Length"] > 82, "Name_Size"] = 8
# %%

# --- Name_Length Distribution ---
# You already have a KDE plot for Name_Length vs Survived, this is a general distribution
plt.figure(figsize=(8, 5))
sns.histplot(train_df["Name_Length"], kde=True, color="purple")
plt.title("Distribution of Name Length")
plt.xlabel("Name Length")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "name_length_distribution.png"))
plt.close()
print("Saved name_length_distribution.png")

# %%
train_df["TicketNumber"] = train_df["Ticket"].apply(
    lambda x: pd.Series({"Ticket": x.split()[-1]})
)
test_df["TicketNumber"] = test_df["Ticket"].apply(
    lambda x: pd.Series({"Ticket": x.split()[-1]})
)
train_df.groupby(["TicketNumber"], as_index=False)["Survived"].agg(
    ["count", "mean"]
).sort_values("count", ascending=False)  # pyright:ignore
train_df.groupby("TicketNumber")["TicketNumber"].transform("count")
train_df["TicketNumberCounts"] = train_df.groupby("TicketNumber")[
    "TicketNumber"
].transform("count")
test_df["TicketNumberCounts"] = test_df.groupby("TicketNumber")[
    "TicketNumber"
].transform("count")
train_df.groupby(["TicketNumberCounts"], as_index=False)["Survived"].agg(
    ["count", "mean"]
).sort_values("count", ascending=False)  # pyright:ignore
train_df["Ticket"].str.split(pat=" ", expand=True)

# %%
train_df["TicketLocation"] = np.where(
    train_df["Ticket"].str.split(pat=" ", expand=True)[1].notna(),
    train_df["Ticket"].str.split(pat=" ", expand=True)[0].apply(lambda x: x.strip()),
    "Blank",
)
test_df["TicketLocation"] = np.where(
    test_df["Ticket"].str.split(pat=" ", expand=True)[1].notna(),
    test_df["Ticket"].str.split(pat=" ", expand=True)[0].apply(lambda x: x.strip()),
    "Blank",
)

train_df["TicketLocation"].value_counts()
# %%


train_df["TicketLocation"] = train_df["TicketLocation"].replace(
    {
        "SOTON/O.Q.": "SOTON/OQ",
        "C.A.": "CA",
        "CA.": "CA",
        "SC/PARIS": "SC/Paris",
        "S.C./PARIS": "SC/Paris",
        "A/4.": "A/4",
        "A/5.": "A/5",
        "A.5.": "A/5",
        "A./5.": "A/5",
        "W./C.": "W/C",
    }
)
test_df["TicketLocation"] = test_df["TicketLocation"].replace(
    {
        "SOTON/O.Q.": "SOTON/OQ",
        "C.A.": "CA",
        "CA.": "CA",
        "SC/PARIS": "SC/Paris",
        "S.C./PARIS": "SC/Paris",
        "A/4.": "A/4",
        "A/5.": "A/5",
        "A.5.": "A/5",
        "A./5.": "A/5",
        "W./C.": "W/C",
    }
)
train_df.groupby(["TicketLocation"], as_index=False)["Survived"].agg(["count", "mean"])
# %%
train_df["Cabin"] = train_df["Cabin"].fillna("U")
train_df["Cabin"] = pd.Series(
    [i[0] if not pd.isnull(i) else "x" for i in train_df["Cabin"]]
)
test_df["Cabin"] = test_df["Cabin"].fillna("U")
test_df["Cabin"] = pd.Series(
    [i[0] if not pd.isnull(i) else "x" for i in test_df["Cabin"]]
)
# Rich people has cabin and poor people doesn't
train_df.groupby(["Cabin"], as_index=False)["Survived"].agg(["count", "mean"])
# %%
train_df["Cabin_Assigned"] = train_df["Cabin"].apply(lambda x: 0 if x in ["U"] else 1)
test_df["Cabin_Assigned"] = test_df["Cabin"].apply(lambda x: 0 if x in ["U"] else 1)
train_df.groupby(["Cabin_Assigned"], as_index=False)["Survived"].agg(["count", "mean"])
# %%
train_df.info()
train_df.shape
test_df.shape
train_df.columns
test_df.info()
# %%
train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)
test_df["Age"].fillna(test_df["Age"].mean(), inplace=True)
test_df["Fare"].fillna(test_df["Fare"].mean(), inplace=True)

# %%
ohe = OneHotEncoder(sparse_output=False)
ode = OrdinalEncoder
SI = SimpleImputer(strategy="most_frequent")

# %%
ode_cols = ["Family_Size_Group"]
ohe_cols = ["Sex", "Embarked"]

# %%
X = train_df.drop(["Survived"], axis=1)
y = train_df["Survived"]
X_test = test_df.drop(["Age_Cut", "Fare_Cut"], axis=1)

# %%
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=21
)

# %%
ordinal_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
)

# %%
ohe_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)
# %%
col_trans = ColumnTransformer(
    transformers=[
        ("impute", SI, ["Age"]),
        ("ord_pipeline", ordinal_pipeline, ode_cols),
        ("ohe_pipeline", ohe_pipeline, ohe_cols),
        (
            "passthrough",
            "passthrough",
            ["Pclass", "TicketNumberCounts", "Cabin_Assigned", "Name_Size"],
        ),
    ],
    remainder="drop",
    n_jobs=-1,
)
# %%
rfc = RandomForestClassifier()
# %%
param_grid = {
    "n_estimators": [100, 150, 200],
    "min_samples_split": [5, 10, 15],
    "max_depth": [8, 9, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}
# %%
CV_rfc = GridSearchCV(
    estimator=rfc,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1,
)
# %%
pipefinalrfc = make_pipeline(col_trans, CV_rfc)
pipefinalrfc.fit(X_train, y_train)

# %%
print(CV_rfc.best_params_)
print(CV_rfc.best_score_)

# %%
dtc = DecisionTreeClassifier()
# %%
param_grid = {
    "min_samples_split": [5, 10, 15],
    "max_depth": [10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}
# %%
CV_dtc = GridSearchCV(
    estimator=dtc,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1,
)
# %%
pipefinaldtc = make_pipeline(col_trans, CV_dtc)
pipefinaldtc.fit(X_train, y_train)

# %%
print(CV_dtc.best_params_)
print(CV_dtc.best_score_)

# %%
knn = KNeighborsClassifier()
# %%
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "p": [1, 2],
}
# %%
CV_knn = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1,
)
# %%
pipefinalknn = make_pipeline(col_trans, CV_knn)
pipefinalknn.fit(X_train, y_train)

# %%
print(CV_knn.best_params_)
print(CV_knn.best_score_)
# %%
svc = SVC(probability=True)
# %%
param_grid = {
    "C": [100, 10, 1.0, 0.1, 0.001, 0.001],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
}
# %%
CV_svc = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1,
)
# %%
pipefinalsvc = make_pipeline(col_trans, CV_svc)
pipefinalsvc.fit(X_train, y_train)
# %%
print(CV_svc.best_params_)
print(CV_svc.best_score_)
# %%
lr = LogisticRegression()
# %%
param_grid = {
    "C": [100, 10, 1.0, 0.1, 0.001, 0.001],
}
# %%
CV_lr = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1,
)
# %%
pipefinallr = make_pipeline(col_trans, CV_lr)
pipefinallr.fit(X_train, y_train)
# %%
print(CV_lr.best_params_)
print(CV_lr.best_score_)
# %%
gnb = GaussianNB()
# %%
param_grid = {
    "var_smoothing": [0.00000001, 0.000000001, 0.00000001],
}
# %%
CV_gnb = GridSearchCV(
    estimator=gnb,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1,
)
# %%
pipefinalgnb = make_pipeline(col_trans, CV_gnb)
pipefinalgnb.fit(X_train, y_train)
# %%
print(CV_gnb.best_params_)
print(CV_gnb.best_score_)

# %%
Y_pred = pipefinaldtc.predict(X_test)

# %%
test_results_df = test_df.copy()
test_results_df["Survived_Prediction"] = Y_pred

desired_order = ["PassengerId", "Survived_Prediction"] + [
    col
    for col in test_results_df.columns
    if col not in ["PassengerId", "Survived_Prediction"]
]
test_results_df = test_results_df[desired_order]

output_filename = "detailed_predictions.csv"  # Choose a descriptive name
test_results_df.to_csv(output_filename, index=False)

# %%
result_df = pd.read_csv("detailed_predictions.csv")
result_df.describe()
result_df.info()
train_df.info()
subset_df = result_df[["PassengerId", "Survived_Prediction"]]
print(subset_df.head())  # Print the first 5 rows of this subset
