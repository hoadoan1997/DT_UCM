import itertools
import pandas as pd
import numpy as np


def entropy(y):
    if isinstance(y, pd.Series):
        a = y.value_counts() / y.shape[0]
        e = np.sum(-a * np.log2(a + 1e-9))
        return e


def variance(y):
    if len(y) == 1:
        return 0
    else:
        return y.var()


def information_gain(y, mask, func=entropy):
    a = sum(mask)
    b = mask.shape[0] - a

    if a == 0 or b == 0:
        ig = 0

    else:
        if y.dtypes != 'O':
            ig = variance(y) - (a / (a + b) * variance(y[mask])) - (b / (a + b) * variance(y[-mask]))
        else:
            ig = func(y) - a / (a + b) * func(y[mask]) - b / (a + b) * func(y[-mask])

    return ig


def categorical_options(a):
    a = a.unique()

    opciones = []
    for L in range(0, len(a) + 1):
        for subset in itertools.combinations(a, L):
            subset = list(subset)
            opciones.append(subset)

    return opciones[1:-1]


def max_information_gain_split(x, y, func=entropy):
    split_value = []
    ig = []
    numeric_variable = not pd.api.types.is_bool_dtype(x)
    if numeric_variable:
        options = x.sort_values().unique()[1:]
    else:
        options = categorical_options(x)

    for val in options:
        mask = x < val if numeric_variable else x.isin(val)
        val_ig = information_gain(y, mask, func)
        ig.append(val_ig)
        split_value.append(val)

    if len(ig) == 0:
        return None, None, None, False

    else:
        best_ig = max(ig)
        best_ig_index = ig.index(best_ig)
        best_split = split_value[best_ig_index]
        return best_ig, best_split, numeric_variable, True


def get_best_split(y, data):
    masks = data.drop(y, axis=1).apply(max_information_gain_split, y=data[y])
    print(masks)
    if sum(masks.loc[3, :]) == 0:
        return None, None, None, None

    else:
        masks = masks.loc[:, masks.loc[3, :]]

        split_variable = masks.iloc[0].astype(np.float32).idxmax()
        split_value = masks[split_variable][1]
        split_ig = masks[split_variable][0]
        split_numeric = masks[split_variable][2]

        return split_variable, split_value, split_ig, split_numeric


def make_split(variable, value, data, is_numeric):
    if is_numeric:
        data_1 = data[data[variable] < value]
        data_2 = data[(data[variable] >= value)]

    else:
        data_1 = data[data[variable].isin(value)]
        data_2 = data[data[variable].isin(value) == False]

    return data_1, data_2


def make_prediction(data, target_factor):
    if target_factor:
        pred = data.value_counts().idxmax()
    else:
        pred = data.mean()

    return pred


def train_tree(data, y, target_factor, max_depth=None, min_samples_split=None, min_information_gain=1e-20, counter=0,
               max_categories=20):
    if counter == 0:
        types = data.dtypes
        check_columns = types[types == "object"].index
        for column in check_columns:
            var_length = len(data[column].value_counts())
            if var_length > max_categories:
                print
                'The variable ' + column + ' has ' + str(
                    var_length) + ' unique values, which is more than the accepted ones: ' + str(max_categories)

    if max_depth == None:
        depth_cond = True

    else:
        if counter < max_depth:
            depth_cond = True

        else:
            depth_cond = False

    if min_samples_split == None:
        sample_cond = True

    else:
        if data.shape[0] > min_samples_split:
            sample_cond = True

        else:
            sample_cond = False

    if depth_cond & sample_cond:

        var, val, ig, var_type = get_best_split(y, data)

        if ig is not None and ig >= min_information_gain:

            counter += 1

            left, right = make_split(var, val, data, var_type)

            split_type = "<=" if var_type else "in"
            question = "{} {}  {}".format(var, split_type, val)
            subtree = {question: []}

            yes_answer = train_tree(left, y, target_factor, max_depth, min_samples_split, min_information_gain, counter)

            no_answer = train_tree(right, y, target_factor, max_depth, min_samples_split, min_information_gain, counter)

            if yes_answer == no_answer:
                subtree = yes_answer

            else:
                subtree[question].append(yes_answer)
                subtree[question].append(no_answer)

        else:
            pred = make_prediction(data[y], target_factor)
            return pred

    else:
        pred = make_prediction(data[y], target_factor)
        return pred

    return subtree


def clasificar_datos(observacion, arbol):
    question = list(arbol.keys())[0]
    if question.split()[1] == '<=':

        if observacion[question.split()[0]] <= float(question.split()[2]):
            answer = arbol[question][0]
        else:
            answer = arbol[question][1]

    else:
        if observacion[question.split()[0]] in (question.split()[2]):
            answer = arbol[question][0]
        else:
            answer = arbol[question][1]

    # If the answer is not a dictionary
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return clasificar_datos(observacion, answer)


from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

df = pd.read_csv("hepatitis.csv")
tf_columns = ['steroid',
              'sex',
              'antivirals',
              'fatigue',
              'malaise',
              'anorexia',
              'liver_big',
              'liver_firm',
              'spleen_palable',
              'spiders',
              'ascites',
              'varices',
              'class',
              'histology']
df = df.reindex(sorted(df.columns), axis=1)
X_train, X_test = train_test_split(df, test_size=0.3, random_state=1)


def seed_continuous_noises(distance, data):
    noises = make_circles(n_samples=30, noise=np.mean(data[distance]), random_state=1)
    return abs(noises[0][:, 1])


continuous_columns = ['alk_phosphate', 'age', 'bilirubin', 'sgot', 'albumin', 'protime']
nosies = dict()
for continuous_column in continuous_columns:
    nosies[continuous_column] = seed_continuous_noises(continuous_column, df)
for format_column in tf_columns:
    nosies[format_column] = np.random.randint(2, size=30)
noises_df = pd.DataFrame(nosies)
noises_df = noises_df.reindex(sorted(noises_df.columns), axis=1)
noises_df = pd.concat([X_train, noises_df])
for column in df.columns:
    if column in tf_columns:
        noises_df[column] = noises_df[column].replace((0, 1), (False, True)).astype(type(False))
        X_test[column] = X_test[column].replace((0, 1), (False, True)).astype(type(False))

get_best_split('class', noises_df)
