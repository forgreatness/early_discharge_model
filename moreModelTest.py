import pandas as pd
import numpy as np
import seaborn as srn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

encountersData = pd.read_csv('./diabetic_data.csv')

""" Data Preprocessing """
# . Remove undefined | null | not a number row
print("what is the size of the data I am working with", encountersData.shape)
encountersData.replace('?', np.nan, inplace=True)
nanCountPerFeature = encountersData.isna().sum()
featuresHavingTooMuchNan = nanCountPerFeature[nanCountPerFeature > 4000].index.values
encountersData.drop(featuresHavingTooMuchNan, axis=1, inplace=True)

encountersData = encountersData.dropna() #remove all rows where there is a column with nan value

# one of the method for turning categorical data into numerical
encountersData.replace({
    'readmitted': {
        'NO': 0,
        '<30': 1,
        '>30': 0
    }
}, inplace=True)

# So far we removed all rows where there are nan, but there are still rows which has value that are unexpecte like gender having unknown
# lets remove these people no ethical regards as they have decided to not know what they are and biologically it doesn't help
encountersData = encountersData.drop(index=encountersData[encountersData['race'] == 'Unknown/Invalid'].index)

# Help me do a replace targeting the age column, convert each value from 60-70 to an average so that we can work with numeric
encountersData = encountersData.replace({
    'age': {
        "[70-80)":75,
        "[60-70)":65,
        "[50-60)":55,
        "[80-90)":85,
        "[40-50)":45,
        "[30-40)":35,
        "[90-100)":95,
        "[20-30)":25,
        "[10-20)":15,
        "[0-10)":5
    }
})
print(encountersData['age'].value_counts())

####      Feature Selection     ####
selectedFeatures = pd.Series(['race', 'gender', 'age', 'time_in_hospital', 'num_procedures', 'num_medications', 'readmitted'])
featureSelectedEncounters = encountersData[selectedFeatures]
print(featureSelectedEncounters)

# print("Show me what the readmitted columns looks like", encountersData['readmitted'].unique())
# print("show me the count for each unique value", encountersData['readmitted'].value_counts())

""" Data Visualization """
def showReadmittedVisualization():
    srn.set_palette('hls')
    readmittedDistributionPlt = srn.countplot(x='readmitted', hue='readmitted', data=encountersData)
    for p in readmittedDistributionPlt.patches:
        readmittedDistributionPlt.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.title('Target Variable Distribution')
    plt.show()
    
def showReadmittedByRaceVisualization():
    srn.set_palette('hls')
    readmittedByRaceDistribution = srn.countplot(x='race', hue='readmitted', data=encountersData)
    for p in readmittedByRaceDistribution.patches:
        readmittedByRaceDistribution.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.xticks(rotation=90)
    plt.title('Readmitted Distribution by Race')
    plt.show()

# showReadmittedByRaceVisualization()
