# Helper file for clustering functions

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

def calculate_zscore(df, columns):
    '''
    scales columns in dataframe using z-score
    '''
    df = df.copy()
    for col in columns:
        df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0) # Delta degrees of freedom = 0 -> divisor is number of datapoints

    return df

def one_hot_encode(df, columns):
    '''
    one-hot encode the categorical columns and concatenate back to the original dataframe
    '''

    concat_df = pd.concat([pd.get_dummies(df[col], drop_first=True, prefix=col) for col in columns], axis=1)
    one_hot_cols = concat_df.columns

    return concat_df, one_hot_cols

def plot_sse(df, start=2, end=11):
    '''
    plot sum of squared error from cluster numbers
    '''
    sse = []
    for k in range(start, end):
        # Assign the labels to the clusters
        kmeans = KMeans(n_clusters=k, random_state=10).fit(df)
        sse.append({"k": k, "sse": kmeans.inertia_})

    sse = pd.DataFrame(sse)
    # Plot the data
    plt.plot(sse.k, sse.sse)
    plt.xlabel("K")
    plt.ylabel("Sum of Squared Errors")
    plt.show()
    
# Group ethnicities in groups for later one-hot encoding (mapping generated with ChatGPT)
ethnicity_mapping = {
    # African descent
    'African Americans': 'African',
    'Afro Trinidadians and Tobagonians': 'African',
    'Black people': 'African',
    'Black Canadians': 'African',
    'Akan people': 'African',
    'Ghanaian Americans': 'African',

    # Asian descent
    'Asian people': 'Asian',
    'Indians': 'Asian',
    'Punjabis': 'Asian',
    'Bengali': 'Asian',
    'Gujarati people': 'Asian',
    'Sindhis': 'Asian',
    'Tamil': 'Asian',
    'Telugu people': 'Asian',
    'Malayali': 'Asian',
    'Indian Americans': 'Asian',
    'Japanese Americans': 'Asian',
    'Japanese people': 'Asian',
    'Koreans': 'Asian',
    'Hongkongers': 'Asian',
    'Malaysian Chinese': 'Asian',
    'Chinese Americans': 'Asian',
    'Chinese Canadians': 'Asian',
    'Taiwanese Americans': 'Asian',
    'Pathani': 'Asian',
    'Afghans in India': 'Asian',
    'Israeli Americans': 'Asian',
    'Palestinians in the United States': 'Asian',
    'Filipino Americans': 'Asian',
    'Indonesian Americans': 'Asian',

    # Hindi descent
    'Marathi people': 'Hindi',
    'Kayastha': 'Hindi',
    'Bunt (RAJPUT)': 'Hindi',
    'Chitrapur Saraswat Brahmin': 'Hindi',
    'Kanyakubja Brahmins': 'Hindi',
    'Parsi': 'Hindi',
    'Bengali Hindus': 'Hindi',

    # European descent
    'Irish Americans': 'European',
    'Irish people': 'European',
    'Irish Australians': 'European',
    'Black Irish': 'European',
    'Scottish Americans': 'European',
    'Scottish people': 'European',
    'Scottish Canadians': 'European',
    'English Americans': 'European',
    'English people': 'European',
    'English Australian': 'European',
    'Welsh people': 'European',
    'Welsh Americans': 'European',
    'White British': 'European',
    'British': 'European',
    'British Americans': 'European',
    'British Chinese': 'European',
    'British Nigerian': 'European',
    'Dutch Americans': 'European',
    'Dutch': 'European',
    'French': 'European',
    'French Americans': 'European',
    'French Canadians': 'European',
    'Germans': 'European',
    'German Americans': 'European',
    'Swiss': 'European',
    'Italians': 'European',
    'Italian Americans': 'European',
    'Italian Australians': 'European',
    'Italian Canadians': 'European',
    'Spanish Americans': 'European',
    'Spaniards': 'European',
    'Swedes': 'European',
    'Swedish Americans': 'European',
    'Norwegians': 'European',
    'Norwegian Americans': 'European',
    'Danish Americans': 'European',
    'Danes': 'European',
    'Polish Americans': 'European',
    'Polish Canadians': 'European',
    'Lithuanian Americans': 'European',
    'Slovak Americans': 'European',
    'Czech Americans': 'European',
    'Hungarians': 'European',
    'Hungarian Americans': 'European',
    'Ukrainians': 'European',
    'Ukrainian Americans': 'European',
    'Russian Americans': 'European',
    'Russian Canadians': 'European',
    'Sicilian Americans': 'European',
    'Greek Americans': 'European',
    'Greek Canadians': 'European',
    'Croatian Americans': 'European',
    'Croatian Australians': 'European',
    'Armenians': 'European',
    'Romani people': 'European',
    'Romanichal': 'European',
    'Austrians': 'European',
    'Albanian Americans': 'European',
    'Scotch-Irish Americans': 'European',
    'Anglo-Irish people': 'European',
    'Anglo-Celtic Australians': 'European',
    'White people': 'European',
    'White Americans': 'European',
    'White Africans of European ancestry': 'European',
    'European Americans': 'European',

    # Latin America
    'Hispanic and Latino Americans': 'Latin American',
    'Mexicans': 'Latin American',
    'Mexican Americans': 'Latin American',
    'Dominican Americans': 'Latin American',
    'Stateside Puerto Ricans': 'Latin American',
    'Puerto Ricans': 'Latin American',
    'Bolivian Americans': 'Latin American',
    'Honduran Americans': 'Latin American',
    'Criollo people': 'Latin American',
    'Latin American British': 'Latin American',

    # Oceania
    'Australians': 'Oceanian',
    'Australian Americans': 'Oceanian',
    'New Zealanders (Kiwi)': 'Oceanian',
    'Māori': 'Oceanian',

    # Indigenous
    'Native Americans in the United States': 'Indigenous',
    'Native Americans': 'Indigenous',
    'Cherokee': 'Indigenous',
    'Indigenous peoples of the Americas': 'Indigenous',
    'Lumbee': 'Indigenous',
    'Native Hawaiians': 'Indigenous',
    'Sámi people':  'Indigenous',
    'Cajun': 'Indigenous',

    # Jewish descent
    'Jewish people': 'Jewish',
    'Ashkenazi Jews': 'Jewish',
    'American Jews': 'Jewish',

    # Other or unclear
    'multiracial American': 'Other',
    'Q31340083': 'Other',
    'Pacific Islander Americans': 'Other',
}
