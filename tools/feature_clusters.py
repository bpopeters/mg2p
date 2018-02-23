#!/usr/bin/env python

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

GEODATA = '/home/bpop/thesis/mg2p/data/latin_geo.csv'

def make_geo_clusters():
    geo = pd.read_csv(GEODATA, 
        usecols=['G_CODE', 'G_LATITUDE', 'G_LONGITUDE'],
        index_col='G_CODE',
        na_values='--')
    # look-up table from iso codes to latitude and longitude where both are defined    
    geo = geo.loc[geo['G_LATITUDE'].notnull() & geo['G_LONGITUDE'].notnull()]
    print(geo)
    
    # square root of the number of languages in the data
    n = int(np.sqrt(geo.size))
    km = KMeans(n_clusters=n, random_state=0).fit(geo)

    labels = pd.Series(km.predict(geo), index=geo.index)
    labels.to_csv('/home/bpop/thesis/mg2p/data/latlongclusters.csv')
    
make_geo_clusters()


