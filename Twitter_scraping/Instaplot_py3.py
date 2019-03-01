import geopandas
import pandas as pd
#import geoplot
import os
import descartes
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
import pyproj
import shapely
from shapely.geometry import Point
import string 
from skimage import data
from skimage import filters
from skimage import exposure


def geo_plot(A,Grid,colors):
    #The file we got from the geoportale is in Gauss-Boaga projection. its EPSG code is 3003 but it does not know it.
    df = geopandas.read_file('carta_sintesi_geo/carta_sintesi_geo.shp')
    #We have to tell it:
    df.crs={'init':'epsg:3003'}
    #And then to plot it in the usual GPS coordinates.
    #df.to_crs(epsg=4326).plot()#.to_crs({'proj': 'merc'}).plot()

    topic1=pd.DataFrame(index=Grid)
    for i in range(len(A.T)):
        data=list(A.T[i])
        Norm_factor=np.sqrt(sum([d**2 for d in data]))
        data=[d/Norm_factor for d in data]
        topic1['data{}'.format(i)] = pd.Series(data, index=Grid)
        #topic1=pd.DataFrame({'data{}'.format(i):list(A.T[i])}, index=Grid)

    #Our coordinates systems are transposed wrt what geopandas expect
    Grid2=[(a[1],a[0]) for a in Grid]
    #Encode points in a way that geopandas understands
    points=pd.Series([Point(a) for a in Grid2],index=Grid)
    topic1['points']=points
    #Our coordinates are in the usual WGS84 encoding (i.e. EPSG: 4326).
    TOP1 = geopandas.GeoDataFrame(topic1, geometry=points)
    TOP1.crs = {'init': 'epsg:4326', 'no_defs': True}

    for i in range(17):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        vmin, vmax= min(TOP1['data{}'.format(i)].tolist()),max(TOP1['data{}'.format(i)].tolist())
        df.to_crs(epsg=4326).plot(linewidth=0.01, ax=ax, edgecolor='gray', color='white')
        TOP1.plot(column='data{}'.format(i), markersize=60, ax=ax, cmap=colors,vmin=vmin,vmax=vmax, alpha=.5)
        ax.axis('off')
        # add a title
        ax.set_title('Intensity of topic {}'.format(i), fontdict={'fontsize': '25', 'fontweight' : '3'})

        # create an annotation for the data source
        ax.annotate('Source: Instagram, 2018',xy=(0.01, 1),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')

        # Create colorbar as a legend
        sm = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(vmin=vmin, vmax=vmax))

        # empty array for the data range
        sm._A = []

        # add the colorbar to the figure
        cbar = fig.colorbar(sm,shrink=.5)
        plt.show()
        fig.savefig('trial_geo/trial{}.pdf'.format(i))
        plt.close()

def geo_plot_one(row,Grid,colors,topic,save=False,folder='',shapefile='carta_sintesi_geo/carta_sintesi_geo.shp'):
    #The file we got from the Torino website is in Gauss-Boaga projection. its EPSG code is 3003 but it does not know it.
    df = geopandas.read_file(shapefile)
    #We have to tell it:
    df.crs={'init':'epsg:3003'}
    #And then to plot it in the usual GPS coordinates.
    #df.to_crs(epsg=4326).plot()#.to_crs({'proj': 'merc'}).plot()

    topic1=pd.DataFrame(index=Grid)
    data=list(row)
    Norm_factor=np.sqrt(sum([d**2 for d in data]))
    data=[d/Norm_factor for d in data]
    topic1['data'] = pd.Series(data, index=Grid)
    #topic1=pd.DataFrame({'data{}'.format(i):list(A.T[i])}, index=Grid)

    #Our coordinates systems are transposed wrt what geopandas expect
    Grid2=[(a[1],a[0]) for a in Grid]
    #Encode points in a way that geopandas understands
    points=pd.Series([Point(a) for a in Grid2],index=Grid)
    topic1['points']=points
    #Our coordinates are in the usual WGS84 encoding (i.e. EPSG: 4326).
    TOP1 = geopandas.GeoDataFrame(topic1, geometry=points)
    TOP1.crs = {'init': 'epsg:4326', 'no_defs': True}

    fig, ax = plt.subplots(1, figsize=(10, 10))
    vmin= min(TOP1['data'].tolist())
    vmax=max(TOP1['data'].tolist())
    df.to_crs(epsg=4326).plot(linewidth=0.01, ax=ax, edgecolor='gray', color='white')
    TOP1.plot(column='data', markersize=100, marker='s',ax=ax, cmap=colors,vmin=vmin,vmax=vmax, alpha=.5)
    ax.axis('off')
    # add a title
    ax.set_title(topic, fontdict={'fontsize': '25', 'fontweight' : '3'})

    # create an annotation for the data source
    ax.annotate('Source: Instagram, 2018',xy=(0.01, 1),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')

    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # empty array for the data range
    sm._A = []

    # add the colorbar to the figure
    cbar = fig.colorbar(sm,shrink=.5)
    plt.show()
    if save:
        print('file saved', folder+'_'.join(topic.split(' ')))
        fig.savefig(folder+'_'.join(topic.split(' '))+'.pdf')
    plt.close()
    
def geo_hm_one(row,Grid,colors,topic,save=False,folder='',shapefile='carta_sintesi_geo/carta_sintesi_geo.shp'):
    #The file we got from the Torino website is in Gauss-Boaga projection. its EPSG code is 3003 but it does not know it.
    df = geopandas.read_file(shapefile)
    df.crs={'init':'epsg:3003'}
    neighbourhoods=df.to_crs(epsg=4326).geometry.tolist()
    #We have to tell it:
    
    #And then to plot it in the usual GPS coordinates.
    #df.to_crs(epsg=4326).plot()#.to_crs({'proj': 'merc'}).plot()

    topic1=pd.DataFrame(index=neighbourhoods)
    data=list(row)
    Norm_factor=np.sqrt(sum([d**2 for d in data]))
    data=[d/Norm_factor for d in data]
    if not type(Grid[0])==shapely.geometry.point.Point:
        coord_data=list(zip([Point(a[1],a[0]) for a in Grid],[d**2 for d in data]))
    else:
        coord_data=list(zip(Grid,data))
    by_ngh=[0 for n in neighbourhoods]
    for n in neighbourhoods:
        for c in coord_data:
            if c[0].within(n):
                by_ngh[neighbourhoods.index(n)]+=c[1]
    topic1['data'] = pd.Series(by_ngh, index=neighbourhoods)
    #Our coordinates are in the usual WGS84 encoding (i.e. EPSG: 4326).
    TOP1 = geopandas.GeoDataFrame(topic1, geometry=neighbourhoods)
    TOP1.crs = {'init': 'epsg:4326', 'no_defs': True}

    fig, ax = plt.subplots(1, figsize=(10, 10))
    vmin= min(by_ngh)
    vmax=max(by_ngh)
    TOP1.to_crs(epsg=4326).plot(column='data', linewidth=0.5, ax=ax, edgecolor='gray', cmap=colors)
    #df.to_crs(epsg=4326).plot(linewidth=0.01, ax=ax, edgecolor='gray', color='white')
    #TOP1.plot(column='data', markersize=100, marker='s',ax=ax, cmap=colors,vmin=vmin,vmax=vmax, alpha=.5)
    ax.axis('off')
    # add a title
    ax.set_title(topic, fontdict={'fontsize': '25', 'fontweight' : '3'})

    # create an annotation for the data source
    ax.annotate('Source: Instagram, 2018',xy=(0.01, 1),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')

    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # empty array for the data range
    sm._A = []

    # add the colorbar to the figure
    cbar = fig.colorbar(sm,shrink=.5)
    plt.show()
    if save:
        print('file saved', folder+string.join(topic.split(' '),'_'))
        fig.savefig(folder+string.join(topic.split(' '),'_')+'_hm.pdf')
    plt.close()

def geo_hm_one_2(row,Grid,colors,topic,ax,save=False,folder='',shapefile='carta_sintesi_geo/carta_sintesi_geo.shp'):
    #The file we got from the Torino website is in Gauss-Boaga projection. its EPSG code is 3003 but it does not know it.
    df = geopandas.read_file(shapefile)
    df.crs={'init':'epsg:3003'}
    neighbourhoods=df.to_crs(epsg=4326).geometry.tolist()
    #We have to tell it:
    
    #And then to plot it in the usual GPS coordinates.
    #df.to_crs(epsg=4326).plot()#.to_crs({'proj': 'merc'}).plot()

    topic1=pd.DataFrame(index=neighbourhoods)
    data=list(row)
    Norm_factor=np.sqrt(sum([d**2 for d in data if str(d)!='nan']))
    data=[d/Norm_factor for d in data]
    if not type(Grid[0])==shapely.geometry.point.Point:
        coord_data=list(zip([Point(a[1],a[0]) for a in Grid],[d**2 for d in data]))
    else:
        coord_data=list(zip(Grid,data))
    by_ngh=[0 for n in neighbourhoods]
    for n in neighbourhoods:
        for c in coord_data:
            if str(c[1])!='nan':
                if c[0].within(n):
                    by_ngh[neighbourhoods.index(n)]+=c[1]
    vmin=min(by_ngh)
    vmax=max(by_ngh)
    by_ngh=np.array(by_ngh)
    val = filters.threshold_otsu(by_ngh)
    print('otsu threshold', val)
    by_ngh=np.ma.masked_where(by_ngh<val,by_ngh)
    topic1['data'] = pd.Series(by_ngh, index=neighbourhoods)
    #Our coordinates are in the usual WGS84 encoding (i.e. EPSG: 4326).
    TOP1 = geopandas.GeoDataFrame(topic1, geometry=neighbourhoods)
    TOP1.crs = {'init': 'epsg:4326', 'no_defs': True}
    TOP1.dropna().to_crs(epsg=4326).plot(column='data', linewidth=0.5, ax=ax, edgecolor='gray', cmap=colors, alpha=1)

    
    def heatmap_2(d, cmap, smoothing=1.3):
        def getx(pt):
            return pt.coords[0][0]

        def gety(pt):
            return pt.coords[0][1]

        x = list(d.geometry.apply(getx))
        y = list(d.geometry.apply(gety))
        dat=d[d.columns[0]].tolist()
        for i in xrange(len(dat)):
            if str(dat[i])=='nan':
                dat[i]=0
        Norm=np.sqrt(sum([d**2 for d in dat]))
        print(Norm)
        dat=[d/Norm for d in dat]
        absc=sorted(list(set(x)))
        ordo=sorted(list(set(y)))
        # Add the end of the last box
        ordo.append(2*ordo[-1]-ordo[-2])
        absc.append(2*absc[-1]-absc[-2])
        #Build a heatmap that shows the NODES and not the BOXES; we have one extra box for each coordinate.
        heatmap = np.zeros((len(absc)-1, len(ordo)-1))
        for i in xrange(len(x)):
            heatmap[ordo.index(y[i]),absc.index(x[i])]=dat[i]
        extent = [min(absc), max(absc), max(ordo), min(ordo)]
        heatmap = ndimage.filters.gaussian_filter(heatmap, smoothing, mode='nearest')   
        val = filters.threshold_otsu(heatmap)
        print(val)
        heatmap = np.ma.masked_where(heatmap<val, heatmap)
        #cmap=plt.cm.jet
        #cmap=plt.cmap=plt.cm.get_cmap('Blues', 6)
        cmap.set_bad(color='white')
        plt.imshow(heatmap, extent=extent,cmap=cmap,alpha=.5)
        plt.colorbar(shrink=.5)
        plt.gca().invert_yaxis()
        #plt.show()
        print (len(set(y)),len(set(x)))