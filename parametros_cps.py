import scipy.io as sio
import pandas as pd
from pandas import DataFrame as DF
from datetime import datetime
import xarray as xr
import cartopy.crs as ccrs
import netCDF4
import math
import numpy as np
from scipy.ndimage import gaussian_filter

from cartopy import config
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.img_tiles as cimgt

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.collections as collections

import IPython.display as IPdisplay, matplotlib.font_manager as fm
from PIL import Image
import glob

from sklearn.linear_model import LinearRegression

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import os
import RussoMod as rm
projection=ccrs.PlateCarree()

dd = display
dirfigs = os.getcwd()+'/../figuras/CFS/'
dirfigsCFSR = dirfigs + 'CFSR/'
dirfigsCFSv2 = dirfigs + 'CFSv2/'



dirdados = os.getcwd()+'/../DADOS/'
dirdadosCFSR = os.getcwd()+'/../DADOS/CFS/CFSR/'
dirdadosCFSv2 = os.getcwd()+'/../DADOS/CFS/CFSv2/'
dirdadosERA5 = os.getcwd()+'/../DADOS/ERA5/'
dirFred = dirdadosERA5 + 'fredericferry_era5_cps_diagram/'



dirCamargoERA5 = dirdados+'ERA5/ExtratropicalCycloneTracks_Gramcianinov_e_Camargo_2020/'
dirCamargoCFS = dirdados+'CFS/ExtratropicalCycloneTracks_Gramcianinov_e_Camargo_2020/'

reanalise_escolhida = 'CFS'

dd(dirdadosCFSR)
dd(dirfigsCFSR)


FS=12




'-------------------'
'''Função para graficos'''
def plota_mapa(LON,LAT,CAMPO,cmap='twilight_shifted',title_right=''):

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
    ax.add_feature(cfeature.COASTLINE)


    plt.pcolormesh(LON,LAT,CAMPO,cmap=cmap)



    xticks=range(-80,   20+1, 5)
    yticks=range(-90, -5+1, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels(xticks, rotation=90, ha='right',fontsize=FS-4)
    ax.set_yticklabels(yticks, rotation=90, ha='right',fontsize=FS-4)

    ax.grid(color='grey', linewidth=0.1)

    plt.title(title_right,loc='right',fontsize=FS-4)
    plt.colorbar()
    plt.tight_layout
    plt.show()

'''Função para graficos'''
def plot_background(ax):
    ax.coastlines("10m", zorder=3, color='grey')
    ax.gridlines()
    ax.set_xticks(np.linspace(-180, 180, 37), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(-90, 90, 37), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    return(ax)

'''Função para o cálculo de distâncias na esfera.'''
def haversine(lon1, lat1, lon2, lat2):
    '''
    Equal to the geopy.distance.
    To validate, run this:
    
        from geopy.distance import distance
        t = (cps.haversine(lon2[:,np.newaxis], lat2, storm_lon_i, storm_lat_i)).T
        print(t[0,0])
        tt = distance((lat2[0], lon2[0]), (storm_lat_i, storm_lon_i)).km
        print(tt)

    '''
    # convert decimal degrees to radians 
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371
    return c * r

'''Função para o cálculo de ângulos na esfera.'''
def get_bearing(lat1, lon1, lat2, lon2):
    '''
    'The horizontal direction from one terrestrial point to another; basically synonymous with azimuth.' --> 
    https://glossarytest.ametsoc.net/wiki/Bearing

    '''
    dLon = (lon2 - lon1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)
    brng = (brng +360) % 360
    return brng

    """Exemplo Prático
    Se você usar essa função para calcular o bearing (azimute) entre dois pontos, 
    o valor de saída será o ângulo de direção, indicando, em graus, 
    qual direção seguir (de norte, no sentido horário) para ir de um ponto ao outro na Terra.

    Resumo
    lat1, lon1: Coordenadas do ponto inicial.
    lat2, lon2: Coordenadas do ponto final.

    Retorna: O azimute (ângulo de direção) em graus, indicando o rumo a seguir do primeiro ponto ao segundo ponto, medido a partir do norte (0°) no sentido horário até 360°."""
    "Finalmente, a função retorna o azimute, que é o ângulo de direção em graus a partir do ponto 1 em direção ao ponto 2."

def bounds():
    lon_i_swAtl,lon_f_swAtl = -70 , -17
 
    lat_i_swAtl, lat_f_swAtl = -65, -20
    return [(lon_i_swAtl,lon_f_swAtl, lat_i_swAtl, lat_f_swAtl)]#[(lonW, lonE, latS, latN)]


def parametros_CPS(GPH,isobaric,time_field,camargo, STORM_ID_ESC,cps_or_radiuscomp,print_check=False):

    '''
    time_field: dict_CFS[str(ANOi)]['time_atm']
        Esse é o daterange dos campos: 
            array(['2001-01-01 00:00:00', '2001-01-01 03:00:00', ..., '2010-12-31 18:00:00', '2010-12-31 21:00:00'], 
                   dtype='<U19')
        Para salvar nesse formato: np.array(daterange.strftime('%Y-%m-%d %H:%M:%S'), dtype='<U19')
    '''
    
    '-----------------'



    #Limites do mapa para plots
    lon_i_swAtl,lon_f_swAtl = -70 , -17
    lat_i_swAtl, lat_f_swAtl = -65, -20

    bounds = [(lon_i_swAtl,lon_f_swAtl, lat_i_swAtl, lat_f_swAtl)]#[(lonW, lonE, latS, latN)]
    

    '-------------'
    cicl_esc = camargo[camargo['ID'] == STORM_ID_ESC] #ESCOLHENDO O CICLONE
    
    if cicl_esc['Time'].iloc[1] - cicl_esc['Time'].iloc[0] != cicl_esc['Time'].iloc[2] - cicl_esc['Time'].iloc[1]:
        cicl_esc = cicl_esc.iloc[1:] #DEALING WITH POSSIBLE GAP IN THE SECOND LINE OF SOME CICLONE DATAFRAME. EX: 2002 ID 589
        
    #NECESSARIO CORTE NOS PRIMEIROS DADOS DE TRACKING CASO A PRIMEIRA DATA DO CICLONE ESCOLHIDO NÃO 
    #COINCIDA COM AS HORAS AMOSTRADAS NOS CAMPOS BAIXADOS
    data_ini_track = cicl_esc['Time'].iloc[0]
    
    # Calcula a diferença absoluta entre cada elemento do array e a data inicial do track
    diferencas = np.abs(np.array([np.datetime64(data_ini_track)] * len(time_field)) - np.array([np.datetime64(d) for d in time_field]))
   
    
    #print('(np.array([np.datetime64(data_ini_track)] * len(time_field)): ',(np.array([np.datetime64(data_ini_track)] * len(time_field))))
    #print('[np.datetime64(d) for d in time_field]: ', [np.datetime64(d) for d in time_field])
    #print('diferencas converted to days, just to visualize and understand better (originally in nanosecs):', 
    #      diferencas / np.timedelta64(1, 'D') )
    

    
    # Exibir o resultado

    
    idx_data_mais_proxima_no_campo = np.argmin(diferencas)  # Encontra o índice do valor mais próximo

    
    #print('Showing, in the inputed variable field, the time which is the nearest to the initial time of the cyclone track: (pd.to_datetime(time_field[idx_data_mais_proxima_no_campo]) = ',pd.to_datetime(time_field[idx_data_mais_proxima_no_campo]))
    verification = (np.where(cicl_esc['Time'] == pd.to_datetime(time_field[idx_data_mais_proxima_no_campo])))[0]
    #print('verification:',verification)

    if len(verification)==0:
        #THIS VERIFICATION IS NEEDED BECAUSE THE LINES ABOVE FIND THE NEAREST DATE, WHICH CAN BE BEFORE THE INITIAL 
        #HOUR OF TRACKING. FOR INSTANCE, IF THE FIRST TIME OF THE TRACK IS 07Z, AND MY GRID DATA HAS 3H RESOLUTION,
        #THE NEAREST WILL BE 06Z, BUT IT IS NOT PRESENT ON THE CYCLONE TRACK ARRAY OF TIME. SO, IT WILL NEED TO GET
        #THE VALUE WHICH IS NOT THE NEAREST: THE 09Z VALUE OF THE TRACK WILL BE THE FIRST. SO WE USE IDX = 2 HERE (BELOW)
        #IN CASE THE verification RETURNED AN EMPTY VALUE FROM NP.WHERE 
        idx_data_mais_proxima_no_tracking = 2
    else:
        idx_data_mais_proxima_no_tracking = (np.where(cicl_esc['Time'] == pd.to_datetime(time_field[idx_data_mais_proxima_no_campo])))[0][0]


    #print('The index  from the cyclone track which is the nearest to the inputed variable field resolution:',idx_data_mais_proxima_no_tracking)
    
    if print_check:
        
        dd('cicl_esc BEFORE: ',cicl_esc) 
        
        print('data_ini_track: ',data_ini_track) 
        diferenca = diferencas[0] / np.timedelta64(1, 'D')    #Calcular a diferença em dias
        dias = int(diferenca)
        horas = int((diferenca - dias) * 24)    
        print(f'A primeira data no array time_field, {time_field[0]}, está cerca de {dias} dias e {horas} horas antes de {data_ini_track}, o primeiro time step do track da storm {str(STORM_ID_ESC)}.')
        print() 
        
        print('The index, from the inputed variable field, of the time which is the nearest to the initial'+
              ' time of the cyclone track (idx_data_mais_proxima_no_campo = '+
              str(idx_data_mais_proxima_no_campo),'. >>> '+
              str(time_field[(idx_data_mais_proxima_no_campo)]),' MUST BE EQUAL TO '+str(data_ini_track)+' (obs: se o primeiro for 1h antes, entrará no verification então esse print não deverá ser EQUAL TO mesmo! Por exemplo 03:00:00 e 04:00:00. Preciso corrigir esse print aqui, mas está funcionando)')
        print()    
        
        print('The index  from the cyclone track which is the nearest to the inputed variable field resolution:',idx_data_mais_proxima_no_tracking)
        print()
        
        
    cicl_esc = cicl_esc.iloc[idx_data_mais_proxima_no_tracking:] #RECORTANDO CASO NAO COINCIDA
    dd('cicl_esc AFTER: ',cicl_esc) if print_check else None
    print('----------------------------------------------------------------') if print_check else None
    
    cicl_esc['Time'] = pd.to_datetime(cicl_esc['Time']).dt.strftime('%Y-%m-%dT%H') #DEIXANDO NO FORMATO DE DATA IGUAL ORIGINAL

    liste_lon = list(cicl_esc['Longitude'].values)
    liste_lat = list(cicl_esc['Latitude'].values)
    liste_time = list(cicl_esc['Time'].values)
    liste_pres = list(cicl_esc['T42 vorticity'].values)

    '------------'
    '''Alteração do intervalo de tempo. CASO NECESSARIO'''
    #interval=int(input("Enter the desired time step (3h or 6h advised): "))
    interval = 3 #MANTEMOS EM 3 POIS O TRACKING É HORARIO MAS OS CAMPOS DE DADOS LIDOS SÃO DE 3horas.

    liste_time=liste_time[::interval]
    liste_lon=liste_lon[::interval]
    liste_lat=liste_lat[::interval]
    liste_pres=liste_pres[::interval]
    """SE A VARIAVEL FOR OCEANICA, USAR [1::interval] ?"""


    '-------------------'    
    '''ENCONTRANDO TEMPO INICIAL DA STORM ID ESCOLHIDA (após ter localizado com o idx_data_mais_proxima_no_campo)'''

    timee = liste_time
    TEMPO_INI = int(np.where(pd.to_datetime(time_field) == pd.to_datetime(timee[0]))[0])
    
    #if TEMPO_INI != idx_data_mais_proxima_no_campo:
    #    print('TEMPO_INI diferente: ',TEMPO_INI)
    #elif TEMPO_INI == idx_data_mais_proxima_no_campo:
    #    print('TEMPO_INI igual: ',TEMPO_INI)
        
    '-------------------'
    '''Definição das camadas verticais (os niveis que o Hart 2003 usa)'''
    #lower_layer = int(input('Enter 1 for 900-600 hPa lower layer and 2 for 925-700 hPa lower layer : ')) 
    lower_layer = 1

    if (lower_layer==1):
        listlev1=[900, 800, 700, 600]
    if (lower_layer==2):
        listlev1=[925, 875, 825, 775, 700]

    #higher_layer = int(input('Enter 1 for 600-300 hPa higher layer and 2 for 700-400 hPa higher layer : ')) 
    higher_layer = 1 

    if (higher_layer==1):
        listlev2=[600, 500, 400, 300]
    if (higher_layer==2):
        listlev2=[700, 600, 500, 400]

    '-----------------'
    '''Calculando as espessuras. O Delta Z_R e o Delta Z_L'''


    idx_levtop = np.where(isobaric == listlev1[-1])[0] #600 hPa NO MEU DADO
    idx_levbott = np.where(isobaric == listlev1[0])[0] #900 hPa NO MEU DADO

    thickness = GPH[:,idx_levtop , :,:] - GPH[:,idx_levbott, :,:] 
    thickness = thickness[:,0,:,:]

    
    '----------------------'
    '''Escolha definitiva do raio máximo para o cálculo dos parâmetros do CPS. Hart (2003) usa 500km"'''
    #max_dist = int(input('Enter the value of the maximum radius (in km) to compute the CPS diagrams : ')) 
    max_dist = 500 #Aqui já é em KM!  #PARA FICAR IGUAL HART

    '''The mean thicknesses are evaluated in semicircles of radius 500km.''' 
    '''https://moe.met.fsu.edu/cyclonephase/help.html'''

    # Niveis para contourf
    lev = isobaric

    # Definindo idate
    idate = 0

    #O listlev[-1] DELE É O NIVEL MAIS ALTO NA ATMOSFERA (MENOR VALOR DE HPA)
    cicl_esc['Time'] = pd.to_datetime(cicl_esc['Time'])
    cicl_esc = cicl_esc[::3]
    
    dd(DF(cicl_esc)) if print_check else ()
    
    if cps_or_radiuscomp == 'cps':
        return thickness,liste_lat,liste_lon,max_dist,TEMPO_INI,liste_time,idate,listlev1,listlev2,lev,cicl_esc
    

    elif cps_or_radiuscomp == 'radiuscomp':
        return liste_lat,liste_lon,max_dist,TEMPO_INI,liste_time   
        

    
    
