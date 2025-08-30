"""PARA IMPORTAR O MODULO"""
#'''
#import os
#import sys
#import importlib
#
## Primeiro adiciona o caminho desejado
#sys.path.append(r'C:\Users\berus\OneDrive\Documentos\Bernardo\BE_RUSSO\MeusModulos_PYTHON')
## Depois força o reload — isso atualiza o módulo mesmo se já estava carregado
#import RussoMod as rm
#importlib.reload(rm)
#
## Verifica se agora está usando o caminho certo
#print(rm.__file__)
#'''
#
#'''
#OUTRA FORMA DE IMPORTAR O MODULO
#cwd = os.getcwd()
#os.chdir('/home2/arofcn/pY/')
#import funcoes_mod, colormaps_LOF
#os.chdir(cwd)
#'''
#
#'''
#%load_ext autoreload
#%autoreload 2
#'''
#'''
#JUPYTER LINUX:
#%%bash
#ls 
#'''
#'''
#Modelo de fill_between:
#
#plt.figure(figsize=(15,5))
#plt.fill_between(ddd[0:365], serieEOF[0:365,ne], 0,
#                 where = (serieEOF[0:365,ne] > 0),
#                 color = 'g',alpha=0.5)
#plt.fill_between(ddd[0:365], serieEOF[0:365,ne], 0,
#                 where = (serieEOF[0:365,ne] < 0),
#                 color = 'r',alpha=0.5)
#plt.plot(ddd[0:365], serieEOF[0:365,ne],alpha=1,lw=2)
#'''

"""----------------------------------------------------------------------------------------------"""
from pandas import DataFrame as DF
import pandas as pd
import numpy as np
npw = np.where
npsh = np.shape
from numpy import shape as npsh
import matplotlib.pyplot as plt
import netCDF4 as nc
#import cartopy.crs as ccrs
import glob
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import scipy.io as sio
import natsort
import logger
import RussoMod as rm
import os
import time
import sys

FS = 18
plt.style.use('default')
plt.rcParams.update({'axes.grid' : True, 'grid.color': 'lightgrey'})
plt.rcParams['figure.figsize']=(15,5)
plt.rcParams['xtick.labelsize']=FS
plt.rcParams['ytick.labelsize']=FS
plt.rcParams["figure.autolayout"] = True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['figure.edgecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

dd=display
dirlagran = os.path.join(os.getcwd(),'..','LAGRANGEANO') #ANTIGO dirdados
diratual = os.path.join(os.getcwd())



def COMANDOCYCLER():
    print("plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0,1, 10))))")
    print("plt.gca().set_prop_cycle(plt.cycler(color = plt.cm.BrBG(np.linspace(0.1,1, len(dados.keys()))),alpha = (np.linspace(1,0.4, len(dados.keys())))))")
 
def COMANDOPLOTFIXO():
    print("mpl.rcParams.update({'axes.grid' : True, 'grid.color': 'lightgrey'})")
    print("plt.rcParams['figure.figsize']=(14,4)")

# Função para imprimir a letra grega com base no nome da letra
def imprimir_letra_grega(nome):
    '''
    # Exemplo de uso
    #nu_letra = imprimir_letra_grega("GREEK SMALL LETTER NU")  # Imprime a letra "nu" minúscula
    #imprimir_letra_grega("GREEK CAPITAL LETTER DELTA")  # Imprime a letra "Delta" maiúscula
    #nu_letra
    '''
    import unicodedata
    letra = unicodedata.lookup(nome)
    print(letra)
    return letra


    
class colors:
    '''Colors class:reset all colors with colors.reset; two
    sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,
    underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold'''

    end = reset='\033[0m'
    bold = '\033[01m'
    disable ='\033[02m'
    underline = under ='\033[04m'
    reverse ='\033[07m'
    strikethrough ='\033[09m'
    invisible ='\033[08m'
    
    class fg: #LETRAS
        black='\033[30m'
        red='\033[31m'
        green='\033[32m'
        orange='\033[33m'
        blue='\033[34m'
        purple='\033[35m'
        cyan='\033[36m'
        lightgrey='\033[37m'
        darkgrey='\033[90m'
        lightred='\033[91m'
        lightgreen='\033[92m'
        yellow='\033[93m'
        lightblue='\033[94m'
        pink='\033[95m'
        lightcyan='\033[96m'
        
    class bg: #BACKGROUND
        black='\033[40m'
        red='\033[41m'
        green=grn='\033[42m'
        orange='\033[43m'
        blue=blu='\033[44m'
        purple='\033[45m'
        cyan='\033[46m'
        lightgrey='\033[47m'
        
bold= colors.bold
B   = colors.bold
end = colors.reset
E   = colors.reset
bgc = colors.bg.cyan
bgb = colors.bg.blue
bgo = colors.bg.orange
bgr = colors.bg.red

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------




import inspect

def retrieve_name(var):
    '''
    Getting a variable name as a string.
    
     Exemplo de uso:
     foo = dict()
     foo['bar'] = 2
     print((retrieve_name(foo)))  # Deve imprimir 'foo'

    '''
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

def legendalegal(ncol=1,FS=12,loc='lower left',facecolor='k',mode=None):
    '''
    mode: 
        'expand' para expandir
    '''
    alpha = 0.5
    leg = plt.legend(frameon=True,facecolor=facecolor,fontsize=FS-2,mode = mode,
                     loc = loc,#bbox_to_anchor = (1.12,1),
                     shadow =True,framealpha = alpha-alpha/1.5,
                     ncol=ncol)

    for text in leg.get_texts():
        text.set_color('w')
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def make_cmap(colors, position=None, bit=False):
    '''
    colors_exemplo = [(0,0,0), (0,0,250), (0,0,0)]
        Cada tupla é uma cor da barra. (R, G, B)
        Se colocar apenas 2 tuplas, o degradê será entre 2 cores.

    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib.colors as mc    
    import numpy as np
    
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mc.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
    
def figuras_animacao(dirfigs :str, formatofigs: str, 
                     diranimacao: str,nomeanimacao: str, 
                     formatoanimacao: str, 
                     duracao=200):
    """
    ***
    Args:
        dirfigs:
            diretorio figuras da animacao
        formato:
            'png', 'jpg',...
        diranimacao:
            diretorio do save
        nomeanimacao:
            nome do arquivo .gif ou .mp4 gerado
        formatoanimacao:
            'gif' or 'mp4'
        duracao:
            Duração em (milisegundos) de cada imagem.
    ***
    """       
    #CRIA PASTA diranimacao CASO NAO EXISTA:
    if not os.path.exists(diranimacao):
        logger.logger.info('Pasta diranimacao não existe. Criando, entao...') 
        os.makedirs(diranimacao)
        
    logger.logger.info('Iniciando função de criação de animação a partir das figuras no diretorio...') 
    
    from PIL import Image
    import glob
    print('Criando lista com os frames (imagens)...')
    #CRIANDO LISTA COM OS 'FRAMES' (IMAGENS)
    frames = []
    imgs = glob.glob(dirfigs+'*.'+formatofigs) #glob.glob: RETORNA LISTA COM OS CAMINHOS DAS IMAGENS QUE VC QUER NO GIF 
                                     #O ASTERISCO SIGNIFICA QUE PEGARÁ TODAS IMAGENS

    imgs = natsort.natsorted(imgs) #ORGANIZA POR Nome

    for i in imgs: #LOOP PARA LER TODOS OS CAMINHOS DO glob.glob E INSERIR NA LISTA frames
        new_frame = Image.open(i)
        frames.append(new_frame) 
        #print(new_frame)
        
    print(DF(frames))
       
    print('Começando o save do gif...')
    # SALVANDO O GIF
    if formatoanimacao == 'gif':
        frames[0].save(diranimacao+nomeanimacao+'.'+formatoanimacao, format='GIF',
                       append_images=frames[0:],
                       save_all=True,
                       duration=duracao,                #DURAÇÃO EM MILISEGUNDOS DE CADA IMAGEM
                       loop=0)                      #loop = 0 => LOOP ETERNO; loop = 1 => APENAS 1 LOOP
                
    elif formatoanimacao == 'mp4':
        frames[0].save(diranimacao+nomeanimacao+'.'+formatoanimacao, format='MP4',
                       append_images=frames[0:],
                       save_all=True,
                       duration=duracao,                #DURAÇÃO EM MILISEGUNDOS DE CADA IMAGEM
                       loop=0)                      #loop = 0 => LOOP ETERNO; loop = 1 => APENAS 1 LOOP
 
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

def grafico_maneiro(style = 'default',
                    axes_grid = True,
                    grid_color = 'lightgrey',
                    figure_figsize=(15,5),
                    FS = 18,
                    xtick_labelsize=18,
                    ytick_labelsize=18,
                    figure_autolayout = True,
                    axes_spines_right = False,
                    axes_spines_top = False,
                    figure_edgecolor = 'white',
                    figure_facecolor = 'white'):
                    #font_sansserif = 'Helvetica'):

    plt.style.use(style)
    plt.rcParams.update({'axes.grid' : axes_grid, 'grid.color': grid_color})
    plt.rcParams['figure.figsize']=figure_figsize
    plt.rcParams['xtick.labelsize']=xtick_labelsize
    plt.rcParams['ytick.labelsize']=ytick_labelsize
    plt.rcParams["figure.autolayout"] = figure_autolayout
    plt.rcParams['axes.spines.right'] = axes_spines_right
    plt.rcParams['axes.spines.top'] = axes_spines_top
    plt.rcParams['figure.edgecolor'] = figure_edgecolor
    plt.rcParams['figure.facecolor'] = figure_facecolor
    #plt.rcParams['font.sans-serif'] = font_sansserif
    
    '''OBS: use the command >plt.rcParams to see the other params you can use for your graphs :)'''

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def everykind_to_datetime(d):
    import datetime as dt
    import cftime
    if isinstance(d, dt.datetime):
        return d
    if isinstance(d, cftime.DatetimeNoLeap):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, cftime.DatetimeGregorian):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, str):
        errors = []
        for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return dt.datetime.strptime(d, fmt)
            except ValueError as e:
                errors.append(e)
                continue
        raise Exception(errors)
    elif isinstance(d, np.datetime64):
        return d.astype(dt.datetime)
    else:
        raise Exception("Unknown value: {} type: {}".format(d, type(d))) 
        
        

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

def latlon_grid(latmin,latmax,lat_resolution,
                lonmin,lonmax,lon_resolution):
    """
    Define os eixos de lat-lon com base na resolução do modelo utilizado
    
    Args:
    latmin/lonmin: 
        valor mínimo de lat/lon
    
    latmax/lonmax: 
        valor máximo de lat/lon
    
    lat_resolution/lon_resolution: 
        Resolução lat/lon do modelo utilizado.
        A resolução é a distancia entre dois pontos do grid!
    """
    logger.logger.info('Iniciando função de definição dos arrays de LATITUDE e LONGITUDE...') #FUNCIONOU!
    
    latr = lat_resolution
    lonr = lon_resolution
    
    lat =np.arange(latmin,latmax+latr,latr)
    
    lon = np.arange(lonmin,lonmax+lonr,lonr)
    
    print('Shape (nc)lat: ',np.shape(lat))

    print('Shape (nc)lon: ',np.shape(lon))

    return np.asarray(lat),np.asarray(lon)

def valorlatlonmesh(pontolat,pontolon,latmesh,lonmesh):
    """
    FUNÇÃO UTILIZADA PARA SABER QUAL O INDICE DO ARRAY QUE EQUIVALE AO PONTO DE LATLON QUE QUEREMOS ESTUDAR!!!
    Retorna o valor de LAT e LON real de um meshgrid.
    ***
    pontolat,pontolon: indice do valor real de lat e lon do dado.
        Exemplo: se o dado vai de 0°S a 30°S, com uma resoluçãode 1°. Colocando pontolat = 0, retornará 30°S.
    latmesh: meshgrid que resulta da def latlon_grid
    lonmesh: meshgrid que resulta da def latlon_grid
    
    ***
    """
    print(bold,'Ponto de LAT, LON: ',latmesh[pontolat,pontolon], ',',lonmesh[pontolat,pontolon])



#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

def cria_arquivo(endereco_arquivo):
    #CRIA ARQUIVO
    file = endereco_arquivo
    try:
        open(file, 'a').close()
    except OSError:
        print('Failed creating the file')
    else:
        print('File created')

def cria_diretorio(folder_path_withbar,folder_name_withoutbars):
    '''
    folder_path_withbar: '...... path/'
    
    folder_name_withoutbars: 'folder_name'
    '''
    if folder_name_withoutbars not in folder_path_withbar:
        os.mkdir(folder_path_withbar + folder_name_withoutbars)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------        
def print_and_clear(text,duracao_para_apagar):
    print(text, end='\r')
    time.sleep(duracao_para_apagar)  # Espera 1 segundo para simular o download
    print(' ' * len(text), end='\r')  # Limpa o texto anterior

    '''Exemplo de uso:
    downloads = ['Download 1', 'Download 2', 'Download 3']

    for download in downloads:
        print_and_clear(f'Downloading: {download}')
    '''        
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------        

def le_1_netcdf(arquivo_nc,var_escolhida = None, printa_tudo = False):
    '''
    Rotina para ler 1 dado NetCDF. Para ler mais dados, inserir em um loop.
    
    Args:
    
    ***arquivo_nc: str
        Localização do arquivo e arquivo. Exemplo: '/home/hycom_operacional/force/ncfile/gfs0001.nc'
    '''
    import netCDF4 as nc
    import numpy as np

    ncData = nc.Dataset(arquivo_nc)
    
    if printa_tudo == True:
        print('Variaveis do arquivo NetCDF:')
        print(ncData.variables.keys())
        #display(ncData.variables) #PARA VER MAIS DETALHES DAS VARIAVEIS
        print('#---------------------------------------------------------')
        print('Variaveis do arquivo NetCDF, uma por uma:')

        for i in ncData.variables :
            print ('VARIAVEL DO ARQUIVO:', i) 

            try:
                print ('UNIDADE:', ncData.variables[i].units) #1 pascal [Pa] = 0,01 hectopascal [hPa]
            except:
                pass
            try:
                print ('SHAPE:',     ncData.variables[i].shape)
            except:
                pass        
            try:
                print ('DIMENSIONS:', ncData.variables[i].dimensions)
            except:
                pass
            try:
                print ('MIN/MAX NON-SPURIOUS VALUES:', ncData.variables[i].valid_min,';',ncData.variables[i].valid_max) 
            except:
                pass
            try:
                print ('LONG_NAME:', ncData.variables[i].long_name)
            except:
                pass
            print()
    
    if var_escolhida != None:
        #lat_recorte = slice(50,120)
        #lon_recorte = slice(202,241)        
        #dado = ncData.variables[var_escolhida][:,lat_recorte,lon_recorte] #USAR ESSAS 3 LINHAS PARA RECORTAR O DADO
        dado = ncData.variables[var_escolhida]#[:,:,:] #USAR ESSA LINHA PARA NAO RECORTAR
        print('Shape dado escolhido:',np.shape(dado)) if printa_tudo == True else None
        print('#---------------------------------------------------------') if printa_tudo == True else None

        return dado,ncData
    
    else:
        return ncData
    
#-------------------#-------------------#-------------------#-------------------#-------------------#-------------------    
#HOW TO USE:
#print(ncData.variables.keys())
#var_escolhida = 'TMP_surface'
#dadoA,ncData = le_1_netcdf(arqA,var_escolhida)
#dadoB,ncData = le_1_netcdf(arqB,var_escolhida)      
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
from scipy.interpolate import griddata
# DEFININDO PARAMETROS DO MODELO
def leitura_binario_acesso_direto(modelo,batimetria = False):
    
    if modelo == 'LSE24':  
        print('Lendo parametros do LSE24')
        IDM=601 #numeros de pontod idm
        JDM=727 #LSE24 TEM 727 JDM #numeros de pontos jdm
        KDM=32  #LSE24 TEM 32 KDM#numeros total de camadas
        gridLSE24='/../DADOS/TOPO/LSE24_V01/regional.grid_LSE24_V01.a'
        grid_fid=open(diratual+gridLSE24,mode='rb'); # endereco do regional grid
        depthLSE24='/../DADOS/TOPO/LSE24_V01/depth_LSE24_V01_03.a'
        depth_fid=open(diratual+depthLSE24,mode='rb'); # endereco do regional grid
        
    elif modelo == 'LSE36':  
        print('Lendo parametros do LSE36')
        IDM=741 #numeros de pontod idm
        JDM=920 #LSE24 TEM 727 JDM #numeros de pontos jdm
        KDM=32  #LSE24 TEM 32 KDM#numeros total de camadas
        gridLSE24='/../DADOS/TOPO/topo_LSE36/regional.grid.a'
        grid_fid=open(diratual+gridLSE24,mode='rb'); # endereco do regional grid
        depthLSE24='/../DADOS/TOPO/topo_LSE36/depth_LSEa0.02.a'
        depth_fid=open(diratual+depthLSE24,mode='rb'); # endereco do regional grid

    if modelo == 'SRP24':  
        print('Lendo parametros do SRP24')
        IDM=601 #numeros de pontod idm
        JDM=558 #numeros de pontos jdm
        KDM=32  #numeros total de camadas
        gridLSE24='/../DADOS/SRP24/topo/regional.grid.a'
        grid_fid=open(diratual+gridLSE24,mode='rb'); # endereco do regional grid
        depthLSE24='/../DADOS/SRP24/topo/depth_SRPa0.04_03.a'
        depth_fid=open(diratual+depthLSE24,mode='rb'); # endereco do regional grid
                
    IJDM=IDM*JDM
    npad=4096-(IJDM%4096) # 2^12
    data_type = np.dtype(np.float32).newbyteorder ('>') # '>' é ieee-be #arquivo binário

    lon = np.fromfile(grid_fid,dtype=data_type,count=IJDM)
    plon=lon.reshape(JDM,IDM)
    grid_fid.seek(4*(npad+IJDM),0) #sets the file's current position at the offset
    lat = np.fromfile (grid_fid,dtype=data_type,count=IJDM)
    plat=lat.reshape(JDM,IDM)

    bat = np.fromfile (depth_fid,dtype=data_type,count=IJDM)
    bat = np.where(bat > 1e10, np.nan,bat) 
    bat = bat.reshape(JDM,IDM)

    # CRIANDO MESHGRID PARA INTERPOLAR SAIDA DO MODELO
    lonn=np.arange(np.min(np.min(plon)), np.max(np.max(plon)), 1/4)
    latn=np.arange(np.min(np.min(plat)), np.max(np.max(plat)), 1/4)
    LON,LAT=np.meshgrid(lonn,latn)
    batim = griddata((plon.ravel(),plat.ravel()),bat.ravel(),(LON, LAT), method='nearest')
    #plt.pcolormesh(LON,LAT,batim,cmap='Blues')
    if batimetria: 
        print('Retornando LON, LAT e batim.')
        return LON,LAT,batim
    else:
        return npad,data_type,plon,plat,IDM,JDM,IJDM

def leitura_LSE24(ano_escolhido,dado = 'CorrentesUeV'):
    """
    Retorna as componentes U e V, a magnitude raiz(u*2 + v*2),a lon e a lat
    para o ano escolhido.
 
    ESSA LEITURA É PARA OS DADOS QUE FORAM EXTRAIDOS ANO A ANO PELO ACESSO DIRETO.
    Por exemplo, os dados que terminam por '..._001a365.a' e '...001a366.a'
    """
    npad,data_type,plon,plat,IDM,JDM,IJDM = leitura_binario_acesso_direto(modelo='LSE24')
    print('ANO: ',ano_escolhido)
    u0m = []
    v0m = []
    ano = ano_escolhido
    inter=1 #intervalo de extracao do acesso direto #####ALTERAR NA ROTINA FORTRAN
    if ano == 2008 or ano == 2012:
        ntot = 366
    else:
        ntot=365#((diaend-diaini)/inter)+1 #número total de dados extraidos
    cam=0 #CAMADA EXTRAIDA 0 => SUPERFICIE

    for comp in ['u','v']:

        if dado == 'VentoUeV':
            print('Lendo ventos...')
            diretorio = dirlagran+'/../DADOS/LSE24/MERRA2_'+str(ano)+'/'
            arquivo = 'forcing.'+comp+'10.a'
            f=open(diretorio+arquivo, mode='rb')
        if dado == 'CorrentesUeV':
            print('Lendo correntes...')
            f=open(dirlagran+'/../DADOS/LSE24/'+(comp).upper()+
                   'interp_LSE24_V01_0000m_'+str(ano)+'_001a'+str(ntot)+'.a', mode='rb')

        for INDEX in range(1,int(ntot)+1): # loop de leitura do arquivo gerado pelo acesso direto
            f.seek((INDEX-1)*4*(npad+IJDM),0)
            var = np.fromfile (f,dtype=data_type,count=IJDM)
            var=np.where(var > 1e10, np.nan,var) # tirando espúrios 
            #lt=np.divide(var,9806) # no caso de camadas - para encontrar valor em metros  ####????????
            variavel = var.reshape(JDM,IDM)
            if comp == 'u':
                u0m.append(variavel)
            if comp == 'v':
                v0m.append(variavel)

    u0m = np.array(u0m)
    v0m = np.array(v0m)

    dd(np.shape(u0m))
    dd(np.shape(v0m))
    c0m = np.sqrt( u0m**2 + v0m**2 )
    return u0m,v0m,c0m,plon,plat

def leitura_TESTE_LSE36_3p01_comMare_semCERES(ano_escolhido,mes_escolhido,dia_escolhido,dado = 'CorrentesUeV',prof=0):
    """
    Retorna as componentes U e V, a magnitude raiz(u*2 + v*2),a lon e a lat
    para o ano escolhido.
    prof: 
        50,800,1200,2000
    
    ESSA LEITURA É PARA OS DADOS QUE FORAM EXTRAIDOS DIA A DIA PELO ACESSO DIRETO.
    Por exemplo, os dados que terminam por '..._20170101.a','...20170102.a',...
    
    """
    modelo = input('Modelo: [SRP24, LSE36, LSE24]')
    npad,data_type,plon,plat,IDM,JDM,IJDM = leitura_binario_acesso_direto(modelo=modelo,batimetria=False)
    print(data_type,IDM,JDM,IJDM)
    print('ANO: ',ano_escolhido)
    print('MES: ',mes_escolhido)
    print('DIA: ',dia_escolhido)
    
    u0m = []
    v0m = []
    ano = ano_escolhido
    mes = mes_escolhido
    diad = dia_escolhido

    inter=1 #intervalo de extracao do acesso direto #####ALTERAR NA ROTINA FORTRAN
    if ano in [2008,2012,2016,2020]:# == 2008 or ano == 2012 :
        ntot = 366
    else:
        ntot=365#((diaend-diaini)/inter)+1 #número total de dados extraidos
    cam=0 #CAMADA EXTRAIDA 0 => SUPERFICIE

    for comp in ['u','v']:
        #fid=fopen([dire,num2str(prof),'m','/Uinterp_LSE36_3p01_V01_comMARE_semCERES_operacional_',num2str(prof, '%04.f'),'m_',num2str(ano),num2str(mes, '%02.f'),num2str(diad,'%02.f'),'.a']); #MATLAB
        
        if dado == 'CorrentesUeV':
            if modelo == 'SRP24':
                print('Lendo correntes SRP24...')

                f=open(dirlagran+'/../DADOS/SRP24/SRP24_3p01_lobo_LIVRE_VAZ_variavel/VEL/'+
                       (comp).upper()+
                       'interp_SRP24_LIVRE_vaz_varia_'
                       +format(prof,'04d')+'m_'+str(ano)+format(mes,'02d')+format(diad,'02d')+'.a',
                       mode='rb')
                
            elif modelo == 'LSE36':
                print('Lendo correntes LSE36...')

                f=open(dirlagran+'/../DADOS/LSE36/'+
                       (comp).upper()+
                       'interp_LSE36_3p01_V01_comMare_semCERES_operacional_'
                       +format(prof,'04d')+'m_'+str(ano)+format(mes,'02d')+format(diad,'02d')+'.a',
                       mode='rb')
            
            for INDEX in range(1,int(ntot)+1): # loop de leitura do arquivo gerado pelo acesso direto
                f.seek((INDEX-1)*4*(npad+IJDM),0)
                var = np.fromfile (f,dtype=data_type,count=IJDM)
                var=np.where(var > 1e10, np.nan,var) # tirando espúrios 
                variavel = var.reshape(JDM,IDM)
                if comp == 'u':
                    u0m.append(variavel)
                if comp == 'v':
                    v0m.append(variavel)
                    
            

    u0m = np.array(u0m)
    v0m = np.array(v0m)

    dd(np.shape(u0m))
    dd(np.shape(v0m))
    c0m = np.sqrt( u0m**2 + v0m**2 )
    return u0m,v0m,c0m,plon,plat




def leitura_SRP24(ano_escolhido,dado = 'CorrentesUeV'):
    """
    Retorna as componentes U e V, a magnitude raiz(u*2 + v*2),a lon e a lat
    para o ano escolhido.
    """
    npad,data_type,plon,plat,IDM,JDM,IJDM = leitura_binario_acesso_direto(modelo='SRP24')
    print('ANO: ',ano_escolhido)
    u0m = []
    v0m = []
    temp =[] 
    ano = ano_escolhido
    inter=1 #intervalo de extracao do acesso direto #####ALTERAR NA ROTINA FORTRAN
    if ano == 2008 or ano == 2012:
        ntot = 366
    else:
        ntot=365#((diaend-diaini)/inter)+1 #número total de dados extraidos
    cam=0 #CAMADA EXTRAIDA 0 => SUPERFICIE
    
    if dado in ['CorrentesUeV','VentoUeV']:
        for comp in ['u','v']:

            #if dado == 'VentoUeV':
            #    print('Lendo ventos...')
            #    diretorio = dirlagran+'/../DADOS/LSE24/MERRA2_'+str(ano)+'/'
            #    arquivo = 'forcing.'+comp+'10.a'
            #    f=open(diretorio+arquivo, mode='rb')
            #    
            if dado == 'CorrentesUeV':
                print('Lendo correntes...')
                f=open(dirlagran+'/../DADOS/SRP24/SRP24_3p01_lobo_LIVRE_VAZ_variavel/VEL/'+
                       'TEMP_SRP24_LIVRE_vaz_varia_prof_000m_'+str(ano), mode='rb')
            #
            for INDEX in range(1,int(ntot)+1): # loop de leitura do arquivo gerado pelo acesso direto
                f.seek((INDEX-1)*4*(npad+IJDM),0)
                var = np.fromfile (f,dtype=data_type,count=IJDM)
                var=np.where(var > 1e10, np.nan,var) # tirando espúrios 
                #lt=np.divide(var,9806) # no caso de camadas - para encontrar valor em metros  ####????????
                variavel = var.reshape(JDM,IDM)
                if comp == 'u':
                    u0m.append(variavel)
                if comp == 'v':
                    v0m.append(variavel)
                    
        u0m = np.array(u0m)
        v0m = np.array(v0m)
        temp = np.array(temp)

        dd(np.shape(u0m))
        dd(np.shape(v0m))
        c0m = np.sqrt( u0m**2 + v0m**2 )
        return u0m,v0m,c0m,plon,plat
                    

    if dado == 'Temperatura':
        print('Lendo temperatura...')
        f=open(dirlagran+'/../DADOS/SRP24/SRP24_3p01_lobo_LIVRE_VAZ_variavel/TEMP/'+
               'TEMP_SRP24_LIVRE_vaz_varia_prof_000m_'+str(ano), mode='rb')

        for INDEX in range(1,int(ntot)+1): # loop de leitura do arquivo gerado pelo acesso direto
            f.seek((INDEX-1)*4*(npad+IJDM),0)
            var = np.fromfile (f,dtype=data_type,count=IJDM)
            var=np.where(var > 1e10, np.nan,var) # tirando espúrios 
            #lt=np.divide(var,9806) # no caso de camadas - para encontrar valor em metros  ####????????
            variavel = var.reshape(JDM,IDM)
            temp.append(variavel)

        temp = np.array(temp)

        dd(np.shape(temp))
        return temp,plon,plat
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def ruidobrancogaussiano(mean,variance,size):
    """GERADOR DE RUÍDO BRANCO GAUSSIANO, White Gaussian Noise (WGN)"""

    std = np.sqrt(variance)  
    wgn = np.random.normal(mean, std, size)
    return wgn

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def filtro_digital(pdc,nf,original_data,filtertype = 'low',show_filter = True, title = None,ylim = None, show_result = None):
    '''    
    FUNÇÃO PARA CRIAÇÃO DE FILTRO DIGITAL EM PYTHON
    Igual a Matlab: funções butter+freqz+filtfilt
    ***
    pdc: int (Periodo de corte)
        Para filtrar com base no período, uma abordagem melhor para estudo de fenomenos de escala temporal de horas ou dias.
        EXEMPLO: Se tivermos 4 dados por dia e queremos atenuar 
        todas as oscilações com período menor que 3 dias (com o filtro 'low'), então usamos pdc = 3*4
        Se o dado for amostragem diária, então basta usar pdc = 3. 
        
    nf: int (Numero de pesos final - Ordem final do filtro)
        Testar valores para não ocorrer o Fenômeno de Gibbs: observar na figura.
        Se a ordem for muito grande, o filtro vai estourar. Se ocorrer reduzir o valor de nf.
        Exemplo: começar com nf = 20 para um dado de amostragem diária
        
    filtertype: 'low','high'
        'low' for lowpass filter
        'high' for highpass filter
        
    original_data: pd.DataFrame or np.array (preferences)
        The data you want to filter

    ***
    --- ENGLISH:
    FUNCTION FOR CREATING A DIGITAL FILTER IN PYTHON
    Equivalent to Matlab: butter+freqz+filtfilt functions
    ***
    pdc: int (Cut-off period)
        To filter based on the period, a better approach for studying phenomena with a temporal scale of hours or days.
        EXAMPLE: If we have 4 data points per day and want to attenuate 
        all oscillations with a period of less than 3 days (with the 'low' filter), then we use pdc = 3*4
        If the data is sampled daily, then just use pdc = 3.

    nf: int (Number of final weights - Final filter order)
        Test values to avoid the Gibbs Phenomenon: observe in the figure.
        If the order is too large, the filter will explode. If this occurs, reduce the value of nf.
        Example: start with nf = 20 for daily sampling data

    filtertype: 'low','high'
        'low' for lowpass filter
        'high' for highpass filter

    original_data: pd.DataFrame or np.array (preferences)
        The data you want to filter
    
    '''
    
    import matplotlib.pyplot as plt
    from pandas import DataFrame as DF
    from scipy.signal import butter, filtfilt,freqz
    np.seterr(divide='ignore', invalid='ignore')

    font_size = 14
    pi = np.pi

    pdc = pdc
    fc = 1/pdc #FREQUENCIA DE CORTE
    fn = 1/2 #FREQUENCIA DE NYQUIST
    #fc/fn #FREQUENCIA DE CORTE NORMALIZADA

    #----------------------------------CRIAÇÃO DO FILTRO
    if show_filter == True:
        plt.figure(dpi=60)
        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0.1, 1, nf)))) #PARA PLOT EM ESCALA DE COR

    for n in range(1,nf):
        b, a = butter(N = n, 
                      Wn = fc/fn, #CUTTOF FREQUENCY #IF Wn IS A 2 ELEMENT VECTOR, RETURNS AN ORDER 2N BANDPASS FILTER W1<W<W2
                      btype = filtertype)#'low' : LOWPASS
        wl, hl = freqz(b, a) #CALCULA A RESPOSTA EM FREQUENCIA DO FILTRO BUTTER
        if show_filter == True:
            plt.semilogx(1/(fn*wl/pi) , abs(hl))

    if show_filter == True:
        plt.semilogx(1/(fn*wl/pi) , abs(hl) ,color='k') #ULTIMA CURVA, SERÁ O FILTRO IDEAL; EIXO X EM PERIODO
        #plt.semilogx((fn*wl/pi) , abs(hl) ,color='k')   #ULTIMA CURVA, SERÁ O FILTRO IDEAL; EIXO X EM FREQUENCIA (ROTAÇÕES POR [UNIDADE DE MEDIDA DO PERIODO])
        plt.xlabel('Period [PERIOD UNIT OF MEASUREMENT]',fontsize=font_size)
        plt.ylabel('Gain [WITHOUT DIMENSION]',fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

    ##----------------------------------FILTRANDO O DADO
    #
    original_data = DF(original_data)
    filtered_data = filtfilt(b, a, original_data,axis=0, padtype='odd', padlen=3*(max(len(b),len(a))-1))
    #signal.filtfilt(B, A, ~np.isnan(values)) #TENTAR ISSO PARA FUNCIONAR COM NANs
    filtered_data = DF(filtered_data,index=original_data.index)
    #
    ##----------------------------------RESULTADO
    #
    if show_result != None:

        plt.figure(figsize=(17.5 , 5))
        plt.plot(original_data,color='gray',label = 'ORIGINAL DATA',lw=3,marker='o')
        plt.plot(filtered_data,color='k',label = 'FILTERED DATA',lw = 3,marker='o',alpha=0.7)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.legend()
        plt.ylim(ylim)
        if title != None:
            plt.title(str(title))
    
        
    return filtered_data

#------------------------
#RUNNING LOWPASS:
#######filtro_digital(pdc=3*4,nf=20,original_data=x,filtertype='low')

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def filtro_digital_campo(array_dado, array_lats, array_lons, depth_index=0,
                            pdc=15, nf=11,
                            show_filter=False,filtertype='high', title=None,ylim=None, show_result=False):
    '''
    Função filtro_digital aplicada à campos completos, e não apenas um ponto.
    Importante verificar como está o filtro criado, se os parâmetros estão correto, como na função original
    ---
    array_dado: np.array
    
        ATENÇÃO AO FORMATO: array_dado[TEMPO,PROFUNDIDADE,LAT,LON]
        OU: array_dado[TEMPO,LAT,LON] caso não tem vertical.
        
    array_lats: 
        array 1D
    
    array_lons: 
        array 1D
    
    depth_index:
        Profundidade escolhida 
        
    pdc: periodo de corte
        Parametro importante para a construção do filtro. Insira show_filter=True para ver
        como está a construção do seu filtro. A ordem nf do filtro tem que ser grande o suficiente para ser um filtro
        eficaz, mas pequena o suficiente para o filtro nao explodir.
    nf: ordem do filtro
        Parametro importante para a construção do filtro. Insira show_filter=True para ver
        como está a construção do seu filtro. A ordem nf do filtro tem que ser grande o suficiente para ser um filtro
        eficaz, mas pequena o suficiente para o filtro nao explodir.        
    ---
    Funcionando em 29/01/2024
    '''
    # Variável local para controle do filtro exibido
    filter_shown = False
    
    zeros = np.zeros(shape=np.shape(array_dado))
    

    for lat_idx in range(len(array_lats)):
        print('lat_idx/len(array_lats): ',lat_idx,'/',len(array_lats))

        for lon_idx in range(len(array_lons)):
            
            if depth_index == None:
                fis_ponto = array_dado[:, lat_idx, lon_idx]
            else:
                fis_ponto = array_dado[:, depth_index, lat_idx, lon_idx]
            
            #TO VISUALIZE THE CREATED FILTER
            #if (lat_idx == len(array_lats) // 2) and (lon_idx == len(array_lons) // 2) and (show_filter == True):
            #    fis_ponto_filtered = rm.filtro_digital(pdc=pdc, nf=nf, original_data= fis_ponto,
            #                                           show_filter = show_filter, filtertype=filtertype, title=title,
            #                                           ylim=ylim, show_result=show_result)[0]
            #else:
            #    fis_ponto_filtered = rm.filtro_digital(pdc=pdc, nf=nf, original_data= fis_ponto,
            #                                           show_filter = show_filter, filtertype=filtertype, title=title,
            #                                           ylim=ylim, show_result=show_result)[0]
            #plt.close()
            # Modificação mínima no bloco if para garantir que o filtro só seja mostrado uma vez
            # Exibir o filtro apenas uma vez
            if (lat_idx == len(array_lats) // 2) and (lon_idx == len(array_lons) // 2) and (show_filter == True) and not filter_shown:
                fis_ponto_filtered = rm.filtro_digital(pdc=pdc, nf=nf, original_data=fis_ponto,
                                                       show_filter=show_filter, filtertype=filtertype, title=title,
                                                       ylim=ylim, show_result=show_result)[0]
                filter_shown = True  # Marca que o filtro foi mostrado uma vez
            else:
                fis_ponto_filtered = rm.filtro_digital(pdc=pdc, nf=nf, original_data=fis_ponto,
                                                       show_filter=False, filtertype=filtertype, title=title,
                                                       ylim=ylim, show_result=show_result)[0]
            plt.close()
            
            if depth_index == None:
                zeros[:, lat_idx, lon_idx] = fis_ponto_filtered
                
            else:
                zeros[:, depth_index, lat_idx, lon_idx] = fis_ponto_filtered

    return zeros


# Usage example:
#fis1g_hpfilt = rm.filtro_digital_espacial(fis1g, lat1g, lon1g, depth_index=0)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def degree_to_rad(valor_em_graus):
    """
    '''Graus para Rads'''
    180graus ------ PIrad
    90graus  ------ Xrad
    180*X = 90*PI
     Xrad = PI*90/180
     Xrad = PI*1/2
     Xrad = PI/2 

    =>pi * fase°/180
    """
    x = valor_em_graus
    rad = (x*pi)/180
    return rad

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def interpola_campo1dia(lon_lowres, lat_lowres,lon_highres, lat_highres,um_campo_highres,printa_shapes):
    '''
    lon_lowres, lat_lowres: needs to be a meshgrid!
        lon_lowres, lat_lowres = np.meshgrid(lon_DataWithLowerRes,lat_DataWithLowerRes)
        
    lon_highres, lat_highres: needs to be a meshgrid too!
        lon_highres, lat_highres = np.meshgrid(lon_DataWithHigherRes,lat_DataWithHigherRes)
        
    um_campo_highres: needs to be a field with the same shape as the meshgrid of lon_highres, lat_highres!
        obs: if the DATA has shape, for instance, (time,lat,lon), you need to 
        use um_campo_highres = DATA[0,lat,lon] to apply this fuction just for the first time.
        So, for all the time steps a for loop is necessary.
        
    ---
    Example of use:
    
    lon_lowres, lat_lowres = np.meshgrid(lon_ostia,lat_ostia)
    lon_highres, lat_highres = np.meshgrid(dict['lon_ocn'],dict['lat_ocn'])
    um_campo_highres = dict['temp_surf'][0,:,:]

    x=interpola_campo1dia(lon_lowres, lat_lowres,lon_highres, lat_highres,um_campo_highres)
    
    
    ---
    Example of use with for loop:
    lon_lowres, lat_lowres = np.meshgrid(lon_ostia,lat_ostia)
    lon_highres, lat_highres = np.meshgrid(dict['lon_ocn'],dict['lat_ocn'])


    dict['temp_surf_interp'] = []

    for time in range(3):#len(dict['temp_surf'])):

        um_campo_highres = dict['temp_surf'][time,:,:]

        x = rm.interpola_campo1dia(lon_lowres, lat_lowres,lon_highres, lat_highres,um_campo_highres)
        dict['temp_surf_interp'].append(x)

    np.shape(dict['temp_surf_interp'])
    >>> Desired shape after interpolation: (46, 55) 
        Shape before interpolation: (111, 127) 
        Shape after interpolation: (46, 55) -> This must be equal to  (46, 55) 

        Desired shape after interpolation: (46, 55) 
        Shape before interpolation: (111, 127) 
        Shape after interpolation: (46, 55) -> This must be equal to  (46, 55) 

        Desired shape after interpolation: (46, 55) 
        Shape before interpolation: (111, 127) 
        Shape after interpolation: (46, 55) -> This must be equal to  (46, 55) 
        (3, 46, 55)
   
    '''
    from scipy.interpolate import griddata 




    interpolated_data = griddata(points = (lon_highres.ravel(),lat_highres.ravel()), #POINTS: COORDENADAS DOS PONTOS DO DADO #LON E LAT EM MESHGRID DO DADO DE MAIOR RESO, EM TRIPA, POR ISSO O ravel()
                                 values = um_campo_highres.ravel(), #VALUE: VALORES DOS DADOS #1 DIA DE 1 CAMPO, DO DADO DE MAIOR RESO, EM TRIPA, POR ISSO O ravel()
                                 xi = (lon_lowres, lat_lowres), #MESHGRID DE LON E LAT DO DADO DE MENOR RESO
                                 method='nearest') 
    
    print('\nDesired shape after interpolation:',np.shape(lon_lowres), 
          #'\nDesired shape after interpolation:',np.shape(lat_lowres),
          '\nShape before interpolation:',np.shape(lon_highres), 
          #'\nShape before interpolation:',np.shape(lat_highres),
          #'\nShape before interpolation:',np.shape(um_campo_highres),
          '\nShape after interpolation:',np.shape(interpolated_data), '-> This must be equal to ',np.shape(lon_lowres),'') if printa_shapes else None
    
    return(interpolated_data)
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#Acho q e isso mesmo dickson, eu q perdi essa formula e nunca mais usei.
#A freq de amostragem em segundos certo? Se meu dado tem:
#dt = 30min
#Entao:
#freq_amostragem = 1/30min = 1/60*30seg
def freqtoHertz(dt_em_segundos,omegas):
    """
    dt_em_segundos: int, intervalo de amostragem em segundos
    Exemplos: 
        se o meu dado é amostrado a cada 30 minutos, dt_em_segundos deve ser 60seg*30min
        se o meu dado é amostrado a cada 1 hora, dt_em_segundos deve ser 60seg*60min
        se o meu dado é amostrado a cada 1 dia, dt_em_segundos deve ser 60seg*60min*24h
        
    omega: w = np.linspace(0 , np.pi , quant_omegas)
        Lista com quantidade de valores de omega, de 0 a pi.


    """
    import numpy as np
    pi = np.pi
    w = omegas
    
    fs = 1/(dt_em_segundos)
    fHz = fs*w/(2*pi)
    return fHz
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def periodograma_padrao(sinal,quant_omegas,printatudo = False):
    """
    Densidade Espectral de Potência pelo método do PERIODOGRAMA PADRÃO.
    ***
    sinal: y(n)
        dado a ser analisado. Irá retornar a sua PSD
    quant_omegas: int 
        quantidade de ômegas (valores de 0 a pi)
    ***
    """
    import numpy as np
    e = np.exp
    y = sinal
    L = np.size(y)
    
    if printatudo == True:
        print('Iniciando calculo da PSD do sinal, printando as etapas.')

        print('O somatório da PSD vai de nu = ['+str(-L+1)+','+str(L-1)+']')
        print('*************************************************************')
        print('*************************************************************')
        print('*************************************************************')
        Ry = []
        for nu in np.arange(-L+1,L): #-L+1 até L-1, mas no python o ultimo ja é ignorado. Por isso deixei L
            print('nu:',nu)
            somadeRdenu = []
            print('O somatório da Ry['+str(nu)+'] vai de n = ['+str(0)+','+str(L-np.abs(nu)-1)+']')
            print('--------------------------------------------------------------')

            for n in np.arange(0,L-np.abs(nu)): #no python o ultimo ja é ignorado
                print('n:',n)
                somadeRdenu.append(y[n]*y[n+np.abs(nu)])
                print('y(n)y(n+|\u03BD|) = y('+str(n)+')y('+str(n)+'+'+str(np.abs(nu))+') : '
                      +str(y[n]*y[n+np.abs(nu)]))    
            print()
            print('\u03A3y(n)y(n+'+str(np.abs(nu))+') =' ,  np.sum(somadeRdenu))
            print()
            Ry.append(( 1/L * np.sum(somadeRdenu)))
            print('Ry['+str(nu)+']:\n 1/L * \u03A3y(n)y(n+'+str(np.abs(nu))+') =' ,( 1/L * np.sum(somadeRdenu)))# Ry[nu])
            """AQUI, JA TEMOS A AUTOCORRELAÇÃO DE y(n) NO INTERVALO"""
            print('_____________________________________________________________')
            print('_____________________________________________________________')
            print('_____________________________________________________________')
            print('_____________________________________________________________')
            print('_____________________________________________________________')


        """TEMOS A AUTOCORRELAÇÃO Ry.
        AGORA, VAMOS ESTIMAR A PSD"""
        Ry = DF(Ry,index=np.arange(-L+1,L),columns=['Ry(nu)'])
        PSDdew = []
        for OMEGA in np.linspace(0 , np.pi , quant_omegas):

            """NO MOMENTO, TEMOS Ry(-3),Ry(-2),...Ry(3)"""
            """VAMOS, ENTÃO, MULTIPLICAR: Ry(-3).e^-jw(-3)"""

            lista=[]
            for NU in np.arange(-L+1,L): #MULTIPLICANDO POR e^(-1j*OMEGA*NU)
                lista.append((Ry.loc[NU].values * e(-1j*OMEGA*NU))[0])        

            """AQUI TEMOS CADA VALOR DE Ry(nu) MULTIPLICADO PELO e^-jw(-nu)"""
            df=DF(np.array(lista),columns=['$Ry(\u03BD) . e^{-j\u03C9\u03BD}$, sendo \u03C9 = '+str(OMEGA.round(3))],index=np.arange(-L+1,L))
            #display(df)
            """AGORA, DEVEMOS SOMAR TODOS ESSES VALORES PARA ESSE OMEGA. 
            DEPOIS, FAREMOS A SOMA PARA OS OUTROS OMEGAS"""
            PSDdew.append((df.cumsum().iloc[-1].values)[0]) #PEGA O ULTIMO VALOR DO df.cumsum

        PSDdew = np.array(PSDdew)
        PSDdew = DF(PSDdew,index=np.linspace(0 , np.pi , quant_omegas),columns=['$\u0393(e^{j\u03C9})$'])
        PSDdew.index.name = '\u03C9'
        dd(PSDdew)

        w = np.linspace(0 , np.pi , quant_omegas)
       
    else:
        print('Iniciando calculo da PSD do sinal, sem printar as etapas.')
        Ry = []
        for nu in np.arange(-L+1,L): #-L+1 até L-1, mas no python o ultimo ja é ignorado. Por isso deixei L
            somadeRdenu = []

            for n in np.arange(0,L-np.abs(nu)): #no python o ultimo ja é ignorado
                somadeRdenu.append(y[n]*y[n+np.abs(nu)])
      
            Ry.append(( 1/L * np.sum(somadeRdenu)))

            """AQUI, JA TEMOS A AUTOCORRELAÇÃO DE y(n) NO INTERVALO"""

        """TEMOS A AUTOCORRELAÇÃO Ry.
        AGORA, VAMOS ESTIMAR A PSD"""
        Ry = DF(Ry,index=np.arange(-L+1,L),columns=['Ry(nu)'])
        PSDdew = []
        for OMEGA in np.linspace(0 , np.pi , quant_omegas):

            """NO MOMENTO, TEMOS Ry(-3),Ry(-2),...Ry(3)"""
            """VAMOS, ENTÃO, MULTIPLICAR: Ry(-3).e^-jw(-3)"""

            lista=[]
            for NU in np.arange(-L+1,L): #MULTIPLICANDO POR e^(-1j*OMEGA*NU)
                lista.append((Ry.loc[NU].values * e(-1j*OMEGA*NU))[0])        

            """AQUI TEMOS CADA VALOR DE Ry(nu) MULTIPLICADO PELO e^-jw(-nu)"""
            df=DF(np.array(lista),columns=['$Ry(\u03BD) . e^{-j\u03C9\u03BD}$, sendo \u03C9 = '+str(OMEGA.round(3))],index=np.arange(-L+1,L))
            """AGORA, DEVEMOS SOMAR TODOS ESSES VALORES PARA ESSE OMEGA. 
            DEPOIS, FAREMOS A SOMA PARA OS OUTROS OMEGAS"""
            PSDdew.append((df.cumsum().iloc[-1].values)[0]) #PEGA O ULTIMO VALOR DO df.cumsum
            print('OMEGA:',OMEGA)
            
        PSDdew = np.array(PSDdew)
        PSDdew = DF(PSDdew,index=np.linspace(0 , np.pi , quant_omegas),columns=['$\u0393(e^{j\u03C9})$'])
        PSDdew.index.name = '\u03C9'
        dd(PSDdew)

        w = np.linspace(0 , np.pi , quant_omegas)


    return PSDdew, w

#EXECUTANDO:
#quant_omegas=200
#PSD,w = periodograma_padrao(sinal,quant_omegas=quant_omegas)

#PLOT:
def plot_periodograma(PSDs: dict, freq, unidade_freq: str,semilogx = False):
    '''
    freq: pode ser em Hz ou rad
        lista de omegas. w que sai do def periodograma_padrao.
        
    unidade_freq: str 
        rad ou Hz (rotações por segundo)
    '''
    plt.figure()
    
    for psd in PSDs.keys():
            
        if semilogx == True:
            plt.semilogx(freq,10*np.log10(np.abs(PSDs[psd]))-10,label=psd,lw=2)
        else:
            plt.plot(freq,10*np.log10(np.abs(PSDs[psd]))-10,label=psd,lw=2)
        
        #plt.xlabel('\u03C9 '+str(unidade_freq))
        plt.xlabel('$Freq ['+str(unidade_freq)+']$')
        plt.ylabel('$|\u0393(e^{j\u03C9})|$\n\n(db , 10 * log10(PSD) - 10)')
        #plt.annotate('Senoides com \nfrequências:'+ 
        #             '\n$f1='+str(f1)+'$Hz '+
        #             '\n$f2='+str(f2)+'$Hz '+
        #             '\n$f3='+str(f3)+'$Hz '+
        #             '\n$f4='+str(f4)+'$Hz '+
        #             '\n$f5='+str(f5)+'$Hz '+
        #             '\n$fs='+str(fs)+'$\namostras/seg'+
        #             '\n$\u03C9 = [0,\u03C0]$'+
        #             '\n$L = '+str(L)+'$',
        #             xy=(780,110),xycoords='figure points',
        #             bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round,pad=0.2',alpha=0.1))

            #LEGENDA

        rm.legendalegal(ncol=len(PSDs.keys()))

        #plt.ylim([-75,18])
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#-------------ANÁLISE HARMÔNICA DE MARÉS - Pacote: ttide---------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def destacando_componentes_ttide(tide_dados_goos):
    """Pegando as componentes com Signal-to-Noise Ratio acima de 2"""

    lat_dados_goos = tide_dados_goos['lat']
    freq_dados_goos = []
    nameu_dados_goos = []
    tidecon_dados_goos = [] 
    synteses_dados_goos = []
    fase_dados_goos = []
    ampli_dados_goos = []

    for i in range(len(tide_dados_goos['fu'])):
        print('i:',i)
        if tide_dados_goos ["snr"][i] >= 2:
            print()
            freq_dados_goos.append(tide_dados_goos['fu'][i])
            print("freq_dados_goos:\n",tide_dados_goos['fu'][i])

            print()
            nameu_dados_goos.append(tide_dados_goos["nameu"][i])
            print("nameu_dados_goos:\n",tide_dados_goos["nameu"][i])

            print()
            tidecon_dados_goos.append(tide_dados_goos ["tidecon"][i])
            print("tidecon_dados_goos:\n",tide_dados_goos ["tidecon"][i])

            print()
            ampli_dados_goos.append(tide_dados_goos ["tidecon"][i][0])
            print("ampli_dados_goos:\n",tide_dados_goos ["tidecon"][i][0])

            print()
            fase_dados_goos.append(tide_dados_goos ["tidecon"][i][2])
            print("fase_dados_goos:\n",tide_dados_goos ["tidecon"][i][2])

        print('--------------------------------------------------------------------------')



    """Transformando de ciclos por hora (cph) para ciclos por dia (cpd)"""
    freq_dados_goos_cpd =[] 
    fase_dados_goos_local = []

    fr = 0
    fase = 0
    for i in range(len(freq_dados_goos)):
        fr = freq_dados_goos[i] * 24
        freq_dados_goos_cpd.append(fr)
    #    
        fase = fr + fase_dados_goos[i]
        if fase >= 360:
            fase = fase - 360
            fase_dados_goos_local.append(fase)
        else:
            fase_dados_goos_local.append(fase)

    info_har_dados_goos = pd.DataFrame({'Componente de Mare':nameu_dados_goos,
                                        'Frequencia (cpd)':freq_dados_goos_cpd,
                                        'Amplitude (m)':ampli_dados_goos,
                                        'Fase (°)':fase_dados_goos_local})
    info_har_dados_goos.index = info_har_dados_goos['Componente de Mare']

    del info_har_dados_goos['Componente de Mare']

    return info_har_dados_goos

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def plot_componentesharmonicas(tide_dados_goos,TITULO):
    ######################################################### PLOT ########################################################
        
    fig = plt.figure(figsize=(11*1.2,5*1.2), dpi=70)
    plt.gca().set_prop_cycle(plt.cycler('color', 
                                        plt.cm.nipy_spectral(
                                            np.linspace(0, 0.98, 
                                                        len((np.array(np.where(
                                                            tide_dados_goos ["snr"] >= 2)))[0,:])))))
    ax1 = fig.add_subplot(111)

    #plt.xticks(x_plot, parametros_plot,fontsize=16, weight='bold')

    ax1.set_title('Análise Harmônica - Marégrafo '+str(TITULO),fontsize=20, weight='bold')

    #loc = 0.9
    for i in range(len(tide_dados_goos['fu'])):
        if tide_dados_goos ["snr"][i] >= 2:
            legenda = tide_dados_goos["nameu"][i]
            legenda = str(legenda)
            legenda = legenda.replace('\'', '')
            legenda = legenda.replace('b', '')
            legenda = legenda.replace('  ', '')
            legenda = legenda.replace(' ', '')
            plt.bar(tide_dados_goos['fu'][i]*24,       #EIXO X
                    tide_dados_goos ["tidecon"][i][0], #EIXO Y
                    align='center',#width=0.08,
                    label=legenda,
                    width=0.15,
                    edgecolor ='w',
                    linewidth =1,
                    alpha=0.8)


            if tide_dados_goos ["tidecon"][i][0] >= 0.05: #PARA COLOCAR LEGENDA SOBRE AS BARRAS
                ax1.annotate(legenda, 
                    xy=(tide_dados_goos['fu'][i]*24,
                        tide_dados_goos ["tidecon"][i][0] + 0.001), 
                    ha='center', va='bottom')

    ####ax1.set_xscale('log')
    ####ax1.legend(['Medido','Modelado | Período cada ponto','Modelado | Período B a C'], ncol=3, loc='upper center',fontsize=20)
    ax1.set_ylabel('Amplitude (m)',fontsize=20)
    ax1.set_xlabel('Frequência (cpd)',fontsize=20)
    ax1.set_xlim(0,8)
    ax1.legend(ncol=4,fontsize=14)
    plt.setp(ax1.get_yticklabels(),fontsize=14)
    plt.setp(ax1.get_xticklabels(),fontsize=14)



    plt.tight_layout()
    ###### Salvando as figuras em png 
    ####plt.savefig(dire_plot + 'histograma_com_30min_atraso_fase/periodo_dados_e_B&C/pto_A_hist.png', bbox_inches='tight', facecolor='w')
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def destacandoK1_O1_S2_M2_N2_K2_AmplitudeFaseeFatordeForma(info_har_dados_goos):
    ak1,pk1, ao1,po1, as2,ps2, am2,pm2, an2,pn2, ak2,pn2, ap1, pp1 = None,None,None,None,None,None,None,None,None,None,None,None,None,None
    for i in range(len(info_har_dados_goos)):
        #print(info_har_dados_goos.index[i])
        if info_har_dados_goos.index[i] == b'K1  ':
            ak1= info_har_dados_goos['Amplitude (m)'][i]
            pk1= info_har_dados_goos['Fase (°)'][i]
            print('ak1:',ak1)
            print('fase:',pk1)
            print('---------------------------------------------')

        if info_har_dados_goos.index[i] == b'O1  ':
            ao1= info_har_dados_goos['Amplitude (m)'][i]
            po1= info_har_dados_goos['Fase (°)'][i]
            print('ao1:',ao1)
            print('fase:',po1)
            print('---------------------------------------------')

        if info_har_dados_goos.index[i] == b'S2  ':
            as2= info_har_dados_goos['Amplitude (m)'][i]
            ps2= info_har_dados_goos['Fase (°)'][i]
            print('as2:',as2)
            print('fase:',ps2)
            print('---------------------------------------------')

        if info_har_dados_goos.index[i] == b'M2  ':
            am2= info_har_dados_goos['Amplitude (m)'][i]
            pm2= info_har_dados_goos['Fase (°)'][i]

            print('am2:',am2)
            print('fase:',pm2)
            print('---------------------------------------------')

        if info_har_dados_goos.index[i] == b'N2  ':
            an2= info_har_dados_goos['Amplitude (m)'][i]
            pn2= info_har_dados_goos['Fase (°)'][i]

            print('an2:',an2)
            print('fase:',pn2)
            print('---------------------------------------------')

        if info_har_dados_goos.index[i] == b'K2  ':
            ak2= info_har_dados_goos['Amplitude (m)'][i]
            pk2= info_har_dados_goos['Fase (°)'][i]
            print('ak2:',ak2)
            print('fase:',pk2)
            print('---------------------------------------------')
            
        if info_har_dados_goos.index[i] == b'K2  ':
            ap1= info_har_dados_goos['Amplitude (m)'][i]
            pp1= info_har_dados_goos['Fase (°)'][i]
            print('ap1:',ap1)
            print('fase:',pp1)
            print('---------------------------------------------')
            
        if info_har_dados_goos.index[i] == b'M4  ':
            am4= info_har_dados_goos['Amplitude (m)'][i]
            pm4= info_har_dados_goos['Fase (°)'][i]

            print('am4:',am4)
            print('fase:',pm4)
            print('---------------------------------------------')


    if ak1 and ao1 and as2 and am2 != None:
        nf_goos = (ak1+ao1)/(as2+am2) #NUMERO DE FORMA
        print(nf_goos)

        if nf_goos <= 0.25:
            print('Maré semi-diurna')

        elif nf_goos > 0.25 and nf_goos <= 1.5:
            print('Maré mista com predominância semi-diurna')

        elif nf_goos > 1.5 and nf_goos <= 3.0:
            print('Maré mista com predominância diurna')

        elif nf_goos > 3.0:
            print('Maré diurna')
        
    return ak1,pk1, ao1,po1, as2,ps2, am2,pm2, an2,pn2, ak2,pn2, ap1,pp1, am4,pm4

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def mapa_inicializa_figura_com_mapa_ARTIGOMONOGRAFIA_RbGf(zoom = True):
    font_size = 10
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    import cartopy.feature as cfeature
    import matplotlib.patheffects as PathEffects
    # get the data
    fn = shpreader.natural_earth(
        resolution='10m', category='cultural', 
        name='admin_1_states_provinces')
    reader = shpreader.Reader(fn)
    states = [x for x in reader.records() if x.attributes["admin"] == "Brazil"]
    states_geom = cfeature.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())

    data_proj = ccrs.PlateCarree()
    
    fig = plt.figure(figsize=(8.66/1.5,6.37/1.5),dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())        

    ax.add_feature(cfeature.STATES,lw=0.5) 
    ax.add_feature(cfeature.LAND,color='lightgrey')
    ax.add_feature(cfeature.COASTLINE,lw=0.5)
    #NOMES DOS ESTADOS
    for state in states:

        lon = state.geometry.centroid.x
        lat = state.geometry.centroid.y
        name = state.attributes["name"] 
        if zoom:
            if name in ['Rio de Janeiro','Minas Gerais','SÃ£o Paulo','EspÃ­rito Santo']:
                if name == 'EspÃ­rito Santo':name = 'ES';
                if name == 'SÃ£o Paulo':name = 'SP';lon = lon + 1.5
                if name == 'Minas Gerais':name = 'MG';
                if name == 'Rio de Janeiro':name = 'RJ';

                ax.text(lon, lat, name, size=font_size-2, transform=data_proj, ha="center", va="center",
                    path_effects=[PathEffects.withStroke(linewidth=5, foreground="lightgrey")])  
                
    return fig,ax

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def mapa_com_estados_ARTIGOMONOGRAFIA_RbGf(ax,zoom = True):
    font_size = 10
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    import cartopy.feature as cfeature
    import matplotlib.patheffects as PathEffects
    # get the data
    fn = shpreader.natural_earth(
        resolution='10m', category='cultural', 
        name='admin_1_states_provinces')
    reader = shpreader.Reader(fn)
    states = [x for x in reader.records() if x.attributes["admin"] == "Brazil"]
    states_geom = cfeature.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())

    data_proj = ccrs.PlateCarree()
    #NOMES DOS ESTADOS
    for state in states:

        lon = state.geometry.centroid.x
        lat = state.geometry.centroid.y
        name = state.attributes["name"] 
        if zoom:
            if name in ['Rio de Janeiro','Minas Gerais','SÃ£o Paulo','EspÃ­rito Santo']:
                if name == 'EspÃ­rito Santo':name = 'ES';
                if name == 'SÃ£o Paulo':name = 'SP';lon = lon + 1.5
                if name == 'Minas Gerais':name = 'MG';
                if name == 'Rio de Janeiro':name = 'RJ';

                ax.text(lon, lat, name, size=font_size-2, transform=data_proj, ha="center", va="center",
                    path_effects=[PathEffects.withStroke(linewidth=5, foreground="lightgrey")])  
      
    return ax

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def near(array, value):
    '''
    Função para comparar dados de lat e lon diferentes.
    ---
    array: 
        vetor de numeros
    value: 
        numero que vc quer encontrar no array
    ------    
    Returns: 
        indice, no array, do numero mais próximo ao requisitado pelo value.
        
    Por exemplo:
    
        array = np.array([1.0 , 2.0 , 3.0])
        value = 1.51 #É UM TESTE
        print('array:',array)
        print('np.abs(array - value):',np.abs(array - value))
        print('(np.abs(array - value)).argmin():',(np.abs(array - value)).argmin())
        
        >>>array: [1. 2. 3.]
        >>>np.abs(array - value): [0.51 0.49 1.49]
        >>>(np.abs(array - value)).argmin(): 1
    '''
    array = np.array(array)
    idx = (np.abs(array - value)).argmin() #SUBTRAI UM DO OUTRO E VE QUAL INDICE DO 
                                           #VALOR MAIS PROXIMO DO REQUISITADO EM value
    return idx

def last_day_of_the_month(year,month):
    '''
    Retorna o numero do ultimo dia do mês do ano fornecido.
    ----
    Exemplo:
        calendar.monthrange(2024,1)[1]
        >>> 31

    '''
    import calendar
    from datetime import datetime
    
    last_day_of_the_month = calendar.monthrange(year,month)[1]
    
    return last_day_of_the_month


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def azul_e_preto():
    import matplotlib.colors as mc
    colors = [(0,0,0), (0,40,60), (0,70,100), (0,100,160), (0,130,200), (0,160,255)]#ORIGINAL:(0,160,255)]#, (0,0,255)]
    #colors = [(40,40,160), (0,130,200), (100,200,200), (200,200,200),(200,200,100),(200,130,0),(160,40,40)]#AZUL E FOGO
    #cmap.set_over('#d1d3d4')
    #cmap.set_under('#404d90')
    #position = [0, 0.5, 1] #POSIÇÃO DO BRANCO
    my_cmap = rm.make_cmap(colors,bit=True)#, position=position)
    return my_cmap

def azul_e_fogo():
    import matplotlib.colors as mc
    colors = [(40,40,160), (0,130,200), (100,200,200), (200,200,200),(200,200,100),(200,130,0),(160,40,40)]#AZUL E FOGO
    #cmap.set_over('#d1d3d4')
    #cmap.set_under('#404d90')
    #position = [0, 0.5, 1] #POSIÇÃO DO BRANCO
    my_cmap = rm.make_cmap(colors,bit=True)#, position=position)
    return my_cmap

def azul_e_fogo_claro():
    import matplotlib.colors as mc
    # Clareando os valores RGB originais, mantendo as mesmas cores
    colors = [(0,0,255),
              (40+50,40+50,160+50), 
              (0+50,130+50,200+50), 
              (100+30,200+20,200+30), 
              (255,255,255),
              (200+30,200+20,100+30),
              (200+50,130+50,0+50),
              (160+50,40+50,40+50),
              (255,0,0)]#AZUL E FOGO

    
    # Fazendo o cmap com as cores clareadas
    my_cmap = rm.make_cmap(colors, bit=True)
    return my_cmap

def azul_escuro_dourado():
    import matplotlib.colors as mc
    import numpy as np

    # Definindo os valores RGB para azul escuro e dourado
    colors = [
        (0, 0, 128),       # Azul escuro
        (0, 0, 255),       # Azul
        (50, 50, 200),     # Azul mais claro
        (150, 150, 255),   # Azul bem claro
        (255, 255, 255),   # Branco
        (255, 215, 0),     # Dourado
        (255, 165, 0),     # Laranja/dourado mais escuro
        (200, 100, 0),     # Bronzeado
        (128, 64, 0)       # Dourado profundo
    ]

    # Normalizando as cores e criando o colormap
    def make_cmap(colors, bit=False):
        if bit:
            colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
        return mc.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    my_cmap = make_cmap(colors, bit=True)
    return my_cmap

def correntes_lof():
    import matplotlib.colors as mc

    # Definindo os valores RGB baseados na paleta do screenshot
    colors = [
        (0, 0, 255),     # Azul
        (0, 128, 255),   # Ciano
        (0, 255, 255),   # Ciano claro
        (255, 255, 0),   # Amarelo
        (255, 165, 0),   # Laranja
        (255, 0, 0)      # Vermelho
    ]

    # Criando o colormap com as cores
    def make_cmap(colors, bit=False):
        if bit:
            colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
        return mc.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    my_cmap = make_cmap(colors, bit=True)
    return my_cmap


def turbo_com_branco():
    import matplotlib.colors as mc
    
    colors = [
        (0, 0, 130),       # azul escuro
        (0, 80, 255),      # azul claro
        (0, 200, 200),     # ciano
        (50, 200, 50),   # verde
        (255, 255, 255),   # branco no centro
        (255, 255, 0),     # amarelo
        (255, 120, 0),     # laranja
        (230, 0, 0),       # vermelho
        (130, 0, 0)        # vermelho escuro
    ]
    
    my_cmap = rm.make_cmap(colors, bit=True)

    return my_cmap

def verde_azul_branco():
    import matplotlib.colors as mc
    
    colors = [
        (0, 50, 0),        # verde escuro
        (0, 120, 80),      # verde médio
        (0, 190, 80),      # verde
        (0, 200, 255),     # azul claro (centro)
        (0, 0, 180),       # azul escuro
        (230, 230, 255)    # branco (extremidade final)
    ]
    
    my_cmap = rm.make_cmap(colors, bit=True)
    
    return my_cmap

def verde_azul_branco():
    import matplotlib.colors as mc
    
    colors = [
        (50, 120, 50),      # verde escuro muito suavizado
        (80, 180, 120),     # verde médio muito suavizado
        (120, 220, 180),    # verde claro ainda mais claro
        (100, 100, 220),    # azul escuro (centro) suavizado
        (150, 200, 255),    # azul claro muito suavizado
        (250, 250, 255)     # branco muito claro (extremidade final)
    ]
    
    my_cmap = rm.make_cmap(colors, bit=True)
    
    return my_cmap


def azul_e_gelo():
    import matplotlib.colors as mc
    colors = [(20,20,120), (20,20,160), (40,40,160),(40,70,160),(40,70,180),
              (40,100,210),(40,100,240),(255,255,255),(80,140,180)]#AZUL E GELO    
    my_cmap = rm.make_cmap(colors,bit=True)#, position=position)
    return my_cmap

def roxo_claro_e_roxo_escuro():
    import matplotlib.colors as mc
    # Definindo a paleta de cores do roxo claro até o roxo escuro
    colors = [(230, 190, 255), (200, 150, 245), (170, 110, 230), (130, 70, 210), (90, 40, 180), (60, 10, 140)] 
    # Fazendo o cmap com as cores definidas
    my_cmap = rm.make_cmap(colors, bit=True)
    return my_cmap


def azul_ver_amarelo(cmap, minval=0.0, maxval=1.0, n=100):
    '''Azul verde e amarelo como Matlab
    '''
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    
    #USO:cmap_name = 'turbo_r'
         #new_cmap = truncate_colormap(plt.get_cmap(cmap_name), 0.25, 1)
    return new_cmap

def nipy_spectral_com_branco():
    import matplotlib.colors as mc
    # Definindo a paleta de cores do nipy_spectral, incluindo o branco no início
    colors = [(255, 255, 255),
              (255, 255, 255),# Branco
              (50, 0, 110),     # Escuro (tom semelhante ao início do nipy_spectral)
              (0, 0, 255),      # Azul forte
              (0, 255, 255),    # Ciano
              (0, 255, 0),      # Verde
              (255, 255, 0),    # Amarelo
              (255, 127, 0),    # Laranja
              (255, 0, 0),      # Vermelho
              (160, 0, 0)]      # Vermelho escuro (fim de nipy_spectral)
    
    # Fazendo o cmap com as cores definidas
    my_cmap = rm.make_cmap(colors, bit=True)
    return my_cmap

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def umidade_windy():
    colors = [
        (173, 85, 56, 255),
        (173, 110, 56, 255),
        (173, 146, 56, 255),
        (105, 173, 56, 255),
        (56, 173, 121, 255),
        (56, 174, 173, 255),
        (56, 160, 173, 255),
        (56, 157, 173, 255),
        (56, 148, 173, 255),
        (56, 135, 173, 255),
        (56, 132, 173, 255),
        (56, 123, 173, 255),
        (56, 98, 157, 255),
        (56, 70, 114, 255)
    ]
    
    # Normaliza as cores
    colors = [(r / 255, g / 255, b / 255, a / 255) for r, g, b, a in colors]
    
    n_bins = len(colors)  # Discretizes the interpolation into bins
    cmap_name = 'umidade_windy'
    my_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return my_cmap

import matplotlib.colors as mcolors

def umidade_windy_com_branco():
    # Definindo cores: lado laranja -> branco -> lado azul
    colors = [
        (173, 85, 56, 255),   # Laranja escuro
        (173, 110, 56, 255),  # Laranja claro
        (173, 146, 56, 255),  # Laranja mais claro
        (150, 200, 100, 255), # Laranja claro esverdeado antes do branco
        #(255, 255, 255, 255), # Branco no centro
        (100, 200, 200, 255), # Azul claro esverdeado depois do branco
        (56, 174, 173, 255),  # Azul claro
        (56, 135, 173, 255),  # Azul mais claro
        (56, 70, 114, 255)    # Azul escuro
    ]

    
    # Normaliza as cores (0-1)
    colors = [(r / 255, g / 255, b / 255, a / 255) for r, g, b, a in colors]
    
    # Criação do colormap com transição suave
    n_bins = len(colors)  # Discretizes the interpolation into bins
    cmap_name = 'umidade_windy'
    my_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return my_cmap

def umidade_windy_flux():
    # Definindo cores: lado laranja -> branco -> lado azul
    colors = [
        (173+50, 85, 56+30, 255),   # Laranja escuro
        (173+50, 110, 56+30, 255),  # Laranja claro
        (173+50, 146, 56+30, 255),  # Laranja mais claro
        (150+50, 200, 100+30, 255), # Laranja claro esverdeado antes do branco
        #(255, 255, 255, 255), # Branco no centro
        (100+50, 200, 200+30, 255), # Azul claro esverdeado depois do branco
        (56+50, 174, 173+30, 255),  # Azul claro
        (56+50, 135, 173+30, 255),  # Azul mais claro
        (56+50, 70, 114+30, 255)    # Azul escuro
    ]

    
    # Normaliza as cores (0-1)
    colors = [(r / 255, g / 255, b / 255, a / 255) for r, g, b, a in colors]
    
    # Criação do colormap com transição suave
    n_bins = len(colors)  # Discretizes the interpolation into bins
    cmap_name = 'umidade_windy'
    my_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return my_cmap
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

import psutil

def info_memoria():
    
    # Obter informações sobre o uso de memória
    mem = psutil.virtual_memory()
    print(f"Uso de memória (total): {mem.total / (1024 ** 3)} GB")
    print(f"Uso de memória (disponível): {mem.available / (1024 ** 3)} GB")
    print(f"Percentual de uso de memória: {mem.percent}%")

    # Obter informações sobre o uso de processamento (CPU)
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"Percentual de uso de CPU: {cpu_percent}%")
    
    
import psutil

def listar_uso_memoria():
    processos = []
    
    for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_info']):
        try:
            info = proc.info
            mem_usage = info['memory_info'].rss / (1024 ** 2)  # Conversão para MB
            processos.append({
                'PID': info['pid'],
                'Nome': info['name'],
                'Usuário': info['username'],
                'Uso de Memória (MB)': mem_usage
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    # Ordenar os processos pelo uso de memória
    processos_ordenados = sorted(processos, key=lambda x: x['Uso de Memória (MB)'], reverse=True)
    
    print(f"{'PID':<10}{'Nome':<30}{'Usuário':<20}{'Uso de Memória (MB)':<20}")
    print("=" * 80)
    
    for proc in processos_ordenados[:10]:  # Mostra os 10 maiores consumidores de memória
        print(f"{proc['PID']:<10}{proc['Nome']:<30}{proc['Usuário']:<20}{proc['Uso de Memória (MB)']:<20.2f}")

# Chamar a função
#listar_uso_memoria()

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------    
    
def le_tracking_camargo(ANOi,ANOf, reanalise_escolhida='CFS'):
    '''
    Usage example:
    camargo,ano_esc = le_tracking_camargo(ANOi,ANOf)

    '''
    dirdados = os.getcwd()+'/../DADOS/'

    dirCamargoERA5 = dirdados+'ERA5/ExtratropicalCycloneTracks_Gramcianinov_e_Camargo_2020/'
    dirCamargoCFS = dirdados+'CFS/ExtratropicalCycloneTracks_Gramcianinov_e_Camargo_2020/'

    dataframes = []  # Inicializa uma lista vazia para armazenar os DataFrames que vamos concatenar

    for ano_esc in range(ANOi, ANOf+1):
        for mes in range(1, 12+1):

            mes_str = str(mes).zfill(2)  # Adiciona um zero à esquerda se 'mes' for menor que 10


            if reanalise_escolhida == 'ERA5':
                csv_mes = dirCamargoERA5 + 'ff_cyc_ExSAt_era5_' + str(ano_esc) + mes_str + '.csv'

                if (str(ano_esc) + mes_str) not in ['198308','201908']: #FALTAM ESSES DADOS NO ERA5 DO CAMARGO
                    camargo = pd.read_csv(csv_mes, header=None)
                    dataframes.append(camargo)  # Adiciona o DataFrame à lista

            if reanalise_escolhida == 'CFS':
                csv_mes = dirCamargoCFS + 'ff_cyc_ExSAt_cfs_' + str(ano_esc) + mes_str + '.csv'
                camargo = pd.read_csv(csv_mes, header=None)
                dataframes.append(camargo)  # Adiciona o DataFrame à lista




    # Concatena todos os DataFrames na lista
    camargo = pd.concat(dataframes)

    camargo.columns = ['ID', 'Time', 'Longitude', 'Latitude', 'T42 vorticity']


    #PARA FICAR IGUAL LODISE #COMANDO GPT #Verifique as longitudes maiores que 180 e subtraia 360 delas
    camargo.loc[camargo['Longitude'] > 180, 'Longitude'] -= 360 


    #COLOCANDO DF camargo IGUAL AO DE lodise:
    camargo['ID'] = camargo['ID'].apply(lambda x: int(str(x)[-4:]))
    # Reordenando as colunas
    camargo = camargo[['ID', 'Longitude', 'Latitude', 'T42 vorticity', 'Time']]
    camargo['Latitude'] = camargo['Latitude'].round(3) #APENAS TIRANDO DIGITOS DESNECESSARIOS
    camargo['Longitude'] = camargo['Longitude'].round(3) #APENAS TIRANDO DIGITOS DESNECESSARIOS
    camargo['T42 vorticity'] = camargo['T42 vorticity'].round(3) #APENAS TIRANDO DIGITOS DESNECESSARIOS
    
    #COLOCANDO COLUNA DE DATA NO FORMATO datetime
    camargo['Time'] = pd.to_datetime(camargo['Time'])

    dd(camargo)
    return camargo,ano_esc


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def normalize(array):
    '''
        Subtract the mean, and divide by the standard deviation of the array.
    '''
    array_normalized = (array - np.nanmean(array))/(np.nanstd(array)) 
    return array_normalized
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# Função para verificar se pontos estão dentro do polígono
def inpolygon(x, y, poly):
    '''
    Inspired on Matlab function inpolygon.
    
    The fuction matplotlib.path.Path requires the shape in two columns of Lon and Lat, in this order
    The values of Lon and Lat to create the polygon need to close. This means that 
    the last value of lon and lat should be the same as the first. That's why in the example of use below it is
    perfoming a concatenation of the first value in the end of the arrays.
    After this, the function np.vstack is used to unify the values of the tuple (Lon and Lat),  
    being able now to transpose it, necessary for the inpolygon.
    After this, the inpolygon will return values of true and false. 
        the x parameter is the vectorized longitudes of all the cyclones;
        the y parameter is the vectorized latitudes of all the cyclones;
        the poly parameter is the 2 columns Lon and Lat array with initial value equal to the last! 
        Example:
         poly = array([[-50.    , -29.5   ],
                       [-51.242 , -32.0025],
                       [-51.198 , -32.0077],
                       .
                       .
                       .
                       [-48.    , -28.5   ],
                       [-50.    , -29.5   ]])
                       
    To plot the values that are the ones inside of the created poly, just use, for example:
        plt.plot(df_storms['Longitude'][in_stf], df_storms['Latitude'][in_stf])
    
    ---
    Example of use:
    
        > Lon_poly_stf = np.concatenate(( orsi['stf']['Longitude'].values, [orsi['stf']['Longitude'].values[0]] ))
        > Lat_poly_stf = np.concatenate((orsi['stf']['Latitude'].values, [orsi['stf']['Latitude'].values[0]] ))

        > polygon_stf = np.vstack((Lon_poly_stf, Lat_poly_stf)).T
        > in_stf = inpolygon(df_storms['Longitude'], df_storms['Latitude'], polygon_stf)

        > plt.plot(Lon_poly_stf, Lat_poly_stf, color = 'red')
        > plt.plot(df_storms['Longitude'][in_stf], df_storms['Latitude'][in_stf], '*',color='tab:red')

    '''
    import matplotlib.path as mpath

    path = mpath.Path(poly)
    return path.contains_points(np.vstack((x, y)).T)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def le_orsi_swatl():
    dirdadosOrsi = os.getcwd()+'/../DADOS/Orsi_1995_OceanicFronts/Archive/'

    orsi = {}
    orsi['stf'] = pd.read_table(dirdadosOrsi + 'stf_SWAtlantic.txt', delim_whitespace=True, names=['Longitude', 'Latitude'])
    orsi['saf'] = pd.read_table(dirdadosOrsi + 'saf_SWAtlantic.txt', delim_whitespace=True, names=['Longitude', 'Latitude'])
    orsi['pf'] = pd.read_table(dirdadosOrsi + 'pf_SWAtlantic.txt', delim_whitespace=True, names=['Longitude', 'Latitude'])
    
    return orsi

def plota_orsi_swatl(ax,alpha=0.8,lw=2,zorder=2):
    orsi = le_orsi_swatl()
    
    ax.plot(orsi['stf']['Longitude'],orsi['stf']['Latitude'], color='k',alpha=alpha,lw=lw,zorder=zorder)
    ax.plot(orsi['saf']['Longitude'],orsi['saf']['Latitude'], color='k',alpha=alpha,lw=lw,zorder=zorder)
    ax.plot(orsi['pf']['Longitude'],orsi['pf']['Latitude'], color='k',alpha=alpha,lw=lw,zorder=zorder)
    

    
#def ZapiolaPolygon(print_polygon=False):
def ZapiolaPolygon(ax,print_polygon=False):

    '''

    
    How to use with the data:
        lon_mydata, lat_mydata = np.meshgrid(lon_mydata, lat_mydata)
        in_zap = rm.inpolygon(lon_mydata.flatten(), lat_mydata.flatten(), rm.ZapiolaPolygon(print_polygon=True))
        plt.plot(lon_mydata.flatten()[in_zap], lat_mydata.flatten()[in_zap], '*',color='tab:blue')
        
    To go back to the previous format and plot the field:
        cutted_grid = np.full(lon_mydata.shape, np.nan)  # Cria um grid vazio com NaNs
        cutted_grid.flat[in_zap] = original_grid.flatten()[in_zap]  # Substitui apenas os pontos dentro do polígono
        plt.pcolormesh(cutted_grid) #original_grid must be the same shape as lon_mydata

    '''
    
    orsi = le_orsi_swatl()

    ##-----------------------------  ZAPIOLA DRIFT
    zap_stf_lon = orsi['stf']['Longitude'].values[47:87]
    zap_stf_lat = orsi['stf']['Latitude'].values[47:87]
    zap_saf_lon = orsi['saf']['Longitude'].values[229:354]
    zap_saf_lat = orsi['saf']['Latitude'].values[229:354]

    Lon_poly_zap = np.concatenate((zap_stf_lon, np.flipud(zap_saf_lon), [zap_stf_lon[0]]))
    Lat_poly_zap = np.concatenate((zap_stf_lat, np.flipud(zap_saf_lat), [zap_stf_lat[0]]))

    polygon_zap = np.vstack((Lon_poly_zap, Lat_poly_zap)).T #ATE AQUI APENAS DADOS ORSI

    #if print_polygon:
    #    plt.figure(5)
    #    plt.plot(Lon_poly_zap, Lat_poly_zap,color='blue')
    #    plt.show()
    if print_polygon:
        #plt.figure(5)
        ax.plot(Lon_poly_zap, Lat_poly_zap,color='blue')
        #plt.show()    
    return polygon_zap

import numpy as np
import matplotlib.pyplot as plt




def NorthSTFPolygon(ax,print_polygon=False,color='yellow'):

    '''
    
    How to use with the data:
        in_zap = rm.inpolygon(lon_mydata, lat_mydata, polygon_zap)
        plt.plot(lon_mydata[in_zap], lat_mydata[in_zap], '*',color='tab:blue')

    '''
    
    orsi = le_orsi_swatl()
    
    lon_i_swAtl,lon_f_swAtl = -70 , -17
    lat_i_swAtl, lat_f_swAtl = -65, -20
    close_stf_lons = np.array([lon_f_swAtl,         -39,   -40  ,  -44,   -47  ,   -48  ])
    close_stf_lats = np.array([lat_f_swAtl, lat_f_swAtl,   -22.5,  -24,   -25.5,   -28.5])
    #close_stf_lons = np.array([                      -39,   -40  ,  -44,   -47  ,   -48  ])
    #close_stf_lats = np.array([              lat_f_swAtl,   -22.5,  -24,   -25.5,   -28.5])
    #------------------------------- NORTH OF SUBTROPICAL FRONT
    #THIS ONE IS DIFFERENT
 

    Lon_poly_stf = np.concatenate( ( orsi['stf']['Longitude'].values, close_stf_lons, [orsi['stf']['Longitude'].values[0]] ))
    Lat_poly_stf = np.concatenate((orsi['stf']['Latitude'].values, close_stf_lats, [orsi['stf']['Latitude'].values[0]] ))

    polygon_stf = np.vstack((Lon_poly_stf, Lat_poly_stf)).T
        
    
    if print_polygon:
        #plt.figure()
        ax.plot(Lon_poly_stf, Lat_poly_stf, color='grey', lw=3.5) 
        ax.plot(Lon_poly_stf, Lat_poly_stf, color = color)
        #plt.show()        
    return polygon_stf


#def ReducedZapiolaPolygon(scale_factor=0.8, print_polygon=False,color='lime'):
def ReducedZapiolaPolygon(ax,scale_factor=0.8, print_polygon=False,color='lime'):

    """
    Cria um polígono reduzido da região de Zapiola, mantendo o mesmo centro.
    
    scale_factor: Fator para reduzir o tamanho do polígono. 1.0 mantém o tamanho original.
    print_polygon: Se True, plota o polígono para visualização.
    """
    orsi = le_orsi_swatl()

    ##-----------------------------  ZAPIOLA DRIFT
    zap_stf_lon = orsi['stf']['Longitude'].values[47:87]
    zap_stf_lat = orsi['stf']['Latitude'].values[47:87]
    zap_saf_lon = orsi['saf']['Longitude'].values[229:354]
    zap_saf_lat = orsi['saf']['Latitude'].values[229:354]

    Lon_poly_zap = np.concatenate((zap_stf_lon, np.flipud(zap_saf_lon), [zap_stf_lon[0]]))
    Lat_poly_zap = np.concatenate((zap_stf_lat, np.flipud(zap_saf_lat), [zap_stf_lat[0]]))

    # Calcula o centro do polígono
    center_lon = np.mean(np.concatenate((zap_stf_lon, zap_saf_lon)))
    center_lat = np.mean(np.concatenate((zap_stf_lat, zap_saf_lat)))
    
    # Ajusta os pontos para aproximar ao centro
    zap_stf_lon_reduced = center_lon + scale_factor * (zap_stf_lon - center_lon)
    zap_stf_lat_reduced = center_lat + scale_factor * (zap_stf_lat - center_lat)
    zap_saf_lon_reduced = center_lon + scale_factor * (zap_saf_lon - center_lon)
    zap_saf_lat_reduced = center_lat + scale_factor * (zap_saf_lat - center_lat)
    
    # Cria o polígono reduzido
    Lon_poly_zap_reduced = np.concatenate((zap_stf_lon_reduced, np.flipud(zap_saf_lon_reduced), [zap_stf_lon_reduced[0]]))
    Lat_poly_zap_reduced = np.concatenate((zap_stf_lat_reduced, np.flipud(zap_saf_lat_reduced), [zap_stf_lat_reduced[0]]))
    polygon_zap_reduced = np.vstack((Lon_poly_zap_reduced, Lat_poly_zap_reduced)).T
    
    #if print_polygon:
    #    plt.plot(Lon_poly_zap_reduced, Lat_poly_zap_reduced, color='grey', lw=3.5)
    #    plt.plot(Lon_poly_zap_reduced, Lat_poly_zap_reduced, color=color)
    #    plt.show()
    if print_polygon:
        ax.plot(Lon_poly_zap_reduced, Lat_poly_zap_reduced, color='grey', lw=3.5)
        ax.plot(Lon_poly_zap_reduced, Lat_poly_zap_reduced, color=color)
    #ATENCAO: POR ALGUM MOTIVO O IF DO PRINT ABAIXO NAO FUNDIONOU, APENAS FORA DO IF!!!!!!!!!!
    #if print_polygon:
        #plt.figure()
        #plt.plot(Lon_poly_zap_reduced, Lat_poly_zap_reduced, color='grey', lw=3.5)        
        #plt.plot(Lon_poly_zap_reduced, Lat_poly_zap_reduced, color=color)#, label='Polígono Reduzido')
        #plt.plot(zap_stf_lon, zap_stf_lat, '--', color='blue', alpha=0.5, label='STF Original')
        #plt.plot(zap_saf_lon, zap_saf_lat, '--', color='green', alpha=0.5, label='SAF Original')
        #plt.legend()
        #plt.xlabel('Longitude')
        #plt.ylabel('Latitude')
        #plt.title('Região de Zapiola Reduzida')
        #plt.show()

    return polygon_zap_reduced


######def NorthACCPolygon(ax, print_polygon=False, color='magenta'):
######    
######    '''
######    
######    How to use with the data:
######        in_zap = rm.inpolygon(lon_mydata, lat_mydata, polygon_zap)
######        plt.plot(lon_mydata[in_zap], lat_mydata[in_zap], '*',color='tab:blue')
######
######    '''
######    
######    orsi = le_orsi_swatl()
######    lon_i_swAtl,lon_f_swAtl = -70 , -17
######    lat_i_swAtl, lat_f_swAtl = -65, -20
######    close_pf_lons = np.array([lon_f_swAtl, lon_i_swAtl])
######    close_pf_lats = np.array([lat_i_swAtl, lat_i_swAtl])
######    ##----------------------------- SOUTH OF POLAR FRONT
######    Lon_poly_acc = np.concatenate((orsi['pf']['Longitude'].values, close_pf_lons, [orsi['pf']['Longitude'].values[0]]))
######    Lat_poly_acc = np.concatenate((orsi['pf']['Latitude'].values, close_pf_lats, [orsi['pf']['Latitude'].values[0]]))
######
######    polygon_acc = np.vstack((Lon_poly_acc, Lat_poly_acc)).T
######
######    #if print_polygon:
######    #    #plt.figure()
######    #    plt.plot(Lon_poly_acc, Lat_poly_acc, color='grey', lw=3.5) 
######    #    plt.plot(Lon_poly_acc, Lat_poly_acc, color = color)
######    #    #plt.show()
######    if print_polygon:
######        ax.plot(Lon_poly_acc, Lat_poly_acc, color='grey', lw=3.5)
######        ax.plot(Lon_poly_acc, Lat_poly_acc, color=color)
######        
######    return polygon_acc
def NorthACCPolygon(ax, scale_factor=1.0, print_polygon=False, color='magenta'):
    """
    Cria um polígono reduzido/ampliado da região North ACC, mantendo o mesmo centro.
    
    scale_factor: Fator para reduzir/ampliar o tamanho do polígono (1.0 mantém o tamanho original).
    print_polygon: Se True, plota o polígono para visualização.
    """
    orsi = le_orsi_swatl()
    lon_i_swAtl, lon_f_swAtl = -70, -17
    lat_i_swAtl, lat_f_swAtl = -65, -20

    # Fechamento manual do polígono
    close_pf_lons = np.array([lon_f_swAtl, lon_i_swAtl])
    close_pf_lats = np.array([lat_i_swAtl, lat_i_swAtl])

    # Polígono original
    Lon_poly_acc = np.concatenate((
        orsi['pf']['Longitude'].values,
        close_pf_lons,
        [orsi['pf']['Longitude'].values[0]]  # fecha o polígono
    ))

    Lat_poly_acc = np.concatenate((
        orsi['pf']['Latitude'].values,
        close_pf_lats,
        [orsi['pf']['Latitude'].values[0]]
    ))

    # Calcula o centro
    center_lon = -38 #np.mean(Lon_poly_acc)
    center_lat = -58 #np.mean(Lat_poly_acc)
    ##########OBS: o calculo da latlon central nao funcionou assim. Dessa forma, -38x-58 foi escolhido visualmente

    # Reduz/amplia em relação ao centro
    Lon_poly_acc_scaled = center_lon + scale_factor * (Lon_poly_acc - center_lon)
    Lat_poly_acc_scaled = center_lat + scale_factor * (Lat_poly_acc - center_lat)

    polygon_acc_scaled = np.vstack((Lon_poly_acc_scaled, Lat_poly_acc_scaled)).T

    if print_polygon:
        ax.plot(Lon_poly_acc_scaled, Lat_poly_acc_scaled, color='grey', lw=3.5)
        ax.plot(Lon_poly_acc_scaled, Lat_poly_acc_scaled, color=color)

    return polygon_acc_scaled



#def NorthSAFPolygon(print_polygon=False,color='orange'):
def NorthSAFPolygon(ax,print_polygon=False,color='orange'):

    '''
    
    How to use with the data:
        in_zap = rm.inpolygon(lon_mydata, lat_mydata, polygon_zap)
        plt.plot(lon_mydata[in_zap], lat_mydata[in_zap], '*',color='tab:blue')

    '''
    
    orsi = le_orsi_swatl()
    close_saf_lons = np.array([-67,-62.5])
    close_saf_lats = np.array([-45,-40])
    #------------------------------- SUBANTARCTIC ZONE
    Lon_poly_saz = np.concatenate((orsi['stf']['Longitude'].values, np.flipud(orsi['saf']['Longitude'].values),close_saf_lons ,[orsi['stf']['Longitude'].values[0]]))
    Lat_poly_saz = np.concatenate((orsi['stf']['Latitude'].values, np.flipud(orsi['saf']['Latitude'].values), close_saf_lats,[orsi['stf']['Latitude'].values[0]]))

    polygon_saz = np.vstack((Lon_poly_saz, Lat_poly_saz)).T

    #if print_polygon:

    #    plt.figure(2)
    #    plt.plot(Lon_poly_saz, Lat_poly_saz, color='grey', lw=3.5) 
    #    plt.plot(Lon_poly_saz, Lat_poly_saz, color = color)
    #    plt.show()
    
    if print_polygon:
        #plt.figure(2)
        ax.plot(Lon_poly_saz, Lat_poly_saz, color='grey', lw=3.5) 
        ax.plot(Lon_poly_saz, Lat_poly_saz, color = color)
        #plt.show()       
    return polygon_saz


#def NorthPFPolygon(ax,print_polygon=False,color='green'):
#
#    '''
#    
#    How to use with the data:
#        in_zap = rm.inpolygon(lon_mydata, lat_mydata, polygon_zap)
#        plt.plot(lon_mydata[in_zap], lat_mydata[in_zap], '*',color='tab:blue')
#
#    '''
#    
#    orsi = le_orsi_swatl()
#    ##----------------------------- POLAR FRONTAL ZONE
#    Lon_poly_pfz = np.concatenate((orsi['saf']['Longitude'].values, np.flipud(orsi['pf']['Longitude'].values), [orsi['saf']['Longitude'].values[0]]))
#    Lat_poly_pfz = np.concatenate((orsi['saf']['Latitude'].values, np.flipud(orsi['pf']['Latitude'].values), [orsi['saf']['Latitude'].values[0]]))
#
#    polygon_pfz = np.vstack((Lon_poly_pfz, Lat_poly_pfz)).T
#    
#    #if print_polygon:
#
#    #    plt.figure(3)
#    #    plt.plot(Lon_poly_pfz, Lat_poly_pfz, color='grey', lw=3.5) 
#    #    plt.plot(Lon_poly_pfz, Lat_poly_pfz, color = color)
#    #    plt.show()
#    if print_polygon:
#
#        #plt.figure(3)
#        ax.plot(Lon_poly_pfz, Lat_poly_pfz, color='grey', lw=3.5) 
#        ax.plot(Lon_poly_pfz, Lat_poly_pfz, color = color)
#        #plt.show()        
#    return polygon_pfz

def NorthPFPolygon(ax, scale_factor=1.0, print_polygon=False, color='blue'):
    """
    Cria um polígono reduzido/ampliado da região North of Polar Front, mantendo o mesmo centro.

    scale_factor: Fator para reduzir/ampliar o tamanho do polígono (1.0 mantém o tamanho original).
    print_polygon: Se True, plota o polígono para visualização.
    """
    orsi = le_orsi_swatl()
    lon_i_swAtl, lon_f_swAtl = -70, -17
    lat_i_swAtl, lat_f_swAtl = -65, -20

    # Polígono original
    Lon_poly_npf = np.concatenate((
        orsi['saf']['Longitude'][215:270].values,
        [orsi['pf']['Longitude'][120]],
        orsi['pf']['Longitude'][121:].values,
        [lon_f_swAtl],
        [-39], [-40], [-44], [-47], [-48],
        [orsi['stf']['Longitude'].values[0]],
        [orsi['saf']['Longitude'][215]],
    ))

    Lat_poly_npf = np.concatenate((
        orsi['saf']['Latitude'][215:270].values,
        [orsi['pf']['Latitude'][120]],
        orsi['pf']['Latitude'][121:].values,
        [lat_f_swAtl],
        [lat_f_swAtl], [-22.5], [-24], [-25.5], [-28.5],
        [orsi['stf']['Latitude'].values[0]],
        [orsi['saf']['Latitude'][215]],
    ))

    # Calcula o centro do polígono
    center_lon = -35#np.mean(Lon_poly_npf)
    center_lat = -35#np.mean(Lat_poly_npf)
    ##########OBS: o calculo da latlon central nao funcionou assim. Dessa forma, -35x-35 foi escolhido visualmente

    # Aplica redução/ampliação
    Lon_poly_npf_scaled = center_lon + scale_factor * (Lon_poly_npf - center_lon)
    Lat_poly_npf_scaled = center_lat + scale_factor * (Lat_poly_npf - center_lat)

    polygon_npf_scaled = np.vstack((Lon_poly_npf_scaled, Lat_poly_npf_scaled)).T

    if print_polygon:
        ax.plot(Lon_poly_npf_scaled, Lat_poly_npf_scaled, color='grey', lw=3.5)
        ax.plot(Lon_poly_npf_scaled, Lat_poly_npf_scaled, color=color)

    #ax.plot(Lon_poly_npf, Lat_poly_npf, color='grey', lw=1,marker='o',ms=2) 
    #ax.plot(Lon_poly_npf, Lat_poly_npf, color = 'blue', marker='o',ms=2)
    #plt.show() 
    return polygon_npf_scaled

def ShelfPolygon(ax, scale_factor=1.0, print_polygon=False, color='brown'):
    """
    Cria um polígono reduzido/ampliado da região North of Polar Front, mantendo o mesmo centro.

    scale_factor: Fator para reduzir/ampliar o tamanho do polígono (1.0 mantém o tamanho original).
    print_polygon: Se True, plota o polígono para visualização.
    """
    orsi = le_orsi_swatl()

    lon_i_swAtl,lon_f_swAtl = -70 , -17
    lat_i_swAtl, lat_f_swAtl = -65, -20
    #------------------------------- NORTH OF POLAR FRONT
    #THIS ONE IS DIFFERENT


    Lon_poly_shelf = np.concatenate( ( 
                                orsi['saf']['Longitude'][215:270].values, 
                                orsi['pf']['Longitude'][120::-1].values,
                                [-70],[-65],[-61],
                                [orsi['stf']['Longitude'].values[10]],
                                [orsi['saf']['Longitude'][215]],
                                #[orsi['pf']['Longitude'][0]]
                               )) 

    Lat_poly_shelf = np.concatenate( ( 
                                orsi['saf']['Latitude'][215:270].values ,
                                orsi['pf']['Latitude'][120::-1].values ,
                                [-52],[-45],[-40],
                                [orsi['stf']['Latitude'].values[10]],
                                [orsi['saf']['Latitude'][215]],
                                #[orsi['pf']['Latitude'][0]] ,
                               ))

    # Calcula o centro do polígono
    center_lon = -62#np.mean(Lon_poly_npf)
    center_lat = -48#np.mean(Lat_poly_npf)
    ##########OBS: o calculo da latlon central nao funcionou assim. Dessa forma, -57x-48 foi escolhido visualmente

    # Aplica redução/ampliação
    Lon_poly_shelf_scaled = center_lon + scale_factor * (Lon_poly_shelf - center_lon)
    Lat_poly_shelf_scaled = center_lat + scale_factor * (Lat_poly_shelf - center_lat)

    polygon_shelf_scaled = np.vstack((Lon_poly_shelf_scaled, Lat_poly_shelf_scaled)).T
    
    if print_polygon:
        ax.plot(Lon_poly_shelf_scaled, Lat_poly_shelf_scaled, color='grey', lw=3.5) 
        ax.plot(Lon_poly_shelf_scaled, Lat_poly_shelf_scaled, color = color)
        
    return polygon_shelf_scaled
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


def wrapTo360(angles):
    '''
    # Exemplo de uso:
    #angles = np.array([-45, 0, 45, 360, 720, -360, -720])
    #wrapped_angles = wrapTo360(angles)
    #print(wrapped_angles)
    '''
    return angles % 360


def wrapTo180(angles):
    '''
    # Exemplo de uso:
    angles = np.array([-190, 0, 45, 180, 190, 360, 720, -360, -720])
    wrapped_angles = wrapTo180(angles)
    print(wrapped_angles)
    '''
    wrapped_angles = angles % 360  # Primeiro, aplique o módulo 360
    wrapped_angles = np.where(wrapped_angles > 180, wrapped_angles - 360, wrapped_angles)
    return wrapped_angles


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def fill_spatial_gaps(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. 
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
        
    How to use:
    xx = (np.full(dict['v_1000hPa'].shape, np.nan)) #dict['v_1000hPa'] IS (TIME,LAT,LON)
    
    for dia in range(len(dict['v_1000hPa'])):
        xx[dia] = fill(data = dict['v_1000hPa'][dia], invalid=np.isnan(dict['v_1000hPa'][dia]))
    
    To visualize the gaps:
        np.where(np.isnan(dict[v_1000hPa]) == True)    
        

    """    
    import numpy as np
    from scipy import ndimage as nd
    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, 
                                    return_distances=False, 
                                    return_indices=True)
    return data[tuple(ind)]


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def mapinha(ax):
    '''
    How to use:
    import cartopy.crs as ccrs
    projecao = ccrs.PlateCarree() 
    fig, axs = plt.subplots(subplot_kw={'projection': projecao}, figsize=(10, 6), dpi=150)
    ax = rm.mapinha(axs)
    contour_1 = ax.contour(lon1,lat1,qq_ang_all,levels=np.arange(0,360,30),cmap='seismic')
    cbar1 = fig.colorbar(contour_1,ax=axs)     
    
    How to use (2)
    import cartopy.crs as ccrs
    projecao = ccrs.PlateCarree() 
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': projecao}, figsize=(20, 6), dpi=150)
    ax = rm.mapinha(axs[0])
    contour_1 = ax.contourf(dict['lon_atm'], dict['lat_atm'], dict['wnd_1000hPa'][d], cmap='Spectral', levels=50)
    cbar1 = fig.colorbar(contour_1,ax=axs[0])
    cbar1.set_label('lh') 
    cbar1.ax.tick_params(labelsize=12,labelrotation=45)  

    ax = rm.mapinha(axs[1])
    contour_2 = ax.contourf(dict['lon_atm'], dict['lat_atm'], dict['wnd_1000hPa_15d'][d], cmap='Spectral', levels=50)
    ax.set_title("Vento 1000 hPa (15d)", fontsize=12)

    '''
    lon_i_swAtl,lon_f_swAtl = -70 , -17
    lat_i_swAtl, lat_f_swAtl = -65, -20
    import cartopy.crs as ccrs
    projecao = ccrs.PlateCarree() 

    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.feature as cfeature
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.BORDERS)
    
    # Configurando ticks e limites do eixo
    xticks = range(int(lon_i_swAtl), int(lon_f_swAtl), 5)
    yticks = range(int(lat_i_swAtl), int(lat_f_swAtl), 5)
    
    ax.set_xticks(xticks, crs=projecao)
    ax.set_yticks(yticks, crs=projecao)
    ax.set_xticklabels(xticks, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(yticks, rotation=45, ha='right', fontsize=8)
    ax.set_xlim([int(lon_i_swAtl), int(lon_f_swAtl)])
    ax.set_ylim([int(lat_i_swAtl), int(lat_f_swAtl)])
    return ax

def mapinha_global(ax):
    '''
    How to use:
    import cartopy.crs as ccrs
    projecao = ccrs.PlateCarree() 
    fig, axs = plt.subplots(subplot_kw={'projection': projecao}, figsize=(10, 6), dpi=150)
    ax = rm.mapinha(axs)
    contour_1 = ax.contour(lon1,lat1,qq_ang_all,levels=np.arange(0,360,30),cmap='seismic')
    cbar1 = fig.colorbar(contour_1,ax=axs)     
    
    How to use (2)
    import cartopy.crs as ccrs
    projecao = ccrs.PlateCarree() 
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': projecao}, figsize=(20, 6), dpi=150)
    ax = rm.mapinha(axs[0])
    contour_1 = ax.contourf(dict['lon_atm'], dict['lat_atm'], dict['wnd_1000hPa'][d], cmap='Spectral', levels=50)
    cbar1 = fig.colorbar(contour_1,ax=axs[0])
    cbar1.set_label('lh') 
    cbar1.ax.tick_params(labelsize=12,labelrotation=45)  

    ax = rm.mapinha(axs[1])
    contour_2 = ax.contourf(dict['lon_atm'], dict['lat_atm'], dict['wnd_1000hPa_15d'][d], cmap='Spectral', levels=50)
    ax.set_title("Vento 1000 hPa (15d)", fontsize=12)

    '''
    #lon_i_swAtl,lon_f_swAtl = -70 , -17
    #lat_i_swAtl, lat_f_swAtl = -65, -20
    import cartopy.crs as ccrs
    projecao = ccrs.PlateCarree() 

    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.feature as cfeature
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.BORDERS)
    
    # Configurando ticks e limites do eixo
    #xticks = range(int(lon_i_swAtl), int(lon_f_swAtl), 5)
    #yticks = range(int(lat_i_swAtl), int(lat_f_swAtl), 5)
    
    #ax.set_xticks(xticks, crs=projecao)
    #ax.set_yticks(yticks, crs=projecao)
    #ax.set_xticklabels(xticks, rotation=45, ha='right', fontsize=8)
    #ax.set_yticklabels(yticks, rotation=45, ha='right', fontsize=8)
    #ax.set_xlim([int(lon_i_swAtl), int(lon_f_swAtl)])
    #ax.set_ylim([int(lat_i_swAtl), int(lat_f_swAtl)])
    return ax

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def calcula_pdf(values_xaxis, numero_de_bins, range_xaxis = None):
    '''
    A PDF é feita normalizando o histograma de acordo com o número total 
    de observações, N, e a largura dos bins, bin_width (esse parametro vc escolhe oq deseja ao ver o hist).
    bin_width = dx !!!
    Essa equação basicamente ajusta as alturas dos bins de 
    modo que a soma das áreas de todos os bins seja 1.

    A equação para a PDF em cada bin é:

    PDF(bin) = (Nº de ocorrencias no bin) / (N ​ bin_width)

    ​ Nº de ocorrências no bin: Número de dados dentro do intervalo correspondente ao bin.
    ​ N: O número total de observações (soma das ocorrências em todos os bins).
    ​ bin_width: A largura de cada bin (diferença entre os limites superior e inferior do bin).
    O resultado dessa equação para cada bin te dá a altura da curva de PDF naquele intervalo.
    
    OBS: a equacao da PDF gaussiana nao é essa. essa é uma normalizacao para ver, empiricamente,
        o tipo de PDF. Se ela for gaussiana, utilizar a equacao da PDF continua gaussiana pode trazer
        vantagens e maior precisao. Mas e necessario verificar empiricamente antes com essa funcao aqui!
    
    ---
    Args:
        values_xaxis: 
            list with sorted values! use values_xaxis.sort()
            
        numero_de_bins: ATENCAO
            Quantidade de barras! Se aumentar, aumentará a resolucao.
            
        range_xaxis: optional, tuple. ATENCAO
            definir visualizando. Se for None, será (0,numero_de_bins/2). Significa
            que o eixo x irá de 0 até metade do numero de bins (logo, 
            cada bin terá valores num range de 0 a 0.5)
            
    Returns: pdf, ocorrencias_cada_bin, bin_edges, bin_width
        pdf: valores de 0 a 1
        ocorrencias_cada_bin: valores reais de ocorrencias em cada intervalo (bin) da resolucao
                              definida pelo numero de bins
                              
        bin_edges: necessario para o plot (é o proprio eixo x) >>> 0,0.5,1.0,... 7.5,8.0,...
        bin_width: dx, tambem necessario para o plot
    ---
    EXAMPLE OF USE AND PLOT:
    
    pdf, ocorrencias_cada_bin, bin_edges, bin_width = calcula_pdf(values_xaxis = list_duration,
                                                                  numero_de_bins = 16,range_xaxis=(0,8))
    # Criar a figura e o primeiro eixo (para a PDF)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(bin_edges[:-1], ocorrencias_cada_bin, width = bin_width, color='grey', edgecolor='white', align='edge', label='Occurrences')
    ax1.set_xlabel('Days in the region', fontsize=16)
    ax1.set_ylabel('Cyclone occurrences', fontsize=16, color='grey')
    ax1.tick_params(axis='y', labelcolor='grey')
    
    # Criar o segundo eixo (para a PDF) e plotar a curva da PDF
    ax2 = ax1.twinx()
    ax2.plot(bin_edges[:-1], pdf, color='blue', lw=5, label='PDF')
    ax2.set_ylabel('PDF', fontsize=16, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Ajustar os ticks e exibir o gráfico
    plt.xticks(np.linspace(0, 8, 17), fontsize=16)
    plt.show()
    
    ---
    TO COMPARE WITH THE GAUSSIAN PDF:
    pdf, ocorrencias_cada_bin, bin_edges, bin_width = calcula_pdf(values_xaxis = list_duration,
                                                              numero_de_bins = 16,range_xaxis=(0,8))
    # Import necessary libraries
    from scipy.stats import norm

    # Média e desvio padrão
    mean = (np.nanmean(values_xaxis))
    std_dev = (np.nanstd(values_xaxis))

    # Valores de 'x' para a PDF gaussiana
    x_values = np.linspace(0, np.nanmax(values_xaxis), 100)  # De 0 a 8 dias

    # PDF da normal gaussiana com a média e desvio padrão dados
    pdf_gauss = norm.pdf(x_values, mean, std_dev)

    # Agora plote o histograma e a PDF normal sobreposta
    plt.figure(figsize=(12, 4))

    plt.plot(bin_edges[:-1], pdf, color='blue', lw=5, label='PDF')

    # PDF gaussiana sobreposta
    plt.plot(x_values, pdf_gauss, 'r-', lw=2, label='PDF Gaussiana')

    plt.xlabel('Days in the region', fontsize=16)
    plt.ylabel('PDF', fontsize=16)
    plt.legend(fontsize=14)
    plt.show()
    
    
    ------------MODIFICACAO 18/01 --> INCLUINDO O soma_pdf
    Returns:
        pdf: PDF normalizada.
        ocorrencias_cada_bin: número de ocorrências em cada bin.
        bin_edges: limites dos bins.
        bin_width: largura de cada bin.
        soma_pdf: verificação da normalização da PDF (deve ser 1).
    '''
    if range_xaxis is None:
        range_xaxis = (0, np.nanmax(values_xaxis))
    
    # Total de observações
    N = len(values_xaxis)
    
    # Histograma
    ocorrencias_cada_bin, bin_edges = np.histogram(
        values_xaxis, bins=numero_de_bins, range=range_xaxis
    )
    bin_width = bin_edges[1] - bin_edges[0]
    
    # PDF
    pdf = ocorrencias_cada_bin / (N * bin_width)
    
    # Verificação de normalização
    soma_pdf = np.sum(pdf * bin_width)
    print(f"Soma da PDF (verificação de normalização): {soma_pdf}, deve ser igual a 1. Se nao, é porque alguns dados em values_xaxis caem fora do intervalo range_xaxis fornecido (eles não serão considerados no cálculo). Isso reduziria a soma da PDF.")
    
    return pdf, ocorrencias_cada_bin, bin_edges, bin_width


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def interpolate_to_new_len(data,new_len,interp_kind):
    '''
    # Função para interpolar um array para um comprimento fixo

    data: array
        Your data
    new_len: int
        The desired new len for your data
    
    '''
    import numpy as np
    from scipy.interpolate import interp1d

    x_original = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, new_len)
    interpolation_function = interp1d(x_original, data, kind=interp_kind)
    
    return interpolation_function(x_new),x_new



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def format_func(value, tick_number):
    '''

    How to use:
    from matplotlib.ticker import FuncFormatter

    theta_interpolados = np.linspace(0, 2*np.pi, 36)  # 360 ângulos
    dd(theta_interpolados)
    plt.plot(theta_interpolados, np.sin(theta_interpolados),marker='o',label='len = '+str(len(theta_interpolados))); plt.legend()
    plt.gca().set_xticks(np.linspace(0, 2*np.pi, 5))  # Ajustando a localização dos ticks
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

    '''
    # Função para formatar os valores em múltiplos de pi
    N = (np.round(value / np.pi,1))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi$"
    elif N == -1:
        return r"-$\pi$"
    else:
        return r"${0}\pi$".format(N)
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def cart2pol(x,y):
    r = np.sqrt( (x**2) + (y**2) )
    phi = np.arctan2(y,x)
    return r,phi


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


# Função para calcular os índices de um mês em um único ano
def get_month_indices_single_year(month, year, freq_per_day):
    """
    Retorna os índices para o início e fim de um mês em um único ano, considerando anos bissextos.
    
    Parâmetros:
        - month: Nome (string) ou número (1-12) do mês.
        - year: Ano para verificar se é bissexto.
        - freq_per_day: Frequência temporal (número de registros por dia, padrão: 8 para dados a cada 3 horas).
        
    Retorna:
        - start_idx, end_idx: Índices iniciais e finais do mês no array para um único ano.
    ---
    A lógica para anos bissextos foi adicionada: um ano é bissexto se:
    É divisível por 4 e não por 100, ou
    É divisível por 400.
    ---
    
    ## Exemplo de uso
    month = "September"  # ou número do mês (1-12)
    year = 2016  # Substitua pelo ano desejado
    dini, dend = get_month_indices(month, year)

    # Plot ajustado
    fig, axs = plt.subplots(subplot_kw={'projection': projecao}, figsize=(10, 6), dpi=150)
    ax = rm.mapinha(axs)
    contour_1 = ax.contourf(dict['lon_ocn'], dict['lat_ocn'],
                            np.nanmean(dict['mld'][dini:dend], axis=0),
                            levels=np.arange(0, 300, 5),
                            cmap='turbo', extend='max')
    cbar1 = fig.colorbar(contour_1, ax=axs)
    rm.plota_orsi_swatl(ax)
    
    """
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    days_per_month = [31, 29 if is_leap_year else 28, 31, 30, 31, 30,
                      31, 31, 30, 31, 30, 31]
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    if isinstance(month, str):
        month = month_names.index(month.capitalize()) + 1
        #print('month:',month)

    start_day = sum(days_per_month[:month-1])
    #print('days_per_month[:month-1]:',days_per_month[:month-1])
    #print('sum(days_per_month[:month-1]):',sum(days_per_month[:month-1]))
    end_day = start_day + days_per_month[month-1]
    #print('days_per_month[month-1]:',days_per_month[month-1])
    #print('start_day + days_per_month[month-1]:',start_day + days_per_month[month-1])
    start_idx = start_day * freq_per_day
    end_idx = end_day * freq_per_day
    #print('start_idx:',start_idx)
    return start_idx, end_idx

# Função para obter índices para o mês ao longo de vários anos
def get_month_indices_all_years(month, start_year, end_year, freq_per_day):
    """
    Retorna os índices correspondentes a um mês específico ao longo de vários anos.
    
    Parâmetros:
        - month: Nome (string) ou número (1-12) do mês.
        - start_year: Ano inicial.
        - end_year: Ano final.
        - freq_per_day: Frequência temporal (número de registros por dia, padrão: 8 para dados a cada 3 horas).
        
    Retorna:
        - indices: Lista com os índices para o mês em todos os anos.
    """
    list_years = np.arange(start_year, end_year + 1)
    #print(list_years)
    #print('len(list_years):', len(list_years))

    indices = []

    # Inicializa o deslocamento em índices (para acumular os dias de anos anteriores)
    offset = 0

    for year in list_years:
        #print('Current year:', year)

        # Verifica se o ano é bissexto
        days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365

        # Calcula os índices ajustados pelo deslocamento
        start_idx, end_idx = get_month_indices_single_year(month, year, freq_per_day)
        indices.append(slice(start_idx + offset, end_idx + offset))  # Ajusta com o deslocamento atual

        #print(f'Year: {year}, Days in Year: {days_in_year}, Start: {start_idx + offset}, End: {end_idx + offset}')
        #print('indices:', indices)
        #print('-----------------------------------')

        # Atualiza o deslocamento para o próximo ano
        offset += days_in_year * freq_per_day

    #print('Final indices:', indices)
    return indices



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------