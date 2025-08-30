import matplotlib.pyplot as plt
FS = 12

lon_i_atl,lon_f_atl = -75, 20   
lon_i_atl_menor = -69
lat_i_atl, lat_f_atl = -90, -5    


#lon_i_ame,lon_f_ame = lon_i_arg, lon_f_br   

#lat_i_ame, lat_f_ame = lat_i_arg, lat_f_plata    


lon_i_br, lon_f_br = -52, -39

lat_i_br, lat_f_br = -38, -20#-22



lon_i_plata, lon_f_plata = -85, -52

lat_i_plata, lat_f_plata = -38, -20#-22



lon_i_arg,lon_f_arg = -85 , -50

lat_i_arg,lat_f_arg = -55 , -38



lon_i_drake,lon_f_drake = -85 , -45
 
lat_i_drake, lat_f_drake = -60, -55



lon_i_ant,lon_f_ant = -85 , -45#-75 , -45
 
lat_i_ant, lat_f_ant = -68, -60


lon_i_paci,lon_f_paci = -170 , -85#-75 , -45
 
lat_i_paci, lat_f_paci = -75, -10

lat_lim_antartica = -60



lat_i_stf, lat_f_stf = -55, -35


lon_i_africa, lon_f_africa = 0, 25

lat_i_africa, lat_f_africa = -40, -20#-22


lon_i_reboita2009, lon_f_reboita2009 = -80, 10
lat_i_reboita2009, lat_f_reboita2009 = -55, -15#-22

lon_i_swAtl,lon_f_swAtl = -70 , -17
 
lat_i_swAtl, lat_f_swAtl = -65, -20

#70-17°W, 65-20°S

from matplotlib.patches import Polygon

def plota_poligonos(escolhidos,ax,alpha=1,zorder=3):
    # Definir as coordenadas dos polígonos
    poligonos = {
        'SE-BR': [[lon_i_br, lat_i_br], [lon_f_br, lat_i_br], [lon_f_br, lat_f_br], [lon_i_br, lat_f_br]],
        'PLATA': [[lon_i_plata, lat_i_plata], [lon_f_plata, lat_i_plata], [lon_f_plata, lat_f_plata], [lon_i_plata, lat_f_plata]],
        'ARG': [[lon_i_arg, lat_i_arg], [lon_f_arg, lat_i_arg], [lon_f_arg, lat_f_arg], [lon_i_arg, lat_f_arg]],
        'DRAKE': [[lon_i_drake, lat_i_drake], [lon_f_drake, lat_i_drake], [lon_f_drake, lat_f_drake], [lon_i_drake, lat_f_drake]],
        'ANT': [[lon_i_ant, lat_i_ant], [lon_f_ant, lat_i_ant], [lon_f_ant, lat_f_ant], [lon_i_ant, lat_f_ant]],
        'AFR': [[lon_i_africa, lat_i_africa], [lon_f_africa, lat_i_africa], [lon_f_africa, lat_f_africa], [lon_i_africa, lat_f_africa]],
        'SWATL': [[lon_i_swAtl, lat_i_swAtl], [lon_f_swAtl, lat_i_swAtl], [lon_f_swAtl, lat_f_swAtl], [lon_i_swAtl, lat_f_swAtl]],
    }

    # Definir as cores dos polígonos
    cores = {
        'SE-BR': 'grey',
        'PLATA': 'grey',
        'ARG': 'grey',
        'DRAKE': 'grey',
        'ANT': 'grey',
        'AFR': 'grey',
        'SWATL': 'red',
    }

    # Criar e adicionar os polígonos ao gráfico com anotações
    for nome, coords in poligonos.items():
        if nome in escolhidos:
            poly = Polygon(coords, closed=True, edgecolor=cores[nome], facecolor='none', linewidth=7,alpha=alpha,zorder=zorder)
            ax.add_patch(poly)

            # Adicionar as anotações aos polígonos
            x_center = coords[0][0]+5
            y_center = coords[3][1]-0.2
            ax.annotate(nome.split(' ')[-1], (x_center, y_center), color=cores[nome], fontsize=FS-4.5, ha='center', va='center',fontweight='bold')
            #ax.annotate(nome.split(' ')[-1], (x_center, y_center), color=cores[nome], fontsize=FS-6, ha='center', va='center')
