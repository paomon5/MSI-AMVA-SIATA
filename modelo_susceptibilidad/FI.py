### LIBRERÍAS ###
import modelo_susceptibilidad
from wmf import wmf
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd

import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cm import get_cmap
import matplotlib.colors
from skimage import io
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature

from shapely.geometry import Point
from shapely.wkb import loads
try:
    from osgeo import ogr
except:
    print("no se pudo importar osgeo")
import fiona
import json
import sys

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph, Table, TableStyle
from IPython.display import IFrame
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
import sys
import os
from matplotlib.patches import Rectangle


from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

#### RUTAS DE OBJETOS VARIOS ####
path = "/home/hidrologia/jupyter/ModeloSusceptibilidadIncendios/AMVA/"
path_ave = path+'reportes/reporte_horario/Tools/'
cu = wmf.SimuBasin(rute=path_ave+'cuenca_amva_incendios_90m_py3.nc',) #Cargar la cuenca
path_fuente = path_ave+'AvenirLTStd-Book.otf'
path_historial = '/var/datos_hidrologia/MSI/LocalModel/Resultados_Modelo/'
path_zonas_pronostico = path_ave+'ZonasPronostico.tif'
path_zona_urbana = path_ave+'urbano_90m.csv'
path_shape_amva = path_ave+'AreaMetropolitana.shp'
path_icono_incendios = path_ave+'icono1.png'
path_shape_antioquia = path_ave+'Antioquia.shp'
path_departamentos = path_ave+'Departamentos.shp'
path_registro = path+'reportes/registro_incendios/registro.csv'
save_path = "/var/datos_hidrologia/MSI/LocalModel/Imagenes/"
path_plantilla = path + 'reportes/reporte_horario/'

model = modelo_susceptibilidad.msi()
rep=pd.read_csv(path_registro)
rep.date = pd.to_datetime(rep.date)

############# FUNCIONES ####################
## Ubicar un punto en un shape ##
#def searchPointInPolygon(latitud, longitud, shapefile):
#    try:
#        shapeData = ogr.Open(shapefile)
#    except:
#        shapeData = fiona.Open(shapefile)
#    layer     = shapeData.GetLayer(0)
#    polygon   = layer.GetNextFeature()
#    point1    = Point(longitud, latitud)
#    ubicacion = []
#    while polygon:
#        geomPolygon = loads(polygon.GetGeometryRef().ExportToWkb())
#        if geomPolygon.contains(point1):
#            ubicacion.append(polygon.ExportToJson())
#        polygon = layer.GetNextFeature()
#    return ubicacion
def searchPointInPolygon(latitud, longitud, shapefile):
    try:
        shapeData = ogr.Open(shapefile)
    except:
        shapeData = fiona.Open(shapefile)
    layer     = shapeData.GetLayer(0)
    polygon   = layer.GetNextFeature()
    point1 = ogr.Geometry(ogr.wkbPoint)
    point1.AddPoint(longitud,latitud)
    ubicacion = []
    for in_feat in layer:
        geomPolygon = in_feat.geometry()
        if geomPolygon.Contains(point1):
            ubicacion.append(in_feat.ExportToJson())
        #polygon = layer.GetNextFeature()
    return ubicacion

## Obtiene las salidas del modelo en las fechas especificadas ##
def read_susc(path, start, end):
    matriz = pd.DataFrame()
    if end.hour < 10:
        hora_actual = "0" + str(end.hour)
    else:
        hora_actual = str(end.hour)
    end = pd.to_datetime(end)
    print("soy end", end)
    delta = end-start
    ## Se generan los días que se van a consultar 
    fechas = [datetime.datetime.strftime((end - datetime.timedelta(days = n)), "%Y-%m-%d") for n in range(delta.days+1)]
    fechas.sort()
    print(fechas)
    for dia in fechas:
        print(path_historial + dia +".csv")
        try:
            ## Lee el resultado de cada día a la hora del reporte actual
            tabla = pd.read_csv(path_historial + dia +".csv", usecols=[str(hora_actual)])#, )
            matriz = pd.concat([matriz, tabla], axis=1)
            print("Leí el día {}".format(dia))
        except:
            tabla = pd.DataFrame(np.zeros(79778)*np.nan)
            matriz = pd.concat([matriz, tabla], axis=1)
            print("fallé en el día{}".format(dia))
            pass
    ## Matríz sería cada uno de los mapas de los días anteriores a la hora actual 
    dias = matriz.shape[1]  
    print(dias)
    matriz.columns = fechas[-dias:]
    vec = list(matriz.values.T)
    return vec

def resumen(start,end , end_14, hour=None, duplas=None):

    end = pd.to_datetime(end)
    start = pd.to_datetime(start)
    if hour == None:
        hour = end.hour
    if hour <14:
        vec = read_susc(path_historial, start, end_14)
        vechoy = read_susc(path_historial, start, end)
        vechoy = vechoy[-1]
        vec[-1] =vechoy
    else:
        vec = read_susc(path_historial, start, end)
    path_tif = path_zonas_pronostico
    mapa_zonas, prop, epsg = wmf.read_map_raster(path_tif) #Leer mapa de regiones
    zonas_basin = cu.Transform_Map2Basin(mapa_zonas, prop) #Escribir el mapa en forma de cuenca
    zonas_basin[zonas_basin < 0] = 0
    model = modelo_susceptibilidad.msi()
    mapa_suscep_agre = np.zeros_like(vec)
    week = np.zeros([len(vec),12])
    mascara=pd.read_csv(path_zona_urbana,header=None, dtype=int)[0].values

    for j in range(len(vec)):
        for i in range(1,13):
            vec[j][mascara] = 0
            pp = np.where(np.array(zonas_basin) == i)[0]
            nula= np.where(vec[0]<0.08)[0]
            hist=np.histogram(vec[j][list(set(pp)-set(mascara))],[0,0.22,0.42,0.62,0.87])  # Anterior
            hist=np.histogram(vec[j][list(set(pp)-set(mascara))],[0,0.10,0.22,0.38,0.95])  # Actual
            hist=np.histogram(vec[j][list(set(pp)-set(mascara))],[0,0.10,0.30,0.38,0.95])  # Actual Pedido por Neider porque Aleja 8A dijo que estaba muy amarillo el mapa
            hist=np.histogram(vec[j][list(set(pp)-set(mascara))],[0,0.10,0.30,0.44,0.95])  # Actual Pedido por Neider porque Aleja 8A dijo que estaba muy rojo el mapa el 3 de julio del 2021
            hist=np.histogram(vec[j][list(set(pp)-set(mascara))],[0,0.10,0.38,0.55,0.95])  # Actual modificado porque estaba muy azul el 22 de Julio del 2021
            #hist=np.histogram(vec[j][list(set(pp)-set(mascara))],[0,0.25,0.5,0.75,0.95])  # Ideal
            posmoda=np.where(hist[0]==max(hist[0]))[0]
            moda=[0.14,0.34,0.54,0.73][posmoda[0]]#hist[1][posmoda]  #Anterior
            moda=[0.05,0.14,0.3,0.73][posmoda[0]]#hist[1][posmoda]  #Actual
            moda=[0.05,0.16,0.45,0.73][posmoda[0]] if hist[0].sum() != 0 else np.nan  #Actual
            #moda=[0.15,0.34,0.65,0.83][posmoda[0]]#hist[1][posmoda]  #Ideal
            #mapa_suscep_agre[j][pp] = i
            week[j,i-1] = moda
    r = pd.date_range(start, end, freq = 'd')
    day_parser = dict(zip([u'Monday',u'Tuesday', u'Wednesday', u'Thursday', u'Friday', u'Saturday', u'Sunday'],['L', 'M', 'M', 'J', 'V', 'S', 'D']))
    days = [str(day_parser[i.day_name()]) + i.strftime('%d') for i in r]
    fig = plt.figure(figsize=(19,10))
    ax =  fig.add_subplot(111)
    rvb =model.colores_modelo
    zonas = ['Caldas','Sabaneta','La Estrella','Envigado','Itagüí','Medellín Occidente','Medellín Oriente','Medellín Centro','Bello','Copacabana','Girardota','Barbosa']
    moda_susc = pd.DataFrame(week.T, columns = days, index =zonas)
    #ax.imshow(week.T,cmap=rvb, extent=(0,week.shape[0],0 ,week.shape[1]),vmax=0.87, vmin = 0)
    semana=np.copy(week.T)
    semana[semana==0.05] = 1
    semana[semana==0.16] = 2
    semana[semana==0.45] = 3
    semana[semana==0.73] = 4
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#62D2DD","#3F7DAA","#EDE214", "#FF0000"])
    #ax.imshow(moda_susc,cmap=rvb, extent=(0,week.shape[0],0 ,week.shape[1]))#,vmax=0.87, vmin = 0)
    ax.imshow(semana,cmap=cmap, extent=(0,week.shape[0],0 ,week.shape[1]),vmax=4, vmin = 1)
    ax.grid(ls='-',color='k',lw=1, zorder=3)
    ax.set_xticks(range(week.shape[0]))
    ax.set_yticks(np.array(range(week.shape[1])))
    ax.set_yticklabels([])
    ax.set_xticklabels(days,ha="left", fontsize = 17)
    ax.set_ylim(0,week.shape[1])
    ax.set_xlabel('Día de la semana', fontsize = 20)
    ax.set_title('Susceptibilidad diaria a\nincendios forestales', fontsize=24)
    ax2=fig.add_axes([0.4,.11,.225,.77] )
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    rect = ax2.patch 
    rect.set_facecolor('white') 
    rect.set_alpha(0)
    ax2.set_xticks([])
    ax2.set_yticks(np.arange(week.shape[1]))
    ax2.set_yticklabels(zonas,ha="right", fontsize = 17)
    ax2.tick_params(color= "#FF000000", pad=10)
    
    icono = (io.imread(path_icono_incendios))
    if len(duplas) >=1:
        fechas_ocurrencia = [i for i in rep[(rep.date<end)&(rep.date>start)].loc[:,'fecha']]
        df_coordenada=pd.DataFrame(duplas, columns= ["longitude", "latitude"])
        df_coordenada["zona"]        = ""

        for index, row in df_coordenada.iterrows():        
            shapefile_zonas = path_ave + "ZonasPronostico.shp"
            puntos    = []
            poligono  = []
            try:
                latitud  = row["latitude"]
                longitud = row["longitude"]
                shp_zonas = searchPointInPolygon(latitud, longitud, shapefile_zonas)
            except Exception as e:
                print (e)
                
            zonas = []
            poly_zonas = []        
        #
            if len(shp_zonas) >= 1:
                for u in shp_zonas:
                    u = json.loads(u)
                    zonas.append(u["properties"]["Zona"])
                    poly_zonas.append(u["geometry"]["coordinates"])
        
            df_coordenada.loc[df_coordenada.index == index, "zona"]        = "--".join(zonas)
        
        df_coordenada.index = pd.to_datetime(fechas_ocurrencia)
        df_coordenada["fin"] = [str(day_parser[i.day_name()]) + i.strftime('%d') for i in df_coordenada.index]
        r = pd.date_range(start, end, freq = 'd')
        day_parser = dict(zip([u'Monday',u'Tuesday', u'Wednesday', u'Thursday', u'Friday', u'Saturday', u'Sunday'],['L', 'M', 'M', 'J', 'V', 'S', 'D']))
        zonas = ['Caldas','Sabaneta','La Estrella','Envigado','Itagüí','Medellín Occidente','Medellín Oriente','Medellín Centro','Bello','Copacabana','Girardota','Barbosa']
        days = [str(day_parser[i.day_name()]) + i.strftime('%d') for i in r]
        dd, zz =np.meshgrid(days,zonas)
        days_r, zonas_r = dd.ravel(), zz.ravel()
        ocurrencia = np.zeros(len(days_r))
        
        for z,d in zip(zonas_r, days_r):
            if z in df_coordenada["zona"].values:
                for j in range(len(df_coordenada)):
                    if z == df_coordenada["zona"][j] and d == df_coordenada["fin"][j]:
                        ocurrencia[np.where((zonas_r == df_coordenada["zona"][j]) & (days_r == df_coordenada["fin"][j]))[0]] = 50
        ax2.scatter(days_r, zonas_r, s=ocurrencia, c="purple", marker="X")
        ax2.set_yticks(np.arange(week.shape[1]))
        ax2.set_yticklabels(zonas,ha="right", fontsize = 17)
        for x0, y0 in zip(days_r[ocurrencia>1], zonas_r[ocurrencia>1]):
            im = OffsetImage(icono, zoom=0.7)
            ab = AnnotationBbox(im, (x0,y0), frameon=False)
            ax2.add_artist(ab)
        
    plt.savefig(save_path + 'Resumen_semana_%s_%s.png'%(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')),bbox_inches='tight')
    return week


def Plot_Maximo_Mes(mapa_final,fecha,duplas=None,n_report='', fil=None,col=None, 
                    pos=None, umbrales = [0,0.10,0.47,0.56,0.95]):
    import matplotlib as mpl
    start = pd.to_datetime(fecha[7:17])
    end = pd.to_datetime(fecha[20:30])
    figsize = 11
    fontsize = 15
    text=fecha
    fig = plt.figure(figsize=(figsize,figsize*(6.515-5.975)/(75.725-75.21)),facecolor='w',edgecolor='w')

    if fil!=None:
        gs = gridspec.GridSpec(fil,col)
        ax=fig.add_subplot(gs[pos[0],pos[1]],projection=ccrs.PlateCarree())
    else:
        gs = gridspec.GridSpec(1,1)
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())

    ax.set_extent([-75.21, -75.725, 5.975, 6.555])
    ax.add_feature(cartopy.feature.BORDERS)
    
    cmap1 = plt.matplotlib.colors.ListedColormap(['darkslategray'])
    mascara=pd.read_csv(path_zona_urbana,header=None, dtype=int)[0].values
    mask = mapa_final *np.nan
    mask[mascara] = 1
    mask, prop = cu.Transform_Basin2Map(mask)
     #[0,0.22,0.42,0.62,87]
    #umbs = [0,0.1,0.22,0.38,0.95]
    #umbrales = [0,0.10,0.3,0.44,0.95]
    #umbrales = [0,0.10,0.20,0.32,0.95]
    #umbrales = [0,0.10,0.33,0.50,0.95]
    #umbrales = [0,0.10,0.47,0.56,0.95]
    reclass = np.copy(mapa_final)
    reclass[np.where((mapa_final >= umbrales[0]) & (mapa_final<umbrales[1]))[0]] = 1
    reclass[np.where((mapa_final >= umbrales[1]) & (mapa_final<umbrales[2]))[0]] = 2
    reclass[np.where((mapa_final >= umbrales[2]) & (mapa_final<umbrales[3]))[0]] = 3
    reclass[np.where(mapa_final >= umbrales[3])[0]] = 4
    Map, prop = cu.Transform_Basin2Map(reclass)
    Map[(Map==-9999)]=np.nan
    mask[(mask==-9999)|(mask==0)]=np.nan
    longitudes=np.array([prop[2] + 0.5*float(prop[4]) + float(prop[4])*i for i in range(prop[0])])
    latitudes=np.array([prop[3]+0.5*prop[-2]+i*float(prop[-2]) for i in range(prop[1])])
    x,y = np.meshgrid(longitudes,latitudes)
    rvb =model.colores_modelo
    rvb.set_bad('w')
    
    cs = ax.contourf(x,y,Map.T[::-1], cmap = rvb ,zorder = 3,vmax=4, vmin = 1)#,levels=[0,0.22,0.42,0.62,0.87],vmax=0.87, vmin = 0,)
    cs1 = ax.contourf(x,y,mask.T[::-1],cmap=cmap1 , zorder =3 )
    #Agregamos el grid
    gl= ax.gridlines(color="black",linestyle="-.", linewidth = 0.2)
    gl.xlabels_bottom=True
    gl.ylabels_left=True
    gl.xlines=True
    gl.ylines=True
    gl.xlabel_style = {'size': 15} #Tamaño etiquetas eje X 
    gl.ylabel_style = {'size': 15} #Tamaño etiquetas eje Y
    ax.set_frame_on(False)
    
    # Colorbar
    cmap = plt.get_cmap(model.colores_modelo,4)
    ax4= fig.add_axes((0.17,0.83,0.33,0.032), zorder=5)
    ticks= ['Nula','Baja','Media','Alta']
    cbar = mpl.colorbar.ColorbarBase(ax4, cmap=cmap, spacing='proportional', orientation='horizontal')
    cbar.set_ticks([0.125,0.375,0.625,0.875])
    cbar.set_ticklabels(ticks)
    cbar.set_label("susceptibilidad a incendios",fontsize=fontsize,color=(0.45, 0.45, 0.45))
    cbar.ax.tick_params(labelsize=fontsize,colors=(0.45, 0.45, 0.45))
    geo_reg = shpreader.Reader(path_shape_amva)
    shapefile = list(shpreader.Reader(path_shape_amva).geometries())
    ax.add_geometries(shapefile, ccrs.PlateCarree(),edgecolor='gray',facecolor='lightslategrey', alpha=0.3, linewidth=1)

    # Ubicar incendio si ocurrio (XY en los argumentos)
    icono = (io.imread(path_icono_incendios))
    n_report = 0
    if duplas != None:
        for xx in range(len(duplas)):
            Xcc, Ycc = duplas[xx][0],duplas[xx][1]
            n_report+=1
            im = OffsetImage(icono, zoom=0.6)
            ab = AnnotationBbox(im, (Xcc,Ycc), frameon=False)
            ax.add_artist(ab)

    plt.text(1.55,-14,'Incendios y columnas\nde humo reportadas\n%s\nTotal:%s'%(text.split('Semana')[1],n_report), ha='center',va='center', fontsize=18,color='k',zorder=4) 
    im = OffsetImage(icono, zoom=1.5)
    ac = AnnotationBbox(im, (-75.35,6.25), frameon=False)
    ax.add_artist(ac)

    ## Colombia
    ax2=fig.add_axes([0.43,.15,.3,.17],projection=ccrs.PlateCarree())
    ax2.set_extent([-65, -81.8, -5., 13])
    ax2.coastlines()
    ax2.add_feature(cartopy.feature.OCEAN)
    ax2.add_feature(cartopy.feature.BORDERS)
    shapefile = list(shpreader.Reader(path_shape_antioquia).geometries())
    ax2.add_geometries(shapefile, ccrs.PlateCarree(),edgecolor='gray',facecolor="green", alpha=0.3, linewidth=1)
    
    ## Antioquia
    ax3=fig.add_axes([0.63,.15,.3,.17],projection=ccrs.PlateCarree())
    ax3.set_extent([-73.76, -77.18, 5.35, 8.9])
    ax3.coastlines()
    ax3.add_feature(cartopy.feature.OCEAN)
    ax3.add_feature(cartopy.feature.BORDERS)
    shapefile = list(shpreader.Reader(path_shape_antioquia).geometries())
    ax3.add_geometries(shapefile, ccrs.PlateCarree(),edgecolor='gray',facecolor='green', alpha=0.4, linewidth=1)
    shapefile = list(shpreader.Reader(path_shape_amva).geometries())
    ax3.add_geometries(shapefile, ccrs.PlateCarree(),edgecolor='gray',facecolor='lightslategrey', alpha=0.4, linewidth=1, zorder=4)
    shapefile = list(shpreader.Reader(path_departamentos).geometries())
    ax3.add_geometries(shapefile, ccrs.PlateCarree(),edgecolor='gray',facecolor='white', alpha=0.3, linewidth=1)

    plt.savefig(save_path + 'Mapa_semana_%s_%s.png'%(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')),dpi=100,bbox_inches='tight')

def Plot_Mapa_ubicacion(mapa_final,fecha,duplas=None,n_report=''):
    start = pd.to_datetime(fecha[7:17])
    end = pd.to_datetime(fecha[20:30])
    figsize = 11
    fontsize = 15
    text= fecha
    fig = plt.figure(figsize=(figsize,figsize),facecolor='w',edgecolor='w')

    gs = gridspec.GridSpec(1,1)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.set_extent([-75.21, -75.725, 5.975, 6.555])
    cmap1 = plt.matplotlib.colors.ListedColormap(['darkslategray'])
    mascara=pd.read_csv(path_zona_urbana,header=None, dtype=int)[0].values
    mask = mapa_final *np.nan
    mask[mascara] = 1
    mask, prop = cu.Transform_Basin2Map(mask)
    mask[(mask==-9999)|(mask==0)]=np.nan
    longitudes=np.array([prop[2] + 0.5*float(prop[4]) + float(prop[4])*i for i in range(prop[0])])
    latitudes=np.array([prop[3]+0.5*prop[-2]+i*float(prop[-2]) for i in range(prop[1])])
    x,y = np.meshgrid(longitudes,latitudes)
    cs1 = ax.contourf(x,y,mask.T[::-1],cmap=cmap1 , zorder =3 )


    gl= ax.gridlines(color="black",linestyle="-.", linewidth = 0.2)
    gl.xlabels_top=True
    gl.ylabels_left=True
    ax.set_frame_on(False)
    
    geo_reg = shpreader.Reader(path_shape_amva)
    shapefile = list(shpreader.Reader(path_shape_amva).geometries())
    ax.add_geometries(shapefile, ccrs.PlateCarree(),edgecolor='gray',facecolor='whitesmoke', alpha=0.3, linewidth=1)
    rectangulo = Rectangle((-75.44,6.1), 0.02, 0.02000, fc = 'darkslategray')
    ax.add_patch(rectangulo)
    ax.text(-75.35,6.109,'Zona urbana', ha='center',va='center', fontsize=22,color='k',zorder=4)
    
    ## Ubicar incendio si ocurrio (XY en los argumentos)
    n_report = 0
    icono = (io.imread(path_icono_incendios))
    if duplas != None:
        for xx in range(len(duplas)):
            Xcc, Ycc = duplas[xx][0],duplas[xx][1]
            n_report+=1
            im = OffsetImage(icono, zoom=0.6)
            ab = AnnotationBbox(im, (Xcc,Ycc), frameon=False)
            ax.add_artist(ab)

    plt.text(-75.35,6.18,'Incendios y columnas\nde humo reportadas\n%s\nTotal:%s'%(text.split('Semana')[1],n_report), ha='center',va='center', fontsize=18,color='k',zorder=40) 
    im = OffsetImage(icono, zoom=1.4)
    ac = AnnotationBbox(im, (-75.35,6.25), frameon=False)
    ax.add_artist(ac)
    plt.savefig(save_path + 'ubicacion_incendios_%s_%s.png'%(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')),dpi=100,bbox_inches='tight')


def report(start, end):
    barcode_font = path_ave+'AvenirLTStd-Book.ttf'
    pdfmetrics.registerFont(TTFont("AvenirBook", barcode_font))

    widthPage =  1157
    heightPage = 955
    pdf = canvas.Canvas(save_path + 'Reporte_incendios5_%s_%s.pdf'%(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')),pagesize=(widthPage,heightPage))

    #pdf.drawImage(path_plantilla + 'PlantillaReporte2.png',0,0,width=widthPage,height=heightPage)
    pdf.drawImage(path_plantilla + 'PlantillaReporte2022.png',0,0,width=widthPage,height=heightPage)
    
    pdf.drawImage(save_path + 'Resumen_semana_%s_%s.png'%(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')),660,-60 ,width=440,preserveAspectRatio=True)
    pdf.drawImage(save_path + 'Mapa_semana_%s_%s.png'%(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')),30,-50 ,width=600,preserveAspectRatio=True) ## antes estaba 30, -60
    print(save_path + 'Mapa_semana_%s_%s.png'%(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')))
    #text_color = '#%02x%02x%02x' % (8,31,45)
    text_color = "#00364d"
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify',\
                              alignment=TA_JUSTIFY,\
                              fontName = "AvenirBook",\
                              fontSize = 30,\
                              textColor = text_color,\
                              leading = 20))
    
    p = Paragraph(end.strftime('Mapa de %Y-%m-%d %H:00:00'), styles["Justify"])
    p.wrapOn(pdf, 720, 200)
    p.drawOn(pdf,438,760)
    
    pdf.showPage()

    pdf.save()
    
    
    
    

def report_bomberos(start, end):   
    
    barcode_font = path_ave+'AvenirLTStd-Book.ttf'
    pdfmetrics.registerFont(TTFont("AvenirBook", barcode_font))

    widthPage =  1157
    heightPage = 850
    pdf = canvas.Canvas(save_path + 'Bomberos/Reporte_incendios_bomberos_%s_%s.pdf'%(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')),pagesize=(widthPage,heightPage))

    #pdf.drawImage(path_plantilla + 'PlantillaReporte2.png',0,0,width=widthPage,height=heightPage)
    pdf.drawImage(path_plantilla + 'Plantilla_bomberos.png',0,0,width=widthPage,height=heightPage)
    
    pdf.drawImage(save_path + 'Resumen_semana_%s_%s.png'%(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')),670,-115 ,width=460,preserveAspectRatio=True)
    pdf.drawImage(save_path + 'Mapa_semana_%s_%s.png'%(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')),20,-95 ,width=670,preserveAspectRatio=True)
    print(save_path + 'Mapa_semana_%s_%s.png'%(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')))
    #text_color = '#%02x%02x%02x' % (8,31,45)
    text_color = "#00364d"
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify',\
                              alignment=TA_JUSTIFY,\
                              fontName = "AvenirBook",\
                              fontSize = 30,\
                              textColor = text_color,\
                              leading = 20))


    pdfmetrics.registerFont(TTFont('Vera', 'Vera.ttf'))
    pdfmetrics.registerFont(TTFont('VeraBd', 'VeraBd.ttf'))
    pdf.setFillColorRGB(30/255, 129/255, 176/255) #Color del cuadro detrás del título
    pdf.rect(0., 755., 1320., 302., fill = True, stroke = False) #Cuadro azul detrás del título #(x,y(de la esquina inferior izquiera), largo, ancho)
    pdf.setFont('VeraBd', 29)
    pdf.setFillColorRGB(1, 1, 1)# Color del título
    pdf.drawString(190,795., end.strftime("Susceptibilidad a ICV para el %Y-%m-%d %H:00:00"))
    #pdf.drawString(280,760.,end.strftime("vegetal para el día %Y-%m-%d %H:00:00"))
    pdf.setFillColorRGB(3/255, 5/255, 138/255)
    #pdf.drawString(400,740., "Reporte para bomberos")
    #p = Paragraph(end.strftime('Reporte de susceptibilidad a incendios de cobertura vegetal para el día %Y-%m-%d %H:00:00'))
    #p.wrap(700, 1850)
    #p.drawOn(pdf, 238,760)
    
    pdf.showPage()

    pdf.save()
