# Desagregación de salidas anuales multianuales InVEST en series anuales

Para el análisis ROI se requieren resultados distribuidos en el tiempo con resolución anual. Dado que los modelos InVEST son de respuesta promedio de largo plazo, se utiliza una función logística para distribuir en el periodo de tiempo de análisis el resultado obtenido con la modelación. En este apartado se presenta la función que realiza el cálculo.

La distribución del resultado InVEST en el tiempo se realizará utilizando una función logística. Dado que el resultado de flujo base InVEST lo presenta originalmente en milímetros, en este paso se propone transformar en m3. Para esto la respuesta de InVEST debe ser multiplicada por el tramaño de la cuenca.

* El parámetro de porcentaje de beneficio en el tiempo t=0 se calcula considerando el porcentaje de beneficio en t=0 de cada actividad del portafolio mediante un promedio ponderado por el área.

<img src="https://latex.codecogs.com/gif.latex?w&space;=&space;\frac{W_{max}}{1&space;&plus;&space;(\frac{W_{max}}{W_{o}}&space;-&space;1)exp(-rt)&space;}" title="w = \frac{W_{max}}{1 + (\frac{W_{max}}{W_{o}} - 1)exp(-rt) }" /></a>


* El parámetro r se calcula considerando el tiempo en alcanzar el 100% del beneficio del portafolio. Este tiempo se calculo mediante un promedio ponderado por el área del tiempo para cada actividad.

<img src="https://latex.codecogs.com/gif.latex?Benefit_{t=0}&space;=&space;\frac{\sum_{n}^{i=1}A_{i}*Benefit_{i}}{\sum_{n^{i=1}A_{i}}}" title="Benefit_{t=0} = \frac{\sum_{n}^{i=1}A_{i}*Benefit_{i}}{\sum_{n^{i=1}A_{i}}}" /></a>

<img src="https://latex.codecogs.com/gif.latex?t_{MaxBenefit}&space;=&space;\frac{\sum_{n}^{i=1}A_{i}*t_{MaxBenefit_i}}{\sum_{n^{i=1}A_{i}}}" title="t_{MaxBenefit} = \frac{\sum_{n}^{i=1}A_{i}*t_{MaxBenefit_i}}{\sum_{n^{i=1}A_{i}}}" /></a>

En el esquema de cálculo, se realiza la distribución con la función logística para el beneficio marginal de cada portafolio y al final se realiza una convolución de los resultados considerando el desplazamiento anual de cada portafolio. El desarrollador de software deberá comprender el esquema de cálculo formulado a partir de este ejemplo y generalizarlo para su implementación

## Código

´´´

    # -*- coding: utf-8 -*-
    # Import Packages
    import numpy as np
    import pandas as pd
    import os

    def Desaggregation_BaU_NBS(PathProject):
        # Funtion Lambda
        Sigmoid_Desaggregation = lambda Wmax, Wo, r, t: Wmax/(1 + (((Wmax/Wo) - 1)*np.exp(-t*r)))

        NameCol     = ['AWY (m3)','Wsed (Ton)','WN (Kg)','WP (kg)','BF (m3)','WC (Ton)']
        Data        = pd.read_csv(os.path.join(PathProject,'01-INPUTS_InVEST.csv'),usecols=NameCol)
        NBS         = pd.read_csv(os.path.join(PathProject,'01-INPUTS_NBS.csv')).values[:,1:]
        Time        = pd.read_csv(os.path.join(PathProject,'01-INPUTS_Time.csv')).values[0][0]

        '''
        Current-BaU
        '''
        nn = nn = np.shape(Data)[1]
        Results_BaU = pd.DataFrame(data=np.empty([Time + 1, nn]), columns=NameCol)
        r = -1*np.log(0.000000001)/Time
        t = np.arange(0,Time + 1)
        for i in range(0,6):
            Results_BaU[NameCol[i]] = Sigmoid_Desaggregation(Data[NameCol[i]][1], Data[NameCol[i]][0], r, t)

        '''
        BaU-NBS
        '''
        # Estimation Time NBS
        n = np.size(NBS[0,2:])
        t_NBS = np.empty([n,1])
        p_NBS = np.empty([n,1])
        for i in range(0,n):
            t_NBS[i] = np.sum(NBS[:,0]* NBS[:,i+2])/np.sum(NBS[:,i+2])
            p_NBS[i] = np.sum(NBS[:,1]* NBS[:,i+2])/np.sum(NBS[:,i+2])

        # Desaggregation
        Results_NBS = pd.DataFrame(data=np.empty([Time + 1, nn]), columns=NameCol)

        # Estimation Diff
        [f,c]       = Data.shape
        Data1       = Data[2:].values
        Data1[0,:]  = Data1[0,:] - Data.loc[1].values
        Tmp         = np.cumsum(Data1,0)
        for i in range(1,f-2):
            Data1[i,:]   = Data1[i,:] - Data.loc[1].values - Tmp[i-1,:]
            Tmp = np.cumsum(Data1,0)

        for i in range(0,nn):
            Tmp = np.zeros((Time+1,n))
            for j in range(0,n):
                t    = np.arange(0, Time + 1 - (j+1))
                Wmax = Data1[j,i]
                tmax = t_NBS[j][0]
                Wo   = p_NBS[j][0]*Data1[j,i]*0.01
                r    = -1*np.log(0.000000001)/tmax

                #print(Wmax,'|', Wo,'|', r)

                Tmp[(j+1):,j] = Sigmoid_Desaggregation(Wmax, Wo, r, t)

            Results_NBS[NameCol[i]] = np.sum(Tmp,1) + Results_BaU[NameCol[i]].values

        '''    
        Save Data
        '''
        Results_BaU.to_csv(os.path.join(PathProject,'02-OUTPUTS_BaU.csv'), index_label='Time')
        Results_NBS.to_csv(os.path.join(PathProject,'02-OUTPUTS_NBS.csv'), index_label='Time')

	
´´´
## Tester 

´´´

    # terter
    PathProject = r'Z:\Box Sync\01-TNC-ThinkPad-P51\28-Project-WaterFund_App\02-Productos-Intermedios\Python_Convolution\Project'
    Desaggregation_BaU_NBS(PathProject)
    
´´´

## Configuración de un proyecto
La función para el cálculo de desagregación requiere 3 entradas separadas en archivos csv, las cuales son:
•	01-INPUTS_InVEST.csv
•	01-INPUTS_NBS.csv
•	01-INPUTS_Time.csv
El nombre de los archivos csv de entradas no pueden cambiar. A continuación, se describe la información que debe contener cada uno de ellos.

### 01-INPUTS_InVEST.csv
Este archivo contiene los resultados de las ejecuciones InVEST para cada uno de los escenarios. La estructura de este archivo es la siguiente.

|Scenario-InVEST|AWY (m3)|Wsed (Ton)|WN (Kg)|WP (kg)|BF (m3)|WC (Ton)|
|--|--|--|--|--|--|--|
|Current|800000000|100000|700000|500000|94822500|150|
|BaU|600000000|200000|800000|700000|50572000|50|
|NBS-Year_1|900000000|70000|600000|400000|120108500|250|
|NBS-Year_2|1000000000|65000|550000|350000|126430000|300|
|NBS-Year_3|1100000000|62000|520000|310000|132751500|330|

En este sentido, el archivo debe contener 7 columnas, donde:
	• **Scenario-InVEST**: Nombre de escenario
	• **AWY (m3)**: Los resultados de volumen de agua anual multianual del modelo Anual Water Yield en m3
	• **Wsed (Ton)**: Los resultados de carga de sedimentos anuales multianuales en toneladas del modelo Sediment Delivery Ratio.
	• **WN (Kg)**: Los resultados de carga de nitrógeno anuales multianuales en toneladas del modelo Nutrient Delivery Ratio.
	• **WP (kg)**: Los resultados de carga de fosforo anuales multianuales en toneladas del modelo Nutrient Delivery Ratio.
	• **BF (m3)**: Los resultados de flujo base anuales multianuales en m3 del modelo Seasonal Water Yield. Es importante tener en cuenta que este modelo arroja los resultados en mm, por lo que se debe realizar la conversión multiplicando por el área de la cuenca.
	• **WC (Ton)**: Los resultados almacenamiento de carbono anual multianual en toneladas del modelo Carbon Storage and Sequestration.

Además, la primera fila debe corresponder a los resultados del escenario Current State, la segunda fila corresponde a los resultados del escenario Bussines as Usual y las siguientes corresponde a los resultados de los años de implementación de las NBS. Es importante tener en cuenta que estos resultados corresponden a las implementaciones acumuladas de las NBS año tras año.

### 01-INPUTS_NBS.csv

