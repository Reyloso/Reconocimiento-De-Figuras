#Algoritmo de deteccion de triangulos
#Por Glar3
#
#
#Detecta triangulos azules
   
#Librerias
import cv2
import numpy as np
   
#Iniciar camara
captura = cv2.VideoCapture(1)

FONTE = cv2.FONT_HERSHEY_SIMPLEX
   
while(1):
   
    #Caputrar una imagen y convertirla a hsv
    valido, imagen = captura.read()
    if valido:
       hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
       
       #Guardamos el rango de colores hsv (azules)
       bajos = np.array([67,40,105], dtype=np.uint8)
       altos = np.array([129, 255, 182], dtype=np.uint8)
       
       #bajos = np.array([255,255,255], dtype=np.uint8)
       #altos = np.array([137, 137, 137], dtype=np.uint8)
       #Crear una mascara que detecte los colores
       mask = cv2.inRange(hsv, bajos, altos)
       
       #Filtrar el ruido con un CLOSE seguido de un OPEN
       kernel = np.ones((9,9),np.uint8)
       mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
       mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
       
       #Difuminamos la mascara para suavizar los contornos y aplicamos filtro canny
       blur = cv2.GaussianBlur(mask, (7, 7), 0)
       edges = cv2.Canny(mask,1,2)
       
       #Si el area blanca de la mascara es superior a 500px, no se trata de ruido
       _,contours,_ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
       areas = [cv2.contourArea(c) for c in contours]
    
       cantriangulos = 0
       canCuadrado = 0
       canCirculo = 0
       i=0
       for extension in areas:
           if extension > 600:
               actual = contours[i]
               approx = cv2.approxPolyDP(actual,0.05*cv2.arcLength(actual,True),True)
               if len(approx)==3:
                   cv2.drawContours(imagen,[actual],0,(0,0,255),5)
                   cv2.drawContours(mask,[actual],0,(0,0,255),5)
                   cantriangulos = cantriangulos + 1
                   forma ="Triangulo : " +str(cantriangulos)
                   cv2.putText(imagen,forma,((20),(25)),FONTE,1,(0,255,0),2)
               elif len(approx)==4:
                  cv2.drawContours(imagen,[actual],0,(0,0,255),6)
                  cv2.drawContours(mask,[actual],0,(0,0,255),6)
                  canCuadrado = canCuadrado + 1
                  forma ="Cuadrado : " +str(canCuadrado)
                  cv2.putText(imagen,forma,((20),(55)),FONTE,1,(0,255,0),2)
               else:
                  cv2.drawContours(imagen,[actual],0,(0,0,255),6)
                  cv2.drawContours(mask,[actual],0,(0,0,255),6)
                  canCirculo = canCirculo + 1
                  forma ="Circulo : " +str(canCirculo)
                  cv2.putText(imagen,forma,((20),(85)),FONTE,1,(0,255,0),2)
               i=i+1
               

       
       cv2.imshow('mask', mask)
       cv2.imshow('Camara', imagen)
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
   

cv2.destroyAllWindows()


