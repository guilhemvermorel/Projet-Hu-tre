import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

########entourer l'huître dans un rectangle########

def condition(car1,car2,car3,indice,val_indice):
  rep = False
  if car1 == 1.0 :
     if car2 == 1.0 :
        if car3 == 1.0 :
          if indice != val_indice :
            rep = True


  return rep



########trouver la longueur maximum de l'huître########
def longueur_max(img):  
  
  larg = len(img)
  long = len(img[0])
  Max = 0
  
  #on regarde toutes les lignes du tableau img, et on compare la longueurs des pixels les plus éloignés, on récupère la longueur la plus grande
  for i in range(larg):
    left_j=0
    while condition(img[i][left_j][0],img[i][left_j][1],img[i][left_j][2],left_j,long-1):
      left_j+=1
    j_max_left = left_j
  
    right_j=long-1
    while condition(img[i][right_j][0],img[i][right_j][1],img[i][right_j][2],right_j,0):
      right_j-=1
    j_max_right = right_j

    if j_max_right - j_max_left >= Max :
      Max =  j_max_right - j_max_left
      (iplus,jplus) = (i,j_max_right)
      (imoins,jmoins) = (i,j_max_left)
  
  return Max,(iplus,jplus),(imoins,jmoins)

#trouver la largeur maximum de l'huître
def largeur_max(img):  
  
  larg = len(img)
  long = len(img[0])
  Max = 0

  #on regarde toutes les colonnes du tableau img, et on compare la longueurs des pixels les plus éloignés, on récupère la longueur la plus grande
  for j in range(long):
    bot_i=0
    while condition(img[bot_i][j][0],img[bot_i][j][1],img[bot_i][j][2],bot_i,larg-1):
      bot_i+=1
    i_max_bot = bot_i
  
    top_i=larg-1
    while condition(img[top_i][j][0],img[top_i][j][1],img[top_i][j][2],top_i,0):
      top_i-=1
    i_max_top = top_i

    
    if i_max_top - i_max_bot >= Max :
      Max = i_max_top - i_max_bot
      (iplus,jplus) = (i_max_top,j)
      (imoins,jmoins) = (i_max_bot,j)
    
  
  return Max,(iplus,jplus),(imoins,jmoins)


########récupérer l'image de l'huître sans fond########

#== Parameters =======================================================================
BLUR = 21
CANNY_THRESH_1 = 45
CANNY_THRESH_2 = 50
MASK_DILATE_ITER = 50
MASK_ERODE_ITER = 50
MASK_COLOR = (0.0,0.0,1.0) # In BGR format

 
#== Processing =======================================================================

#--Run algorithm for all pictures-------------------------------------------------------
#on créer le tableau qui contiendra les longueurs et largeurs max de chaque images
array_dimension = []

for k in range(1,137):
  if k!=92:
    if k <100 :
        if k<10 :
            numImage = '00'+str(k)+'d'
        else :
            numImage = '0'+str(k)+'d'
    else :
        numImage = str(k)+'d'

    path ='/content/drive/MyDrive/Projet Entreprise/Ingenierie Projet/database_v1/'+str(numImage)+'.jpg'
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((c,cv2.isContourConvex(c),cv2.contourArea(c),))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]
    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
    #-- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
    #-- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices,
    img = img.astype('float32') / 255.0                 #  for easy blending

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit

    # split image into channels
    c_red, c_green, c_blue = cv2.split(img)

    # merge with mask got on one of a previous steps
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))


    # creation de l'image b qui réorganise l'image de façon à avoir le fond en blanc
    img_b = []
    longueur = len(img_a)
    largeur = len(img_a[1])

    for i in range(longueur):
      long = []
      for j in range(largeur):
        if img_a[i][j][3]==0.0:
          long.append([1.0,1.0,1.0])
        else :
          long.append(img_a[i][j][:3])
      img_b.append(long)

    mg_b=np.array(img_b)

    print(k)
    #on ajoute les longueurs et largeurs max des images
    array_dimension.append([longueur_max(img_b)[0],largeur_max(img_b)[0]])

#on créer csv du tableau précédent
array_dimension = np.array(array_dimension)
df_dimension = pd.DataFrame(array_dimension,columns=['longueur','largeur'])

df_dimension.to_csv('/content/drive/MyDrive/Projet Entreprise/Ingenierie Projet/Database des dimensions.csv')



#########lecture des données#########
df_dimension = pd.read_csv('/content/drive/MyDrive/Projet Entreprise/Ingenierie Projet/Database des dimensions.csv',',')
df_dimension = df_dimension.drop('Unnamed: 0',axis=1)
df_poids = pd.read_excel('/content/drive/MyDrive/Projet Entreprise/Ingenierie Projet/Solution technique/Poids_huitres.xlsx')
df_dimension['classe'] = np.append(df_poids['Classe'].to_numpy()[:92],df_poids['Classe'].to_numpy()[93:])



#########afficher les huîtres en fonction de leur dimension selon leur classe d'appartenance (couleurs)#########
plt.scatter(df_dimension['longueur'][df_dimension['classe']==3], df_dimension['largeur'][df_dimension['classe']==3],c='red',label='Classe 3')
plt.scatter(df_dimension['longueur'][df_dimension['classe']==4], df_dimension['largeur'][df_dimension['classe']==4],c='green',label='Classe 4')
plt.scatter(df_dimension['longueur'][df_dimension['classe']==5], df_dimension['largeur'][df_dimension['classe']==5],c='blue',label='Classe 5')
plt.scatter(df_dimension['longueur'][df_dimension['classe']==6], df_dimension['largeur'][df_dimension['classe']==6],c='yellow',label='Classe 6')
plt.legend()
plt.title("Clusters selon les classes d'huîtres")
plt.xlabel('Longueur')
plt.ylabel('Largeur')
plt.ioff()
plt.savefig('/content/drive/MyDrive/Projet Entreprise/Programmation /clusters.jpg')


#########Knn classifier#########
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import statistics

#On mets en place la méthode du KNN Classifier
X = df_dimension.drop('classe',axis=1).to_numpy()
y = df_dimension['classe'].to_numpy()

#on rééchelonne les données 
scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X, y)


#RENTRER LES DONNEES DE X à PREDIRE, exemple :
img_pred = cv2.imread('/content/img.jpg')
X = np.asarray([longueur_max(img_pred)[0],largeur_max(img_pred)[0]])
X_pred = scaler.transform(X_pred)
y_pred = classifier.predict(X_pred)

average_accuracy_all=list()
average_accuracy_class3=list()
average_accuracy_class4=list()
average_accuracy_class5=list()
average_accuracy_class6=list()


#########On test la méthode du KNN Classifier et on calcule la moyenne de son accurracy pour 1000 tests différents#########
for i in range(1000):
  X = df_dimension.drop('classe',axis=1).to_numpy()
  y = df_dimension['classe'].to_numpy()
  #on divise le train/test en 80/20
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  
  scaler = StandardScaler()
  scaler.fit(X_train)
  #on rééchelonne les données 
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)
  classifier = KNeighborsClassifier(n_neighbors=5)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)

  class_report = classification_report(y_test, y_pred,output_dict=True)


  if '3' in class_report :
    average_accuracy_class3.append(class_report['3']['precision'])
  if '4' in class_report :
    average_accuracy_class4.append(class_report['4']['precision'])
  if '5' in class_report :
    average_accuracy_class5.append(class_report['5']['precision'])
  if '6' in class_report :
    average_accuracy_class6.append(class_report['6']['precision'])
  average_accuracy_all.append(accuracy_score(y_test,y_pred))


#Résultats du test
print('RESULTATS DU KNearestNeighboor Classifier')
print('')
print('accuracy moyenne de toutes les classes : ',statistics.mean(average_accuracy_all))
print('accuracy moyenne de la classe 3 : ',statistics.mean(average_accuracy_class3))
print('accuracy moyenne de la classe 4 : ',statistics.mean(average_accuracy_class4))
print('accuracy moyenne de la classe 5 : ',statistics.mean(average_accuracy_class5))
print('accuracy moyenne de la classe 6 : ',statistics.mean(average_accuracy_class6))