# Projet : Classification d'huîtres par leur poids en Machine Learning #

Le projet a commencé le 20 septembre 2021 et pris fin le 10 décembre 2021. 


<B> Contexte : </B>

  Une entreprise d'ostréicole de petite taille située sur l'Ile d'Oléron cherche à automatiser sa chaîne de traitement d'huîtres. Elle produis les différentes types d'huitres correspondant aux labels "Marennes-Oléron": fines de claires, spéciales de claires et pousses en claire. Du fait de sa petite taille, elle n'a pas accès au tri mécaniques et est donc contraint de l'effectuer manuellement. Comme le tri se fait par leur observation il est possible de remplacer se processus fastidieux avec l'aide d'algorithmes de Machine Learning.


<B> Problématique : </B>
  
  Les huîtres sont vendues avec un numéro correspondant au poids. Ainsi, l'objectif du projet serait à partir de photos d'huîtres prise par un appareil installé dans la chaîne de traitement, de pouvoir classer l'huître en plusieurs catégorie en fonction de leur poids. 
  

<B> Réalisation : </B>
  
  Ce git contient l'implémentation d'un algorithme de clustering plutôt "simple", le KNN algorithm. Il prends en entrée des photos des huîtres de la variété "huîtres Marennes Oléros" et les classifie selon 4 classes différentes. Après traitement de l'image via la librairie python CV2, l'algorithme entoure l'huître grossièrement dans un rectangle pour en récupérer ses dimensions de longueurs et largeurs. Ensuite, le KNN classifier réalise la classification selon la méthode des plus proches voisins selon les 4 classes labélisés. Le tableau csv contient le jeu d'entraînement des huîtres. L'accuracy de l'algorithme est de 50%.
  
Il existe également un algorithme plus complexe de réseau de neurones (VGG16) implémenté qui obtient une accuracy de 60%, mais sous licence de l'entreprise. 
 
