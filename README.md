# TP_HSP_ESQ_HEM 

 ## Partie 1 - Prise en main de Cuda : Multiplication de matrices

 ### Addition de matrices
 
 ### Compléxité et temps de calcul : 
 Nous avons mesuré les temps d'éxécution des addition et des multiplications pour le CPU et le GPU. Lorsque la taille de la matrice reste petite, les calculs prennent environ le même temps pour le CPU et le GPU. Nous avons remarqué que lorsque la taille de la matrice devient conséquente ( ex n=1000 ), les additions gardent environ le même temps d'éxécution mais les multiplications sur CPU devienent beaucoup plus lentes que celles sur GPU ( environ 8 secondes contre 0,03 secondes ). 
 C'est cohérent avec le cours car nous savons que le GPU est beaucoup plus compétent sur des calculs complexes. Les temps d'éxécution sont réduits grâce au GPU. 
 

## Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

L'architecture du réseau LeNet-5 est composé de plusieurs couches :

 - Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST 
 - Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.
 - Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.
 
 ### Layer 1 - Génération des données de test : 
 
 
 ### Layer 2 - Convolution 2D : 
 
 ### Layer 3 - Sous-échantillonnage
 
 
 ## Partie 3 - Un peu de Python
 
 Fonctions manquantes : 
