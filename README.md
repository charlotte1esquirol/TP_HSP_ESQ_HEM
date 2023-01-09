# TP_HSP_ESQ_HEM 

 ## Partie 1 - Prise en main de Cuda : Multiplication de matrices

 ### Addition de matrices
 
#### Sur CPU

Addition de deux matrices M1 et M2 : 

<img width="284" alt="Capture d’écran 2023-01-09 à 19 28 05" src="https://user-images.githubusercontent.com/94962700/211380979-61669054-1f03-4c6f-97a9-f135def4790a.png">


On vient additionner deux matrices comme à notre habitude sur le CPU. Voici un exemple sur des matrices 3x3 :

<img width="734" alt="Capture d’écran 2023-01-09 à 19 27 17" src="https://user-images.githubusercontent.com/94962700/211380871-d8852c24-0998-463f-a87f-37e986c46150.png">

On fait aussi afficher le temps de calcul de notre opération sur le CPU en secondes : 

<img width="273" alt="image" src="https://user-images.githubusercontent.com/94962700/211380825-dee78ed9-241a-49e3-877f-c6e3fd0cfaad.png">

 #### Sur GPU

On vient maintenant réaliser la même opération mais cette fois-ci sur le GPU. Pour cela, on doit savoir comment définir nos indices des coefficients matriciels, désormais elle se fait sur 3 dimensions. Par exemple on définira l’indice de ligne comme ci-dessous : 

<img width="164" alt="image" src="https://user-images.githubusercontent.com/94962700/211381183-65b48141-60ca-4d92-b23f-1792533ff3c0.png">



Où : threadIdx.x donne le numéro de la ligne du thread appartenant au block.

On vient ensuite définir la fonction global cudaMatrixAdd afin d’effectuer les calculs sur le 
GPU : 

![image](https://user-images.githubusercontent.com/94962700/211381237-c2743767-e6ab-45a2-9909-cd97370fef23.png)


Cette fonction sera nécessaire pour toutes les fonctions où l’on souhaite que le calcul soit réaliser sur le GPU. 
Pour finir on vient allouer les mémoires des matrices sur le GPU à l’aide de cudaMalloc et copiées depuis le CPU vers le GPU à l’aide de cudaMemcpy. Et l’on vient aussi définir les dimensions des variables dim3 dans notre main : 

<img width="217" alt="image" src="https://user-images.githubusercontent.com/94962700/211381312-1343b32f-87c0-4b55-8e58-c2c724e118b1.png">


Exemple d’addition de deux matrices 3x3 sur le GPU : 

<img width="656" alt="Capture d’écran 2023-01-09 à 19 30 19" src="https://user-images.githubusercontent.com/94962700/211381408-be3dcd0d-6be9-4933-ae9e-68587f30250b.png">


On fait aussi afficher le temps de calcul de notre opération sur le GPU en secondes : 

<img width="278" alt="image" src="https://user-images.githubusercontent.com/94962700/211381337-4f5421f1-2e8e-436e-b975-6ae8d01f59d2.png">


En comparant les temps de calculs pour l’addition sur GPU et CPU pour un petit n on peut voir que ce sont presque les mêmes. 


### Multiplication de matrice 

#### Sur CPU 
Pour la multiplication, la seule difficulté est de faire attention à l’indexage des coefficients que l’on cherche à obtenir.  

Exemple de multiplication de deux matrices 3x3 sur le CPU: 

<img width="656" alt="Capture d’écran 2023-01-09 à 19 30 19" src="https://user-images.githubusercontent.com/94962700/211381907-c0d1f4ce-022b-48e3-81ff-619837dafef8.png">

Comme avant on vient aussi afficher le temps de calcul de notre opération sur le CPU en secondes : 
<img width="322" alt="image" src="https://user-images.githubusercontent.com/94962700/211381947-cb12de54-0d5a-44ca-a91a-728ec93f6405.png">


#### Sur GPU 
Pour la multiplication sur GPU on procède de la même manière que pour l’addition sur GPU. 

Exemple de multiplications de deux matrices 3x3 sur GPU : 

<img width="835" alt="Capture d’écran 2023-01-09 à 19 33 45" src="https://user-images.githubusercontent.com/94962700/211382129-8fb6cd76-0204-488b-9dfc-1509b309d210.png">

Affichage du temps de calcul de notre opération sur le GPU en secondes : 

<img width="328" alt="image" src="https://user-images.githubusercontent.com/94962700/211382166-b062bf2b-43ed-4ba9-9e25-b92d02671304.png">


En comparant les temps de calculs pour la multiplication sur GPU et CPU pour un petit n on peut voir que pour le GPU le temps est plus faible.


 ### Compléxité et temps de calcul : 
Nous avons mesuré les temps d'exécution des addition et des multiplications pour le CPU et le GPU. Lorsque la taille de la matrice reste petite, les calculs prennent environ le même temps pour le CPU et le GPU. Nous avons remarqué que lorsque la taille de la matrice devient conséquente ( voir exemple ci dessous avec n=1000 ), les additions gardent environ le même temps d'exécution mais les multiplications sur CPU deviennent beaucoup plus lentes que celles sur GPU ( environ 8 secondes contre 4 microsecondes ). C'est cohérent avec le cours car nous savons que le GPU est beaucoup plus compétent sur des calculs complexes. Les temps d'exécution sont réduits grâce au GPU.

Exemple temps de calcul pour n =1000 :

<img width="369" alt="image" src="https://user-images.githubusercontent.com/94962700/211382262-abf90c26-38d0-4b86-b2bf-5503dd7d041a.png">
 
## Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

L'architecture du réseau LeNet-5 est composé de plusieurs couches :

 - Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST 
 - Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.
 - Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.
 
 ### Layer 1 - Génération des données de test : 
 
La génération des données consiste en la création des matrices suivantes sous la forme de tableaux à une dimension:
 - __raw_data__ : la matrice d'entrée dans le réseau 32x32 initialisée avec des valeurs aléatoires entre 0 et 1.
 - __C1_data__ : la matrice 6x28x28 résultante de la convolution 2D initialisée avec des valeurs nulles.
 - __S1_data__ : la matrice 6x14x14 résultante du sous-échantillonnage initialisée avec des valeurs nulles.
 - __C1_kernel__ : le kernel 6x5x5 permettant la convolution de la layer 1 et initialisé entre 0 et 1.

 ### Layer 2 - Convolution 2D : 
La convolution se fait exclusivement sur le GPU. De façon analogue à la multiplication, on fait glisser un kernel __C1_kernel__ sur la totalité de la matrice raw_data pour obtenir la matrice résultante __C1_data__.

<img width="400" alt="image" src="https://user-images.githubusercontent.com/94962700/211378877-f6ce6469-a3e9-4818-b073-9363abd03f86.png">

 ### Layer 3 - Sous-échantillonnage
Cette troisième layer va nous permettre de sous échantillonner notre image par Mean Pooling 2x2 (=moyennage sur 4 pixels vers 1). Ce sous échantillonnage va nous permettre de réduire d’un facteur 2 les dimensions de __C1_data__. 

<img width="440" alt="Capture d’écran 2023-01-09 à 19 18 55" src="https://user-images.githubusercontent.com/94962700/211379329-e8993429-da05-4df4-bec5-22a5ca9b0f9a.png">

*__Figure 1 : Exemple de sous-échantillonnage d’une matrice 8x8 par moyennage de 2x2 pixels vers un pixel.__*

Comme précédemment, on réalise le calcul sur le GPU depuis un appel du CPU.

 ### TEST:
On vient maintenant tester notre layer de convolution 2D ainsi que la layer de sous-échantillonnage avec des valeurs « simples » afin de vérifier leur bon fonctionnement : 

Pour : 
- Une matrice initiale taille 5x5 
- Un kernel de 2x2 

Nous obtenons les résultats suivant :

<img width="329" alt="image" src="https://user-images.githubusercontent.com/94962700/211379833-b3553205-bb78-4252-bcef-cc840559c7fb.png">

 ### Fonction d'activation:
Comme nous le savons une couche de convolution est toujours suivi d’une fonction d’activation on va donc venir en rajouter une après notre layer de Convolution 2D. Pour cela on vient créer une fonction *__activation_tanh(float M)__* , de manière à ce qu’elle puisse être appelé par chaque kernel du GPU : 
<img width="366" alt="image" src="https://user-images.githubusercontent.com/94962700/211380164-2e9cb426-d784-4890-9a13-f11ff481059f.png">

Nous avons rajouté ___device___ afin d’effectuer les calculs sur le GPU depuis un appel du GPU. Notre couche d’activation va nous retourner une matrice de même dimension que celle qui lui est fournie. 
On vient ensuite tester notre fonction d’activation sur une matrice de petite taille :

<img width="327" alt="image" src="https://user-images.githubusercontent.com/94962700/211380359-72629c9c-6f51-475c-96e4-86dcb1ae9765.png">



 ## Partie 3 - Un peu de Python
 
 Fonctions manquantes : Il nous manque actuellement une deuxième convolution et sa fonction d'activation suivie d'un sous-échantillonage. Ensuite, il nous manque une couche Flatten et 3 couches Dense dont une suivie d'une fonction d'activation de type softmax. Pour la couche Flatten, ici dans notre cas nous n'avons pas besoin de modifier la sortie de notre deuxième échantillonage car celle ci est un tableau de valeurs (1x400)=(1x(5x5x16)). De ce fait, nous n'avons pas besoin de modifier. Puis, pour la couche Dense, nous savons que cela consiste en la multiplication d'une matrice d'entrée X avec les poids W correspondants puis à l'addition de ce produit avec une matrice B représentant les biais. On a donc SortieDeCoucheDense=W*X+B.
 
 Cependant, nous avons rencontré beaucoup de problèmes avec la couche Dense ce qui nous a empeché de continuer le reste du TP. Actuellement, nos deux premières couches Dense ne posent plus de problème et ne provoquent pas la mise à 0 de nos matrices précédentes mais la troisième si. Nous n'avons pas réussi à régler le soucis. 
 
 
