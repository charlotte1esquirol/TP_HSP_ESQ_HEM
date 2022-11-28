#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// TP HSP1 : Implémentation d'un CNN - LeNet-5 sur GPU


// Partie 1 - Prise en main de Cuda : Multiplication de matrices

// Initialisation de matrice de taille n*p avec des valeurs aléatoires entre -1 et 1
void MatrixInit(float *M, int n, int p){

    float random_value;

    for (int i=0; i<n*p; i++){ //n*p la taille de la matrice, on parcourt toute la matrice
        random_value = (float)rand() / (float)((RAND_MAX*1.0)-1);
        M[i] = random_value;
        //printf("%f", random_value);
    }
    //printf("%d", RAND_MAX);
}

//Affichage d'une matrice de taille n*p
void MatrixPrint(float *M, int n, int p){
    for (int i=0; i<n; i++){
        for (int j=i*p; j<(i+1)*p; j++){
            printf("%1.3f ", M[j]);
        }
        printf("\n");
    }
    printf("\n");
}

//Addition de deux matrices sur CPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i=0; i<n*p; i++){ //n*p la taille de la matrice, on parcourt toute la matrice
        Mout[i] = M1[i] + M2[i];
    }
}

//Addition de deux matrices sur GPU
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int i = threadIdx.x;
    int j = threadIdx.y;

    
    if (i<n && j<p){
        Mout[i*p+j]=M1[i*p+j]+M2[i*p+j];
    }

}

//Multiplication de deux matrices sur CPU
void MatrixMult(float *M1, float *M2, float *Mout, int n){
    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            float sum=0.0;
            for(int k=0; k<n; k++){
                sum+=M1[i*n+k]*M2[k*n+j];
            }

            Mout[i*n+j]=sum;
            
        }
    }
}

//Multiplication de deux matrices sur GPU
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    int i = threadIdx.x;
    int j = threadIdx.y;

    float sum=0.0;
    
    if (i<n && j<n){
        for(int k=0; k<n; k++){
            sum+=M1[i*n+k]*M2[k*n+j];
        }
            Mout[i*n+j]=sum;
    }

}


int main(){
    // Test initialisation de matrices + Affichage + Addition CPU + Multiplication CPU
  
    int n=1000;
    int p=2;
    double time_spentaddCPU=0.0;
    double time_spentmulCPU=0.0;
    double time_spentaddGPU=0.0;
    double time_spentmulGPU=0.0;
  
    //Addition

    float *M1;
    float *M2;
    float *Mout;
    

    M1 = (float*)malloc(n*p*sizeof(float));
    M2 = (float*)malloc(n*p*sizeof(float));
    Mout = (float*)malloc(n*p*sizeof(float));
    

    MatrixInit(M1,n,p);
    //MatrixPrint(M1,n,p);

    MatrixInit(M2,n,p);
    //MatrixPrint(M2,n,p);

    clock_t begin = clock(); // Mesure du temps d'execution d'une addition CPU
    MatrixAdd(M1,M2,Mout,n,p);
    clock_t end = clock();
    time_spentaddCPU+=(double)(end-begin)/CLOCKS_PER_SEC;
    printf("Time for addition CPU : %f\n",time_spentaddCPU);
    //MatrixPrint(Mout,n,p);


    //Multiplication

    float *M1bis;
    float *M2bis;
    float *Moutbis;

    M1bis = (float*)malloc(n*n*sizeof(float));
    M2bis = (float*)malloc(n*n*sizeof(float));
    Moutbis = (float*)malloc(n*n*sizeof(float));

    MatrixInit(M1bis,n,n);
    //MatrixPrint(M1bis,n,n);

    MatrixInit(M2bis,n,n);
    //MatrixPrint(M2bis,n,n);


    clock_t begin1 = clock(); // Mesure du temps d'execution d'une addition CPU
    MatrixMult(M1bis,M2bis,Moutbis,n);
    clock_t end1 = clock();
    time_spentmulCPU+=(double)(end1-begin1)/CLOCKS_PER_SEC;
    printf("Time for multiplication CPU : %f\n",time_spentmulCPU);
    
    //MatrixPrint(Moutbis,n,n);

  

    // Test initialisation de matrices, additions et multiplications sur GPU

    // Additions

     float *MG1;
     float *MG2;
     float *MGout;


     cudaMalloc((void**)&MG1, n*p*sizeof(float));
     cudaMalloc((void**)&MG2, n*p*sizeof(float));
     cudaMalloc((void**)&MGout, n*p*sizeof(float));

     cudaMemcpy(MG1, M1, n*p*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(MG2, M2, n*p*sizeof(float), cudaMemcpyHostToDevice);
     

     dim3 gridDim(1,1);
     dim3 blockDim(n,p,1);

     clock_t begin2 = clock(); // Mesure du temps d'execution d'une addition CPU
     cudaMatrixAdd<<<gridDim,blockDim>>>(MG1, MG2, MGout, n, p);
     clock_t end2 = clock();
     time_spentaddGPU+=(double)(end2-begin2)/CLOCKS_PER_SEC;
     printf("Time for addition GPU : %f\n",time_spentaddGPU);
    
    

     cudaDeviceSynchronize();
     cudaMemcpy(Mout, MGout, n*p*sizeof(float), cudaMemcpyDeviceToHost);
     cudaDeviceSynchronize();

     //MatrixPrint(M1,n,p);
     //MatrixPrint(M2,n,p);
     //MatrixPrint(Mout,n,p);

     
     // Multiplications
     
     
     float *MG1bis;
     float *MG2bis;
     float *MGoutbis;

     cudaMalloc((void**)&MG1bis, n*n*sizeof(float));
     cudaMalloc((void**)&MG2bis, n*n*sizeof(float));
     cudaMalloc((void**)&MGoutbis, n*n*sizeof(float));

     cudaMemcpy(MG1bis, M1bis, n*n*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(MG2bis, M2bis, n*n*sizeof(float), cudaMemcpyHostToDevice);


     dim3 gridDimbis(1,1);
     dim3 blockDimbis(n,n,1);

     clock_t begin3 = clock(); // Mesure du temps d'execution d'une addition CPU
     cudaMatrixMult<<<gridDimbis,blockDimbis>>>(MG1bis, MG2bis, MGoutbis, n);
     clock_t end3 = clock();
     time_spentmulGPU+=(double)(end3-begin3)/CLOCKS_PER_SEC;
     printf("Time for multiplication GPU : %f\n",time_spentmulGPU);
     
     cudaDeviceSynchronize();
     cudaMemcpy(Moutbis, MGoutbis, n*n*sizeof(float), cudaMemcpyDeviceToHost);
     cudaDeviceSynchronize();

     //MatrixPrint(M1bis,n,n);
     //MatrixPrint(M2bis,n,n);
     //MatrixPrint(Moutbis,n,n);

    return 0;
}
