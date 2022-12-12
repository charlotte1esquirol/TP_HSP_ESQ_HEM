#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// TP HSP1 : Implémentation d'un CNN - LeNet-5 sur GPU


// Partie 2 - Premières couches du reseau de neuronnes LeNet-5 : Convolution 2D et subsampling

// 3.1 Layer 1 - Génération des données de test


void MatrixInit(float *M, int n, int p, int m, int init){

    float random_value;

    if( init==0){
        for (int i=0; i<n*m*p; i++){ //n*p la taille de la matrice, on parcourt toute la matrice
            M[i] = 0;
        }
    }

    else {
        for (int i=0; i<n*p; i++){ //n*p la taille de la matrice, on parcourt toute la matrice
            random_value = (float)rand() / (float)((RAND_MAX*1.0)-1);
            M[i] = random_value;
        }//printf("%f", random_value);
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

// 3.2 Layer 2 - Convolution 2D

__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_nb_raw, int M_nb_column, int kernel_size,  int nb_kernel, int Mout_nb_raw, int Mout_nb_column){

    int i = threadIdx.x;
    int j = threadIdx.y;

    float sum;

    if (i<Mout_nb_raw && j<Mout_nb_column){


        for(int k=0;k<nb_kernel;k++){
            sum=0.0;
            for(int ki=0;ki<kernel_size;ki++){
                for(int kj=0;kj<kernel_size;kj++){

                    sum += M[(i+ki)*M_nb_column+(j+kj)] * kernel[ki*kernel_size+kj+k*kernel_size*kernel_size]; 


                }
            }

            Mout[i*Mout_nb_column+j+k*Mout_nb_raw*Mout_nb_column]=sum;
        }


    }

}



__global__ void cudaSubsampling(float* M, float* Mout, int M_nb_raw, int M_nb_column, int M_nb_filtre, int subsampling_size, int Mout_nb_raw, int Mout_nb_column){

    int i = threadIdx.x;
    int j = threadIdx.y;

    if(i%subsampling_size==0 && j%subsampling_size==0){

        float sum;
        float average;


        for(int k=0;k<M_nb_filtre;k++){

            sum=0.0;

            for(int di=0;di<subsampling_size;di++){
                for(int dj=0; dj<subsampling_size;dj++){
                    sum += M[(i+di)*M_nb_column+(j+dj)+k*M_nb_column*M_nb_raw];
                    average=sum/(subsampling_size*subsampling_size);
                }
            }

            if(i==0){
                Mout[i*Mout_nb_column+(j/subsampling_size)+k*Mout_nb_column*Mout_nb_raw]=average;
            }
            else if(j==0){
                Mout[(i/subsampling_size)*Mout_nb_column+(j)+k*Mout_nb_column*Mout_nb_raw]=average;
            }
            else{
                Mout[(i/subsampling_size)*Mout_nb_column+(j/subsampling_size)+k*Mout_nb_column*Mout_nb_raw]=average;
            }     
            
        }
    }
}





int main(){
    
    //int argc, char **argv[]

    float *raw_data;
    raw_data = (float*)malloc(32*32*1*sizeof(float));
    MatrixInit(raw_data,32,32,1,1);
    MatrixPrint(raw_data,32,32);

    float *C1_data;
    C1_data = (float*)malloc(28*28*6*sizeof(float));
    MatrixInit(C1_data,28,28,6,0);
    MatrixPrint(C1_data,28,28);

    float *S1_data;
    S1_data = (float*)malloc(14*14*6*sizeof(float));
    MatrixInit(S1_data,14,14,6,0);
    MatrixPrint(S1_data,14,14);

    float *C1_kernel;
    C1_kernel = (float*)malloc(5*5*6*sizeof(float));
    MatrixInit(C1_kernel,5,5,6,1);
    MatrixPrint(C1_kernel,5,5);

    // Test sur matrices plus petites pour vérifier les valeurs :

    float *raw_data2;
    raw_data2 = (float*)malloc(5*5*1*sizeof(float));
    MatrixInit(raw_data2,5,5,1,1);
    printf("Matrice initiale:\n");
    MatrixPrint(raw_data2,5,5);

    float *C1_data2;
    C1_data2 = (float*)malloc(4*4*1*sizeof(float));
    MatrixInit(C1_data2,4,4,1,0);
    printf("Matrice après convolution 2D:\n");
    MatrixPrint(C1_data2,4,4);

    float *C1_kernel2;
    C1_kernel2 = (float*)malloc(2*2*1*sizeof(float));
    MatrixInit(C1_kernel2,2,2,1,1);
    printf("Matrice kernel:\n");
    MatrixPrint(C1_kernel2,2,2);

    float *S1_data2;
    S1_data2 = (float*)malloc(2*2*1*sizeof(float));
    MatrixInit(S1_data2,2,2,1,0);
    printf("Matrice après sous-échantillonnage:\n");
    MatrixPrint(S1_data2,2,2);













    // Tests sur GPU

     float *raw_data_g;
     float *C1_data_g;
     float *S1_data_g;
     float *C1_kernel_g;


     cudaMalloc((void**)&raw_data_g, 32*32*1*sizeof(float));
     cudaMalloc((void**)&C1_data_g, 28*28*6*sizeof(float));
     cudaMalloc((void**)&S1_data_g, 14*14*6*sizeof(float));
     cudaMalloc((void**)&C1_kernel_g, 5*5*6*sizeof(float));

     cudaMemcpy(raw_data_g, raw_data, 32*32*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(C1_data_g, C1_data, 28*28*6*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(S1_data_g, S1_data, 14*14*6*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(C1_kernel_g, C1_kernel, 5*5*6*sizeof(float), cudaMemcpyHostToDevice);
     

     dim3 gridDim2(1,1);
     dim3 blockDim2(32,32);

     cudaConv2D<<<gridDim2,blockDim2>>>(raw_data_g, C1_kernel_g, C1_data_g, 32, 32, 5, 6, 28, 28);
     cudaDeviceSynchronize();
     cudaSubsampling<<<gridDim2,blockDim2>>>(C1_data_g, S1_data_g, 28, 28, 6, 2, 14, 14);
     cudaDeviceSynchronize();

     cudaMemcpy(C1_data, C1_data_g, 28*28*6*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(S1_data, S1_data_g, 14*14*6*sizeof(float), cudaMemcpyDeviceToHost);
     cudaDeviceSynchronize();


     MatrixPrint(raw_data,32,32);
     MatrixPrint(C1_kernel,5,5);
     MatrixPrint(C1_data,28,28);
     MatrixPrint(S1_data,14,14);
     
     cudaFree(raw_data_g);
     cudaFree(C1_kernel);
     cudaFree(C1_data);
     cudaFree(S1_data);

     free(raw_data);
     free(C1_kernel);
     free(C1_data);
     free(S1_data);


    // Test sur plus petite matrice pour vérifier les valeurs

     float *raw_data2_g;
     float *C1_data2_g;
     float *C1_kernel2_g;
     float *S1_data2_g;


     cudaMalloc((void**)&raw_data2_g, 5*5*1*sizeof(float));
     cudaMalloc((void**)&C1_data2_g, 4*4*1*sizeof(float));
     cudaMalloc((void**)&S1_data2_g, 2*2*1*sizeof(float));
     cudaMalloc((void**)&C1_kernel2_g, 2*2*1*sizeof(float));

     cudaMemcpy(raw_data2_g, raw_data2, 5*5*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(C1_data2_g, C1_data2, 4*4*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(S1_data2_g, S1_data2, 2*2*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(C1_kernel2_g, C1_kernel2, 2*2*1*sizeof(float), cudaMemcpyHostToDevice);
     

     dim3 gridDim(1,1);
     dim3 blockDim(5,5);

     cudaConv2D<<<gridDim,blockDim>>>(raw_data2_g, C1_kernel2_g, C1_data2_g, 5, 5, 2, 1, 4, 4);
     cudaSubsampling<<<gridDim,blockDim>>>(C1_data2_g, S1_data2_g, 4, 4, 1, 2, 2, 2);
     cudaDeviceSynchronize();

    
     cudaMemcpy(C1_data2, C1_data2_g, 4*4*1*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(S1_data2, S1_data2_g, 2*2*1*sizeof(float), cudaMemcpyDeviceToHost);
     cudaDeviceSynchronize();


     printf("Matrice initiale:\n");
     MatrixPrint(raw_data2,5,5);
     printf("Matrice kernel:\n");
     MatrixPrint(C1_kernel2,2,2);
     printf("Matrice après convolution 2D:\n");
     MatrixPrint(C1_data2,4,4);
     printf("Matrice après sous-échantillonnage:\n");
     MatrixPrint(S1_data2,2,2);
     
     cudaFree(raw_data2_g);
     cudaFree(C1_kernel2);
     cudaFree(C1_data2);
     cudaFree(S1_data2);

     free(raw_data2);
     free(C1_kernel2);
     free(C1_data2);
     free(S1_data2);


    return 0;
    }







