#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// TP HSP1 : Implémentation d'un CNN - LeNet-5 sur GPU


// Partie 3 - Un peu de Python

void MatrixInit(float *M, int n, int p, int m, int init){

    float random_value;

    if( init==0){
        for (int i=0; i<n*m*p; i++){ //n*p la taille de la matrice, on parcourt toute la matrice
            M[i] = 0;
        }
    }

    else {
        for (int i=0; i<n*p; i++){ //n*p la taille de la matrice, on parcourt toute la matrice
            random_value = ((float)rand() / (float)((RAND_MAX))*2-1);
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


//Addition de deux matrices sur GPU
__device__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int m){
    int i = threadIdx.x;//0
    int j = threadIdx.y;//0->399
    
    if (i<n && j<m){
        Mout[i*m+j]=M1[i*m+j]+M2[i*m+j];
    }

}



//Multiplication de deux matrices sur GPU
__device__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n, int p, int q){
    int i = threadIdx.x;//0
    int j = threadIdx.y;//0-399

    float sum=0.0;
    
    for(int k=0; k<p; k++){
        sum+=M1[i*p+k]*M2[k*q+j];
    }
    Mout[i*q+j]=sum;

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

        
            Mout[(i/subsampling_size)*Mout_nb_column+(j/subsampling_size)+k*Mout_nb_column*Mout_nb_raw]=average;
               
            
        }
    }
}

__device__ float* activation_tanh(float* M, int M_nb_raw, int M_nb_column, int M_nb_filtre){

    int i = threadIdx.x;
    int j = threadIdx.y;

    if(i<M_nb_raw && j<M_nb_column){

        for(int k=0;k<M_nb_filtre;k++){

            M[i*M_nb_column+j+k*M_nb_raw*M_nb_column]=tanh(M[i*M_nb_column+j+k*M_nb_raw*M_nb_column]);

        }
    }

    return M;

}

__device__ int activation_softmax(float* M, int M_nb_column){


    float max;
    int a;

    max=0.0;
    a=0;

        for(int k=0;k<M_nb_column;k++){

            if(M[k]>max){
                a=k;
            }

        }

    return a;

}

__global__ void cudaActivation_tanh(float* M,  int M_nb_raw, int M_nb_column, int M_nb_filtre){
    activation_tanh(M, M_nb_raw, M_nb_column, M_nb_filtre);
}

__global__ void cudaActivation_softmax(float* M, int M_nb_column){
    activation_softmax(M,M_nb_column);
}


__global__ void cudaDense(float* M, float* Mout, float* W, float* B, int M_nb_raw, int M_nb_column, int W_nb_column){
    float *Mout_bis;
    Mout_bis = (float*)malloc(M_nb_raw*W_nb_column*1*sizeof(float));
    cudaMatrixMult(M, W, Mout_bis, M_nb_raw, M_nb_column, W_nb_column);
    cudaMatrixAdd(Mout_bis, B, Mout, M_nb_raw, W_nb_column);
}



int main(){
    
//Initialisation des  matrices sur le CPU

    // Initialisation de nos matrices réelles

    //printf("Matrices initialisées : \n");

    float *raw_data;
    raw_data = (float*)malloc(32*32*1*sizeof(float));
    MatrixInit(raw_data,32,32,1,1);
    //MatrixPrint(raw_data,32,32);

    float *C1_data;
    C1_data = (float*)malloc(28*28*6*sizeof(float));
    MatrixInit(C1_data,28,28,6,0);
    //MatrixPrint(C1_data,28,28);

    float *S1_data;
    S1_data = (float*)malloc(14*14*6*sizeof(float));
    MatrixInit(S1_data,14,14,6,0);
    //MatrixPrint(S1_data,14,14);

    float *C1_kernel;
    C1_kernel = (float*)malloc(5*5*6*sizeof(float));
    MatrixInit(C1_kernel,5,5,6,1);
    //MatrixPrint(C1_kernel,5,5);

    float *C2_kernel;
    C2_kernel = (float*)malloc(5*5*16*sizeof(float));
    MatrixInit(C2_kernel,5,5,16,1);
    //MatrixPrint(C2_kernel,5,5);

    float *C2_data;
    C2_data = (float*)malloc(10*10*16*sizeof(float));
    MatrixInit(C2_data,10,10,16,0);
    //MatrixPrint(C1_data,10,10);

    float *S2_data;
    S2_data = (float*)malloc(5*5*16*sizeof(float));
    MatrixInit(S2_data,5,5,6,0);
    //MatrixPrint(S1_data,5,5);

    float *F1_data;
    F1_data = (float*)malloc(400*sizeof(float));
    MatrixInit(F1_data,1,400,1,0);
    //MatrixPrint(F1_data,1,400);

    float *W1_kernel;
    W1_kernel = (float*)malloc(400*120*sizeof(float));
    MatrixInit(W1_kernel,400,120,1,1);
    //MatrixPrint(W1_kernel,400,120);

    float *B1_kernel;
    B1_kernel = (float*)malloc(120*sizeof(float));
    MatrixInit(B1_kernel,1,120,1,1);
    //MatrixPrint(B1_kernel,1,120);

    float *D1_data;
    D1_data = (float*)malloc(120*sizeof(float));
    MatrixInit(D1_data,1,120,1,0);
    //MatrixPrint(D1_data,1,120);

    float *W2_kernel;
    W2_kernel = (float*)malloc(120*84*sizeof(float));
    MatrixInit(W2_kernel,120,84,1,1);
    //MatrixPrint(W2_kernel,120,84);

    float *B2_kernel;
    B2_kernel = (float*)malloc(84*sizeof(float));
    MatrixInit(B2_kernel,1,84,1,1);
    //MatrixPrint(B2_kernel,1,84);

    float *D2_data;
    D2_data = (float*)malloc(84*sizeof(float));
    MatrixInit(D2_data,1,84,1,0);
    //MatrixPrint(D2_data,1,84);



    float *W3_kernel;
    W3_kernel = (float*)malloc(84*10*sizeof(float));
    MatrixInit(W3_kernel,84,10,1,1);
    //MatrixPrint(W3_kernel,84,10);

    float *B3_kernel;
    B3_kernel = (float*)malloc(10*sizeof(float));
    MatrixInit(B3_kernel,1,10,1,1);
    //MatrixPrint(B3_kernel,1,10);

    float *D3_data;
    D3_data = (float*)malloc(10*sizeof(float));
    MatrixInit(D3_data,1,10,1,0);
    //MatrixPrint(D3_data,1,10);



// Tests de nos layers sur GPU

    // Tests sur nos matrices réelles

     float *raw_data_g;
     float *C1_data_g;
     float *S1_data_g;
     float *C1_kernel_g;
     float *C2_kernel_g;
     float *C2_data_g;
     float *S2_data_g;
     float *F1_data_g;
     float *W1_kernel_g;
     float *B1_kernel_g;
     float *D1_data_g;
     float *W2_kernel_g;
     float *B2_kernel_g;
     float *D2_data_g;
     float *W3_kernel_g;
     float *B3_kernel_g;
     float *D3_data_g;

     cudaMalloc((void**)&raw_data_g, 32*32*1*sizeof(float));
     cudaMalloc((void**)&C1_data_g, 28*28*6*sizeof(float));
     cudaMalloc((void**)&S1_data_g, 14*14*6*sizeof(float));
     cudaMalloc((void**)&C1_kernel_g, 5*5*6*sizeof(float));
     cudaMalloc((void**)&C2_kernel_g, 5*5*16*sizeof(float));
     cudaMalloc((void**)&C2_data_g, 10*10*16*sizeof(float));
     cudaMalloc((void**)&S2_data_g, 5*5*16*sizeof(float));
     cudaMalloc((void**)&B1_kernel_g, 1*120*1*sizeof(float));
     cudaMalloc((void**)&D1_data_g, 1*120*1*sizeof(float));
     cudaMalloc((void**)&W2_kernel_g, 120*84*1*sizeof(float));
     cudaMalloc((void**)&B2_kernel_g, 1*84*1*sizeof(float));
     cudaMalloc((void**)&D2_data_g, 1*84*1*sizeof(float));
     cudaMalloc((void**)&W3_kernel_g, 84*10*1*sizeof(float));
     cudaMalloc((void**)&B3_kernel_g, 1*10*1*sizeof(float));
     cudaMalloc((void**)&D3_data_g, 1*10*1*sizeof(float));
     
     cudaMemcpy(raw_data_g, raw_data, 32*32*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(C1_data_g, C1_data, 28*28*6*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(S1_data_g, S1_data, 14*14*6*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(C1_kernel_g, C1_kernel, 5*5*6*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(C2_kernel_g, C2_kernel, 5*5*16*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(C2_data_g, C2_data, 10*10*16*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(S2_data_g, S2_data, 5*5*16*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(F1_data_g, F1_data, 1*400*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(W1_kernel_g, W1_kernel, 400*120*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(B1_kernel_g, B1_kernel, 1*120*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(D1_data_g, D1_data, 1*120*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(W2_kernel_g, W2_kernel, 120*84*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(B2_kernel_g, B2_kernel, 1*84*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(D2_data_g, D2_data, 1*84*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(W3_kernel_g, W3_kernel, 84*10*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(B3_kernel_g, B3_kernel, 1*10*1*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(D3_data_g, D3_data, 1*10*1*sizeof(float), cudaMemcpyHostToDevice);

     dim3 gridDim1(1,1);
     dim3 blockDim1(28,28);

    
     // Application de la première convolution 2D
     cudaConv2D<<<gridDim1,blockDim1>>>(raw_data_g, C1_kernel_g, C1_data_g, 32, 32, 5, 6, 28, 28);
     cudaDeviceSynchronize();    

     // Application de la fonction d'activation Tanh
     cudaActivation_tanh<<<gridDim1,blockDim1>>>(C1_data_g, 28, 28, 6);
     cudaDeviceSynchronize();
     // Application du sous-échantillonnage
     cudaSubsampling<<<gridDim1,blockDim1>>>(C1_data_g, S1_data_g, 28, 28, 6, 2, 14, 14);
     cudaDeviceSynchronize();
     
      dim3 gridDim2(1,1);
     dim3 blockDim2(10,10);

     // Application de la deuxième convolution
     cudaConv2D<<<gridDim2,blockDim2>>>(S1_data_g, C2_kernel_g, C2_data_g, 14, 14, 5, 16, 10, 10);
     cudaDeviceSynchronize();
     // Application de la fonction d'activation Tanh
     cudaActivation_tanh<<<gridDim2,blockDim2>>>(C2_data_g, 10, 10, 16);
     cudaDeviceSynchronize();
     // Application du sous-échantillonnage
     cudaSubsampling<<<gridDim2,blockDim2>>>(C2_data_g, S2_data_g, 10, 10, 16, 2, 5, 5);
     cudaDeviceSynchronize();

     // Application de Flatten
     // Ici on sait que C2_data_g est comme flattend


     


     dim3 gridDim4(1,1);
     dim3 blockDim4(1,120*400);

     // Application de Dense1
     cudaDense<<<gridDim4,blockDim4>>>(S2_data_g, D1_data_g, W1_kernel_g, B1_kernel_g, 1, 400, 120);
     cudaDeviceSynchronize();

     
     dim3 gridDim5(1,1);
     dim3 blockDim5(1,120*84);

     //Application de la fonction d'activation Tanh
     cudaActivation_tanh<<<gridDim1,blockDim1>>>(D1_data_g, 1, 120, 1);
     cudaDeviceSynchronize();


     
     
     


     // Application de Dense2
     cudaDense<<<gridDim5,blockDim5>>>(D1_data_g, D2_data_g, W2_kernel_g, B2_kernel_g, 1, 120, 84);
     cudaDeviceSynchronize();

     dim3 gridDim6(1,1);
     dim3 blockDim6(1,84*10);

     //Application de la fonction d'activation Tanh
     cudaActivation_tanh<<<gridDim6,blockDim6>>>(D2_data_g, 1, 84, 1);
     cudaDeviceSynchronize();

     cudaMemcpy(C1_data, C1_data_g, 28*28*6*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(S1_data, S1_data_g, 14*14*6*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(C2_data, C2_data_g, 10*10*16*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(S2_data, S2_data_g, 5*5*16*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(D1_data, D1_data_g, 1*120*1*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(D2_data, D2_data_g, 1*84*1*sizeof(float), cudaMemcpyDeviceToHost);
     printf("Matrice après convolution 2 2D et fonction d'activation :\n");
     MatrixPrint(C2_data,10,10);
     printf("Matrice après sous-échantillonnage:\n");
     MatrixPrint(S2_data,5,5);
     printf("Matrice après sous-échantillonnage:\n");
     MatrixPrint(S2_data,5,5);
     printf("Matrice flatten:\n");
     MatrixPrint(S2_data,1,400);
     printf("Matrice après Dense 1:\n");
     MatrixPrint(D1_data,1,120);
     printf("Matrice après Dense 2:\n");
     MatrixPrint(D2_data,1,84);

     // Application de Dense3
     cudaDense<<<gridDim6,blockDim6>>>(D2_data_g, D3_data_g, W3_kernel_g, B3_kernel_g, 1, 84, 10);
     cudaDeviceSynchronize();
     //Application de la fonction d'activation Softmax
     cudaActivation_softmax<<<gridDim1,blockDim1>>>(D3_data_g, 10);
     cudaDeviceSynchronize();


     cudaMemcpy(C1_data, C1_data_g, 28*28*6*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(S1_data, S1_data_g, 14*14*6*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(C2_data, C2_data_g, 10*10*16*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(S2_data, S2_data_g, 5*5*16*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(F1_data, F1_data_g, 1*400*1*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(D1_data, D1_data_g, 1*120*1*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(D2_data, D2_data_g, 1*84*1*sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(D3_data, D3_data_g, 1*10*1*sizeof(float), cudaMemcpyDeviceToHost);
     cudaDeviceSynchronize();


     // Affichage des matrices après chaque layer 
     printf("Matrice initiale:\n");
     MatrixPrint(raw_data,32,32);
     printf("Matrice kernel:\n");
     MatrixPrint(C1_kernel,5,5);
     printf("Matrice après convolution 1 2D et fonction d'activation :\n");
     MatrixPrint(C1_data,28,28);
     printf("Matrice après sous-échantillonnage:\n");
     MatrixPrint(S1_data,14,14);
     printf("Matrice kernel 2:\n");
     MatrixPrint(C2_kernel,5,5);
     printf("Matrice après convolution 2 2D et fonction d'activation :\n");
     MatrixPrint(C2_data,10,10);
     printf("Matrice après sous-échantillonnage:\n");
     MatrixPrint(S2_data,5,5);
     printf("Matrice flatten:\n");
     MatrixPrint(S2_data,1,400);
     printf("Matrice kernel W1:\n");
     //MatrixPrint(W1_kernel,400,120);
     printf("Matrice kernel B1:\n");
     MatrixPrint(B1_kernel,1,120);
     printf("Matrice après Dense 1:\n");
     MatrixPrint(D1_data,1,120);
     printf("Matrice kernel W2:\n");
     /*MatrixPrint(W2_kernel,120,84);
     printf("Matrice kernel B2:\n");
     MatrixPrint(B2_kernel,1,84);
     printf("Matrice après Dense 2:\n");
     MatrixPrint(D2_data,1,84);
     printf("Matrice kernel W3:\n");
     MatrixPrint(W3_kernel,84,10);
     printf("Matrice kernel B3:\n");
     MatrixPrint(B3_kernel,1,10);
     printf("Matrice après Dense 3:\n");
     MatrixPrint(D3_data,1,10);
    */
     

     cudaFree(raw_data_g);
     cudaFree(C1_kernel_g);
     cudaFree(C1_data_g);
     cudaFree(S1_data_g);
     cudaFree(C2_data_g);
     cudaFree(S2_data_g);
     cudaFree(F1_data_g);
     cudaFree(W1_kernel_g);
     cudaFree(B1_kernel_g);
     cudaFree(D1_data_g);
     cudaFree(W2_kernel_g);
     cudaFree(B2_kernel_g);
     cudaFree(D2_data_g);
     cudaFree(W3_kernel_g);
     cudaFree(B3_kernel_g);
     cudaFree(D3_data_g);

     free(raw_data);
     free(C1_kernel);
     free(C1_data);
     free(S1_data);
     free(C2_data);
     free(S2_data);
     free(F1_data);
     free(W1_kernel);
     free(B1_kernel);
     free(D1_data);
     free(W2_kernel);
     free(B2_kernel);
     free(D2_data);
     free(W3_kernel);
     free(B3_kernel);
     free(D3_data);


    return 0;
    }

