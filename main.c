
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// Function to generate a Gaussian kernel matrix
float* generateGaussianKernel(int n, float sigma) {
    // Allocate memory for the kernel matrix
    float *kernel = (float*)malloc(n * n * sizeof(float));
    if (kernel == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    // Calculate center of the kernel
    int center = n / 2;

    // Calculate normalization factor
    float sum = 0.0f;

    // Generate kernel values based on Gaussian function
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int x = i - center;
            int y = j - center;
            kernel[i * n + j] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[i * n + j];
        }
    }

    // Normalize kernel
    for (int i = 0; i < n * n; ++i) {
        kernel[i] /= sum;
    }

    return kernel;
}

// Function to print kernel matrix (for debugging)
void printKernel(float* kernel, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.6f\t", kernel[i * n + j]);
        }
        printf("\n");
    }
}

// Function to apply Gaussian blur to an image
void gaussianBlur(float* image, int width, int height, float* kernel, int kernelSize) {
    // Create temporary image buffer for storing blurred image
    float* blurredImage = (float*)malloc(width * height * sizeof(float));
    if (blurredImage == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }

    // Convolve image with Gaussian kernel
    int kernelRadius = kernelSize / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
                for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        float pixelValue = image[ny * width + nx];
                        float kernelValue = kernel[(ky + kernelRadius) * kernelSize + (kx + kernelRadius)];
                        sum += pixelValue * kernelValue;
                    }
                }
            }
            blurredImage[y * width + x] = sum;
        }
    }

    // Copy blurred image back to original image buffer
    for (int i = 0; i < width * height; ++i) {
        image[i] = blurredImage[i];
    }

    // Free temporary image buffer
    free(blurredImage);
}

// Function to compute gradient magnitude using Sobel operators
void sobelGradient(float* image, float* gradient, int width, int height) {
    // Sobel operators for gradient calculation
    float sobel_x[] = { -1, 0, 1,
    					-2, 0, 2,
    					-1, 0, 1};
    float sobel_y[] = {	-1, -2, -1,
    					0,  0,  0,
    					1,  2,  1};
    // Compute gradient magnitude for each pixel
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float gx = 0, gy = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int index = (y + ky) * width + (x + kx);
                    gx += image[index] * sobel_x[(ky + 1) * 3 + (kx + 1)];
                    gy += image[index] * sobel_y[(ky + 1) * 3 + (kx + 1)];
                }
            }
            gradient[y * width + x] = sqrt(gx * gx + gy * gy);
        }
    }
}

// Function to perform non-maximum suppression
void nonMaximumSuppression(float* gradient, float* edges, int width, int height) {
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float angle = atan2(gradient[y * width + x + 1] - gradient[y * width + x - 1], gradient[(y + 1) * width + x] - gradient[(y - 1) * width + x]);
            float mag = gradient[y * width + x];

            // Determine neighboring pixels based on gradient direction
            float p1, p2;
            if ((angle >= -M_PI_4 && angle < M_PI_4) || (angle >= 3 * M_PI_4 || angle < -3 * M_PI_4)) {
                p1 = gradient[y * width + x + 1];
                p2 = gradient[y * width + x - 1];
            } else if ((angle >= M_PI_4 && angle < 3 * M_PI_4) || (angle >= -3 * M_PI_4 && angle < -M_PI_4)) {
                p1 = gradient[(y - 1) * width + x + 1];
                p2 = gradient[(y + 1) * width + x - 1];
            } else if ((angle >= 3 * M_PI_4 && angle < M_PI) || (angle >= -M_PI && angle < -3 * M_PI_4)) {
                p1 = gradient[(y - 1) * width + x];
                p2 = gradient[(y + 1) * width + x];
            } else {
                p1 = gradient[(y + 1) * width + x + 1];
                p2 = gradient[(y - 1) * width + x - 1];
            }

            // Suppress non-maximum pixels
            if (mag >= p1 && mag >= p2) {
                edges[y * width + x] = mag;
            } else {
                edges[y * width + x] = 0;
            }
        }
    }
}

// Function to perform thresholding for edge detection
void threshold(float* gradient, float* edges, int width, int height, float lowThreshold, float highThreshold) {
    // Apply thresholding to identify edges
    for (int i = 0; i < width * height; ++i) {
        if (gradient[i] >= highThreshold) {
            edges[i] = 255; // Strong edge
        } else if (gradient[i] >= lowThreshold) {
            edges[i] = 128; // Weak edge
        } else {
            edges[i] = 0; // Non-edge
        }
    }
}

// Function to perform hysteresis thresholding
void hysteresisThresholding(float* edges, int width, int height, float lowThreshold, float highThreshold) {
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            if (edges[y * width + x] == 128) { // Check if current pixel is a weak edge
                // Check neighboring pixels for strong edges
                for (int ny = -1; ny <= 1; ++ny) {
                    for (int nx = -1; nx <= 1; ++nx) {
                        if (edges[(y + ny) * width + (x + nx)] == 255) { // Check if neighbor is a strong edge
                            edges[y * width + x] = 255; // Convert weak edge to strong edge
                            break; // Exit loop once a strong edge is found
                        }
                    }
                    if (edges[y * width + x] == 255) { // Break outer loop if a strong edge is found
                        break;
                    }
                }
                // If no strong edge found among neighbors, suppress weak edge
                if (edges[y * width + x] == 128) {
                    edges[y * width + x] = 0; // Suppress weak edge
                }
            }
        }
    }
}

// Function to print image (for debugging)
void printImage(float* image, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%.1f\t", image[y * width + x]);
        }
        printf("\n");
    }
}

float * load_image(const char* filename, int w, int h) {
    // Load raw RGB888 image from file
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening image file.\n");
        return 0;
    }

    unsigned char *raw_image = (unsigned char *)malloc(w * h * 3); // RGB888 format
    if (!raw_image) {
        printf("Error allocating memory for raw image.\n");
        fclose(file);
        return 0;
    }

    fread(raw_image, sizeof(unsigned char), w * h * 3, file);
    fclose(file);

    // Convert raw RGB888 image to grayscale
    unsigned char *gray_image = (unsigned char *)malloc(w * h * sizeof(unsigned char));
    if (!gray_image) {
        printf("Error allocating memory for grayscale image.\n");
        free(raw_image);
        return 0;
    }

    for (int i = 0; i < w * h; ++i) {
        // Convert RGB to grayscale using luminance method
        gray_image[i] = (unsigned char)(0.299 * raw_image[3 * i] + 0.587 * raw_image[3 * i + 1] + 0.114 * raw_image[3 * i + 2]);
    }

    // Convert grayscale image to float buffer
    float *float_image = (float *)malloc(w * h * sizeof(float));
    if (!float_image) {
        printf("Error allocating memory for float image.\n");
        free(raw_image);
        free(gray_image);
        return 0;
    }

    for (int i = 0; i < w * h; ++i) {
        float_image[i] = (float)gray_image[i];
    }

    // Free memory
    free(raw_image);
    free(gray_image);

    return float_image;
}

void writeGrayscaleRawImage(const char* filename, float* image, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing.\n");
        return;
    }

    // Write grayscale raw image data
    for (int i = 0; i < width * height; ++i) {
        unsigned char pixel = (unsigned char)(255.0f - image[i]);// Scale float value to [0, 255]
        fwrite(&pixel, sizeof(unsigned char), 1, file);
        //printf(" %03u", pixel); if(i%30==0) printf("\n");
    }

    fclose(file);
}

int main(const int argc, const char ** const argv) {
    if (argc < 4) {
        printf("Usage: %s RGB24.raw w h\n", argv[0]);
        return 1;
    }

    /*
    int width = 5;
    int height = 5;
    // Example image data (grayscale)
    float image[] = {
        0.2, 0.4, 0.6, 0.8, 1.0,
        0.3, 0.5, 0.7, 0.9, 1.1,
        0.4, 0.6, 0.8, 1.0, 1.2,
        0.5, 0.7, 0.9, 1.1, 1.3,
        0.6, 0.8, 1.0, 1.2, 1.4
    };*/
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
	float* image = load_image(argv[1], width, height);
	if (image == NULL) {
		printf("image load error.\n");
		return 1;
	}
	clock_t start = clock();

	//>>>>>>>>>>>>>>caucassian blur
    // Kernel parameters
    int kernelSize = 3;//3; // Kernel size (odd number)
    float sigma = 1.0f; // Standard deviation
    // Generate Gaussian kernel
    float* kernel = generateGaussianKernel(kernelSize, sigma);
    if (kernel == NULL) {
        return 1;
    }
    // Apply Gaussian blur to image
    gaussianBlur(image, width, height, kernel, kernelSize);
    // Print blurred image (for debugging)
    //printImage(image, width, height);
    // Free allocated memory
    free(kernel);
	//writeGrayscaleRawImage("blur.raw", image, width, height);
	
    //>>>>>>>>>>>>>>>Gradient Calculation
    //Allocate memory for gradient and edges
    float *gradient = (float*)malloc(width * height * sizeof(float));
    float *edges = (float*)malloc(width * height * sizeof(float));
    if (gradient == NULL || edges == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }
    // Perform gradient calculation
    sobelGradient(image, gradient, width, height);

	//>>>>>>>>>>>>>>>Apply non-maximum suppression
	nonMaximumSuppression(gradient, edges, width, height);

    //>>>>>>>>>>>>>>>Apply double thresholding
    float lowThreshold = 50;//0.2;
    float highThreshold = 55.0;//0.8;
    threshold(gradient, edges, width, height, lowThreshold, highThreshold);

	//>>>>>>>>>>>>>>>Apply hysteresis
	hysteresisThresholding(edges, width, height, lowThreshold, highThreshold);
    // Print resulting edges (for debugging)
    //printImage(edges, width, height);

	//get cpu time
    printf("%f seconds to execute.\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);

	writeGrayscaleRawImage("canny.raw", edges, width, height);
    // Free allocated memory
    free(gradient);
    free(edges);
	free(image);

    return 0;
}
