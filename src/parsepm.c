#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned char* read_ppm(const char* filename, int* width, int* height) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    
    char format[3];
    fscanf(f, "%2s", format);
    if (strcmp(format, "P6") != 0) {
        fclose(f);
        return NULL;
    }
    
    // Skip comments
    int c;
    do {
        c = fgetc(f);
        if (c == '#') {
            // Consume entire comment line
            while ((c = fgetc(f)) != '\n' && c != EOF);
        } else {
            ungetc(c, f);  // Push back the non-# character
            break;
        }
    } while (1);
    
    fscanf(f, "%d %d", width, height);
    int maxval;
    fscanf(f, "%d", &maxval);
    fgetc(f); // skip one byte (newline after header)
    
    int size = (*width) * (*height) * 3;
    unsigned char* data = (unsigned char*)malloc(size);
    fread(data, 1, size, f);
    fclose(f);
    return data;
}

void savePGM(float* data, size_t width, size_t height, char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }
    
    // Write PGM header (P5 format)
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    
    // Convert float data back to uint8_t and write
    // Assuming your float data is already in [0, 255] range from casting unsigned char
    for (int i = 0; i < width * height; i++) {
        unsigned char pixel = (unsigned char)(data[i] + 0.5f); // round to nearest
        fwrite(&pixel, 1, 1, f);
    }
    
    fclose(f);
}
