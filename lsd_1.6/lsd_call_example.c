#include <stdio.h>
#include <stdlib.h>
#include "lsd.h"

#define FILENAME "D:\Google Drive\Tesla\papers\Line segment detection\lsd_1.6\usa_gray.bmp"

unsigned char* readBMP(char* filename)
{
    int i;
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);

    return data;
}

int main(void)
{
  double * image;
  double * out;
  int x,y,i,j,n;
  int X = 128;  /* x image size */
  int Y = 128;  /* y image size */

  /* create a simple image: left half black, right half gray */
  image = (double *) malloc( X * Y * sizeof(double) );
  if( image == NULL )
    {
      fprintf(stderr,"error: not enough memory\n");
      exit(EXIT_FAILURE);
    }
  for(x=0;x<X;x++)
    for(y=0;y<Y;y++)
      image[x+y*X] = x<X/2 ? 0.0 : 64.0; /* image(x,y) */


  /* LSD call */
  out = lsd(&n,image,X,Y);


  /* print output */
  printf("%d line segments found:\n",n);
  for(i=0;i<n;i++)
    {
      for(j=0;j<7;j++)
        printf("%f ",out[7*i+j]);
      printf("\n");
    }

  /* free memory */
  free( (void *) image );
  free( (void *) out );

  return EXIT_SUCCESS;
}
