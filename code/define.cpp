#include <stdio.h>

#define PI 3.1415
#define area(x) PI*x*x
main()
{
  double r = 2.6;
  printf("%.2f", area(r));
}
