/* C driver for the real utf8_ok — differential-twin of diff_utf8.ml.
 * Input: [u32 n][bytes]. Prints utf8_ok verdict (0/1). */
#define main caiw_main_
#include "../caiw.c"
#undef main
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static uint32_t rd32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "short\n"); exit(2); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16)
           | ((uint32_t)b[3] << 24);
}

static int run1(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); return 2; }
    uint32_t n = rd32(f);
    char *s = malloc(n ? n : 1);
    if (!s) return 2;
    if (fread(s, 1, n, f) != n) return 2;
    fclose(f);
    printf("%d\n", utf8_ok(s, n));
    free(s);
    return 0;
}

int main(int argc, char **argv) {
    int rc = 0;
    for (int i = 1; i < argc; i++)
        if (run1(argv[i])) rc = 2;
    return rc;
}
