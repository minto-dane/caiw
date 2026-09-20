/* C driver for the real norm_ctx — differential-twin of diff_norm.ml.
 * Input: [u32 aw][aw x u64 LE histogram]. Prints f[i] one per line
 * framed by 'B'/'E' markers so several cases can share one stream. */
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
    uint32_t aw = rd32(f);
    if (aw < 1 || aw > 65536) return 2;
    uint64_t *h = malloc((size_t)aw * 8);
    uint16_t *ft = malloc((size_t)aw * 2);
    if (!h || !ft) return 2;
    for (uint32_t i = 0; i < aw; i++) {
        unsigned char b[8];
        if (fread(b, 1, 8, f) != 8) return 2;
        h[i] = 0;
        for (int j = 0; j < 8; j++) h[i] |= (uint64_t)b[j] << (8 * j);
    }
    fclose(f);
    norm_ctx(h, ft, (int)aw);
    puts("B");
    for (uint32_t i = 0; i < aw; i++) printf("%u\n", ft[i]);
    puts("E");
    free(h); free(ft);
    return 0;
}

int main(int argc, char **argv) {
    int rc = 0;
    for (int i = 1; i < argc; i++)
        if (run1(argv[i])) rc = 2;
    return rc;
}
