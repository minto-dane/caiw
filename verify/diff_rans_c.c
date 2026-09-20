/* C driver for real rANS enc/dec — twin of diff_rans.ml.
 * Input: [u64 x][u64 f][u64 c] (LE; f,c truncated to u32 as in caiw.c).
 * Prints: x' after enc, emitted stream bytes (low->high address = the
 * order dec consumes them), then x'' after dec of the same stream. */
#define main caiw_main_
#include "../caiw.c"
#undef main
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static uint64_t rd64(FILE *f) {
    unsigned char b[8];
    if (fread(b, 1, 8, f) != 8) { fprintf(stderr, "short\n"); exit(2); }
    uint64_t v = 0;
    for (int j = 0; j < 8; j++) v |= (uint64_t)b[j] << (8 * j);
    return v;
}

static int run1(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); return 2; }
    uint64_t x = rd64(f);
    uint32_t fl = (uint32_t)rd64(f), c = (uint32_t)rd64(f);
    fclose(f);
    if (!fl) { puts("skip"); return 0; }   /* enc() dies on f==0 */
    uint8_t buf[64];
    uint8_t *pp = buf + 64;
    uint64_t x2 = enc(x, fl, c, &pp);
    printf("%llu\n", (unsigned long long)x2);
    for (const uint8_t *q = pp; q < buf + 64; q++) printf("%02x", *q);
    putchar('\n');
    const uint8_t *rp = pp;
    uint64_t x3 = dec(x2, fl, c, &rp, buf + 64);
    printf("%llu\n", (unsigned long long)x3);
    return 0;
}

int main(int argc, char **argv) {
    int rc = 0;
    for (int i = 1; i < argc; i++)
        if (run1(argv[i])) rc = 2;
    return rc;
}
