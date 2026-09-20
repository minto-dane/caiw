/* C driver for real pack_enc+pack_dec — twin of diff_pack.ml.
 * Input: [u8 bsz][u64 n][n bytes data].
 * Prints: record bytes as hex ("-" if pack_enc rejects), then the
 * pack_dec round-trip output as hex. */
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

static void phex(const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%02x", p[i]);
    putchar('\n');
}

static int run1(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); return 2; }
    int bsz = fgetc(f);
    uint64_t n = rd64(f);
    if (bsz != 1 && bsz != 2) return 2;      /* model covers 1/2 only */
    if (n > (1 << 20)) return 2;             /* corpus scale */
    uint8_t *data = malloc(n ? n : 1);
    uint8_t *rec = malloc(4 + 512 + n * 2 + 64);
    uint8_t *back = malloc(n ? n : 1);
    if (!data || !rec || !back) return 2;
    if (fread(data, 1, n, f) != n) return 2;
    fclose(f);
    size_t rl = pack_enc(data, n, bsz, rec);
    if (!rl) { puts("-"); puts("-"); }
    else {
        phex(rec, rl);
        pack_dec(rec, rec + rl, n, bsz, back);
        phex(back, n);
    }
    free(data); free(rec); free(back);
    return 0;
}

int main(int argc, char **argv) {
    int rc = 0;
    for (int i = 1; i < argc; i++)
        if (run1(argv[i])) rc = 2;
    return rc;
}
