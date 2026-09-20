/* C driver for the real jkey — differential-twin of diff_jkey.ml.
 * Input file: [u32 olen][u32 rlen][u32 klen][obj][rest][key] (LE u32).
 * Buffer = obj ++ rest ++ NUL; jkey scans [buf, buf+olen) with the
 * NUL-driven helpers free to read into rest (exactly like st_load,
 * where the whole header is NUL-terminated).  Prints the returned
 * object-relative offset, or -1 on failure. */
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
    uint32_t olen = rd32(f), rlen = rd32(f), klen = rd32(f);
    char *buf = malloc((size_t)olen + rlen + 1);
    char *key = malloc((size_t)klen + 1);
    if (!buf || !key) return 2;
    if (fread(buf, 1, (size_t)olen + rlen, f) != (size_t)olen + rlen) return 2;
    if (fread(key, 1, klen, f) != klen) return 2;
    fclose(f);
    buf[olen + rlen] = 0;
    key[klen] = 0;
    const char *r = jkey(buf, buf + olen, key);
    if (!r) puts("-1"); else printf("%ld\n", (long)(r - buf));
    free(buf);
    free(key);
    return 0;
}

int main(int argc, char **argv) {
    int rc = 0;
    for (int i = 1; i < argc; i++)
        if (run1(argv[i])) rc = 2;
    return rc;
}
