/* Frama-C EVA + RTE harness — runtime-error / memory-safety analysis on
 * caiw's pure codec and parsing kernels over bounded nondeterministic
 * inputs.  Scope is deliberately the pointer/integer-dense functions;
 * filesystem, mmap and pthread paths need an OS model and are covered by
 * sanitizers + fuzzing instead (honest boundary — see docs/design.md). */
#ifdef __FRAMAC__
/* Frama-C's bundled libc lacks madvise/MADV_* — declare the hints so the
 * parse succeeds; eva_main never reaches mmap paths anyway. */
#ifndef MADV_SEQUENTIAL
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED 3
#define MADV_DONTNEED 4
#define MADV_HUGEPAGE 14
#endif
int madvise(void *addr, unsigned long length, int advice);
int Frama_C_interval(int min, int max);
#endif
#define main caiw_main_
#include "../caiw.c"
#undef main

int eva_main(void) {
    /* utf8_ok: length-bounded scan of arbitrary bytes */
    {
        unsigned char s[64];
        for (int i = 0; i < 64; i++) s[i] = (unsigned char)Frama_C_interval(0, 255);
        utf8_ok((char *)s, Frama_C_interval(0, 64));
    }
    /* JSON scanners on arbitrary NUL-terminated text */
    {
        unsigned char j[64];
        for (int i = 0; i < 63; i++) j[i] = (unsigned char)Frama_C_interval(0, 255);
        j[63] = 0;
        const char *e = jstr_skip((char *)j);
        if (e) jspan(e);
        jkey((char *)j, (char *)j + 63, "k");
    }
    /* hex4 escape reader */
    {
        unsigned char e[8];
        for (int i = 0; i < 7; i++) e[i] = (unsigned char)Frama_C_interval(0, 255);
        e[7] = 0;
        unsigned out;
        hex4((char *)e, &out);
    }
    /* norm_ctx: small alphabet, arbitrary counts (u64 path) */
    {
        uint64_t h[8]; uint16_t f[8];
        for (int i = 0; i < 8; i++) h[i] = (uint64_t)Frama_C_interval(0, 100000);
        norm_ctx(h, f, 8);
    }
    /* dsym on a well-formed normalized table */
    {
        uint16_t f[32]; uint32_t cum[33]; cum[0] = 0;
        for (int i = 0; i < 32; i++) {
            f[i] = (uint16_t)Frama_C_interval(0, 1024);
            cum[i + 1] = cum[i] + f[i];
        }
        cum[32] = TOT;
        uint32_t fc;
        dsym(f, cum, 32, (uint32_t)Frama_C_interval(0, TOT - 1), &fc);
    }
    /* rANS single-symbol enc/dec inside a scratch buffer */
    {
        uint8_t buf[64]; uint8_t *pp = buf + sizeof buf;
        uint64_t x = LOWER + (uint64_t)Frama_C_interval(0, (int)(255 * LOWER));
        uint32_t f = (uint32_t)Frama_C_interval(1, TOT);
        uint32_t c = (uint32_t)Frama_C_interval(0, (int)(TOT - f));
        uint64_t x1 = enc(x, f, c, &pp);
        const uint8_t *rp = pp;
        dec(x1, f, c, &rp, buf + sizeof buf);
    }
    /* kmap16 bijection cell */
    {
        uint16_t u = (uint16_t)Frama_C_interval(0, 65535);
        (void)kmap16_inv(kmap16(u));
    }
    /* pack decode on an arbitrary blob — exercises every guard path */
    {
        uint8_t blob[96]; uint8_t o[64];
        for (int i = 0; i < 96; i++) blob[i] = (uint8_t)Frama_C_interval(0, 255);
        pack_dec(blob, blob + sizeof blob,
                 (uint64_t)Frama_C_interval(0, 64), 1, o);
    }
    return 0;
}
