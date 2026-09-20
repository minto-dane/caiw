/* Exhaustive C sweeps — the checked-in reproducer for the sweep claims in
 * AGENTS.md / docs/design.md.  Dynamic evidence (not proofs), but EXHAUSTIVE
 * over the stated domains, and deterministic (fixed-seed splitmix64):
 *
 *   1. rANS round trip — every f in [1,TOT] x every emit-boundary point of x.
 *      enc() emits a byte while x >= XMAX(f)=2^24*f, so for fixed f the
 *      emitted-byte count changes ONLY at x = XMAX(f)*256^k and at the
 *      domain edges: testing each boundary +-1 plus both extremes covers
 *      every byte-count class on the whole domain x in [LOWER, 256*LOWER).
 *      Slot c is sampled at its boundaries 0 / (TOT-f)/2 / TOT-f (the slot
 *      s = x%f + c can never reach TOT, so interior c are interchangeable).
 *      Asserts: x round-trips exactly, dec consumes EXACTLY the emitted
 *      bytes (rp == end), enc stays inside the scratch buffer.
 *
 *   2. dsym cell selection — 3,264 deterministic normalized tables
 *      (aw across {2,3,5,17,64,255,256,1024,4096,32768}, mixed composition
 *      shapes) x ALL v in [0,TOT): asserts cum[s] <= v < cum[s+1] and
 *      f[s] == *fc > 0 (the property dec() needs to never divide by zero).
 *
 * Exit 0 iff every case passes. */
#define main caiw_main_
#include "../caiw.c"
#undef main
#include <stdio.h>

static uint64_t rs;
static uint64_t nextr(void) {          /* splitmix64 — deterministic */
    uint64_t z = (rs += 0x9e3779b97f4a7c15ull);
    z = (z ^ z >> 30) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ z >> 27) * 0x94d049bb133111ebull;
    return z ^ z >> 31;
}

static int rans_sweep(void) {
    static uint8_t buf[64];
    uint64_t cases = 0, bad = 0;
    for (uint32_t f = 1; f <= TOT; f++) {
        uint64_t xm = XMAX(f);
        uint64_t xs[16]; int nx = 0;
        xs[nx++] = LOWER; xs[nx++] = LOWER + 1; xs[nx++] = (LOWER << 8) - 1;
        for (int k = 0; k < 3; k++) {           /* x = XMAX * 256^k ±1 */
            uint64_t b = xm << (8 * k);
            if (b > LOWER && b < (LOWER << 8)) {
                xs[nx++] = b - 1; xs[nx++] = b; xs[nx++] = b + 1;
            } else if (b == LOWER) xs[nx++] = b;   /* LOWER itself already in */
        }
        uint32_t cs[3] = {0, (TOT - f) / 2, TOT - f};
        for (int i = 0; i < nx; i++) for (int j = 0; j < 3; j++) {
            uint64_t x = xs[i]; uint32_t c = cs[j];
            uint8_t *pp = buf + sizeof buf;
            uint64_t x1 = enc(x, f, c, &pp);
            const uint8_t *rp = pp;
            uint64_t x2 = dec(x1, f, c, &rp, buf + sizeof buf);
            cases++;
            if (x2 != x || rp != buf + sizeof buf || pp < buf) {
                if (bad++ < 5)
                    fprintf(stderr, "rans FAIL f=%u c=%u x=%llu\n",
                            f, c, (unsigned long long)x);
            }
        }
    }
    printf("rans boundary sweep: %llu cases, %llu bad\n",
           (unsigned long long)cases, (unsigned long long)bad);
    return bad ? 1 : 0;
}

static int dsym_sweep(void) {
    static const int aws[] = {2, 3, 5, 17, 64, 255, 256, 1024, 4096, 32768};
    static uint16_t f[32768]; static uint32_t cum[32769];
    uint64_t tables = 0, sels = 0, bad = 0;
    for (int t = 0; t < 3264; t++) {
        int aw = aws[t % 10];
        uint32_t rem = TOT;
        int spike = (int)(nextr() % (uint64_t)aw);   /* case 2 hot cell */
        for (int i = 0; i < aw; i++) {
            uint32_t d;
            int left = aw - 1 - i;
            switch (t & 3) {                     /* deterministic shapes */
            case 0:                              /* random composition */
                d = left ? (uint32_t)(nextr() % (rem + 1)) : rem;
                break;
            case 1:                              /* uniform + noise */
                d = rem / (uint32_t)(left + 1);
                if (left && (nextr() & 1)) d += (uint32_t)(nextr() % (rem - d + 1)) / 4;
                if (d > rem || !left) d = rem;
                break;
            case 2:                              /* single spike + spread */
                d = (i == spike) ? rem : 0;
                break;
            default:                             /* sparse: few nonzero */
                d = (nextr() % 8) ? 0 : (left ? (uint32_t)(nextr() % (rem + 1)) : rem);
                if (!left) d = rem;
                break;
            }
            if (d > rem) d = rem;                /* composition safety */
            f[i] = (uint16_t)d;
            rem -= d;
        }
        cum[0] = 0;
        for (int i = 0; i < aw; i++) cum[i + 1] = cum[i] + f[i];
        tables++;
        for (uint32_t v = 0; v < TOT; v++) {
            uint32_t fc; uint32_t s = dsym(f, cum, aw, v, &fc);
            sels++;
            if (!(cum[s] <= v && v < cum[s + 1]) || fc != f[s] || !f[s]) {
                if (bad++ < 5)
                    fprintf(stderr, "dsym FAIL t=%d aw=%d v=%u s=%u\n",
                            t, aw, v, s);
            }
        }
    }
    printf("dsym sweep: %llu tables x %u values = %llu selections, %llu bad\n",
           (unsigned long long)tables, TOT,
           (unsigned long long)sels, (unsigned long long)bad);
    return bad ? 1 : 0;
}

int main(void) {
    rs = 0x243F6A8885A308D3ull;                  /* fixed seed */
    int rc = rans_sweep() | dsym_sweep();
    puts(rc ? "sweeps: FAIL" : "sweeps: all cases pass");
    return rc;
}
