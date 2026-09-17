/* caiw: compressor for AI weights — lossless compressor for safetensors.
 *
 * Core: factored rANS over float fields with rolling block tables.
 * Each block's frequency table is derived identically on both sides from
 * the cumulative histogram of all previously coded data in that channel —
 * zero table bytes, self-adapting across tensors.
 *
 * methods (per tensor, chosen by real encoded size):
 *   RAW      verbatim
 *   PACK     dict + bitpack (restricted alphabets)
 *   REF      exact duplicate of an earlier tensor in this archive
 *   FIELD    S|E|M factored rANS (16-bit floats; f32 uses 5-field chain)
 *   FIELDPOS FIELD + exact/grouped column-position context on SE
 *   FIELDROW FIELD + exact/grouped row-position context on SE
 *   DELTA    K-residual vs earlier same-name/family tensor (f16/bf16, f32 xor planes)
 *   DELTAX   DELTA vs external --ref model; with several --ref files the one
 *            with lowest sampled escape rate is used (recorded in archive)
 *   F32      f32 5-field factored rANS
 *   U8       order-0 rANS over raw bytes (ints, packed formats, other)
 *
 * usage:
 *   caiw c out.caiw [--ref base.st ...] in1.st [in2.st ...]
 *   caiw d out.caiw outdir/ [--ref base.st ...]   (same --ref order as c)
 *   caiw v out.caiw [--ref base.st ...] in1.st [...]
 */
#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <pthread.h>

#define BS 17
#define BLK (1u << BS)
#define SB 15
#define TOT (1u << SB)
#define LOWER (1ull << 31)
#define SCRSZ (BLK * 16 + 64)   /* worst-case renorm bytes: 3B per enc call, <=5 calls/elem */
#define XMAX(f) ((((LOWER) << 8) / TOT) * (uint64_t)(f))

enum { M_RAW = 0, M_PACK, M_REF, M_FIELD, M_FIELDPOS, M_DELTA, M_U8, M_F32, M_FIELDROW, M_DELTAX };
#define MF_BAT 0x80   /* method-byte flag: parallel batch member (CAI4) */
#define MF_BH  0x40   /* batch head: first member opens a new batch */

static void die(const char *m) __attribute__((noreturn));
static void die(const char *m) { fprintf(stderr, "caiw: %s\n", m); exit(1); }
/* d-mode: remove partial outputs if we exit on any failure */
static char **g_outp; static int g_outn, g_outok;
static void out_cleanup(void) { if (!g_outok) for (int i = 0; i < g_outn; i++) unlink(g_outp[i]); }

/* ================= CRC32 (integrity) ================= */
static uint32_t crc_tab[256], crc_ready = 0;
static void crc_setup(void) {
    for (int i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_tab[i] = c;
    }
    crc_ready = 1;
}
static uint32_t crc32_of(const uint8_t *p, uint64_t n) {
    if (!crc_ready) crc_setup();
    uint32_t c = 0xFFFFFFFFu;
    while (n--) c = crc_tab[(c ^ *p++) & 255] ^ (c >> 8);
    return ~c;
}
static void *xm(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("oom"); return p; }
static void *xc(size_t a, size_t b) { void *p = calloc(a ? a : 1, b ? b : 1); if (!p) die("oom"); return p; }

static uint64_t fnv(const void *d, size_t n) {
    const uint8_t *p = d; uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

/* ================= safetensors ================= */
typedef struct Tensor {
    char *name, *dtype;
    int64_t shape[8]; int nd;
    uint64_t off, len;
    uint8_t *data;
    int file;
    uint64_t dataoff;           /* decode: output offset within file data region */
    int method; uint32_t ref; uint64_t plen; uint32_t crc;
    int bat;                    /* decoded: member of a parallel batch */
    struct Tensor *xreft;       /* external --ref match, or NULL */
    int xfile;                  /* which --ref file xreft came from */
    uint32_t candref[4]; int ncand; /* intra DELTA ref candidates */
} Tensor;

typedef struct { char *path; Tensor *t; int n; uint8_t *base; uint64_t bsz_; int mapped; } InFile;
static InFile **g_refs = NULL; static int g_nref = 0;

static const char *jget(const char *obj, const char *oend, const char *key, char *buf, int bs) {
    /* find key within [obj,oend) */
    size_t kl = strlen(key);
    const char *p = obj;
    while ((p = memchr(p, '"', oend - p))) {
        if ((size_t)(oend - p) > kl && !memcmp(p, key, kl)) break;
        p++;
    }
    if (!p || p >= oend) return 0;
    p = memchr(p + kl, ':', oend - p - kl);
    if (!p) return 0;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n') p++;
    int i = 0;
    if (*p == '"') { p++; while (*p && *p != '"' && i < bs - 1) buf[i++] = *p++; }
    else if (*p == '[') { while (*p && *p != ']' && i < bs - 1) buf[i++] = *p++; if (*p == ']' && i < bs - 1) buf[i++] = *p++; }
    else while (*p && *p != ',' && *p != '}' && *p != ']' && i < bs - 1) buf[i++] = *p++;
    buf[i] = 0;
    return buf;
}

static InFile *st_load(const char *path, int fidx) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open input");
    uint64_t hl;
    if (fread(&hl, 8, 1, f) != 1) die("hdr");
    if (hl > (1u << 30)) die("unreasonable header");   /* safetensors headers are small */
    char *hdr = xm(hl + 1);
    if (fread(hdr, 1, hl, f) != hl) die("hdr2");
    hdr[hl] = 0;
    int nt = 0;
    for (char *p = hdr; (p = strstr(p, "\"data_offsets\"")); p++) nt++;
    InFile *inf = xc(1, sizeof(InFile));
    inf->path = strdup(path);
    inf->t = xc(nt ? nt : 1, sizeof(Tensor));
    char *p = hdr, kb[600], vb[4200];
    while ((p = strchr(p, '"')) && inf->n < nt) {
        char *e = strchr(p + 1, '"');
        if (!e) break;
        int kl = e - p - 1;
        if (kl <= 0 || kl >= 590) { p = e + 1; continue; }
        memcpy(kb, p + 1, kl); kb[kl] = 0;
        char *np = e + 1;
        while (*np == ' ' || *np == '\t' || *np == '\n') np++;
        if (*np != ':') { p = np; continue; }
        np++;
        while (*np == ' ' || *np == '\t' || *np == '\n') np++;
        if (*np != '{') { p = np; continue; }
        char *ob = np;
        char *oend = strchr(ob, '}');   /* flat objects: first } ends it */
        if (!oend) break;
        if (!jget(ob, oend, "\"data_offsets\"", vb, sizeof vb)) { p = oend + 1; continue; }
        Tensor *t = &inf->t[inf->n];
        t->name = strdup(kb);
        t->file = fidx;
        char dbuf[64];
        t->dtype = jget(ob, oend, "\"dtype\"", dbuf, sizeof dbuf) ? strdup(dbuf) : strdup("?");
        char sbuf[600];
        t->nd = 0;
        if (jget(ob, oend, "\"shape\"", sbuf, sizeof sbuf)) {
            char *q = sbuf;
            while (*q && t->nd < 8) {
                while (*q && (*q < '0' || *q > '9')) q++;
                if (!*q) break;
                t->shape[t->nd++] = strtoll(q, &q, 10);
            }
        }
        uint64_t a = 0, b = 0;
        sscanf(vb, "[%llu,%llu]", (unsigned long long *)&a, (unsigned long long *)&b);
        if (b < a) die("bad data_offsets");
        t->off = a; t->len = b - a;
        inf->n++;
        p = oend + 1;
    }
    free(hdr);
    uint64_t dsz = 0;
    for (int i = 0; i < inf->n; i++) {
        if (inf->t[i].len > UINT64_MAX - inf->t[i].off) die("bad data_offsets");
        if (inf->t[i].off + inf->t[i].len > dsz) dsz = inf->t[i].off + inf->t[i].len;
    }
    /* data region must lie inside the file (else mmap hits SIGBUS).
       Compare by subtraction: 8+hl+dsz itself can wrap to a small value. */
    {
        struct stat st_;
        if (fstat(fileno(f), &st_) || (uint64_t)st_.st_size < 8 + hl ||
            dsz > (uint64_t)st_.st_size - 8 - hl)
            die("truncated input");
    }
    /* mmap data region (handles >RAM inputs); fall back to fread.
       NOTE: mmap offset must be page-aligned — map from the aligned
       boundary and shift base by the delta, else EINVAL always triggers
       the malloc+fread fallback and loads the whole file into RAM. */
    int fd = open(path, O_RDONLY);
    inf->mapped = 0; inf->bsz_ = dsz;
    if (fd >= 0) {
        uint64_t dstart = 8 + hl;
        off_t aoff = (off_t)(dstart & ~4095ull);
        void *m = mmap(NULL, dsz + (dstart - (uint64_t)aoff), PROT_READ, MAP_PRIVATE, fd, aoff);
        if (m != MAP_FAILED) {
            inf->base = (uint8_t *)m + (dstart - (uint64_t)aoff);
            inf->mapped = 1;
            madvise(m, dsz + (dstart - (uint64_t)aoff), MADV_SEQUENTIAL);
        }
    }
    if (!inf->base) {
        fseeko(f, 8 + hl, SEEK_SET);
        inf->base = xm(dsz ? dsz : 1);
        if (dsz && fread(inf->base, 1, dsz, f) != dsz) die("data");
    }
    fclose(f);
    if (fd >= 0) close(fd);
    for (int i = 0; i < inf->n; i++) inf->t[i].data = inf->base + inf->t[i].off;
    return inf;
}

static int dtb(const char *d) {
    if (!strcmp(d, "F64") || !strcmp(d, "I64") || !strcmp(d, "U64")) return 8;
    if (!strcmp(d, "F32") || !strcmp(d, "I32") || !strcmp(d, "U32")) return 4;
    if (!strcmp(d, "BF16") || !strcmp(d, "F16") || !strcmp(d, "I16") || !strcmp(d, "U16")) return 2;
    return 1;
}
static int is_bf(const char *d) { return !strcmp(d, "BF16"); }
static int is_f16(const char *d) { return !strcmp(d, "F16"); }
static int is_f32(const char *d) { return !strcmp(d, "F32"); }
static int is_flt16(const char *d) { return is_bf(d) || is_f16(d); }
static int mbits_of(const char *d) { return is_bf(d) ? 7 : 10; }

/* ================= rolling model =================
 * one cumulative histogram per channel; block tables derived by norm_ctx.
 * layouts depend on (ew,mw) -> separate channels per dtype family.
 */
#define CBF   0   /* bf16: [S:2][E|S:512][M|SE:512*128] */
#define CFP   1   /* f16 : [S:2][E|S:64][M|SE:64*1024] */
#define CBPOS 2   /* bf16 pos: [SE|pos:K*512][M|SE:512*128], K<=KCAP/512 */
#define CFPOS 3   /* f16 pos: [SE|pos:K*64][M|SE:64*1024],  K<=KCAP/64 */
#define C8    4   /* byte stream: [v:256] */
#define CD    5   /* delta residuals: [refctx:128][sym:4098] */
#define C32   6   /* f32: [S:2][E|S:512][m1|SE:512*128][m2|SE:512*256][m3|SE:512*256] */
#define C32D  7   /* f32 delta: 4 planes x [refE:256][v:256] */
#define NCH   8
#define PCAP (1 << 21)
static const int csz[NCH] = {
    2 + 512 + 512 * 128,
    2 + 64 + 64 * 1024,
    PCAP + 512 * 128,
    PCAP + 64 * 1024,
    256,
    128 * 4098,
    2 + 512 + 512 * 128 + 512 * 256 + 512 * 256,
    4 * 256 * 256,
};
static uint32_t *gch[NCH];                /* committed model state (main thread) */
static _Thread_local uint32_t **gh;       /* active channel array; NULL = unmaterialized */
static _Thread_local uint32_t **gsnp;     /* batch snapshot backing lazy clones (workers) */
static int g_threads = 1;

static void model_init(void) {
    for (int c = 0; c < NCH; c++) {
        gh[c] = xm(csz[c] * 4);
        for (int i = 0; i < csz[c]; i++) gh[c][i] = 1;
    }
}
/* channel access for writing: materializes a lazy worker clone from the
   batch snapshot on first touch. Main thread channels are always live. */
static uint32_t *H(int c) {
    uint32_t *p = gh[c];
    if (!p) { p = xm(csz[c] * 4); memcpy(p, gsnp[c], csz[c] * 4); gh[c] = p; }
    return p;
}
static uint32_t *hclone(int c) {
    const uint32_t *s = gh[c] ? gh[c] : gsnp[c];
    uint32_t *h = xm(csz[c] * 4); memcpy(h, s, csz[c] * 4); return h;
}
static void hcommit(int c, uint32_t *h) {
    if (!gh[c]) gh[c] = xm(csz[c] * 4);
    memcpy(gh[c], h, csz[c] * 4); free(h);
}

static void norm_ctx(const uint32_t *h, uint16_t *f, int aw) {
    uint64_t tot = 0; int nz = 0;
    for (int i = 0; i < aw; i++) { tot += h[i]; if (h[i]) nz++; }
    if (!nz) {
        uint16_t v = TOT / aw;
        for (int i = 0; i < aw; i++) f[i] = v;
        f[aw - 1] += TOT - (uint32_t)v * aw;
        return;
    }
    int64_t rem = TOT, bud = TOT - nz; int bi = 0;
    for (int i = 0; i < aw; i++) {
        if (h[i] > h[bi]) bi = i;
        uint32_t q = h[i] ? (uint32_t)((uint64_t)h[i] * bud / tot) + 1 : 0;
        f[i] = (uint16_t)q;
        rem -= q;
    }
    f[bi] += (uint32_t)rem;   /* rem>=0 by construction; deficit to dominant cell */
}

/* ================= rANS ================= */
static inline uint64_t enc(uint64_t x, uint32_t f, uint32_t c, uint8_t **pp) {
    while (x >= XMAX(f)) { *--(*pp) = (uint8_t)x; x >>= 8; }
    return ((x / f) << SB) + (x % f) + c;
}
static inline uint64_t dec(uint64_t x, uint32_t f, uint32_t c, const uint8_t **rp, const uint8_t *end) {
    x = (uint64_t)f * (x >> SB) + (x & (TOT - 1)) - c;
    while (x < LOWER && *rp < end) x = (x << 8) | *(*rp)++;
    return x;
}
static inline uint32_t dsym(const uint16_t *f, const uint32_t *cum, int aw, uint32_t v, uint32_t *fc) {
    int lo = 0, hi = aw - 1;
    while (lo < hi) { int m = (lo + hi) >> 1; if (v >= cum[m + 1]) lo = m + 1; else hi = m; }
    *fc = f[lo];
    return lo;
}
/* emit one rANS block: [u32 len][8B state][bytes] */
static uint8_t *emit_blk(uint8_t *o, uint8_t *scr_end, uint8_t *pp, uint64_t x) {
    for (int b = 0; b < 8; b++) o[4 + b] = (uint8_t)(x >> (8 * (7 - b)));
    uint32_t bl = (uint32_t)(scr_end - pp);
    memcpy(o, &bl, 4);
    memcpy(o + 12, pp, bl);
    return o + 12 + bl;
}
static const uint8_t *read_blk(const uint8_t *rp, const uint8_t *lim, uint64_t *x, const uint8_t **end) {
    if (rp > lim || (uint64_t)(lim - rp) < 12) die("corrupt archive");
    uint32_t bl; memcpy(&bl, rp, 4); rp += 4;
    *x = 0; for (int b = 0; b < 8; b++) *x = (*x << 8) | rp[b]; rp += 8;
    if ((uint64_t)bl > (uint64_t)(lim - rp)) die("corrupt archive");
    *end = rp + bl;
    return rp;
}

/* ============ FIELD 16-bit ============ */
static size_t f16_enc(const uint16_t *s, uint64_t n, int mb, uint8_t *out, uint32_t *h) {
    int ew = 1 << (15 - mb), mw = 1 << mb;
    uint8_t *o = out;
    uint16_t *ft = xm((2 + 2 * ew + (size_t)2 * ew * mw) * 2);
    uint32_t *cum = xm((4 + 2 * (ew + 1) + (size_t)2 * ew * (mw + 1)) * 4);
    uint8_t *scr = xm(SCRSZ);
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        uint16_t *fS = ft, *fE = ft + 2, *fM = ft + 2 + 2 * ew;
        norm_ctx(h, fS, 2);
        for (int c = 0; c < 2; c++) norm_ctx(h + 2 + c * ew, fE + c * ew, ew);
        for (int c = 0; c < 2 * ew; c++) norm_ctx(h + 2 + 2 * ew + (size_t)c * mw, fM + (size_t)c * mw, mw);
        uint32_t *cS = cum, *cE = cum + 4, *cM = cum + 4 + 2 * (ew + 1);
        cS[0] = 0; cS[1] = fS[0]; cS[2] = (uint32_t)fS[0] + fS[1];
        for (int c = 0; c < 2; c++) { uint32_t *b = cE + c * (ew + 1); b[0] = 0; for (int i = 0; i < ew; i++) b[i + 1] = b[i] + fE[c * ew + i]; }
        for (int c = 0; c < 2 * ew; c++) { uint32_t *b = cM + (size_t)c * (mw + 1); b[0] = 0; for (int i = 0; i < mw; i++) b[i + 1] = b[i] + fM[(size_t)c * mw + i]; }
        uint8_t *pp = scr + SCRSZ;
        uint64_t x = LOWER;
        for (uint64_t i = bn; i-- > 0;) {
            uint32_t v = s[b0 + i], S = v >> 15, E = (v >> mb) & (ew - 1), Mv = v & (mw - 1);
            size_t se = (size_t)S * ew + E;
            x = enc(x, fM[se * mw + Mv], cM[se * (mw + 1) + Mv], &pp);
            x = enc(x, fE[S * ew + E], cE[S * (ew + 1) + E], &pp);
            x = enc(x, fS[S], cS[S], &pp);
        }
        o = emit_blk(o, scr + SCRSZ, pp, x);
        for (uint64_t i = 0; i < bn; i++) {
            uint32_t v = s[b0 + i], S = v >> 15, E = (v >> mb) & (ew - 1), Mv = v & (mw - 1);
            h[S]++; h[2 + S * ew + E]++; h[2 + 2 * ew + (size_t)(S * ew + E) * mw + Mv]++;
        }
    }
    free(ft); free(cum); free(scr);
    return o - out;
}
static void f16_dec(const uint8_t *in, const uint8_t *lim, uint64_t n, int mb, uint16_t *s, uint32_t *h) {
    int ew = 1 << (15 - mb), mw = 1 << mb;
    const uint8_t *rp = in;
    uint16_t *ft = xm((2 + 2 * ew + (size_t)2 * ew * mw) * 2);
    uint32_t *cum = xc((size_t)2 + 2 * ew + 2 * ew * mw + 8, 4);
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        uint16_t *fS = ft, *fE = ft + 2, *fM = ft + 2 + 2 * ew;
        norm_ctx(h, fS, 2);
        for (int c = 0; c < 2; c++) norm_ctx(h + 2 + c * ew, fE + c * ew, ew);
        for (int c = 0; c < 2 * ew; c++) norm_ctx(h + 2 + 2 * ew + (size_t)c * mw, fM + (size_t)c * mw, mw);
        uint32_t cS1 = fS[0];
        uint32_t *cE = cum, *cM = cum + 2 * ew + 4;
        for (int c = 0; c < 2; c++) { cE[c * ew] = 0; for (int i = 0; i < ew; i++) cE[c * ew + i + 1] = cE[c * ew + i] + fE[c * ew + i]; }
        for (int c = 0; c < 2 * ew; c++) { size_t b = (size_t)c * mw; cM[b] = 0; for (int i = 0; i < mw; i++) cM[b + i + 1] = cM[b + i] + fM[b + i]; }
        uint64_t x; const uint8_t *end;
        rp = read_blk(rp, lim, &x, &end);
        for (uint64_t i = 0; i < bn; i++) {
            uint32_t v = (uint32_t)(x & (TOT - 1)), fc;
            uint32_t S = v >= cS1;
            x = dec(x, fS[S], S ? cS1 : 0, &rp, end);
            v = (uint32_t)(x & (TOT - 1));
            uint32_t E = dsym(fE + S * ew, cE + S * ew, ew, v, &fc);
            x = dec(x, fc, cE[S * ew + E], &rp, end);
            v = (uint32_t)(x & (TOT - 1));
            size_t mb2 = (size_t)(S * ew + E) * mw;
            uint32_t M = dsym(fM + mb2, cM + mb2, mw, v, &fc);
            x = dec(x, fc, cM[mb2 + M], &rp, end);
            s[b0 + i] = (uint16_t)((S << 15) | (E << mb) | M);
        }
        rp = end;
        for (uint64_t i = 0; i < bn; i++) {
            uint32_t v = s[b0 + i], S = v >> 15, E = (v >> mb) & (ew - 1), Mv = v & (mw - 1);
            h[S]++; h[2 + S * ew + E]++; h[2 + 2 * ew + (size_t)(S * ew + E) * mw + Mv]++;
        }
    }
    free(ft); free(cum);
}

/* ============ FIELD+POS: 16-bit, exact position contexts ============
 * SE joint | position ctx;  M | SE.
 * position p(i): colwise  p = i % P   (P = last dim)
 *                rowwise  p = i / D   (P = first dim, D = n/P)
 * K = min(P, KCAP) ctxs, ctx = p*K/P — exact column when K==P.
 * hist: [SE|pos: K*sew][M|SE: 2*ew*mw]
 */
static size_t pos_enc(const uint16_t *s, uint64_t n, int mb, int64_t P, int rowwise,
                      uint8_t *out, uint32_t *h) {
    if (P <= 0) return 0;   /* degenerate shape: not applicable */
    int ew = 1 << (15 - mb), mw = 1 << mb, sew = 2 * ew;
    int64_t K = P < (int64_t)(PCAP / sew) ? P : (int64_t)(PCAP / sew);
    int64_t D = n / P;
    if (D < 1) D = 1;
    uint8_t *o = out;
    size_t tSE = (size_t)K * sew, tM = (size_t)2 * ew * mw;
    uint16_t *ft = xm((tSE + tM) * 2);
    uint32_t *cum = xm(((size_t)K * (sew + 1) + (size_t)2 * ew * (mw + 1)) * 4);
    uint8_t *scr = xm(SCRSZ);
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        for (int64_t c = 0; c < K; c++) norm_ctx(h + c * sew, ft + c * sew, sew);
        for (int c = 0; c < 2 * ew; c++) norm_ctx(h + tSE + (size_t)c * mw, ft + tSE + (size_t)c * mw, mw);
        uint32_t *cSE = cum, *cM = cum + (size_t)K * (sew + 1);
        for (int64_t c = 0; c < K; c++) { uint32_t *b = cSE + c * (sew + 1); b[0] = 0; for (int i = 0; i < sew; i++) b[i + 1] = b[i] + ft[c * sew + i]; }
        for (int c = 0; c < 2 * ew; c++) { uint32_t *b = cM + (size_t)c * (mw + 1); b[0] = 0; for (int i = 0; i < mw; i++) b[i + 1] = b[i] + ft[tSE + (size_t)c * mw + i]; }
        uint8_t *pp = scr + SCRSZ;
        uint64_t x = LOWER;
        for (uint64_t i = bn; i-- > 0;) {
            uint64_t gi = b0 + i;
            int64_t po = rowwise ? (int64_t)(gi / D) : (int64_t)(gi % P);
            int64_t ctx = po * K / P;
            uint32_t v = s[gi];
            uint32_t SE = v >> mb, Mv = v & (mw - 1);
            x = enc(x, ft[tSE + (size_t)SE * mw + Mv], cM[(size_t)SE * (mw + 1) + Mv], &pp);
            x = enc(x, ft[ctx * sew + SE], cSE[ctx * (sew + 1) + SE], &pp);
        }
        o = emit_blk(o, scr + SCRSZ, pp, x);
        for (uint64_t i = 0; i < bn; i++) {
            uint64_t gi = b0 + i;
            int64_t po = rowwise ? (int64_t)(gi / D) : (int64_t)(gi % P);
            int64_t ctx = po * K / P;
            uint32_t v = s[gi];
            uint32_t SE = v >> mb, Mv = v & (mw - 1);
            h[ctx * sew + SE]++;
            h[tSE + (size_t)SE * mw + Mv]++;
        }
    }
    free(ft); free(cum); free(scr);
    return o - out;
}
static void pos_dec(const uint8_t *in, const uint8_t *lim, uint64_t n, int mb, int64_t P, int rowwise,
                    uint16_t *s, uint32_t *h) {
    if (P <= 0) die("bad shape");
    int ew = 1 << (15 - mb), mw = 1 << mb, sew = 2 * ew;
    int64_t K = P < (int64_t)(PCAP / sew) ? P : (int64_t)(PCAP / sew);
    int64_t D = n / P;
    if (D < 1) D = 1;
    const uint8_t *rp = in;
    size_t tSE = (size_t)K * sew, tM = (size_t)2 * ew * mw;
    uint16_t *ft = xm((tSE + tM) * 2);
    uint32_t *cum = xc(tSE + (size_t)K * sew + tM + 8, 4);
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        for (int64_t c = 0; c < K; c++) norm_ctx(h + c * sew, ft + c * sew, sew);
        for (int c = 0; c < 2 * ew; c++) norm_ctx(h + tSE + (size_t)c * mw, ft + tSE + (size_t)c * mw, mw);
        uint32_t *cSE = cum, *cM = cum + tSE + (size_t)K * sew;
        for (int64_t c = 0; c < K; c++) { cSE[c * sew] = 0; for (int i = 0; i < sew; i++) cSE[c * sew + i + 1] = cSE[c * sew + i] + ft[c * sew + i]; }
        for (int c = 0; c < 2 * ew; c++) { size_t b = (size_t)c * mw; cM[b] = 0; for (int i = 0; i < mw; i++) cM[b + i + 1] = cM[b + i] + ft[tSE + b + i]; }
        uint64_t x; const uint8_t *end;
        rp = read_blk(rp, lim, &x, &end);
        for (uint64_t i = 0; i < bn; i++) {
            uint64_t gi = b0 + i;
            int64_t po = rowwise ? (int64_t)(gi / D) : (int64_t)(gi % P);
            int64_t ctx = po * K / P;
            uint32_t v = (uint32_t)(x & (TOT - 1)), fc;
            uint32_t SE = dsym(ft + ctx * sew, cSE + ctx * sew, sew, v, &fc);
            x = dec(x, fc, cSE[ctx * sew + SE], &rp, end);
            v = (uint32_t)(x & (TOT - 1));
            size_t mb2 = (size_t)SE * mw;
            uint32_t M = dsym(ft + tSE + mb2, cM + mb2, mw, v, &fc);
            x = dec(x, fc, cM[mb2 + M], &rp, end);
            s[gi] = (uint16_t)((SE << mb) | M);
        }
        rp = end;
        for (uint64_t i = 0; i < bn; i++) {
            uint64_t gi = b0 + i;
            int64_t po = rowwise ? (int64_t)(gi / D) : (int64_t)(gi % P);
            int64_t ctx = po * K / P;
            uint32_t v = s[gi];
            uint32_t SE = v >> mb, Mv = v & (mw - 1);
            h[ctx * sew + SE]++;
            h[tSE + (size_t)SE * mw + Mv]++;
        }
    }
    free(ft); free(cum);
}

/* ============ F32: S|E|m1(7)|m2(8)|m3(8) ============ */
static size_t f32_enc(const uint32_t *s, uint64_t n, uint8_t *out, uint32_t *h) {
    uint8_t *o = out;
    size_t nM1 = 512 * 128, nM2 = 512 * 256, nM3 = 512 * 256;
    uint16_t *ft = xm((2 + 512 + nM1 + nM2 + nM3) * 2);
    uint32_t *cum = xm((4 + 2 * 257 + 512 * 129 + 2 * 512 * 257) * 4);
    uint8_t *scr = xm(SCRSZ);
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        norm_ctx(h, ft, 2);
        for (int c = 0; c < 2; c++) norm_ctx(h + 2 + c * 256, ft + 2 + c * 256, 256);
        uint16_t *f1 = ft + 2 + 512, *f2 = f1 + nM1, *f3 = f2 + nM2;
        for (int c = 0; c < 512; c++) norm_ctx(h + 2 + 512 + (size_t)c * 128, f1 + (size_t)c * 128, 128);
        for (int c = 0; c < 512; c++) norm_ctx(h + 2 + 512 + nM1 + (size_t)c * 256, f2 + (size_t)c * 256, 256);
        for (int c = 0; c < 512; c++) norm_ctx(h + 2 + 512 + nM1 + nM2 + (size_t)c * 256, f3 + (size_t)c * 256, 256);
        uint32_t *cS = cum, *cE = cum + 4, *c1 = cE + 2 * 257, *c2 = c1 + 512 * 129, *c3 = c2 + 512 * 257;
        cS[0] = 0; cS[1] = ft[0]; cS[2] = (uint32_t)ft[0] + ft[1];
        for (int c = 0; c < 2; c++) { uint32_t *b = cE + c * 257; b[0] = 0; for (int i = 0; i < 256; i++) b[i + 1] = b[i] + ft[2 + c * 256 + i]; }
        for (int c = 0; c < 512; c++) { uint32_t *b = c1 + c * 129; b[0] = 0; for (int i = 0; i < 128; i++) b[i + 1] = b[i] + f1[c * 128 + i]; }
        for (int c = 0; c < 512; c++) { uint32_t *b = c2 + c * 257; b[0] = 0; for (int i = 0; i < 256; i++) b[i + 1] = b[i] + f2[c * 256 + i]; }
        for (int c = 0; c < 512; c++) { uint32_t *b = c3 + c * 257; b[0] = 0; for (int i = 0; i < 256; i++) b[i + 1] = b[i] + f3[c * 256 + i]; }
        uint8_t *pp = scr + SCRSZ;
        uint64_t x = LOWER;
        for (uint64_t i = bn; i-- > 0;) {
            uint32_t v = s[b0 + i], S = v >> 31, E = (v >> 23) & 255;
            uint32_t m1 = (v >> 16) & 127, m2 = (v >> 8) & 255, m3 = v & 255;
            size_t se = S * 256 + E;
            x = enc(x, f3[se * 256 + m3], c3[se * 257 + m3], &pp);
            x = enc(x, f2[se * 256 + m2], c2[se * 257 + m2], &pp);
            x = enc(x, f1[se * 128 + m1], c1[se * 129 + m1], &pp);
            x = enc(x, ft[2 + S * 256 + E], cE[S * 257 + E], &pp);
            x = enc(x, ft[S], cS[S], &pp);
        }
        o = emit_blk(o, scr + SCRSZ, pp, x);
        for (uint64_t i = 0; i < bn; i++) {
            uint32_t v = s[b0 + i], S = v >> 31, E = (v >> 23) & 255;
            size_t se = S * 256 + E;
            h[S]++; h[2 + S * 256 + E]++;
            h[2 + 512 + se * 128 + ((v >> 16) & 127)]++;
            h[2 + 512 + nM1 + se * 256 + ((v >> 8) & 255)]++;
            h[2 + 512 + nM1 + nM2 + se * 256 + (v & 255)]++;
        }
    }
    free(ft); free(cum); free(scr);
    return o - out;
}
static void f32_dec(const uint8_t *in, const uint8_t *lim, uint64_t n, uint32_t *s, uint32_t *h) {
    const uint8_t *rp = in;
    size_t nM1 = 512 * 128, nM2 = 512 * 256, nM3 = 512 * 256;
    uint16_t *ft = xm((2 + 512 + nM1 + nM2 + nM3) * 2);
    uint32_t *cum = xc(2 + 512 + nM1 + nM2 + nM3 + 2048, 4);
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        norm_ctx(h, ft, 2);
        for (int c = 0; c < 2; c++) norm_ctx(h + 2 + c * 256, ft + 2 + c * 256, 256);
        uint16_t *f1 = ft + 2 + 512, *f2 = f1 + nM1, *f3 = f2 + nM2;
        for (int c = 0; c < 512; c++) norm_ctx(h + 2 + 512 + (size_t)c * 128, f1 + (size_t)c * 128, 128);
        for (int c = 0; c < 512; c++) norm_ctx(h + 2 + 512 + nM1 + (size_t)c * 256, f2 + (size_t)c * 256, 256);
        for (int c = 0; c < 512; c++) norm_ctx(h + 2 + 512 + nM1 + nM2 + (size_t)c * 256, f3 + (size_t)c * 256, 256);
        uint32_t cS1 = ft[0];
        uint32_t *cE = cum, *c1 = cum + 520, *c2 = c1 + nM1 + 512, *c3 = c2 + nM2 + 512;
        for (int c = 0; c < 2; c++) { cE[c * 256] = 0; for (int i = 0; i < 256; i++) cE[c * 256 + i + 1] = cE[c * 256 + i] + ft[2 + c * 256 + i]; }
        for (int c = 0; c < 512; c++) { size_t b = (size_t)c * 128; c1[b] = 0; for (int i = 0; i < 128; i++) c1[b + i + 1] = c1[b + i] + f1[b + i]; }
        for (int c = 0; c < 512; c++) { size_t b = (size_t)c * 256; c2[b] = 0; for (int i = 0; i < 256; i++) c2[b + i + 1] = c2[b + i] + f2[b + i]; }
        for (int c = 0; c < 512; c++) { size_t b = (size_t)c * 256; c3[b] = 0; for (int i = 0; i < 256; i++) c3[b + i + 1] = c3[b + i] + f3[b + i]; }
        uint64_t x; const uint8_t *end;
        rp = read_blk(rp, lim, &x, &end);
        for (uint64_t i = 0; i < bn; i++) {
            uint32_t v = (uint32_t)(x & (TOT - 1)), fc;
            uint32_t S = v >= cS1;
            x = dec(x, ft[S], S ? cS1 : 0, &rp, end);
            v = (uint32_t)(x & (TOT - 1));
            uint32_t E = dsym(ft + 2 + S * 256, cE + S * 256, 256, v, &fc);
            x = dec(x, fc, cE[S * 256 + E], &rp, end);
            size_t se = S * 256 + E;
            v = (uint32_t)(x & (TOT - 1));
            uint32_t m1 = dsym(f1 + se * 128, c1 + se * 128, 128, v, &fc);
            x = dec(x, fc, c1[se * 128 + m1], &rp, end);
            v = (uint32_t)(x & (TOT - 1));
            uint32_t m2 = dsym(f2 + se * 256, c2 + se * 256, 256, v, &fc);
            x = dec(x, fc, c2[se * 256 + m2], &rp, end);
            v = (uint32_t)(x & (TOT - 1));
            uint32_t m3 = dsym(f3 + se * 256, c3 + se * 256, 256, v, &fc);
            x = dec(x, fc, c3[se * 256 + m3], &rp, end);
            s[b0 + i] = (S << 31) | (E << 23) | (m1 << 16) | (m2 << 8) | m3;
        }
        rp = end;
        for (uint64_t i = 0; i < bn; i++) {
            uint32_t v = s[b0 + i], S = v >> 31, E = (v >> 23) & 255;
            size_t se = S * 256 + E;
            h[S]++; h[2 + S * 256 + E]++;
            h[2 + 512 + se * 128 + ((v >> 16) & 127)]++;
            h[2 + 512 + nM1 + se * 256 + ((v >> 8) & 255)]++;
            h[2 + 512 + nM1 + nM2 + se * 256 + (v & 255)]++;
        }
    }
    free(ft); free(cum);
}

/* ============ U8 ============ */
static size_t u8_enc(const uint8_t *s, uint64_t n, uint8_t *out, uint32_t *h) {
    uint8_t *o = out, *scr = xm(SCRSZ);
    uint16_t ft[256]; uint32_t cum[257];
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        norm_ctx(h, ft, 256);
        cum[0] = 0; for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + ft[i];
        uint8_t *pp = scr + SCRSZ;
        uint64_t x = LOWER;
        for (uint64_t i = bn; i-- > 0;)
            x = enc(x, ft[s[b0 + i]], cum[s[b0 + i]], &pp);
        o = emit_blk(o, scr + SCRSZ, pp, x);
        for (uint64_t i = 0; i < bn; i++) h[s[b0 + i]]++;
    }
    free(scr);
    return o - out;
}
static const uint8_t *u8_dec(const uint8_t *in, const uint8_t *lim, uint64_t n, uint8_t *s, uint32_t *h) {
    const uint8_t *rp = in;
    uint16_t ft[256]; uint32_t cum[257];
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        norm_ctx(h, ft, 256);
        cum[0] = 0; for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + ft[i];
        uint64_t x; const uint8_t *end;
        rp = read_blk(rp, lim, &x, &end);
        for (uint64_t i = 0; i < bn; i++) {
            uint32_t v = (uint32_t)(x & (TOT - 1)), fc;
            uint32_t sym = dsym(ft, cum, 256, v, &fc);
            x = dec(x, fc, cum[sym], &rp, end);
            s[b0 + i] = (uint8_t)sym;
        }
        rp = end;
        for (uint64_t i = 0; i < bn; i++) h[s[b0 + i]]++;
    }
    return rp;
}

/* ============ DELTA (16-bit K-residual) ============ */
static inline int64_t kmap16(uint16_t u) {
    return (u & 0x8000) ? (int64_t)(uint16_t)(~u) : (int64_t)(u ^ 0x8000);
}
static inline uint16_t kmap16_inv(int64_t k) {
    k &= 0xFFFF;
    return (k & 0x8000) ? (uint16_t)(k ^ 0x8000) : (uint16_t)(~k);
}
static inline int64_t kmap32(uint32_t u) {   /* ordering map for delta_ok pretest */
    return (u & 0x80000000u) ? (int64_t)(uint32_t)(~u) : (int64_t)(u ^ 0x80000000u);
}
#define DR 2048
#define DESC (2 * DR)
#define DSYMS (2 * DR + 1)
#define DCTX 128               /* residual ctx = top 7 bits of ref element */
static size_t dlt_enc(const uint16_t *cur, const uint16_t *ref, uint64_t n,
                      uint8_t *out, uint32_t *h, int mb, uint32_t *hesc) {
    uint8_t *o = out, *scr = xm(SCRSZ);
    uint16_t *ft = xm(DCTX * DSYMS * 2);
    uint32_t *dcum = xm(DCTX * (DSYMS + 1) * 4);
    uint16_t *sB = xm(BLK * 2); uint8_t *cB = xm(BLK), *used = xm(DCTX);
    uint64_t esc_cap = n / 8 + 1024, esc_n = 0;
    uint16_t *escbuf = xm(esc_cap * 2);
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        /* fused forward pass: sym + refctx per elem, used-ctx mask, escapes */
        memset(used, 0, DCTX);
        for (uint64_t i = 0; i < bn; i++) {
            int c = ref[b0 + i] >> 9;
            int64_t d = kmap16(cur[b0 + i]) - kmap16(ref[b0 + i]);
            uint32_t sym = (d < -DR || d >= DR) ? DESC : (uint32_t)(d + DR);
            cB[i] = (uint8_t)c; sB[i] = (uint16_t)sym; used[c] = 1;
            if (sym == DESC) {
                if (esc_n >= esc_cap) { esc_cap *= 2; escbuf = realloc(escbuf, esc_cap * 2); if (!escbuf) die("oom"); }
                escbuf[esc_n++] = cur[b0 + i];
            }
        }
        /* normalize only contexts present in this block (decoder derives the
           same mask from the ref tensor it already has) */
        for (int c = 0; c < DCTX; c++) if (used[c]) {
            norm_ctx(h + c * DSYMS, ft + c * DSYMS, DSYMS);
            uint32_t *cu = dcum + c * (DSYMS + 1);
            cu[0] = 0;
            for (int i = 0; i < DSYMS; i++) cu[i + 1] = cu[i] + ft[c * DSYMS + i];
        }
        uint8_t *pp = scr + SCRSZ;
        uint64_t x = LOWER;
        for (uint64_t i = bn; i-- > 0;) {
            uint32_t sym = sB[i]; int c = cB[i];
            x = enc(x, ft[c * DSYMS + sym], dcum[c * (DSYMS + 1) + sym], &pp);
        }
        o = emit_blk(o, scr + SCRSZ, pp, x);
        for (uint64_t i = 0; i < bn; i++) h[cB[i] * DSYMS + sB[i]]++;
    }
    memcpy(o, &esc_n, 8); o += 8;
    o += f16_enc(escbuf, esc_n, mb, o, hesc);  /* escapes through FIELD channel */
    free(scr); free(escbuf); free(ft); free(dcum); free(sB); free(cB); free(used);
    return o - out;
}
static void dlt_dec(const uint8_t *in, const uint8_t *lim, const uint16_t *ref, uint64_t n,
                    uint16_t *cur, uint32_t *h, int mb, uint32_t *hesc) {
    const uint8_t *rp = in;
    uint16_t *ft = xm(DCTX * DSYMS * 2);
    uint32_t *cum = xm(DCTX * (DSYMS + 1) * 4);
    uint8_t *used = xm(DCTX);
    /* first pass: decode symbols; record ESC positions */
    uint64_t *escpos = xm((n / 8 + 1024) * 8); uint64_t escn = 0, escap = n / 8 + 1024;
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        memset(used, 0, DCTX);
        for (uint64_t i = 0; i < bn; i++) used[ref[b0 + i] >> 9] = 1;
        for (int c = 0; c < DCTX; c++) if (used[c]) {
            norm_ctx(h + c * DSYMS, ft + c * DSYMS, DSYMS);
            uint32_t *cu = cum + c * (DSYMS + 1);
            cu[0] = 0;
            for (int i = 0; i < DSYMS; i++) cu[i + 1] = cu[i] + ft[c * DSYMS + i];
        }
        uint64_t x; const uint8_t *end;
        rp = read_blk(rp, lim, &x, &end);
        for (uint64_t i = 0; i < bn; i++) {
            uint32_t v = (uint32_t)(x & (TOT - 1)), fc;
            int c = ref[b0 + i] >> 9;
            uint32_t sym = dsym(ft + c * DSYMS, cum + c * (DSYMS + 1), DSYMS, v, &fc);
            x = dec(x, fc, cum[c * (DSYMS + 1) + sym], &rp, end);
            uint64_t gi = b0 + i;
            h[c * DSYMS + sym]++;
            if (sym == DESC) {
                if (escn >= escap) { escap *= 2; escpos = realloc(escpos, escap * 8); if (!escpos) die("oom"); }
                escpos[escn++] = gi;
                cur[gi] = 0;
            } else {
                int64_t k = kmap16(ref[gi]) + (int64_t)sym - DR;
                cur[gi] = kmap16_inv(k);
            }
        }
        rp = end;
    }
    uint64_t ne; memcpy(&ne, rp, 8); rp += 8;
    if (ne != escn) die("delta esc count mismatch");
    if (escn) {
        uint16_t *eb = xm(escn * 2);
        f16_dec(rp, lim, escn, mb, eb, hesc);   /* escapes from FIELD channel */
        for (uint64_t i = 0; i < escn; i++) cur[escpos[i]] = eb[i];
        free(eb);
    }
    free(escpos); free(ft); free(cum); free(used);
}

/* ============ DELTA32: xor byte planes (f32 pairs) ============
 * measured on real F32 checkpoint pairs: int32-ordered K-residuals are
 * near-uniform over any window (mantissa diffs are high-entropy); XOR
 * planes win (~16.5b vs ~18b/elem). */
#define D32CN (1u << 24)   /* elements per plane chunk */
/* each XOR plane coded conditioned on ref exponent byte (ref>>23):
 * ~0.6b/elem gain on real F32 checkpoint deltas */
static size_t dlt32_enc(const uint32_t *cur, const uint32_t *ref, uint64_t n,
                        uint8_t *out, uint32_t *h) {
    uint8_t *pl = xm(D32CN), *cb = xm(D32CN), *o = out, *scr = xm(SCRSZ);
    uint16_t *ft = xm(256 * 256 * 2); uint32_t *cum = xm(256 * 257 * 4);
    for (int p = 0; p < 4; p++) {
        int sh = p * 8;
        for (uint64_t c0 = 0; c0 < n; c0 += D32CN) {
            uint64_t cn = n - c0 < D32CN ? n - c0 : D32CN;
            for (uint64_t i = 0; i < cn; i++) {
                pl[i] = (uint8_t)((cur[c0 + i] ^ ref[c0 + i]) >> sh);
                cb[i] = (uint8_t)(ref[c0 + i] >> 23);
            }
            for (uint64_t b0 = 0; b0 < cn; b0 += BLK) {
                uint64_t bn = cn - b0 < BLK ? cn - b0 : BLK;
                uint8_t used[256] = {0};
                for (uint64_t i = 0; i < bn; i++) used[cb[b0 + i]] = 1;
                for (int c = 0; c < 256; c++) if (used[c]) {
                    norm_ctx(h + (p * 256 + c) * 256, ft + c * 256, 256);
                    uint32_t *cu = cum + c * 257;
                    cu[0] = 0;
                    for (int i = 0; i < 256; i++) cu[i + 1] = cu[i] + ft[c * 256 + i];
                }
                uint8_t *pp = scr + SCRSZ;
                uint64_t x = LOWER;
                for (uint64_t i = bn; i-- > 0;) {
                    int c = cb[b0 + i], sym = pl[b0 + i];
                    x = enc(x, ft[c * 256 + sym], cum[c * 257 + sym], &pp);
                }
                o = emit_blk(o, scr + SCRSZ, pp, x);
                for (uint64_t i = 0; i < bn; i++)
                    h[(p * 256 + cb[b0 + i]) * 256 + pl[b0 + i]]++;
            }
        }
    }
    free(pl); free(cb); free(scr); free(ft); free(cum);
    return o - out;
}
static void dlt32_dec(const uint8_t *in, const uint8_t *lim, const uint32_t *ref, uint64_t n,
                      uint32_t *cur, uint32_t *h) {
    uint8_t *pl = xm(BLK), *cb = xm(BLK);
    uint16_t *ft = xm(256 * 256 * 2); uint32_t *cum = xm(256 * 257 * 4);
    for (int p = 0; p < 4; p++) {
        int sh = p * 8;
        for (uint64_t c0 = 0; c0 < n; c0 += D32CN) {
            uint64_t cn = n - c0 < D32CN ? n - c0 : D32CN;
            for (uint64_t b0 = 0; b0 < cn; b0 += BLK) {
                uint64_t bn = cn - b0 < BLK ? cn - b0 : BLK;
                for (uint64_t i = 0; i < bn; i++) cb[i] = (uint8_t)(ref[c0 + b0 + i] >> 23);
                uint8_t used[256] = {0};
                for (uint64_t i = 0; i < bn; i++) used[cb[i]] = 1;
                for (int c = 0; c < 256; c++) if (used[c]) {
                    norm_ctx(h + (p * 256 + c) * 256, ft + c * 256, 256);
                    uint32_t *cu = cum + c * 257;
                    cu[0] = 0;
                    for (int i = 0; i < 256; i++) cu[i + 1] = cu[i] + ft[c * 256 + i];
                }
                uint64_t x; const uint8_t *end;
                in = read_blk(in, lim, &x, &end);
                for (uint64_t i = 0; i < bn; i++) {
                    uint32_t v = (uint32_t)(x & (TOT - 1)), fc;
                    int c = cb[i];
                    uint32_t sym = dsym(ft + c * 256, cum + c * 257, 256, v, &fc);
                    x = dec(x, fc, cum[c * 257 + sym], &in, end);
                    h[(p * 256 + c) * 256 + sym]++;
                    pl[i] = (uint8_t)sym;
                }
                in = end;
                if (p == 0)
                    for (uint64_t i = 0; i < bn; i++)
                        cur[c0 + b0 + i] = ref[c0 + b0 + i] ^ ((uint32_t)pl[i]);
                else
                    for (uint64_t i = 0; i < bn; i++)
                        cur[c0 + b0 + i] ^= ((uint32_t)pl[i]) << sh;
            }
        }
    }
    free(pl); free(cb); free(ft); free(cum);
}

/* ============ PACK (dict + bitpack) ============ */
static size_t pack_enc(const uint8_t *data, uint64_t n, int bsz, uint8_t *out) {
    /* gather unique elements (as bsz-byte atoms) */
    int aw;            /* hash space */
    if (bsz == 2) aw = 65536;
    else if (bsz == 1) aw = 256;
    else aw = 65536;   /* wider atoms: hash first 2 bytes — rare, fallback RAW anyway */
    int *cnt = xc(aw, 4);
    uint64_t ne = n / bsz;
    if (bsz == 2) { const uint16_t *s = (const uint16_t *)data; for (uint64_t i = 0; i < ne; i++) cnt[s[i]]++; }
    else { for (uint64_t i = 0; i < ne; i++) cnt[data[i]]++; }
    int k = 0;
    for (int i = 0; i < aw; i++) if (cnt[i]) k++;
    if (k < 1 || k > 256) { free(cnt); return 0; }
    int ib = 0; while ((1 << ib) < k) ib++;
    uint8_t *o = out;
    memcpy(o, &k, 4); o += 4;
    uint16_t dict[256] = { 0 }; int dk = 0;
    for (int i = 0; i < aw; i++) if (cnt[i]) dict[dk++] = (uint16_t)i;
    memcpy(o, dict, k * 2); o += k * 2;
    /* build reverse map */
    int *rmap = xc(aw, 4);
    for (int i = 0; i < k; i++) rmap[dict[i]] = i;
    /* bitpack ib-bit indices, MSB-first */
    uint64_t nbits = ne * ib, nbytes = (nbits + 7) / 8;
    memset(o, 0, nbytes);
    uint64_t bit = 0;
    for (uint64_t i = 0; i < ne; i++) {
        uint32_t idx = rmap[bsz == 2 ? ((const uint16_t *)data)[i] : data[i]];
        for (int b = ib - 1; b >= 0; b--) {
            if ((idx >> b) & 1) o[bit >> 3] |= 0x80 >> (bit & 7);
            bit++;
        }
    }
    o += nbytes;
    free(cnt); free(rmap);
    return o - out;
}
static void pack_dec(const uint8_t *in, const uint8_t *lim, uint64_t n, int bsz, uint8_t *data) {
    if (lim - in < 4) die("corrupt pack");
    int k; memcpy(&k, in, 4); in += 4;
    if (k < 1 || k > 256) die("corrupt pack");
    if ((uint64_t)(lim - in) < (uint64_t)k * 2) die("corrupt pack");
    uint16_t dict[256];
    memcpy(dict, in, k * 2); in += k * 2;
    int ib = 0; while ((1 << ib) < k) ib++;
    uint64_t ne = n / bsz, bit = 0;
    for (uint64_t i = 0; i < ne; i++) {
        uint32_t idx = 0;
        if (in + ((bit + ib - 1) >> 3) >= lim) die("corrupt pack");
        for (int b = 0; b < ib; b++) {
            idx = (idx << 1) | ((in[bit >> 3] >> (7 - (bit & 7))) & 1);
            bit++;
        }
        if (idx >= (uint32_t)k) die("corrupt pack");
        if (bsz == 2) ((uint16_t *)data)[i] = dict[idx];
        else data[i] = (uint8_t)dict[idx];
    }
}

/* ================= archive ================= */
/* archive integers are explicitly little-endian (portable format) */
static void w64(FILE *f, uint64_t v) { uint8_t b[8]; for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i)); fwrite(b, 8, 1, f); }
static void w32(FILE *f, uint32_t v) { uint8_t b[4]; for (int i = 0; i < 4; i++) b[i] = (uint8_t)(v >> (8 * i)); fwrite(b, 4, 1, f); }
static void w16(FILE *f, uint16_t v) { uint8_t b[2] = { (uint8_t)v, (uint8_t)(v >> 8) }; fwrite(b, 2, 1, f); }
static void w8(FILE *f, uint8_t v) { fwrite(&v, 1, 1, f); }
static uint64_t r64(const uint8_t **p) { uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)(*p)[i] << (8 * i); *p += 8; return v; }
static uint32_t r32(const uint8_t **p) { uint32_t v = 0; for (int i = 0; i < 4; i++) v |= (uint32_t)(*p)[i] << (8 * i); *p += 4; return v; }
static uint16_t r16(const uint8_t **p) { uint16_t v = (uint16_t)((*p)[0] | ((uint16_t)(*p)[1] << 8)); *p += 2; return v; }
static uint8_t r8(const uint8_t **p) { return *(*p)++; }

static void emit_rec(FILE *of, Tensor *t, int bat) {   /* bat: 0 solo, 1 head, 2 member */
    uint16_t nl = (uint16_t)strlen(t->name), dl = (uint16_t)strlen(t->dtype);
    w16(of, nl); fwrite(t->name, 1, nl, of);
    w16(of, dl); fwrite(t->dtype, 1, dl, of);
    w8(of, (uint8_t)t->nd);
    for (int d = 0; d < t->nd; d++) w64(of, (uint64_t)t->shape[d]);
    w16(of, (uint16_t)t->file);
    int m = t->method;
    if (m == M_DELTAX) t->ref = (uint32_t)t->xfile;
    w8(of, (uint8_t)(m | (bat ? MF_BAT | (bat == 1 ? MF_BH : 0) : 0)));
    w64(of, t->len);
    if (m == M_REF || m == M_DELTA || m == M_DELTAX) w32(of, t->ref);
    w64(of, t->plen);
    w32(of, crc32_of(t->data, t->len));
}

static int g_amap; /* set by slurp when archive buffer is mmap-backed */

/* mmap a whole file read-only; falls back to malloc+fread */
static uint8_t *slurp(const char *path, uint64_t *sz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("in");
    struct stat st;
    if (fstat(fd, &st)) die("stat");
    *sz = (uint64_t)st.st_size;
    void *m = mmap(NULL, *sz ? *sz : 1, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m != MAP_FAILED) { close(fd); g_amap = 1; madvise(m, *sz, MADV_SEQUENTIAL); return m; }
    uint8_t *b = xm(*sz ? *sz : 1);
    if (pread(fd, b, *sz, 0) != (ssize_t)*sz) die("read");
    close(fd);
    return b;
}

/* best-effort release of file-backed pages already consumed (keeps RSS low
   on >RAM workloads; re-access simply re-faults from the file) */
static void drop_pages(const void *p, uint64_t n, int mapped) {
    if (!mapped || !p || n < (1u << 20)) return;
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)4095;
    madvise((void *)a, n + ((uintptr_t)p - a), MADV_DONTNEED);
}

static const char *bname(const char *p) {
    const char *s = strrchr(p, '/');
    return s ? s + 1 : p;
}

/* digit runs -> '#': groups tensors into families (layers.N.*, experts.N.*) */
static char *nnorm(const char *s) {
    char *r = xm(strlen(s) + 1), *d = r;
    while (*s) {
        if (*s >= '0' && *s <= '9') { *d++ = '#'; while (*s >= '0' && *s <= '9') s++; }
        else *d++ = *s++;
    }
    *d = 0;
    return r;
}

/* typed element access needs natural alignment, but safetensors data
   regions can sit at odd file offsets: hand out an aligned copy then */
static uint8_t *alview(const uint8_t *d, uint64_t n, int w, uint8_t **hold) {
    *hold = 0;
    if (w > 1 && n && ((uintptr_t)d & (uintptr_t)(w - 1))) {
        *hold = xm(n); memcpy(*hold, d, n);
        return *hold;
    }
    return (uint8_t *)d;
}

/* cheap pretest for cross-tensor refs: sample K-residual escape rate.
   returns 1 if a DELTA trial encode is likely to pay off. */
static int delta_ok(const uint8_t *cur, const uint8_t *ref, uint64_t nbytes, int bsz) {
    uint64_t n = nbytes / bsz;
    if (!n) return 0;
    uint64_t S = n < 65536 ? n : 65536, step = n / S, esc = 0;
    for (uint64_t i = 0; i < n; i += step) {
        int64_t d;
        if (bsz == 2) {
            uint16_t cv, rv;
            memcpy(&cv, cur + i * 2, 2); memcpy(&rv, ref + i * 2, 2);
            d = kmap16(cv) - kmap16(rv);
        } else {
            uint32_t cv, rv;
            memcpy(&cv, cur + i * 4, 4); memcpy(&rv, ref + i * 4, 4);
            d = kmap32(cv) - kmap32(rv);
        }
        esc += (d < -DR || d >= DR);
    }
    return esc * 4 < (n / step) * 3;   /* <75% escapes => worth a trial encode */
}

/* try to encode tensor into buf; returns size or 0 if inapplicable.
   channel hist clones written to hout[0..1] with channel ids in
   chout[0..1] (-1 = unused); caller commits or frees each.
   ri: intra-archive ref index for M_DELTA. */
static uint64_t try_method(int m, Tensor *t, Tensor *all, uint8_t *scr, uint32_t **hout, int *chout, uint32_t ri) {
    hout[0] = 0; hout[1] = 0; chout[0] = -1; chout[1] = -1;
    int bsz = dtb(t->dtype);
    uint64_t ne = bsz ? t->len / bsz : 0;
    switch (m) {
    case M_RAW:
        memcpy(scr, t->data, t->len);
        return t->len;
    case M_PACK:
        if (bsz <= 2 && ne) return pack_enc(t->data, t->len, bsz, scr);
        return 0;
    case M_FIELD: {
        if (!is_flt16(t->dtype) || !ne) return 0;
        int c = is_bf(t->dtype) ? CBF : CFP;
        uint32_t *h = hclone(c);
        uint64_t sz = f16_enc((uint16_t *)t->data, ne, mbits_of(t->dtype), scr, h);
        hout[0] = h; chout[0] = c;
        return sz;
    }
    case M_FIELDPOS: {
        if (!is_flt16(t->dtype) || !ne || t->nd < 1) return 0;
        int c = is_bf(t->dtype) ? CBPOS : CFPOS;
        uint32_t *h = hclone(c);
        uint64_t sz = pos_enc((uint16_t *)t->data, ne, mbits_of(t->dtype), t->shape[t->nd - 1], 0, scr, h);
        hout[0] = h; chout[0] = c;
        return sz;
    }
    case M_FIELDROW: {
        if (!is_flt16(t->dtype) || !ne || t->nd < 2) return 0;
        int c = is_bf(t->dtype) ? CBPOS : CFPOS;
        uint32_t *h = hclone(c);
        uint64_t sz = pos_enc((uint16_t *)t->data, ne, mbits_of(t->dtype), t->shape[0], 1, scr, h);
        hout[0] = h; chout[0] = c;
        return sz;
    }
    case M_F32: {
        if (!is_f32(t->dtype)) return 0;
        uint32_t *h = hclone(C32);
        uint64_t sz = f32_enc((uint32_t *)t->data, ne, scr, h);
        hout[0] = h; chout[0] = C32;
        return sz;
    }
    case M_U8: {
        if (!t->len) return 0;
        uint32_t *h = hclone(C8);
        uint64_t sz = u8_enc(t->data, t->len, scr, h);
        hout[0] = h; chout[0] = C8;
        return sz;
    }
    case M_DELTA: {
        if (!ne || ri == 0xFFFFFFFFu) return 0;
        Tensor *r = &all[ri];
        uint8_t *rh = 0;
        const uint8_t *rd = alview(r->data, r->len, bsz, &rh);
        if (is_f32(t->dtype)) {
            uint32_t *h = hclone(C32D);
            uint64_t sz = dlt32_enc((const uint32_t *)t->data, (const uint32_t *)rd, ne, scr, h);
            free(rh); hout[0] = h; chout[0] = C32D;
            return sz;
        }
        if (!is_flt16(t->dtype)) { free(rh); return 0; }
        {
            int ec = is_bf(t->dtype) ? CBF : CFP;
            uint32_t *h = hclone(CD), *h2 = hclone(ec);
            uint64_t sz = dlt_enc((const uint16_t *)t->data, (const uint16_t *)rd, ne, scr, h, mbits_of(t->dtype), h2);
            free(rh); hout[0] = h; chout[0] = CD; hout[1] = h2; chout[1] = ec;
            return sz;
        }
    }
    case M_DELTAX: {
        if (!ne || !t->xreft) return 0;
        uint8_t *rh = 0;
        const uint8_t *rd = alview(t->xreft->data, t->xreft->len, bsz, &rh);
        if (is_f32(t->dtype)) {
            uint32_t *h = hclone(C32D);
            uint64_t sz = dlt32_enc((const uint32_t *)t->data, (const uint32_t *)rd, ne, scr, h);
            free(rh); hout[0] = h; chout[0] = C32D;
            return sz;
        }
        if (!is_flt16(t->dtype)) { free(rh); return 0; }
        {
            int ec = is_bf(t->dtype) ? CBF : CFP;
            uint32_t *h = hclone(CD), *h2 = hclone(ec);
            uint64_t sz = dlt_enc((const uint16_t *)t->data, (const uint16_t *)rd, ne, scr, h, mbits_of(t->dtype), h2);
            free(rh); hout[0] = h; chout[0] = CD; hout[1] = h2; chout[1] = ec;
            return sz;
        }
    }
    }
    return 0;
}

/* cheap pre-check: estimate H(SE) - H(SE|pos) on a sample; used to prune
 * FIELDPOS/FIELDROW candidates so ordinary matrices skip the slow encodes */
static double pos_gain(const uint16_t *s, uint64_t n, int mb, int64_t P, int rowwise) {
    if (P <= 0) return 0;   /* degenerate shape: no positional gain */
    int ew = 1 << (15 - mb), sew = 2 * ew;
    int64_t K = P < (int64_t)(PCAP / sew) ? P : (int64_t)(PCAP / sew);
    int64_t D = n / P;
    if (D < 1) D = 1;
    uint64_t ns = n < (4u << 20) ? n : (4u << 20);
    uint64_t *jh = xc((size_t)K * sew, 8), *mh = xc(sew, 8);
    for (uint64_t i = 0; i < ns; i++) {
        int64_t po = rowwise ? (int64_t)(i / D) : (int64_t)(i % P);
        int64_t ctx = po * K / P;
        uint32_t SE = s[i] >> mb;
        jh[ctx * sew + SE]++; mh[SE]++;
    }
    double H0 = 0, H1 = 0;
    for (int i = 0; i < sew; i++) if (mh[i]) { double p = (double)mh[i] / ns; H0 -= p * log2(p); }
    for (int64_t c = 0; c < K; c++) {
        uint64_t tot = 0;
        for (int i = 0; i < sew; i++) tot += jh[c * sew + i];
        if (!tot) continue;
        for (int i = 0; i < sew; i++)
            if (jh[c * sew + i]) { double p = (double)jh[c * sew + i] / tot; H1 -= ((double)tot / ns) * p * log2(p); }
    }
    free(jh); free(mh);
    return H0 - H1;
}

static Tensor *ref_find(const char *name) {
    for (int r = 0; r < g_nref; r++)
        for (int j = 0; j < g_refs[r]->n; j++)
            if (!strcmp(g_refs[r]->t[j].name, name)) return &g_refs[r]->t[j];
    return NULL;
}

/* best of several external refs sharing the same tensor name:
 * pick the one with the lowest sampled K-residual escape rate. */
static Tensor *ref_find_best(Tensor *t, int *ri) {
    Tensor *best = NULL;
    uint64_t bestsc = ~0ull;
    for (int r = 0; r < g_nref; r++)
        for (int j = 0; j < g_refs[r]->n; j++) {
            Tensor *xr = &g_refs[r]->t[j];
            if (strcmp(xr->name, t->name) || xr->len != t->len || strcmp(xr->dtype, t->dtype)) continue;
            uint64_t n = t->len / dtb(t->dtype);
            if (!n) continue;
            uint64_t S = n < 32768 ? n : 32768, st = n / S, esc = 0;
            if (dtb(t->dtype) == 2) {
                for (uint64_t i = 0; i < n; i += st) {
                    uint16_t cv, qv;
                    memcpy(&cv, t->data + i * 2, 2); memcpy(&qv, xr->data + i * 2, 2);
                    int64_t d = kmap16(cv) - kmap16(qv); esc += (d < -DR || d >= DR);
                }
            } else if (dtb(t->dtype) == 4) {
                for (uint64_t i = 0; i < n; i += st) {
                    uint32_t cv, qv;
                    memcpy(&cv, t->data + i * 4, 4); memcpy(&qv, xr->data + i * 4, 4);
                    int64_t d = kmap32(cv) - kmap32(qv); esc += (d < -DR || d >= DR);
                }
            } else esc = 0;
            if (!best || esc < bestsc) { best = xr; bestsc = esc; *ri = r; }
        }
    return best;
}

/* decode-side: resolve DELTAX tensor, preferring the recorded ref file index */
static Tensor *ref_resolve(Tensor *t) {
    if (t->ref != 0xFFFFFFFFu && t->ref < (uint32_t)g_nref) {
        InFile *f = g_refs[t->ref];
        for (int j = 0; j < f->n; j++)
            if (!strcmp(f->t[j].name, t->name) && f->t[j].len == t->len &&
                !strcmp(f->t[j].dtype, t->dtype)) return &f->t[j];
    }
    for (int r = 0; r < g_nref; r++)
        for (int j = 0; j < g_refs[r]->n; j++)
            if (!strcmp(g_refs[r]->t[j].name, t->name) &&
                g_refs[r]->t[j].len == t->len &&
                !strcmp(g_refs[r]->t[j].dtype, t->dtype)) return &g_refs[r]->t[j];
    return NULL;
}

/* ================= threading =================
 * Two levels, both exact:
 *  - parallel candidate tries inside one tensor: each try clones the same
 *    starting hist state, encodes deterministically -> byte-identical to
 *    sequential competition; only the winner's hist is committed.
 *  - parallel batches of whole tensors (method byte | MF_BAT): members
 *    encode/decode against the model state AT BATCH START. Histogram
 *    updates are pure symbol counts, so merging each member's delta into
 *    the committed state reproduces the sequential end state exactly.
 *    Members may only use refs in already-committed (earlier) ranges —
 *    in-batch ref candidates are dropped before competition.
 */
#define BCAP ((uint64_t)1 << 30)          /* max input bytes per batch */
#define TRYBUD ((uint64_t)3 << 30)        /* scratch budget for parallel tries */

typedef struct {
    int m; Tensor *t; Tensor *all; uint32_t ri;
    uint32_t **ghs, **snp;               /* owner channel state (read-only here) */
    uint64_t sz; uint8_t *buf;
    uint32_t *ho[2]; int ch[2];
} TJob;
/* worst-case payload for method m on a len-byte tensor. Every enc() call
   emits <=2B (x<2^39 vs XMAX>=2^24). Per-element call counts: FIELD 3
   (=>3*len), POS 2, DELTA16 1 call + escapes recoded through f16_enc
   (<=3*2B/elem => sym len + esc 3*len = 4*len), DELTA32 4 planes
   (=>2*len), F32 5 (=>2.5*len), U8 1 (=>2*len), PACK <=len+dict. Plus
   <=12B per emitted block and small fixed tails (esc_n, dict).
   4*len + len/1024 covers every method incl. DELTA's escape channel. */
static uint64_t ebound(uint64_t len) {
    return 4 * len + (len >> 10) + (64u << 20);
}
static void *tjob_run(void *a) {
    TJob *j = a;
    gh = j->ghs; gsnp = j->snp;
    j->buf = xm(ebound(j->t->len));
    j->sz = try_method(j->m, j->t, j->all, j->buf, j->ho, j->ch, j->ri);
    return 0;
}

/* compete all applicable methods for tensor t; sets method/plen/ref and
   returns the winner payload in *outp (NULL => RAW: write t->data, or
   REF: no payload). refcut forbids intra refs >= refcut. */
static void compete(Tensor *t, Tensor *all, uint32_t refcut, uint8_t **outp, int ntry) {
    *outp = 0;
    t->method = M_RAW; t->plen = t->len;
    uint8_t *ahold = 0, *aorig = t->data;
    t->data = alview(aorig, t->len, dtb(t->dtype), &ahold);
    uint32_t dup = t->ref;
    t->ref = 0xFFFFFFFFu;
    if (dup != 0xFFFFFFFFu && dup < refcut &&
        !memcmp(all[dup].data, t->data, t->len)) {
        t->method = M_REF; t->plen = 0; t->ref = dup;
        t->data = aorig; free(ahold); return;
    }
    int cand[16]; uint32_t cref[16]; int nc = 0;
    memset(cref, 0xFF, sizeof cref);
    if (is_flt16(t->dtype)) {
        cand[nc++] = M_FIELD;
        int mb = mbits_of(t->dtype);
        uint64_t ne = t->len / 2;
        if (t->nd >= 1 && pos_gain((uint16_t *)t->data, ne, mb, t->shape[t->nd - 1], 0) > 0.06)
            cand[nc++] = M_FIELDPOS;
        if (t->nd >= 2 && pos_gain((uint16_t *)t->data, ne, mb, t->shape[0], 1) > 0.06)
            cand[nc++] = M_FIELDROW;
    }
    else if (is_f32(t->dtype)) { cand[nc++] = M_F32; cand[nc++] = M_U8; }
    else cand[nc++] = M_U8;
    cand[nc++] = M_PACK;
    for (int k = 0; k < t->ncand; k++)
        if (t->candref[k] < refcut) { cref[nc] = t->candref[k]; cand[nc++] = M_DELTA; }
    if (t->xreft) { cref[nc] = 0xFFFFFFFFu; cand[nc++] = M_DELTAX; }

    TJob *jb = xc(nc, sizeof(TJob));
    pthread_t *th = xc(nc, sizeof(pthread_t));
    int nsp = ntry < nc ? ntry : nc;
    uint64_t per = ebound(t->len);
    if (nsp > 1 && per * (uint64_t)nsp > TRYBUD) nsp = (int)(TRYBUD / per);
    if (nsp < 1) nsp = 1;
    for (int k = 0; k < nc; k += nsp) {
        int e = k + nsp < nc ? k + nsp : nc;
        for (int k2 = k; k2 < e; k2++) {
            TJob *j = &jb[k2];
            j->m = cand[k2]; j->t = t; j->all = all; j->ri = cref[k2];
            j->ghs = gh; j->snp = gsnp;
            if (nsp == 1) tjob_run(j);
            else if (pthread_create(&th[k2], 0, tjob_run, j)) tjob_run(j);
        }
        for (int k2 = k; k2 < e; k2++) if (nsp > 1) pthread_join(th[k2], 0);
    }
    int w = -1; uint64_t best = t->len;
    for (int k = 0; k < nc; k++)
        if (jb[k].sz && jb[k].sz < best) { best = jb[k].sz; w = k; }
    if (w >= 0) {
        t->method = jb[w].m; t->plen = jb[w].sz;
        t->ref = (jb[w].m == M_DELTA) ? jb[w].ri : 0xFFFFFFFFu;
        *outp = jb[w].buf;
        for (int s = 0; s < 2; s++) if (jb[w].ho[s]) hcommit(jb[w].ch[s], jb[w].ho[s]);
    }
    for (int k = 0; k < nc; k++) if (k != w) {
        free(jb[k].buf);
        for (int s = 0; s < 2; s++) if (jb[k].ho[s]) free(jb[k].ho[s]);
    }
    free(jb); free(th);
    t->data = aorig; free(ahold);
}

/* a tensor may join the batch starting at bstart only if every possible
   intra-archive ref (exact-dup target and all DELTA candidates) lies in
   already-committed ranges — in-batch refs can't be resolved in parallel */
static int joinable(const Tensor *t, uint32_t bstart) {
    if (t->ref != 0xFFFFFFFFu && t->ref >= bstart) return 0;
    for (int k = 0; k < t->ncand; k++) if (t->candref[k] >= bstart) return 0;
    return 1;
}

/* batch worker: encode one tensor with hist state lazily cloned from the
   batch snapshot; its hist delta is merged into gch by the main thread. */
typedef struct {
    Tensor *t; Tensor *all; uint32_t refcut;
    uint32_t **snp;
    uint32_t *ch[NCH];
    uint8_t *out;
} EJob;
static void *ejob_run(void *a) {
    EJob *j = a;
    gh = j->ch; gsnp = j->snp;
    compete(j->t, j->all, j->refcut, &j->out, 1);
    return 0;
}

/* merge worker hist deltas: gch += (worker - snap), channel-wise.
   counts commute, so member order is irrelevant. */
static void hmerge(uint32_t *const wch[NCH], uint32_t *const snap[NCH]) {
    for (int c = 0; c < NCH; c++) if (wch[c]) {
        uint32_t *w = wch[c], *g = gch[c], *s = snap[c];
        for (int i = 0; i < csz[c]; i++) g[i] += w[i] - s[i];
        free(w);
    }
}
static void hsnap(uint32_t *snap[NCH]) {
    for (int c = 0; c < NCH; c++) { snap[c] = xm(csz[c] * 4); memcpy(snap[c], gch[c], csz[c] * 4); }
}

/* ================= decode ================= */
static void dec_tensor(Tensor *t, const uint8_t *payload, Tensor *all) {
    t->data = xc(t->len ? t->len : 1, 1);   /* zeroed: malformed short decodes can't leave indeterminate bytes */
    int bsz = dtb(t->dtype);
    uint64_t ne = t->len / bsz;
    const uint8_t *lim = payload + t->plen;
    switch (t->method) {
    case M_RAW:
        if (t->plen != t->len) die("corrupt raw");
        memcpy(t->data, payload, t->len); break;
    case M_PACK:
        if (bsz > 2) die("bad pack dtype");
        pack_dec(payload, lim, t->len, bsz, t->data); break;
    case M_REF:
        if (all[t->ref].len != t->len) die("bad ref");
        memcpy(t->data, all[t->ref].data, t->len); break;
    case M_U8: u8_dec(payload, lim, t->len, t->data, H(C8)); break;
    case M_FIELD:
        if (bsz != 2) die("bad field dtype");
        f16_dec(payload, lim, ne, mbits_of(t->dtype), (uint16_t *)t->data, H(is_bf(t->dtype) ? CBF : CFP)); break;
    case M_FIELDPOS:
        if (bsz != 2 || t->nd < 1) die("bad shape");
        pos_dec(payload, lim, ne, mbits_of(t->dtype), t->shape[t->nd-1], 0, (uint16_t *)t->data, H(is_bf(t->dtype) ? CBPOS : CFPOS)); break;
    case M_FIELDROW:
        if (bsz != 2 || t->nd < 2) die("bad shape");
        pos_dec(payload, lim, ne, mbits_of(t->dtype), t->shape[0], 1, (uint16_t *)t->data, H(is_bf(t->dtype) ? CBPOS : CFPOS)); break;
    case M_F32:
        if (bsz != 4) die("bad f32 dtype");
        f32_dec(payload, lim, ne, (uint32_t *)t->data, H(C32)); break;
    case M_DELTA:
        if (all[t->ref].len != t->len) die("bad ref");
        if (bsz != 2 && bsz != 4) die("bad delta dtype");
        if (is_f32(t->dtype))
            dlt32_dec(payload, lim, (uint32_t *)all[t->ref].data, ne, (uint32_t *)t->data, H(C32D));
        else
            dlt_dec(payload, lim, (uint16_t *)all[t->ref].data, ne, (uint16_t *)t->data, H(CD), mbits_of(t->dtype), H(is_bf(t->dtype) ? CBF : CFP));
        break;
    case M_DELTAX: {
        if (bsz != 2 && bsz != 4) die("bad delta dtype");
        Tensor *xr = ref_resolve(t);
        if (!xr) die("deltax: --ref tensor missing");
        uint8_t *xh = 0;
        const uint8_t *xd = alview(xr->data, xr->len, bsz, &xh);
        if (is_f32(t->dtype))
            dlt32_dec(payload, lim, (uint32_t *)xd, ne, (uint32_t *)t->data, H(C32D));
        else
            dlt_dec(payload, lim, (uint16_t *)xd, ne, (uint16_t *)t->data, H(CD), mbits_of(t->dtype), H(is_bf(t->dtype) ? CBF : CFP));
        free(xh);
        break;
    }
    }
}

typedef struct {
    Tensor *t; const uint8_t *pl; Tensor *all;
    uint32_t **snp;
    uint32_t *ch[NCH];
} DJob;
static void *djob_run(void *a) {
    DJob *j = a;
    gh = j->ch; gsnp = j->snp;
    dec_tensor(j->t, j->pl, j->all);
    if (crc32_of(j->t->data, j->t->len) != j->t->crc)
        die("crc mismatch: archive corrupt");
    return 0;
}
/* decode a flagged batch run all[i..j): members decode against the
   batch-start snapshot in parallel; hist deltas merge order-free. */
static void dec_batch(Tensor *all, uint32_t i, uint32_t j, const uint8_t *buf) {
    for (uint32_t k = i; k < j; k++)
        if ((all[k].method == M_REF || all[k].method == M_DELTA) && all[k].ref >= i)
            die("batch ref into open batch");
    uint32_t *snap[NCH]; hsnap(snap);
    for (uint32_t k = i; k < j; k += (uint32_t)g_threads) {
        uint32_t e = k + (uint32_t)g_threads < j ? k + (uint32_t)g_threads : j;
        uint32_t cnt = e - k;
        DJob *dj = xc(cnt, sizeof(DJob));
        pthread_t *th = xc(cnt, sizeof(pthread_t));
        for (uint32_t m = 0; m < cnt; m++) {
            dj[m].t = &all[k + m]; dj[m].pl = buf + all[k + m].off;
            dj[m].all = all; dj[m].snp = snap;
            if (pthread_create(&th[m], 0, djob_run, &dj[m])) djob_run(&dj[m]);
        }
        for (uint32_t m = 0; m < cnt; m++) pthread_join(th[m], 0);
        for (uint32_t m = 0; m < cnt; m++) hmerge(dj[m].ch, snap);
        free(dj); free(th);
    }
    for (int c = 0; c < NCH; c++) free(snap[c]);
}

int main(int argc, char **argv) {
    if (argc < 3) die("usage: caiw c|d|v ... [--ref base.st] [-j N] ...");
    {
        long np = sysconf(_SC_NPROCESSORS_ONLN);
        g_threads = np > 0 ? (int)(np < 8 ? np : 8) : 1;
    }
    /* extract --ref PATH and -j N wherever they appear */
    for (int i = 2; i < argc; i++)
        if (!strcmp(argv[i], "--ref") && i + 1 < argc) {
            g_refs = realloc(g_refs, (g_nref + 1) * sizeof(InFile *));
            if (!g_refs) die("oom");
            g_refs[g_nref++] = st_load(argv[i + 1], 255);
            memmove(&argv[i], &argv[i + 2], (argc - i - 2) * sizeof(char *));
            argc -= 2; i--;
        } else if (!strncmp(argv[i], "-j", 2) &&
                   (!argv[i][2] || (argv[i][2] >= '0' && argv[i][2] <= '9'))) {
            const char *v = argv[i] + 2; int eat = 1;
            if (!*v) { if (i + 1 >= argc) die("-j needs a count"); v = argv[i + 1]; eat = 2; }
            g_threads = atoi(v);
            if (g_threads < 1) g_threads = 1;
            if (g_threads > 64) g_threads = 64;
            memmove(&argv[i], &argv[i + eat], (argc - i - eat) * sizeof(char *));
            argc -= eat; i--;
        }
    gh = gch;          /* main thread uses the committed state directly */
    crc_setup();
    if (!strcmp(argv[1], "c")) {
        if (argc < 4) die("caiw c out.caiw in.st...");
        FILE *of = fopen(argv[2], "wb");
        if (!of) die("out");
        int nf = argc - 3;
        InFile **ins = xc(nf, sizeof(InFile *));
        int NT = 0;
        for (int i = 0; i < nf; i++) { ins[i] = st_load(argv[3 + i], i); NT += ins[i]->n; }
        Tensor *all = xc(NT, sizeof(Tensor));
        int ti = 0;
        for (int i = 0; i < nf; i++)
            for (int j = 0; j < ins[i]->n; j++) all[ti++] = ins[i]->t[j];
        /* dedup + delta ref resolution (earlier tensors only) */
        uint64_t *thash = xc(NT, 8);
        for (int i = 0; i < NT; i++) thash[i] = fnv(all[i].data, all[i].len);
        /* normalized names for family matching (experts.N.*, layers.N.*) */
        char **nn = xc(NT, sizeof(char *));
        for (int i = 0; i < NT; i++) nn[i] = nnorm(all[i].name);
        for (int i = 0; i < NT; i++) {
            all[i].ref = 0xFFFFFFFFu;
            all[i].xreft = NULL; all[i].xfile = 0;
            if (g_nref) {
                int ri = 0;
                Tensor *xr = g_nref > 1 ? ref_find_best(&all[i], &ri) : ref_find(all[i].name);
                if (xr && xr->len == all[i].len && !strcmp(xr->dtype, all[i].dtype)) {
                    all[i].xreft = xr; all[i].xfile = ri;
                }
            }
            int dup = -1, dref = -1, dref1 = -1, fref = -1;
            for (int j = 0; j < i; j++) {
                if (all[j].len == all[i].len && thash[j] == thash[i] &&
                    !memcmp(all[j].data, all[i].data, all[i].len)) dup = j;
                if (!strcmp(all[j].name, all[i].name) &&
                    all[j].nd == all[i].nd && all[j].len == all[i].len &&
                    !strcmp(all[j].dtype, all[i].dtype)) { dref1 = dref; dref = j; }
                if (strcmp(all[j].name, all[i].name) && !strcmp(nn[j], nn[i]) &&
                    all[j].nd == all[i].nd && all[j].len == all[i].len &&
                    !strcmp(all[j].dtype, all[i].dtype) &&
                    (is_flt16(all[i].dtype) || is_f32(all[i].dtype))) fref = j;
            }
            all[i].ncand = 0;
            if (dup >= 0) all[i].ref = dup;          /* exact dup -> REF */
            else {
                /* delta candidates: nearest same-name first, then the one
                   before it (checkpoint chains), then last family member */
                if (dref >= 0)  all[i].candref[all[i].ncand++] = dref;
                if (dref1 >= 0) all[i].candref[all[i].ncand++] = dref1;
                if (fref >= 0 && all[i].ncand < 4 &&
                    delta_ok(all[i].data, all[fref].data, all[i].len, dtb(all[i].dtype)))
                    all[i].candref[all[i].ncand++] = fref;
            }
        }
        for (int i = 0; i < NT; i++) free(nn[i]);
        free(nn);
        model_init();
        /* header */
        fwrite("CAI4", 4, 1, of);
        w32(of, nf);
        for (int i = 0; i < nf; i++) {
            const char *bn = bname(ins[i]->path);
            for (int j = 0; j < i; j++)
                if (!strcmp(bn, bname(ins[j]->path))) die("duplicate input basename");
            w32(of, strlen(bn)); fwrite(bn, 1, strlen(bn), of);
        }
        w32(of, NT);
        uint64_t tin = 0, tout = 0;
        uint32_t i = 0;
        while (i < (uint32_t)NT) {
            /* parallel batch: consecutive tensors sharing batch-start model
               state; members' intra refs are filtered to < i in compete() */
            uint32_t nm = 0; uint64_t blen = 0;
            if (g_threads > 1)
                while (nm < (uint32_t)g_threads && i + nm < (uint32_t)NT &&
                       all[i + nm].len && joinable(&all[i + nm], i) &&
                       blen + all[i + nm].len <= BCAP) {
                    blen += all[i + nm].len; nm++;
                }
            if (nm > 1) {
                uint32_t *snap[NCH]; hsnap(snap);
                EJob *ej = xc(nm, sizeof(EJob));
                pthread_t *th = xc(nm, sizeof(pthread_t));
                for (uint32_t k = 0; k < nm; k++) {
                    ej[k].t = &all[i + k]; ej[k].all = all;
                    ej[k].refcut = i; ej[k].snp = snap;
                    if (pthread_create(&th[k], 0, ejob_run, &ej[k])) ejob_run(&ej[k]);
                }
                for (uint32_t k = 0; k < nm; k++) pthread_join(th[k], 0);
                for (uint32_t k = 0; k < nm; k++) hmerge(ej[k].ch, snap);
                for (uint32_t k = 0; k < nm; k++) {
                    Tensor *t = ej[k].t;
                    emit_rec(of, t, k ? 2 : 1);
                    if (t->plen) fwrite(ej[k].out ? ej[k].out : t->data, 1, t->plen, of);
                    free(ej[k].out);
                    tin += t->len; tout += t->plen;
                    drop_pages(t->data, t->len, ins[t->file]->mapped);
                }
                for (int c = 0; c < NCH; c++) free(snap[c]);
                free(ej); free(th);
                i += nm;
                continue;
            }
            Tensor *t = &all[i];
            uint8_t *buf = 0;
            if (t->len) compete(t, all, 0xFFFFFFFFu, &buf, g_threads);
            else { t->method = M_RAW; t->plen = 0; t->ref = 0xFFFFFFFFu; }
            emit_rec(of, t, 0);
            if (t->plen) fwrite(buf ? buf : t->data, 1, t->plen, of);
            free(buf);
            tin += t->len; tout += t->plen;
            /* input data pages consumed; release them (later REF/DELTA
               re-reads simply re-fault from the input file) */
            drop_pages(t->data, t->len, ins[t->file]->mapped);
            i++;
        }
        if (ferror(of) || fclose(of)) die("write failed (disk full?)");
        static const char *mn[] = {"RAW","PACK","REF","FIELD","FIELDPOS","DELTA","U8","F32","FIELDROW","DELTAX"};
        uint64_t mc[10] = {0}, mb2[10] = {0};
        for (int ti = 0; ti < NT; ti++) { mc[all[ti].method]++; mb2[all[ti].method] += all[ti].plen; }
        fprintf(stderr, "in=%llu out=%llu ratio=%.3f\n",
                (unsigned long long)tin, (unsigned long long)tout,
                tin ? (double)tout / tin : 0);
        for (int m = 0; m < 10; m++) if (mc[m])
            fprintf(stderr, "  %-8s n=%-5llu payload=%.1fMB\n", mn[m],
                    (unsigned long long)mc[m], (double)mb2[m] / 1048576);
        return 0;
    }
    if (!strcmp(argv[1], "d")) {
        if (argc < 4) die("caiw d out.caiw outdir/");
        mkdir(argv[3], 0755);
        uint64_t fsz; uint8_t *buf = slurp(argv[2], &fsz);
        const uint8_t *p = buf, *bend = buf + fsz;
#define BND(q, need) do { if ((uint64_t)(need) > (uint64_t)(bend - (q))) die("truncated archive"); } while (0)
        BND(p, 4);
        if (memcmp(p, "CAI3", 4) && memcmp(p, "CAI4", 4)) die("bad magic");
        p += 4;
        BND(p, 4); uint32_t nf = r32(&p);
        if (nf > 65535) die("too many files");
        char **fnames = xc(nf, sizeof(char *));
        for (uint32_t i = 0; i < nf; i++) {
            BND(p, 4); uint32_t l = r32(&p); BND(p, l);
            fnames[i] = xm(l + 1); memcpy(fnames[i], p, l); fnames[i][l] = 0; p += l;
            /* output filename must not escape outdir */
            if (!l || strchr(fnames[i], '/') || strchr(fnames[i], '\\') ||
                !strcmp(fnames[i], ".") || !strcmp(fnames[i], ".."))
                die("bad filename");
            for (uint32_t j = 0; j < i; j++) if (!strcmp(fnames[i], fnames[j])) die("dup filename");
        }
        BND(p, 4); uint32_t NT = r32(&p);
        Tensor *all = xc(NT, sizeof(Tensor));
        /* pass 1: metadata walk — no decode */
        const uint8_t *q = p;
        uint32_t bstart = 0;   /* head index of currently open batch */
        for (uint32_t i = 0; i < NT; i++) {
            Tensor *t = &all[i];
            uint16_t nl = 0, dl = 0;
            BND(q, 2); memcpy(&nl, q, 2); q += 2;
            BND(q, nl); t->name = xm(nl + 1); memcpy(t->name, q, nl); t->name[nl] = 0; q += nl;
            BND(q, 2); memcpy(&dl, q, 2); q += 2;
            BND(q, dl); t->dtype = xm(dl + 1); memcpy(t->dtype, q, dl); t->dtype[dl] = 0; q += dl;
            BND(q, 1); t->nd = r8(&q);
            if (t->nd > 8) die("bad nd");
            BND(q, 8 * t->nd);
            for (int d = 0; d < t->nd; d++) t->shape[d] = (int64_t)r64(&q);
            BND(q, 2); t->file = r16(&q);
            if (t->file >= (int)nf) die("bad file idx");
            BND(q, 1); int mraw = r8(&q);
            t->bat = (mraw & MF_BAT) != 0 ? ((mraw & MF_BH) ? 1 : 2) : 0;
            t->method = mraw & 0x3F;
            if (t->method > M_DELTAX) die("bad method");
            BND(q, 8); t->len = r64(&q);
            t->ref = 0xFFFFFFFFu;
            if (t->method == M_REF || t->method == M_DELTA) {
                BND(q, 4); t->ref = r32(&q);
                if (t->ref >= i) die("bad ref");
                if (t->bat == 2 && t->ref >= bstart) die("ref into open batch");
            } else if (t->method == M_DELTAX) { BND(q, 4); t->ref = r32(&q); }
            BND(q, 8); t->plen = r64(&q);
            BND(q, 4); t->crc = r32(&q);
            t->off = q - buf;              /* payload offset */
            BND(q, t->plen); q += t->plen;
            if (t->bat == 1) bstart = i;
            else if (!t->bat) bstart = i + 1;
        }
        /* per-file offsets + write headers + open outfiles */
        uint64_t *foff = xc(nf, 8), *hbase = xc(nf, 8);
        FILE **ofs = xc(nf, sizeof(FILE *));
        char tmp[4096];
        g_outp = xc(nf, sizeof(char *)); g_outn = nf; g_outok = 0;
        atexit(out_cleanup);
        for (uint32_t i = 0; i < nf; i++) {
            size_t hcap = 1 << 20; char *j = xm(hcap); size_t jl = 0;
            jl += snprintf(j + jl, hcap - jl, "{");
            int first = 1;
            for (uint32_t k = 0; k < NT; k++) {
                Tensor *t = &all[k];
                if (t->file != (int)i) continue;
                while (jl + 4096 > hcap) { hcap *= 2; j = realloc(j, hcap); if (!j) die("oom"); }
                t->dataoff = foff[i];
                jl += snprintf(j + jl, hcap - jl, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                               first ? "" : ",", t->name, t->dtype);
                for (int d = 0; d < t->nd; d++) jl += snprintf(j + jl, hcap - jl, "%s%lld", d ? "," : "", (long long)t->shape[d]);
                jl += snprintf(j + jl, hcap - jl, "],\"data_offsets\":[%llu,%llu]}",
                               (unsigned long long)foff[i], (unsigned long long)(foff[i] + t->len));
                foff[i] += t->len;
                first = 0;
            }
            jl += snprintf(j + jl, hcap - jl, "}");
            snprintf(tmp, sizeof tmp, "%s/%s", argv[3], fnames[i]);
            g_outp[i] = strdup(tmp);
            ofs[i] = fopen(tmp, "wb");
            if (!ofs[i]) die("out file");
            w64(ofs[i], jl); fwrite(j, 1, jl, ofs[i]);
            hbase[i] = 8 + jl;
            fprintf(stderr, "wrote %s (%lluB data)\n", tmp, (unsigned long long)foff[i]);
            free(j);
        }
        /* keep flags: tensors referenced by REF/DELTA must stay resident */
        uint8_t *keep = xc(NT ? NT : 1, 1);
        for (uint32_t i = 0; i < NT; i++)
            if ((all[i].method == M_REF || all[i].method == M_DELTA) && all[i].ref < NT)
                keep[all[i].ref] = 1;
        /* pass 2: decode + stream out (flagged runs decode in parallel) */
        model_init();
        for (uint32_t i = 0; i < NT;) {
            if (all[i].bat) {
                if (all[i].bat != 1) die("orphan batch member");
                uint32_t j = i + 1; while (j < NT && all[j].bat == 2) j++;
                dec_batch(all, i, j, buf);
                for (uint32_t k = i; k < j; k++) {
                    Tensor *t = &all[k];
                    FILE *of = ofs[t->file];
                    fseeko(of, hbase[t->file] + t->dataoff, SEEK_SET);
                    if (t->len && fwrite(t->data, 1, t->len, of) != t->len) die("write");
                    drop_pages(buf + t->off, t->plen, g_amap);
                    if (!keep[k]) { free(t->data); t->data = 0; }
                }
                i = j; continue;
            }
            Tensor *t = &all[i];
            dec_tensor(t, buf + t->off, all);
            if (crc32_of(t->data, t->len) != t->crc) die("crc mismatch: archive corrupt");
            FILE *of = ofs[t->file];
            fseeko(of, hbase[t->file] + t->dataoff, SEEK_SET);
            if (t->len && fwrite(t->data, 1, t->len, of) != t->len) die("write");
            drop_pages(buf + t->off, t->plen, g_amap);
            if (!keep[i]) { free(t->data); t->data = 0; }
            i++;
        }
        for (uint32_t i = 0; i < nf; i++)
            if (ferror(ofs[i]) || fclose(ofs[i])) die("write");
        g_outok = 1;
        return 0;
    }
    if (!strcmp(argv[1], "v")) {
        /* verify: decode archive in-memory, memcmp each tensor vs source files */
        uint64_t fsz; uint8_t *buf = slurp(argv[2], &fsz);
        int nf = argc - 3;
        InFile **ins = xc(nf, sizeof(InFile *));
        for (int i = 0; i < nf; i++) ins[i] = st_load(argv[3 + i], i);
        const uint8_t *p = buf, *bend = buf + fsz;
        BND(p, 4);
        if (memcmp(p, "CAI3", 4) && memcmp(p, "CAI4", 4)) die("bad magic");
        p += 4;
        BND(p, 4); uint32_t nfa = r32(&p);
        for (uint32_t i = 0; i < nfa; i++) { BND(p, 4); uint32_t l = r32(&p); BND(p, l); p += l; }
        BND(p, 4); uint32_t NT = r32(&p);
        model_init();
        Tensor *all = xc(NT, sizeof(Tensor));
        /* pass 1: metadata + payload offsets, keep flags */
        const uint8_t *q = p;
        uint32_t bstart = 0;   /* head index of currently open batch */
        for (uint32_t i = 0; i < NT; i++) {
            Tensor *t = &all[i];
            uint16_t nl, dl;
            BND(q, 2); memcpy(&nl, q, 2); q += 2;
            BND(q, nl); t->name = xm(nl + 1); memcpy(t->name, q, nl); t->name[nl] = 0; q += nl;
            BND(q, 2); memcpy(&dl, q, 2); q += 2;
            BND(q, dl); t->dtype = xm(dl + 1); memcpy(t->dtype, q, dl); t->dtype[dl] = 0; q += dl;
            BND(q, 1); t->nd = r8(&q);
            if (t->nd > 8) die("bad nd");
            BND(q, 8 * t->nd);
            for (int d = 0; d < t->nd; d++) t->shape[d] = (int64_t)r64(&q);
            BND(q, 2); t->file = r16(&q);
            if (t->file >= nf) die("bad file idx");
            BND(q, 1); int mraw = r8(&q);
            t->bat = (mraw & MF_BAT) != 0 ? ((mraw & MF_BH) ? 1 : 2) : 0;
            t->method = mraw & 0x3F;
            if (t->method > M_DELTAX) die("bad method");
            BND(q, 8); t->len = r64(&q);
            t->ref = 0xFFFFFFFFu;
            if (t->method == M_REF || t->method == M_DELTA) {
                BND(q, 4); t->ref = r32(&q);
                if (t->ref >= i) die("bad ref");
                if (t->bat == 2 && t->ref >= bstart) die("ref into open batch");
            } else if (t->method == M_DELTAX) { BND(q, 4); t->ref = r32(&q); }
            BND(q, 8); t->plen = r64(&q);
            BND(q, 4); t->crc = r32(&q);
            t->off = q - buf;
            BND(q, t->plen); q += t->plen;
            if (t->bat == 1) bstart = i;
            else if (!t->bat) bstart = i + 1;
        }
        uint8_t *keep = xc(NT ? NT : 1, 1);
        for (uint32_t i = 0; i < NT; i++)
            if ((all[i].method == M_REF || all[i].method == M_DELTA) && all[i].ref < NT)
                keep[all[i].ref] = 1;
        int bad = 0, checked = 0;
        for (uint32_t i = 0; i < NT;) {
            uint32_t i0 = i, i1 = i + 1;
            if (all[i].bat) {
                if (all[i].bat != 1) die("orphan batch member");
                i1 = i + 1; while (i1 < NT && all[i1].bat == 2) i1++;
                dec_batch(all, i, i1, buf);
            } else {
                Tensor *t = &all[i];
                dec_tensor(t, buf + t->off, all);
                if (crc32_of(t->data, t->len) != t->crc) die("crc mismatch: archive corrupt");
            }
            /* post: compare each decoded tensor against the source */
            for (uint32_t k = i0; k < i1; k++) {
                Tensor *t = &all[k];
                InFile *src = ins[t->file];
                Tensor *st = 0;
                for (int j = 0; j < src->n; j++) if (!strcmp(src->t[j].name, t->name)) st = &src->t[j];
                if (!st) { fprintf(stderr, "MISS %s\n", t->name); bad++; continue; }
                checked++;
                if (st->len != t->len || memcmp(st->data, t->data, t->len)) {
                    fprintf(stderr, "DIFF %s method=%d\n", t->name, t->method);
                    bad++;
                }
                drop_pages(buf + t->off, t->plen, g_amap);
                drop_pages(st->data, st->len, src->mapped);
                if (!keep[k]) { free(t->data); t->data = 0; }
            }
            i = i1;
        }
        fprintf(stderr, "verify: %d tensors, %d bad\n", checked, bad);
        return bad ? 1 : 0;
    }
    die("unknown cmd");
}
