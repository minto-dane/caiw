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
#include <stddef.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <libgen.h>
#include <errno.h>
#include <ctype.h>
#include <stdarg.h>
#include <pthread.h>

/* length arithmetic is u64 throughout; size_t must not truncate it */
typedef char caiw_needs_64bit_size_t[(sizeof(size_t) >= 8) ? 1 : -1];

/* every archive field is explicitly little-endian — the format must not
   depend on host byte order */
/*@ requires \valid(o + (0 .. 1));
    assigns o[0 .. 1];
*/
static void p16le(uint8_t *o, uint16_t v) { o[0] = (uint8_t)v; o[1] = (uint8_t)(v >> 8); }
/*@ requires \valid(o + (0 .. 3));
    assigns o[0 .. 3];
*/
static void p32le(uint8_t *o, uint32_t v) {
    /*@ loop invariant 0 <= i <= 4;
        loop assigns o[0 .. 3], i;
        loop variant 4 - i;
    */
    for (int i = 0; i < 4; i++) o[i] = (uint8_t)(v >> (8 * i));
}
/*@ requires \valid(o + (0 .. 7));
    assigns o[0 .. 7];
*/
static void p64le(uint8_t *o, uint64_t v) {
    /*@ loop invariant 0 <= i <= 8;
        loop assigns o[0 .. 7], i;
        loop variant 8 - i;
    */
    for (int i = 0; i < 8; i++) o[i] = (uint8_t)(v >> (8 * i));
}
/*@ requires \valid_read(p + (0 .. 3));
    assigns \nothing;
*/
static uint32_t g32le(const uint8_t *p) {
    uint32_t v = 0;
    /*@ loop invariant 0 <= i <= 4;
        loop assigns v, i;
        loop variant 4 - i;
    */
    for (int i = 0; i < 4; i++) v |= (uint32_t)p[i] << (8 * i);
    return v;
}
/*@ requires \valid_read(p + (0 .. 7));
    assigns \nothing;
*/
static uint64_t g64le(const uint8_t *p) {
    uint64_t v = 0;
    /*@ loop invariant 0 <= i <= 8;
        loop assigns v, i;
        loop variant 8 - i;
    */
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

#define BS 17
#define BLK (1u << BS)
#define SB 15
#define TOT (1u << SB)
#define LOWER (1ull << 31)
#define SCRSZ (BLK * 16 + 64)   /* worst-case renorm bytes: 3B per enc call, <=5 calls/elem */
#define XMAX(f) ((((LOWER) << 8) / TOT) * (uint64_t)(f))

enum { M_RAW = 0, M_PACK, M_REF, M_FIELD, M_FIELDPOS, M_DELTA, M_U8, M_F32, M_FIELDROW, M_DELTAX, M_PRW };
#define MF_BAT 0x80   /* method-byte flag: parallel batch member (CAI4) */
#define MF_BH  0x40   /* batch head: first member opens a new batch */

/*@ terminates \false;
    assigns \exit_status \from \nothing;
    exits \exit_status == 1;
    ensures never_terminates: \false;
*/
static void die(const char *m) __attribute__((noreturn));
static void die(const char *m) {
    static volatile int dying;
    if (__sync_lock_test_and_set(&dying, 1)) _exit(1);  /* a worker already exiting: don't re-run atexit */
    fprintf(stderr, "caiw: %s\n", m);
    exit(1);
}
/* d-mode: remove partial outputs if we exit on any failure */
static char **g_outp; static int g_outn, g_outok;
static void out_cleanup(void) { if (!g_outok) for (int i = 0; i < g_outn; i++) if (g_outp[i]) unlink(g_outp[i]); }

/* ================= CRC32 (integrity) ================= */
static uint32_t crc_tab[256], crc_ready = 0;
/*@ assigns crc_tab[0 .. 255], crc_ready;
    ensures crc_ready == 1;
*/
static void crc_setup(void) {
    /*@ loop invariant 0 <= i <= 256;
        loop assigns crc_tab[0 .. 255], i;
        loop variant 256 - i;
    */
    for (int i = 0; i < 256; i++) {
        uint32_t c = i;
        /*@ loop invariant 0 <= k <= 8;
            loop assigns c, k;
            loop variant 8 - k;
        */
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_tab[i] = c;
    }
    crc_ready = 1;
}
/*@ requires \valid_read(p + (0 .. n - 1));
    assigns crc_tab[0 .. 255], crc_ready;
    ensures crc_ready != 0;
*/
static uint32_t crc32_of(const uint8_t *p, uint64_t n) {
    if (!crc_ready) crc_setup();
    uint32_t c = 0xFFFFFFFFu;
    /*@ loop invariant 0 <= i <= n;
        loop assigns c, i;
        loop variant n - i;
    */
    for (uint64_t i = 0; i < n; i++) c = crc_tab[(c ^ p[i]) % 256] ^ (c >> 8);
    return ~c;
}
/* slicing-by-8 fast path: T[0]=crc_tab, T[k][b]=T[k-1][b]>>8^T[0][T[k-1][b]&FF].
   Same CRC as crc32_of by construction; guarded by a one-time differential
   self-check that permanently falls back to the proved scalar loop on
   mismatch (crc_fast_ok = -1). Little-endian hosts only — same constraint
   as the element payloads. */
static uint32_t crc_t8[8][256];
static int crc_fast_ok = 0;   /* 0 = untested, 1 = slice path verified, -1 = mismatch */
static uint32_t crc32_slice(const uint8_t *p, uint64_t n) {
    uint32_t c = 0xFFFFFFFFu;
    while (n >= 8) {
        uint32_t w;
        memcpy(&w, p, 4);   /* little-endian load — BE hosts fail the self-test
                               below and stay on the scalar path */
        c ^= w;
        c = crc_t8[7][c & 0xFF] ^ crc_t8[6][(c >> 8) & 0xFF] ^
            crc_t8[5][(c >> 16) & 0xFF] ^ crc_t8[4][c >> 24] ^
            crc_t8[3][p[4]] ^ crc_t8[2][p[5]] ^ crc_t8[1][p[6]] ^ crc_t8[0][p[7]];
        p += 8; n -= 8;
    }
    while (n--) c = crc_tab[(c ^ *p++) & 0xFF] ^ (c >> 8);
    return ~c;
}
static void crc_setup8(void) {
    for (int b = 0; b < 256; b++) crc_t8[0][b] = crc_tab[b];
    for (int k = 1; k < 8; k++)
        for (int b = 0; b < 256; b++)
            crc_t8[k][b] = crc_t8[k - 1][b] >> 8 ^ crc_tab[crc_t8[k - 1][b] & 0xFF];
    /* differential self-test on a deterministic 4KB+tail pattern: any bug in
       table construction or the fold loop shows up here before real data;
       on mismatch (e.g. big-endian host) stay on the proved scalar loop */
    static uint8_t pat[4101];
    uint32_t s = 0x9E3779B9u;
    for (int i = 0; i < 4101; i++) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; pat[i] = (uint8_t)s; }
    crc_fast_ok = crc32_slice(pat, sizeof pat) == crc32_of(pat, sizeof pat) ? 1 : -1;
}
static uint32_t crc32b(const uint8_t *p, uint64_t n) {
    if (!crc_ready) crc_setup();
    if (!crc_fast_ok) crc_setup8();
    return crc_fast_ok > 0 ? crc32_slice(p, n) : crc32_of(p, n);
}
static void *xm(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("oom"); return p; }
static void *xc(size_t a, size_t b) { void *p = calloc(a ? a : 1, b ? b : 1); if (!p) die("oom"); return p; }
static char *xstrdup(const char *s) { char *p = strdup(s); if (!p) die("oom"); return p; }

/*@ requires \valid_read(p + (0 .. n - 1));
    assigns \nothing;
*/
static uint64_t fnv(const uint8_t *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    /*@ loop invariant 0 <= i <= n;
        loop assigns h, i;
        loop variant n - i;
    */
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

/* ================= safetensors ================= */
typedef struct Tensor {
    char *name, *dtype;
    int64_t shape[64]; int nd;  /* max dims; inputs with more are rejected */
    uint64_t off, len;
    uint8_t *data;
    int file;
    int matched;                /* v-mode: this source tensor was found in the archive */
    uint64_t dataoff;           /* decode: output offset within file data region */
    int method; uint32_t ref; uint64_t plen; uint32_t crc;
    int bat;                    /* decoded: member of a parallel batch */
    uint64_t dlivc;             /* decode: bytes charged against g_dlive */
    struct Tensor *xreft;       /* external --ref match, or NULL */
    int xfile;                  /* which --ref file xreft came from */
    uint32_t candref[4]; int ncand; /* intra DELTA ref candidates */
} Tensor;

typedef struct {
    char *path; Tensor *t; int n;
    uint8_t *base; int mapped;
    char *meta;                 /* verbatim __metadata__ object text or NULL */
} InFile;
static InFile **g_refs = NULL; static int g_nref = 0;

/* ---- minimal JSON for safetensors headers ---- */
/*@ requires valid_read_string(p);
    requires \valid(out);
    assigns *out;
    ensures \result == 0 || \result == 1;
*/
static int hex4(const char *p, unsigned *out) {
    unsigned v = 0;
    /*@ loop invariant 0 <= k <= 4;
        loop invariant \forall integer j; 0 <= j < k ==> p[j] != 0;
        loop assigns v, k;
        loop variant 4 - k;
    */
    for (int k = 0; k < 4; k++) {
        char c = p[k]; v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    *out = v; return 1;
}
/* parse a JSON string at p ('"'); unescape into buf (\u -> UTF-8).
   returns char past the closing quote, NULL on malformed input. */
static const char *jstr(const char *p, char *buf, size_t bs) {
    if (*p != '"') return 0;
    p++;
    size_t i = 0;
    for (;;) {
        unsigned char c = (unsigned char)*p;
        if (!c) return 0;
        if (c == '"') { buf[i] = 0; return p + 1; }
        if (c == '\\') {
            p++;
            switch (*p) {
            case '"': case '\\': case '/': c = (unsigned char)*p; break;
            case 'b': c = '\b'; break; case 'f': c = '\f'; break;
            case 'n': c = '\n'; break; case 'r': c = '\r'; break; case 't': c = '\t'; break;
            case 'u': {
                unsigned cp;
                if (!hex4(p + 1, &cp)) return 0;
                p += 5;
                if (cp >= 0xD800 && cp <= 0xDBFF && p[0] == '\\' && p[1] == 'u') {
                    unsigned lo;
                    if (hex4(p + 2, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        p += 6;
                    }
                }
                if (i + 4 >= bs) return 0;
                if (!cp) return 0;              /* embedded NUL: C-string names can't carry it */
                if (cp < 0x80) buf[i++] = (char)cp;
                else if (cp < 0x800) {
                    buf[i++] = (char)(0xC0 | (cp >> 6));
                    buf[i++] = (char)(0x80 | (cp & 63));
                } else if (cp < 0x10000) {
                    buf[i++] = (char)(0xE0 | (cp >> 12));
                    buf[i++] = (char)(0x80 | ((cp >> 6) & 63));
                    buf[i++] = (char)(0x80 | (cp & 63));
                } else {
                    buf[i++] = (char)(0xF0 | (cp >> 18));
                    buf[i++] = (char)(0x80 | ((cp >> 12) & 63));
                    buf[i++] = (char)(0x80 | ((cp >> 6) & 63));
                    buf[i++] = (char)(0x80 | (cp & 63));
                }
                continue;
            }
            default: return 0;
            }
            p++;                    /* past the escape letter */
            if (i + 1 >= bs) return 0;
            buf[i++] = (char)c;
            continue;
        }
        if (i + 1 >= bs) return 0;
        buf[i++] = (char)c;
        p++;
    }
}
/* skip a JSON string at p; returns char past the closing quote */
static const char *jstr_skip(const char *p) {
    for (p++; *p && *p != '"'; p++) if (*p == '\\' && p[1]) p++;
    return *p ? p + 1 : 0;
}
/* skip a balanced {..} / [..] starting at p (string-aware) */
static const char *jspan(const char *p) {
    char open = *p, close = open == '{' ? '}' : ']';
    if (open != '{' && open != '[') return 0;
    int depth = 0;
    for (;; p++) {
        if (!*p) return 0;
        if (*p == '"') {
            const char *e = jstr_skip(p);
            if (!e) return 0;
            p = e - 1;
        } else if (*p == open) depth++;
        else if (*p == close && --depth == 0) return p + 1;
    }
}
static int dtb(const char *d);
static void ck_shape_len(const Tensor *t);
static void ck_meta(const uint8_t *m, uint32_t ml);
static uint64_t nfh(const char *name, int file);
static void ck_dupname(char **seen, const char *name, int file, uint32_t n, uint32_t cap);
/* strict UTF-8: no overlongs, no surrogates, <=U+10FFFF, no truncation.
   Names/dtype strings are replayed raw into regenerated JSON (jesc passes
   bytes >=0x80 through), so a non-UTF-8 name would produce a header that
   strict parsers reject. Validate on decode AND on archive load. */
/*@ requires n <= 1073741824;
    requires \valid_read(s + (0 .. n - 1));
    assigns \nothing;
    ensures \result == 0 || \result == 1;
*/
static int utf8_ok(const char *s, size_t n) {
    /*@ loop invariant 0 <= i <= n;
        loop assigns i;
        loop variant n - i;
    */
    for (size_t i = 0; i < n;) {
        uint8_t c = (uint8_t)s[i];
        if (c < 0x80) { i++; continue; }
        uint32_t cp; int w;
        if (c < 0xC0) return 0;
        else if (c < 0xE0) { cp = c & 0x1F; w = 2; if (cp < 2) return 0; }
        else if (c < 0xF0) { cp = c & 0x0F; w = 3; }
        else if (c < 0xF5) { cp = c & 7; w = 4; }
        else return 0;
        if (i + w > n) return 0;
        /*@ loop invariant 1 <= k <= w;
            loop invariant w <= n - i;
            loop assigns k, cp;
            loop variant w - k;
        */
        for (int k = 1; k < w; k++) {
            uint8_t t = (uint8_t)s[i + k];
            if ((t & 0xC0) != 0x80) return 0;
            cp = (cp << 6) | (t & 0x3F);
        }
        if (cp >= 0xD800 && cp <= 0xDFFF) return 0;      /* lone surrogate */
        if (w == 3 && cp < 0x800) return 0;            /* overlong */
        if (w == 4 && (cp < 0x10000 || cp > 0x10FFFF)) return 0;  /* overlong / >U+10FFFF */
        i += w;
    }
    return 1;
}

/* locate key's value within flat object [obj,oend); returns value start
   (guaranteed inside oend) or NULL. Value contents are never copied. */
static const char *jkey(const char *obj, const char *oend, const char *key) {
    const char *p = obj;
    char *kb = xm((size_t)(oend - obj) + 1);   /* decoded key <= object span */
    while (p < oend) {
        const char *q = memchr(p, '"', oend - p);
        if (!q) { free(kb); return 0; }
        const char *e = jstr(q, kb, (size_t)(oend - obj) + 1);
        if (!e || e > oend) { free(kb); return 0; }
        const char *np = e;
        while (np < oend && (*np == ' ' || *np == '\t' || *np == '\n' || *np == '\r')) np++;
        p = e;
        if (np >= oend || *np != ':') continue;
        np++;
        while (np < oend && (*np == ' ' || *np == '\t' || *np == '\n' || *np == '\r')) np++;
        if (np >= oend) { free(kb); return 0; }
        if (!strcmp(kb, key)) { free(kb); return np; }
        if (*np == '"') { const char *ve = jstr_skip(np); p = ve ? ve : oend; }
        else if (*np == '{' || *np == '[') { const char *ve = jspan(np); p = ve ? ve : oend; }
        else { while (np < oend && *np != ',') np++; p = np; }
    }
    free(kb);
    return 0;
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
    InFile *inf = xc(1, sizeof(InFile));
    inf->path = xstrdup(path);
    int tcap = 0;   /* grow-on-demand: escaped keys can't be precounted by strstr */
    uint32_t scap = 128; char **seen = xc(scap, sizeof(char *));   /* dup-name hash (file=0) */
    const char *p = hdr; char *kb = xm(hl + 1);   /* decoded names <= raw header len */
    while ((p = strchr(p, '"'))) {
        const char *e = jstr(p, kb, hl + 1);
        if (!e) die("bad header string");
        const char *np = e;
        while (*np == ' ' || *np == '\t' || *np == '\n' || *np == '\r') np++;
        if (*np != ':') { p = e; continue; }
        np++;
        while (*np == ' ' || *np == '\t' || *np == '\n' || *np == '\r') np++;
        if (!utf8_ok(kb, strlen(kb))) die("non-utf8 key");
        if (!strcmp(kb, "__metadata__")) {   /* keep value verbatim for byte-faithful replay */
            const char *ve;
            if (*np == '{' || *np == '[') ve = jspan(np);
            else if (*np == '"') ve = jstr_skip(np);
            else { ve = np; while (ve < hdr + hl && *ve != ',' && *ve != '}') ve++; }
            if (!ve || ve > hdr + hl) die("bad metadata");
            if (memchr(np, 0, (size_t)(ve - np))) die("bad metadata");
            free(inf->meta);
            inf->meta = xm((size_t)(ve - np) + 1);
            memcpy(inf->meta, np, (size_t)(ve - np));
            inf->meta[ve - np] = 0;
            ck_meta((const uint8_t *)inf->meta, (uint32_t)(ve - np));
            p = ve; continue;
        }
        if (*np == '"') {           /* plain string value: skip whole string */
            const char *ve = jstr_skip(np);
            if (!ve) die("bad header string");
            p = ve; continue;
        }
        if (*np == '[') {           /* top-level array: skip whole, never descend */
            const char *ve = jspan(np);
            if (!ve) die("bad header array");
            p = ve; continue;
        }
        if (*np != '{') {           /* scalar value: skip the token so junk
                                       inside it can't be mistaken for a key */
            p = np;
            while (p < hdr + hl && *p != ',' && *p != '}') p++;
            continue;
        }
        const char *oend = jspan(np);   /* char past matching '}' */
        if (!oend) die("bad header object");
        const char *offs = jkey(np, oend, "data_offsets");
        if (!offs) { p = oend; continue; }
        if (inf->n >= tcap) {
            tcap = tcap ? tcap * 2 : 64;
            inf->t = realloc(inf->t, (size_t)tcap * sizeof(Tensor));
            if (!inf->t) die("oom");
            memset(inf->t + inf->n, 0, (size_t)(tcap - inf->n) * sizeof(Tensor));
            if ((uint32_t)tcap * 2 > scap) {   /* keep seen-table load < 1/2 */
                uint32_t ncap = scap * 2;
                char **ns = xc(ncap, sizeof(char *));
                for (uint32_t s = 0; s < scap; s++) if (seen[s]) {
                    int fl = (uint8_t)seen[s][0] | ((uint8_t)seen[s][1] << 8);
                    uint64_t m2 = ncap - 1, sl = nfh(seen[s] + 2, fl) & m2;
                    while (ns[sl]) sl = (sl + 1) & m2;
                    ns[sl] = seen[s];
                }
                free(seen); seen = ns; scap = ncap;
            }
        }
        ck_dupname(seen, kb, 0, (uint32_t)inf->n, scap);
        if (strlen(kb) > 65535) die("name too long");   /* record field is u16 */
        Tensor *t = &inf->t[inf->n];
        t->name = xstrdup(kb);
        t->file = fidx;
        const char *dv = jkey(np, oend, "dtype");
        char dbuf[64];
        if (!dv || *dv != '"' || !jstr(dv, dbuf, sizeof dbuf)) die("bad dtype");
        if (!utf8_ok(dbuf, strlen(dbuf))) die("non-utf8 dtype");
        t->dtype = xstrdup(dbuf);
        const char *sv = jkey(np, oend, "shape");
        if (!sv) die("bad shape");
        t->nd = 0;
        {   /* strict: shape must be an array of nonneg integers */
            const char *q = sv;
            if (*q++ != '[') die("bad shape");
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
            if (*q == ']') q++;
            else for (;;) {
                while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
                if (*q < '0' || *q > '9') die("bad shape");
                if (t->nd >= 64) die("too many dims");
                char *e2; errno = 0;
                t->shape[t->nd] = strtoll(q, &e2, 10);
                if (errno == ERANGE) die("bad shape");  /* out-of-range digits
                    would silently clamp to LLONG_MAX, corrupting the replayed
                    header value */
                t->nd++;
                q = e2;
                while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
                if (*q == ']') { q++; break; }
                if (*q != ',') die("bad shape");
                q++;
            }
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
            if (*q != ',' && *q != '}') die("bad shape");
        }
        uint64_t a = 0, b = 0;
        {
            const char *q = offs; char *e2;
            if (*q != '[') die("bad data_offsets");
            q++;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
            if (*q < '0' || *q > '9') die("bad data_offsets");   /* no +/-, no empty */
            errno = 0; a = strtoull(q, &e2, 10);
            if (errno == ERANGE) die("bad data_offsets");
            for (q = e2; *q == ' ' || *q == '\t' || *q == '\n' || *q == '\r'; q++) ;
            if (*q != ',') die("bad data_offsets");
            q++;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
            if (*q < '0' || *q > '9') die("bad data_offsets");
            errno = 0; b = strtoull(q, &e2, 10);
            if (errno == ERANGE) die("bad data_offsets");
            for (q = e2; *q == ' ' || *q == '\t' || *q == '\n' || *q == '\r'; q++) ;
            if (*q != ']') die("bad data_offsets");
            q++;
            for (; *q == ' ' || *q == '\t' || *q == '\n' || *q == '\r'; q++) ;
            if (*q != ',' && *q != '}') die("bad data_offsets");
        }
        if (b < a) die("bad data_offsets");
        t->off = a; t->len = b - a;
        ck_shape_len(t);
        inf->n++;
        p = oend;
    }
    free(hdr); free(kb);
    for (uint32_t s = 0; s < scap; s++) free(seen[s]);
    free(seen);
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
       Map the ALREADY-OPEN fileno(f): a second open(path) would let a
       racing rename hand us a different inode than the one whose header
       we just parsed (size mismatch → SIGBUS, or silent mis-read).
       NOTE: mmap offset must be page-aligned — map from the aligned
       boundary and shift base by the delta, else EINVAL always triggers
       the malloc+fread fallback and loads the whole file into RAM. */
    int fd = fileno(f);
    inf->mapped = 0;
    {
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
        if (fseeko(f, (off_t)(8 + hl), SEEK_SET)) die("seek");
        inf->base = xm(dsz ? dsz : 1);
        if (dsz && fread(inf->base, 1, dsz, f) != dsz) die("data");
    }
    fclose(f);
    for (int i = 0; i < inf->n; i++) inf->t[i].data = inf->base + inf->t[i].off;
    return inf;
}

/*@ requires valid_read_string(d);
    assigns \nothing;
    ensures \result == 1 || \result == 2 || \result == 4 || \result == 8;
*/
static int dtb(const char *d) {
    if (!strcmp(d, "F64") || !strcmp(d, "I64") || !strcmp(d, "U64")) return 8;
    if (!strcmp(d, "F32") || !strcmp(d, "I32") || !strcmp(d, "U32")) return 4;
    if (!strcmp(d, "BF16") || !strcmp(d, "F16") || !strcmp(d, "I16") || !strcmp(d, "U16")) return 2;
    return 1;
}
/* dtype width in BITS (packed sub-byte dtypes exist in the spec); -1 = unknown */
/*@ requires valid_read_string(d);
    assigns \nothing;
    ensures \result == -1 || \result == 4 || \result == 6 || \result == 8 ||
            \result == 16 || \result == 32 || \result == 64;
*/
static int dbits(const char *d) {
    if (!strcmp(d, "F64") || !strcmp(d, "I64") || !strcmp(d, "U64")) return 64;
    if (!strcmp(d, "F32") || !strcmp(d, "I32") || !strcmp(d, "U32")) return 32;
    if (!strcmp(d, "BF16") || !strcmp(d, "F16") || !strcmp(d, "I16") || !strcmp(d, "U16")) return 16;
    if (!strcmp(d, "BOOL") || !strcmp(d, "U8") || !strcmp(d, "I8") ||
        !strcmp(d, "F8_E4M3") || !strcmp(d, "F8_E5M2") || !strcmp(d, "F8_E8M0")) return 8;
    if (!strcmp(d, "F6_E2M3") || !strcmp(d, "F6_E3M2")) return 6;
    if (!strcmp(d, "F4") || !strcmp(d, "I4") || !strcmp(d, "U4")) return 4;
    return -1;
}
/* safetensors invariant: len == prod(shape) * dtype_bits / 8 — the spec
   uses integer division, so sub-byte dtypes (F4/F6) whose product isn't a
   multiple of 8 are still valid (the remainder bits are simply absent).
   Also bounds pos_dec's ctx index (po*K/P) to < K. Unknown dtypes are
   treated as opaque bytes (round-trip is still byte-exact). */
/*@ requires \valid_read(t);
    requires 0 <= t->nd <= 64;
    requires valid_read_string(t->dtype);
    terminates \false;
    assigns \nothing;
    exits \exit_status == 1;
*/
static void ck_shape_len(const Tensor *t) {
    int bits = dbits(t->dtype); if (bits < 0) bits = 8;
    unsigned __int128 prod = 1;
    /*@ loop invariant 0 <= d <= t->nd;
        loop assigns d, prod;
        loop variant t->nd - d;
    */
    for (int d = 0; d < t->nd; d++) {
        if (t->shape[d] < 0) die("bad shape");
        prod *= (uint64_t)t->shape[d];
        if (prod > UINT64_MAX) die("bad shape");
    }
    if (prod * (unsigned)bits / 8 != (unsigned __int128)t->len) die("shape/len mismatch");
}
/*@ requires valid_read_string(d);
    assigns \nothing;
    ensures \result == 0 || \result == 1;
*/
static int is_bf(const char *d) { return !strcmp(d, "BF16"); }
/*@ requires valid_read_string(d);
    assigns \nothing;
    ensures \result == 0 || \result == 1;
*/
static int is_f16(const char *d) { return !strcmp(d, "F16"); }
/*@ requires valid_read_string(d);
    assigns \nothing;
    ensures \result == 0 || \result == 1;
*/
static int is_f32(const char *d) { return !strcmp(d, "F32"); }
/*@ requires valid_read_string(d);
    assigns \nothing;
    ensures \result == 0 || \result == 1;
*/
static int is_flt16(const char *d) { return is_bf(d) || is_f16(d); }
/*@ requires valid_read_string(d);
    assigns \nothing;
    ensures \result == 7 || \result == 10;
*/
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
static uint64_t *gch[NCH];                /* committed model state (main thread) */
static _Thread_local uint64_t **gh;       /* active channel array; NULL = unmaterialized */
static _Thread_local uint64_t **gsnp;     /* batch snapshot backing lazy clones (workers) */
static int g_threads = 1;
/* resident decoded bytes (d/v): bounded so a hostile archive dies loudly
   instead of letting overcommit turn xc() success into a SIGKILL */
static volatile uint64_t g_dlive;
static uint64_t g_dlim = UINT64_MAX;
static void dlim_init(void) {
    /* MemAvailable counts reclaimable page cache; _SC_AVPHYS_PAGES is only
       freeram, which Linux keeps near zero (cache uses the rest) — using it
       would spuriously kill legitimate decodes */
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char ln[256];
        while (fgets(ln, sizeof ln, f))
            if (!strncmp(ln, "MemAvailable:", 13)) {
                errno = 0;
                uint64_t kb = strtoull(ln + 13, 0, 10);
                if (!errno && kb <= UINT64_MAX / 1024) g_dlim = kb * 1024;
                break;
            }
        fclose(f);
    }
    if (g_dlim == UINT64_MAX) {
        long ap = sysconf(_SC_AVPHYS_PAGES), ps = sysconf(_SC_PAGESIZE);
        if (ap > 0 && ps > 0) g_dlim = (uint64_t)ap * (uint64_t)ps;
    }
}
static void mkpath(const char *p) {   /* mkdir -p semantics */
    char tmp[4096]; size_t l = strlen(p);
    if (!l || l >= sizeof tmp) die("outdir path too long");
    memcpy(tmp, p, l + 1);
    for (char *s = tmp + 1; *s; s++)
        if (*s == '/') { *s = 0; mkdir(tmp, 0755); *s = '/'; }
    mkdir(tmp, 0755);
}

static void model_init(void) {
    for (int c = 0; c < NCH; c++) {
        gh[c] = xm(csz[c] * 8);
        for (int i = 0; i < csz[c]; i++) gh[c][i] = 1;
    }
}
/* channel access for writing: materializes a lazy worker clone from the
   batch snapshot on first touch. Main thread channels are always live. */
static uint64_t *H(int c) {
    uint64_t *p = gh[c];
    if (!p) { p = xm(csz[c] * 8); memcpy(p, gsnp[c], csz[c] * 8); gh[c] = p; }
    return p;
}
static uint64_t *hclone(int c) {
    const uint64_t *s = gh[c] ? gh[c] : gsnp[c];
    uint64_t *h = xm(csz[c] * 8); memcpy(h, s, csz[c] * 8); return h;
}
static void hcommit(int c, uint64_t *h) {
    if (!gh[c]) gh[c] = xm(csz[c] * 8);
    memcpy(gh[c], h, csz[c] * 8); free(h);
}

/*@ requires \valid_read(h + (0 .. aw - 1));
    requires \valid(f + (0 .. aw - 1));
    requires 1 <= aw <= 32768;
    requires \forall integer i; 0 <= i < aw ==> h[i] <= 281474976710655;
    assigns f[0 .. aw - 1];
    ensures (\exists integer j; 0 <= j < aw && h[j] > 0) ==>
            \forall integer i; 0 <= i < aw && h[i] == 0 ==> f[i] == 0;
    ensures \forall integer i; 0 <= i < aw && h[i] > 0 ==> f[i] >= 1;
    ensures \forall integer i; 0 <= i < aw ==> f[i] <= 32768;
    ensures !(\exists integer j; 0 <= j < aw && h[j] > 0) ==>
            \forall integer i; 0 <= i < aw ==> f[i] >= 1;
*/
static void norm_ctx(const uint64_t *h, uint16_t *f, int aw) {
    uint64_t tot = 0; int nz = 0;
    /*@ loop invariant 0 <= i <= aw;
        loop invariant 0 <= nz <= i;
        loop invariant tot <= (uint64_t)i * 281474976710655;
        loop invariant \forall integer j; 0 <= j < i ==> h[j] <= tot;
        loop invariant nz > 0 ==> \exists integer j; 0 <= j < i && h[j] > 0;
        loop invariant nz == 0 ==> \forall integer j; 0 <= j < i ==> h[j] == 0;
        loop assigns tot, nz, i;
        loop variant aw - i;
    */
    for (int i = 0; i < aw; i++) { tot += h[i]; if (h[i]) nz++; }
    if (!nz) {
        uint16_t v = TOT / aw;
        /*@ assert v >= 1 && (uint32_t)v * aw <= TOT; */
        /*@ loop invariant 0 <= i <= aw;
            loop invariant \forall integer j; 0 <= j < i ==> f[j] == v;
            loop assigns f[0 .. aw - 1], i;
            loop variant aw - i;
        */
        for (int i = 0; i < aw; i++) f[i] = v;
        f[aw - 1] += TOT - (uint32_t)v * aw;
        return;
    }
    int64_t rem = TOT, bud = TOT - nz; int bi = 0;
    /*@ assert bud >= 0; */
    /*@ ghost int64_t sq = 0; */
    if (tot < ((uint64_t)1 << 49)) {   /* h[i]<=tot => h[i]*bud < 2^64: u64 mul safe */
        /*@ loop invariant 0 <= i <= aw;
            loop invariant 0 <= bi < aw;
            loop invariant \forall integer j; 0 <= j < i ==> h[j] <= h[bi];
            loop invariant \forall integer j; 0 <= j < i && h[j] == 0 ==> f[j] == 0;
            loop invariant \forall integer j; 0 <= j < i && h[j] > 0 ==> f[j] >= 1 && f[j] <= 32768;
            loop invariant rem <= TOT && rem == TOT - sq && rem >= TOT - (int64_t)i * 4294967295;
            loop invariant sq >= 0;
            loop invariant \forall integer j; 0 <= j < i ==> f[j] <= sq;
            loop assigns f[0 .. aw - 1], bi, rem, i, sq;
            loop variant aw - i;
        */
        for (int i = 0; i < aw; i++) {
            if (h[i] > h[bi]) bi = i;
            /*@ assert h[i] <= tot; */
            /*@ assert h[i] * (uint64_t)bud <= tot * (uint64_t)bud; */
            /*@ assert h[i] * (uint64_t)bud < tot * 4294967296; */
            /*@ assert h[i] * (uint64_t)bud / tot <= (uint64_t)bud; */
            uint32_t q = h[i] ? (uint32_t)(h[i] * (uint64_t)bud / tot) + 1 : 0;
            /*@ assert h[i] > 0 ==> q >= 1 && q <= (uint32_t)bud + 1; */
            f[i] = (uint16_t)q;
            rem -= q;
            /*@ ghost sq += q; */
        }
    } else {   /* 2^49 <= tot < 2^64: u128 keeps h[i]*bud exact.
                * (tot itself is still u64 — beyond 2^64 it wraps and
                * every quotient inflates; real per-block histograms
                * never approach it.) */
        /*@ loop invariant 0 <= i <= aw;
            loop invariant 0 <= bi < aw;
            loop invariant \forall integer j; 0 <= j < i ==> h[j] <= h[bi];
            loop invariant \forall integer j; 0 <= j < i && h[j] == 0 ==> f[j] == 0;
            loop invariant \forall integer j; 0 <= j < i && h[j] > 0 ==> f[j] >= 1 && f[j] <= 32768;
            loop invariant rem <= TOT && rem == TOT - sq && rem >= TOT - (int64_t)i * 4294967295;
            loop invariant sq >= 0;
            loop invariant \forall integer j; 0 <= j < i ==> f[j] <= sq;
            loop assigns f[0 .. aw - 1], bi, rem, i, sq;
            loop variant aw - i;
        */
        for (int i = 0; i < aw; i++) {
            if (h[i] > h[bi]) bi = i;
            /*@ assert h[i] <= tot; */
            /*@ assert h[i] * (uint64_t)bud <= tot * (uint64_t)bud; */
            /*@ assert h[i] * (uint64_t)bud < tot * 4294967296; */
            /*@ assert h[i] * (uint64_t)bud / tot <= (uint64_t)bud; */
            uint32_t q = h[i] ? (uint32_t)(((unsigned __int128)h[i] * (uint64_t)bud) / tot) + 1 : 0;
            /*@ assert h[i] > 0 ==> q >= 1 && q <= (uint32_t)bud + 1; */
            f[i] = (uint16_t)q;
            rem -= q;
            /*@ ghost sq += q; */
        }
    }
    /*@ assert f[bi] <= sq; */
    /*@ assert rem <= TOT - f[bi]; */
    /*@ assert 0 <= bi < aw; */
    /*@ assert \forall integer j; 0 <= j < aw ==> h[j] <= h[bi]; */
    /*@ assert \exists integer j; 0 <= j < aw && h[j] > 0; */
    /*@ assert h[bi] > 0; */
    /* rem>=0 by construction (sum of floors <= bud) — proven at model level
       in Norm.v and swept on 3,264 tables.  WP needs the (rem>0) gate to
       prove f[bi]>=q_bi>=1 without the quotient-sum bound; CBMC needs the
       unconditional add so sum==TOT stays structural rather than requiring
       the solver to prove rem>=0 through the division circuit.  The two
       forms differ only on rem<0 — unreachable per Norm.v. */
#ifdef __CPROVER__
    f[bi] += (uint32_t)rem;
#else
    f[bi] += (uint32_t)rem * (rem > 0);
#endif
}

/* ================= rANS ================= */
/*@ requires 1 <= f <= TOT;
    requires 0 <= c <= TOT - f;
    requires LOWER <= x && x < 256 * LOWER;
    requires \valid(pp);
    requires \valid(*pp + (-2 .. -1));
    assigns *pp, (*pp)[-2 .. -1] \from *pp, x, f;
    ensures LOWER <= \result && \result < 256 * LOWER;
    ensures \at(*pp, Pre) - 2 <= *pp && *pp <= \at(*pp, Pre);
*/
static inline uint64_t enc(uint64_t x, uint32_t f, uint32_t c, uint8_t **pp) {
    /* f==0 (a zeroed cell) would loop forever marching pp below the scratch
       buffer — impossible under the all-ones model invariant; guard anyway */
    if (!f) die("zero freq");
    uint8_t *w = *pp;
    /*@ loop invariant (uint64_t)f * 65536 <= x && x < 256 * LOWER;
        loop invariant (w == \at(*pp, Pre) && x == \at(x, Pre)) ||
                       (w == \at(*pp, Pre) - 1 && 256 * x <= \at(x, Pre) && \at(x, Pre) <= 256 * x + 255) ||
                       (w == \at(*pp, Pre) - 2 && 65536 * x <= \at(x, Pre) && \at(x, Pre) <= 65536 * x + 65535);
        loop assigns w, \at(*pp, Pre)[-2 .. -1], x;
        loop variant x;
    */
    while (x >= XMAX(f)) { *--w = (uint8_t)x; x /= 256; }
    *pp = w;
    uint64_t xf = x / f;
    /*@ assert f * xf <= x && x < f * (xf + 1); */
    /*@ assert xf >= 65536; */
    /*@ assert xf <= 16777215; */
    /*@ assert xf * TOT >= 2147483648; */
    /*@ assert (x % f) + c <= TOT - 1; */
    return xf * TOT + (x % f) + c;
}
/*@ requires 1 <= f <= TOT;
    requires \valid(rp);
    requires 0 <= end - *rp;
    requires \valid_read(*rp + (0 .. end - *rp - 1));
    assigns *rp \from *rp, x, f, c, end, (*rp)[0 .. end - *rp - 1];
    ensures 0 <= *rp - \at(*rp, Pre) && *rp - \at(*rp, Pre) <= end - \at(*rp, Pre);
    ensures \base_addr(*rp) == \base_addr(\at(*rp, Pre));
*/
static inline uint64_t dec(uint64_t x, uint32_t f, uint32_t c, const uint8_t **rp, const uint8_t *end) {
    x = (uint64_t)f * (x / TOT) + (x % TOT) - c;
    const uint8_t *r = *rp;
    const ptrdiff_t cap = end - r;
    ptrdiff_t n = 0;
    /*@ loop invariant 0 <= n && n <= cap;
        loop assigns x, n;
        loop variant cap - n;
    */
    while (x < LOWER && n < cap) { x = x * 256 + r[n]; n++; }
    *rp = r + n;
    return x;
}
/* decode-side symbol lookup: binary search for the unique cell with
   cum[s] <= v < cum[s+1].  fc>0 is load-bearing: dec() divides by it. */
/*@ requires 1 <= aw <= 65536;
    requires \valid_read(f + (0 .. aw - 1));
    requires \valid_read(cum + (0 .. aw));
    requires \valid(fc);
    requires cum[0] == 0;
    requires v < cum[aw];
    requires \forall integer i; 0 <= i < aw ==> cum[i] <= cum[i + 1];
    requires \forall integer i; 0 <= i < aw && cum[i] < cum[i + 1] ==> f[i] > 0;
    assigns *fc;
    ensures 0 <= \result < aw;
    ensures *fc == f[\result];
    ensures *fc > 0;
*/
static inline uint32_t dsym(const uint16_t *f, const uint32_t *cum, int aw, uint32_t v, uint32_t *fc) {
    int lo = 0, hi = aw - 1;
    /*@ loop invariant 0 <= lo <= hi <= aw - 1;
        loop invariant cum[lo] <= v < cum[hi + 1];
        loop assigns lo, hi;
        loop variant hi - lo;
    */
    while (lo < hi) { int m = (lo + hi) >> 1; if (v >= cum[m + 1]) lo = m + 1; else hi = m; }
    uint32_t r = lo;
    //@ assert cum[r] <= v < cum[r + 1];
    //@ assert cum[r] < cum[r + 1] ==> f[r] > 0;
    *fc = f[r];
    return r;
}
/* memory-safety-only variant of dsym: same binary search, but requires no
   cumulative-table invariants and guarantees only an in-range index (the
   monotone-cum property that makes the index *correct* is a construction
   invariant proved at model level; the caller runtime-checks *fc before
   dec() divides by it). */
/*@ requires 1 <= aw <= 65536;
    requires \valid_read(f + (0 .. aw - 1));
    requires \valid_read(cum + (0 .. aw));
    requires \valid(fc);
    assigns *fc;
    ensures 0 <= \result < aw;
    ensures *fc == f[\result];
*/
static inline uint32_t dsym_raw(const uint16_t *f, const uint32_t *cum, int aw, uint32_t v, uint32_t *fc) {
    int lo = 0, hi = aw - 1;
    /*@ loop invariant 0 <= lo <= hi <= aw - 1;
        loop assigns lo, hi;
        loop variant hi - lo;
    */
    while (lo < hi) { int m = (lo + hi) >> 1; if (v >= cum[m + 1]) lo = m + 1; else hi = m; }
    *fc = f[lo];
    return (uint32_t)lo;
}
/* emit one rANS block: [u32 len][8B state][bytes] */
/*@ requires 0 <= scr_end - pp <= 4294967295;
    requires \valid_read(pp + (0 .. scr_end - pp - 1));
    requires \valid(o + (0 .. 11));
    requires \valid(o + (12 .. 11 + (scr_end - pp)));
    requires \separated(o + (0 .. 11 + (scr_end - pp)), pp + (0 .. scr_end - pp - 1));
    assigns o[0 .. 11 + (scr_end - pp)] \from o, scr_end, pp, x,
        pp[0 .. scr_end - pp - 1];
    assigns \result \from o, scr_end, pp;
    ensures \result == o + 12 + (scr_end - pp);
*/
static uint8_t *emit_blk(uint8_t *o, uint8_t *scr_end, uint8_t *pp, uint64_t x) {
    /*@ loop invariant 0 <= b <= 8;
        loop assigns o[4 .. 11], b;
        loop variant 8 - b;
    */
    for (int b = 0; b < 8; b++) o[4 + b] = (uint8_t)(x >> (8 * (7 - b)));
    const ptrdiff_t bl = scr_end - pp;
    p32le(o, (uint32_t)bl);
    /*@ loop invariant 0 <= i <= bl;
        loop invariant \forall integer j; 0 <= j < i ==> o[12 + j] == pp[j];
        loop assigns o[12 .. 11 + bl], i;
        loop variant bl - i;
    */
    for (ptrdiff_t i = 0; i < bl; i++) o[12 + i] = pp[i];
    return o + 12 + bl;
}
/*@ requires 0 <= lim - rp;
    requires \valid_read(rp + (0 .. lim - rp - 1));
    requires \valid(x);
    requires \valid(end);
    terminates \false;
    assigns *x \from rp[0 .. 11];
    assigns *end \from rp, lim, rp[0 .. 3];
    assigns \result \from rp;
    exits \exit_status == 1;
    ensures \result == rp + 12;
    ensures 12 <= *end - rp && *end - rp <= lim - rp;
    ensures \base_addr(\result) == \base_addr(rp);
    ensures \base_addr(*end) == \base_addr(rp);
*/
static const uint8_t *read_blk(const uint8_t *rp, const uint8_t *lim, uint64_t *x, const uint8_t **end) {
    if (rp > lim || (uint64_t)(lim - rp) < 12) die("corrupt archive");
    /*@ assert 12 <= lim - rp; */
    uint32_t bl = (uint32_t)rp[0] + 256u * rp[1] + 65536u * rp[2] + 16777216u * rp[3];
    rp += 4;
    *x = 0;
    /*@ loop invariant 0 <= b <= 8;
        loop assigns *x, b;
        loop variant 8 - b;
    */
    for (int b = 0; b < 8; b++) *x = (*x << 8) | rp[b];
    rp += 8;
    if ((uint64_t)bl > (uint64_t)(lim - rp)) die("corrupt archive");
    *end = rp + bl;
    /*@ assert 0 <= *end - rp && 0 <= lim - *end; */
    return rp;
}

/* ============ FIELD 16-bit ============ */
/* Each emitted block is a self-contained bitstream (x starts at LOWER, the
   block header carries x + payload len), so once a block's ft snapshot is
   known its rANS encode is independent of every other block.  The parallel
   path runs per wave: [norm block ft's from running h + count block into h]
   sequential (the model dependency chain), [backward rANS encode] parallel,
   [emit_blk] sequential.  ft/cum are identical to the serial pass, so the
   output bytes are identical. */
typedef struct {
    const uint16_t *s; uint64_t n; int mb;
    const uint16_t *fts; size_t fsn;
    uint64_t blo, bhi;                 /* block index range (absolute) */
    uint8_t **bufs; uint32_t *lens; uint64_t *xs;   /* wave-local outputs */
    uint32_t *h32;                     /* wave-local per-block count hists */
    uint8_t *useds;                    /* wave-local [blk][2*ew] ctx masks */
    int mode;                          /* 0 = count, 1 = encode */
    int spawned;
} F16W;
static void *f16w_run(void *a) {
    F16W *w = a;
    int ew = 1 << (15 - w->mb), mw = 1 << w->mb;
    if (!w->mode) {
        /* phase A1: per-block histograms into u32 (block <= BLK counts) */
        for (uint64_t b = w->blo; b < w->bhi; b++) {
            uint64_t b0 = b * BLK, bn = w->n - b0 < BLK ? w->n - b0 : BLK;
            uint32_t *bh = w->h32 + (size_t)(b - w->blo) * w->fsn;
            uint8_t *used = w->useds + (size_t)(b - w->blo) * 2 * ew;
            memset(bh, 0, w->fsn * 4); memset(used, 0, 2 * (size_t)ew);
            for (uint64_t i = 0; i < bn; i++) {
                uint32_t v = w->s[b0 + i], S = v >> 15, E = (v >> w->mb) & (ew - 1), Mv = v & (mw - 1);
                used[S * ew + E] = 1;
                bh[S]++; bh[2 + S * ew + E]++; bh[2 + 2 * ew + (size_t)(S * ew + E) * mw + Mv]++;
            }
        }
        return 0;
    }
    uint8_t *scr = xm(SCRSZ);
    uint32_t *cum = xm((4 + 2 * (ew + 1) + (size_t)2 * ew * (mw + 1)) * 4);
    for (uint64_t b = w->blo; b < w->bhi; b++) {
        uint64_t b0 = b * BLK, bn = w->n - b0 < BLK ? w->n - b0 : BLK;
        const uint16_t *fS = w->fts + (size_t)(b - w->blo) * w->fsn;
        const uint16_t *fE = fS + 2, *fM = fS + 2 + 2 * ew;
        uint32_t *cS = cum, *cE = cum + 4, *cM = cum + 4 + 2 * (ew + 1);
        cS[0] = 0; cS[1] = fS[0]; cS[2] = (uint32_t)fS[0] + fS[1];
        for (int c = 0; c < 2; c++) { uint32_t *u = cE + c * (ew + 1); u[0] = 0; for (int i = 0; i < ew; i++) u[i + 1] = u[i] + fE[c * ew + i]; }
        const uint8_t *used = w->useds + (size_t)(b - w->blo) * 2 * ew;
        for (int c = 0; c < 2 * ew; c++) if (used[c]) { uint32_t *u = cM + (size_t)c * (mw + 1); u[0] = 0; for (int i = 0; i < mw; i++) u[i + 1] = u[i] + fM[(size_t)c * mw + i]; }
        uint8_t *pp = scr + SCRSZ;
        uint64_t x = LOWER;
        for (uint64_t i = bn; i-- > 0;) {
            uint32_t v = w->s[b0 + i], S = v >> 15, E = (v >> w->mb) & (ew - 1), Mv = v & (mw - 1);
            size_t se = (size_t)S * ew + E;
            x = enc(x, fM[se * mw + Mv], cM[se * (mw + 1) + Mv], &pp);
            x = enc(x, fE[S * ew + E], cE[S * (ew + 1) + E], &pp);
            x = enc(x, fS[S], cS[S], &pp);
        }
        uint64_t bl = (uint64_t)(scr + SCRSZ - pp);
        uint64_t lb = b - w->blo;             /* wave-local slot */
        w->bufs[lb] = xm(bl ? bl : 1);
        memcpy(w->bufs[lb], pp, bl);
        w->lens[lb] = (uint32_t)bl; w->xs[lb] = x;
    }
    free(scr); free(cum);
    return 0;
}
static size_t f16_enc(const uint16_t *s, uint64_t n, int mb, uint8_t *out, uint64_t *h, int nthr) {
    int ew = 1 << (15 - mb), mw = 1 << mb;
    uint8_t *o = out;
    uint64_t nblk = (n + BLK - 1) / BLK;
    if (nthr > 1 && nblk >= 2) {
        /* wave-bounded parallel path: WCAP blocks at a time.
           A1 parallel per-block counting (u32 hists, increments commute) ->
           A2 serial norm(ft from running h) + merge counts -> B parallel
           rANS encode -> C serial in-order emit.  ft/cum identical to the
           serial pass -> identical bytes. */
        size_t fsn = 2 + 2 * (size_t)ew + 2 * (size_t)ew * mw;
        uint64_t wcap = (uint64_t)nthr * 8;
        if (wcap > nblk) wcap = nblk;
        uint16_t *fts = xm((size_t)wcap * fsn * 2);
        uint32_t *h32 = xc((size_t)wcap * fsn, 4);
        uint8_t *useds = xc((size_t)wcap * 2 * ew, 1);
        uint8_t **bufs = xc(wcap, sizeof(uint8_t *));
        uint32_t *lens = xc(wcap, 4);
        uint64_t *xs = xc(wcap, 8);
        F16W *wj = xc(nthr, sizeof(F16W));
        pthread_t *th = xc(nthr, sizeof(pthread_t));
        for (uint64_t bs = 0; bs < nblk; bs += wcap) {
            uint64_t wn = nblk - bs < wcap ? nblk - bs : wcap;
            int sp = (uint64_t)nthr < wn ? nthr : (int)wn;
            uint64_t per = (wn + sp - 1) / sp;
            for (int phase = 0; phase < 2; phase++) {
                uint64_t lo = bs;
                for (int k = 0; k < sp; k++) {
                    F16W *w = &wj[k];
                    w->s = s; w->n = n; w->mb = mb; w->mode = phase;
                    w->fts = fts + (size_t)(lo - bs) * fsn;
                    w->h32 = h32 + (size_t)(lo - bs) * fsn;
                    w->fsn = fsn; w->blo = lo;
                    w->bhi = lo + per < bs + wn ? lo + per : bs + wn;
                    w->bufs = bufs + (lo - bs); w->lens = lens + (lo - bs);
                    w->xs = xs + (lo - bs); w->spawned = 0;
                    w->useds = useds + (size_t)(lo - bs) * 2 * ew;
                    lo = w->bhi;
                    if (pthread_create(&th[k], 0, f16w_run, w)) f16w_run(w);
                    else w->spawned = 1;
                }
                for (int k = 0; k < sp; k++) if (wj[k].spawned) pthread_join(th[k], 0);
                if (phase) continue;
                /* A2: ft[b] = norm(h) then h += hist32[b], in block order */
                for (uint64_t b = 0; b < wn; b++) {
                    uint16_t *ft = fts + (size_t)b * fsn;
                    const uint32_t *bh = h32 + (size_t)b * fsn;
                    const uint8_t *u = useds + (size_t)b * 2 * ew;
                    norm_ctx(h, ft, 2);
                    for (int c = 0; c < 2; c++) norm_ctx(h + 2 + c * ew, ft + 2 + c * ew, ew);
                    for (int c = 0; c < 2 * ew; c++) if (u[c])
                        norm_ctx(h + 2 + 2 * ew + (size_t)c * mw, ft + 2 + 2 * ew + (size_t)c * mw, mw);
                    for (size_t i = 0; i < fsn; i++) h[i] += bh[i];
                }
            }
            for (uint64_t b = 0; b < wn; b++) {          /* emit, in order */
                uint8_t *bf = bufs[b];
                o = emit_blk(o, bf + lens[b], bf, xs[b]);
                free(bf);
            }
        }
        free(fts); free(h32); free(useds); free(bufs); free(lens); free(xs); free(wj); free(th);
        return o - out;
    }
    uint16_t *ft = xm((2 + 2 * ew + (size_t)2 * ew * mw) * 2);
    uint32_t *cum = xm((4 + 2 * (ew + 1) + (size_t)2 * ew * (mw + 1)) * 4);
    uint8_t *scr = xm(SCRSZ);
    uint8_t *used = xm(2 * (size_t)ew);
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        uint16_t *fS = ft, *fE = ft + 2, *fM = ft + 2 + 2 * ew;
        /* used-se mask: real tensors touch only a few dozen of the 2*ew
           exponent contexts — norm only those mantissa rows (unused rows'
           ft values are never read by either side) */
        memset(used, 0, 2 * (size_t)ew);
        for (uint64_t i = 0; i < bn; i++)
            used[(s[b0 + i] >> mb) & (2 * ew - 1)] = 1;
        norm_ctx(h, fS, 2);
        for (int c = 0; c < 2; c++) norm_ctx(h + 2 + c * ew, fE + c * ew, ew);
        for (int c = 0; c < 2 * ew; c++) if (used[c])
            norm_ctx(h + 2 + 2 * ew + (size_t)c * mw, fM + (size_t)c * mw, mw);
        uint32_t *cS = cum, *cE = cum + 4, *cM = cum + 4 + 2 * (ew + 1);
        cS[0] = 0; cS[1] = fS[0]; cS[2] = (uint32_t)fS[0] + fS[1];
        for (int c = 0; c < 2; c++) { uint32_t *b = cE + c * (ew + 1); b[0] = 0; for (int i = 0; i < ew; i++) b[i + 1] = b[i] + fE[c * ew + i]; }
        for (int c = 0; c < 2 * ew; c++) if (used[c]) { uint32_t *b = cM + (size_t)c * (mw + 1); b[0] = 0; for (int i = 0; i < mw; i++) b[i + 1] = b[i] + fM[(size_t)c * mw + i]; }
        uint8_t *pp = scr + SCRSZ;
        uint64_t x = LOWER;
        for (uint64_t i = bn; i-- > 0;) {
            uint32_t v = s[b0 + i], S = v >> 15, E = (v >> mb) & (ew - 1), Mv = v & (mw - 1);
            size_t se = (size_t)S * ew + E;
            x = enc(x, fM[se * mw + Mv], cM[se * (mw + 1) + Mv], &pp);
            x = enc(x, fE[S * ew + E], cE[S * (ew + 1) + E], &pp);
            x = enc(x, fS[S], cS[S], &pp);
            /* hist update fused into the encode pass — increments commute,
               and the encoder reads only the ft/cum snapshots, never h */
            h[S]++; h[2 + S * ew + E]++; h[2 + 2 * ew + se * mw + Mv]++;
        }
        o = emit_blk(o, scr + SCRSZ, pp, x);
    }
    free(ft); free(cum); free(scr); free(used);
    return o - out;
}
/*@ requires 1 <= ew <= 32768;
    requires 1 <= mw <= 32768 && ew * mw == 32768;
    requires 0 <= mb <= 15;
    requires \valid_read(ft + (0 .. 2 * ew + 65537));
    requires \valid_read(cum + (0 .. 4 * ew + 65537));
    requires \valid(rp);
    requires 0 <= end - *rp;
    requires \valid_read(*rp + (0 .. end - *rp - 1));
    requires \valid(si);
    requires \valid(h + (0 .. 2 * ew + 65537));
    requires \forall integer j; 0 <= j < 2 * ew + 65538 ==>
        h[j] <= 281474976710655;
    terminates \false;
    exits \exit_status == 1;
    assigns *rp \from *rp, x, end, ew, mw, mb, ft[0 .. 2 * ew + 65537],
        cum[0 .. 4 * ew + 65537], (*rp)[0 .. end - *rp - 1];
    assigns *si, h[0 .. 2 * ew + 65537];
    ensures 0 <= *rp - \at(*rp, Pre) && *rp - \at(*rp, Pre) <= end - \at(*rp, Pre);
    ensures \base_addr(*rp) == \base_addr(\at(*rp, Pre));
    ensures \forall integer j; 0 <= j < 2 * ew + 65538 ==>
        h[j] <= \at(h[j], Pre) + 1;
*/
static uint64_t f16_elem(uint64_t x, const uint8_t **rp, const uint8_t *end,
                         const uint16_t *ft, const uint32_t *cum,
                         int ew, int mw, int mb, uint16_t *si, uint64_t *h) {
    const uint16_t *fS = ft, *fE = ft + 2, *fM = ft + 2 + 2 * ew;
    const uint32_t *cE = cum, *cM = cum + 2 * ew + 2;
    const uint8_t *r = *rp;
    uint32_t v = (uint32_t)(x & (TOT - 1)), fc, S, E, M;
    int se;
    if (v >= fS[0]) {
        S = 1;
        if (!fS[1] || fS[1] > TOT) die("corrupt f16");
        x = dec(x, fS[1], fS[0], &r, end);
        v = (uint32_t)(x & (TOT - 1));
        E = dsym_raw(fE + ew, cE + ew + 1, ew, v, &fc);
        if (!fc || fc > TOT) die("corrupt f16");
        x = dec(x, fc, cE[ew + 1 + E], &r, end);
        se = ew + (int)E;
    } else {
        S = 0;
        if (!fS[0] || fS[0] > TOT) die("corrupt f16");
        x = dec(x, fS[0], 0, &r, end);
        v = (uint32_t)(x & (TOT - 1));
        E = dsym_raw(fE, cE, ew, v, &fc);
        if (!fc || fc > TOT) die("corrupt f16");
        x = dec(x, fc, cE[E], &r, end);
        se = (int)E;
    }
    v = (uint32_t)(x & (TOT - 1));
    /*@ assert 0 <= se && se + 1 <= 2 * ew; */
    if ((int64_t)se * mw + mw > 65536 ||
        (int64_t)se * (mw + 1) + mw > (int64_t)2 * ew * (mw + 1) - 1)
        die("corrupt f16");
    M = dsym_raw(fM + se * mw, cM + se * (mw + 1), mw, v, &fc);
    if (!fc || fc > TOT) die("corrupt f16");
    x = dec(x, fc, cM[se * (mw + 1) + M], &r, end);
    *si = (uint16_t)((S << 15) | (E << mb) | M);
    /*@ assert 2 + S * ew + E < 2 + 2 * ew; */
    /*@ assert 2 + 2 * ew + se * mw + M < 2 * ew + 65538; */
    h[S]++; h[2 + S * ew + E]++; h[2 + 2 * ew + se * mw + M]++;
    *rp = r;
    return x;
}
/*@ requires 1 <= ew <= 32768;
    requires 1 <= mw <= 32768 && ew * mw == 32768;
    requires 0 <= mb <= 15;
    requires \valid_read(ft + (0 .. 2 * ew + 65537));
    requires \valid_read(cum + (0 .. 4 * ew + 65537));
    requires \valid_read(r + (0 .. end - r - 1));
    requires 0 <= end - r;
    requires \valid(s + (0 .. bn - 1));
    requires \valid(h + (0 .. 2 * ew + 65537));
    requires 1 <= bn <= 131072;
    requires \forall integer j; 0 <= j < 2 * ew + 65538 ==>
        h[j] + bn <= 281474976710655;
    terminates \false;
    exits \exit_status == 1;
    assigns s[0 .. bn - 1], h[0 .. 2 * ew + 65537];
    ensures \forall integer j; 0 <= j < 2 * ew + 65538 ==>
        h[j] <= \at(h[j], Pre) + bn;
*/
static void f16_blk(uint64_t x, const uint8_t *r, const uint8_t *end,
                    const uint16_t *ft, const uint32_t *cum,
                    int ew, int mw, int mb,
                    uint64_t bn, uint16_t *s, uint64_t *h) {
    /*@ loop invariant 0 <= i <= bn;
        loop invariant 0 <= end - r;
        loop invariant \valid_read(r + (0 .. end - r - 1));
        loop invariant \forall integer j; 0 <= j < 2 * ew + 65538 ==>
            h[j] + bn - i <= 281474976710655;
        loop invariant \forall integer j; 0 <= j < 2 * ew + 65538 ==>
            h[j] <= \at(h[j], Pre) + i;
        loop assigns x, r, s[0 .. bn - 1], h[0 .. 2 * ew + 65537], i;
        loop variant bn - i;
    */
    for (uint64_t i = 0; i < bn; i++)
        x = f16_elem(x, &r, end, ft, cum, ew, mw, mb, s + i, h);
}

/*@ requires \valid(cc + (0 .. aw));
    requires \valid_read(f + (0 .. aw - 1));
    requires 1 <= aw <= 32768;
    terminates \false;
    assigns cc[0 .. aw];
    exits \exit_status == 1;
*/
static void fill_ctx(uint32_t *cc, const uint16_t *f, int aw) {
    cc[0] = 0;
    /*@ loop invariant 0 <= i <= aw;
        loop invariant \forall integer j; 0 <= j <= i ==> cc[j] <= 65535 * j;
        loop assigns cc[1 .. aw], i;
        loop variant aw - i;
    */
    for (int i = 0; i < aw; i++)
        cc[i + 1] = cc[i] + f[i];
    if (cc[aw] != TOT) die("norm bug");
}

/*@ requires 1 <= ew <= 256;
    requires 1 <= mw <= 32768 && ew * mw == 32768;
    requires \valid_read(h + (0 .. 2 * ew + 65537));
    requires \forall integer j; 0 <= j < 2 * ew + 65538 ==>
        h[j] <= 281474976710655;
    requires \valid(fM + (0 .. 65535));
    terminates \false;
    assigns fM[0 .. 65535];
    exits \exit_status == 1;
*/
static void norm_fM(const uint64_t *h, uint16_t *fM, int ew, int mw) {
    /*@ loop invariant 0 <= c <= 2 * ew;
        loop assigns fM[0 .. 65535], c;
        loop variant 2 * ew - c;
    */
    for (int c = 0; c < 2 * ew; c++) {
        if ((int64_t)c * mw + mw > 65536) die("f16 layout");
        /*@ assert c * mw + mw <= 65536; */
        norm_ctx(h + 2 + 2 * ew + (size_t)c * mw, fM + (size_t)c * mw, mw);
    }
}

/*@ requires 1 <= ew <= 256;
    requires 1 <= mw <= 32768 && ew * mw == 32768;
    requires \valid_read(h + (0 .. 2 * ew + 65537));
    requires \forall integer j; 0 <= j < 2 * ew + 65538 ==>
        h[j] <= 281474976710655;
    requires \valid(ft + (0 .. 2 * ew + 65537));
    requires \valid(cum + (0 .. 4 * ew + 65537));
    terminates \false;
    assigns ft[0 .. 2 * ew + 65537], cum[0 .. 4 * ew + 65537];
    exits \exit_status == 1;
*/
static void f16_tab(const uint64_t *h, uint16_t *ft, uint32_t *cum, int ew, int mw) {
    uint16_t *fS = ft, *fE = ft + 2, *fM = ft + 2 + 2 * ew;
    norm_ctx(h, fS, 2);
    /*@ loop invariant 0 <= c <= 2;
        loop assigns fE[0 .. 2 * ew - 1], c;
        loop variant 2 - c;
    */
    for (int c = 0; c < 2; c++) norm_ctx(h + 2 + c * ew, fE + c * ew, ew);
    norm_fM(h, fM, ew, mw);
    /* Σf == TOT exactly (Norm.v); guards dsym's v < cum[aw] domain */
    if (fS[0] + fS[1] != TOT) die("norm bug");
    uint32_t *cE = cum, *cM = cum + 2 * ew + 2;
    /*@ loop invariant 0 <= c <= 2;
        loop assigns cE[0 .. 2 * ew + 1], c;
        loop variant 2 - c;
    */
    for (int c = 0; c < 2; c++) fill_ctx(cE + c * (ew + 1), fE + c * ew, ew);
    int k = 0;
    /*@ loop invariant 0 <= c <= 2 * ew;
        loop invariant 0 <= k;
        loop assigns cM[0 .. 2 * ew * (mw + 1) - 1], c, k;
        loop variant 2 * ew - c;
    */
    for (int c = 0; c < 2 * ew; c++) {
        if ((int64_t)k + mw + 1 > (int64_t)2 * ew * (mw + 1) ||
            (int64_t)c * mw + mw > 65536)
            die("f16 layout");
        /*@ assert k + mw + 1 <= 2 * ew * (mw + 1) && c * mw + mw <= 65536; */
        fill_ctx(cM + k, fM + c * mw, mw);
        k += mw + 1;
    }
}

/*@ requires \valid_read(in + (0 .. lim - in - 1));
    requires 0 <= lim - in;
    requires mb == 7 || mb == 10;
    requires \valid(s + (0 .. n - 1));
    requires \valid(h + (0 .. (mb == 7 ? 66050 : 65602) - 1));
    requires n <= 281474976710655;
    requires \forall integer j; 0 <= j < (mb == 7 ? 66050 : 65602) ==>
        h[j] + n <= 281474976710655;
    requires \separated(s + (0 .. n - 1), h + (0 .. (mb == 7 ? 66050 : 65602) - 1));
    requires \separated(s + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(h + (0 .. (mb == 7 ? 66050 : 65602) - 1), in + (0 .. lim - in - 1));
    terminates \false;
    assigns s[0 .. n - 1], h[0 .. (mb == 7 ? 66050 : 65602) - 1];
    exits \exit_status == 1;
    ensures \forall integer j; 0 <= j < (mb == 7 ? 66050 : 65602) ==>
        h[j] <= \at(h[j], Pre) + n;
*/
static void f16_dec(const uint8_t *in, const uint8_t *lim, uint64_t n, int mb, uint16_t *s, uint64_t *h) {
    int ew, mw;
    if (mb == 7) { ew = 256; mw = 128; } else { ew = 32; mw = 1024; }
    /*@ assert ew * mw == 32768; */
    /*@ assert 2 * ew + 65538 <= (mb == 7 ? 66050 : 65602); */
    /*@ assert \forall integer j; 0 <= j < 2 * ew + 65538 ==>
        h[j] <= 281474976710655; */
    const uint8_t *rp = in;
    uint16_t ft[66050];              /* max cells over mb in {7,10} */
    uint32_t cum[66566];             /* 2(ew+1) + 2ew(mw+1) at mb=7 */
    /*@ loop invariant 0 <= b0;
        loop invariant 0 <= lim - rp;
        loop invariant in <= rp;
        loop invariant \base_addr(rp) == \base_addr(in);
        loop invariant \valid_read(rp + (0 .. lim - rp - 1));
        loop invariant \forall integer j; 0 <= j < (mb == 7 ? 66050 : 65602) ==>
            h[j] <= \at(h[j], Pre) + b0;
        loop invariant \forall integer j; 0 <= j < (mb == 7 ? 66050 : 65602) ==>
            h[j] <= \at(h[j], Pre) + n;
        loop assigns rp, s[0 .. n - 1], h[0 .. (mb == 7 ? 66050 : 65602) - 1], b0,
            ft[0 .. 66049], cum[0 .. 66565];
        loop variant n - b0;
    */
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        f16_tab(h, ft, cum, ew, mw);
        uint64_t x; const uint8_t *end;
        /*@ assert 0 <= lim - rp; */
        rp = read_blk(rp, lim, &x, &end);
        f16_blk(x, rp, end, ft, cum, ew, mw, mb, bn, s + b0, h);
        rp = end;
    }
    if (rp != lim) die("corrupt field");   /* every caller hands the exact stream end */
}

/* ============ FIELD+POS: 16-bit, exact position contexts ============
 * SE joint | position ctx;  M | SE.
 * position p(i): colwise  p = i % P   (P = last dim)
 *                rowwise  p = i / D   (P = first dim, D = n/P)
 * K = min(P, KCAP) ctxs, ctx = p*K/P — exact column when K==P.
 * hist: [SE|pos: K*sew][M|SE: 2*ew*mw]
 */
static size_t pos_enc(const uint16_t *s, uint64_t n, int mb, int64_t P, int rowwise,
                      uint8_t *out, uint64_t *h) {
    if (P <= 0 || P > ((int64_t)1 << 40)) return 0;  /* degenerate/absurd; po*K must fit i64 */
    int ew = 1 << (15 - mb), mw = 1 << mb, sew = 2 * ew;
    int64_t K = P < (int64_t)(PCAP / sew) ? P : (int64_t)(PCAP / sew);
    int64_t D = n / P;
    if (D < 1) D = 1;
    uint8_t *o = out;
    size_t tSE = (size_t)K * sew, tM = (size_t)2 * ew * mw;
    uint16_t *ft = xm((tSE + tM) * 2);
    uint32_t *cum = xm(((size_t)K * (sew + 1) + (size_t)2 * ew * (mw + 1)) * 4);
    uint8_t *scr = xm(SCRSZ);
    uint8_t *used = xm(2 * (size_t)ew);
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        /* the block's po values are contiguous (rowwise: po=gi/D monotone;
           column: a contiguous run, or a wraparound pair); ctx=po*K/P is
           monotone in po → used ctxs form a contiguous range (or two).
           norm/cum only those — untouched ctx tables are never read this
           block; the decoder normalizes per-ctx independently so the
           emitted bitstream is unchanged. */
        int64_t r0lo, r0hi, r1lo = -1, r1hi = -1;
        if (rowwise) { r0lo = (int64_t)(b0 / (uint64_t)D); r0hi = (int64_t)((b0 + bn - 1) / (uint64_t)D); }
        else if ((uint64_t)P <= bn) { r0lo = 0; r0hi = P - 1; }
        else {
            int64_t p0 = (int64_t)(b0 % (uint64_t)P), p1 = (int64_t)((b0 + bn - 1) % (uint64_t)P);
            if (p1 >= p0) { r0lo = p0; r0hi = p1; }
            else { r0lo = p0; r0hi = P - 1; r1lo = 0; r1hi = p1; }
        }
        int64_t c0lo = r0lo * K / P, c0hi = r0hi * K / P;
        int64_t c1lo = r1lo < 0 ? 0 : r1lo * K / P, c1hi = r1lo < 0 ? -1 : r1hi * K / P;
        /* po can reach P when P∤n (ctx=K aliases the mantissa table area —
           pre-existing self-consistent behavior); K's cells are rewritten by
           the mantissa norms below, so clamp it out of the ctx range */
        if (c0hi > K - 1) c0hi = K - 1;
        if (c1hi > K - 1) c1hi = K - 1;
        for (int64_t c = c0lo; c <= c0hi; c++) norm_ctx(h + c * sew, ft + c * sew, sew);
        for (int64_t c = c1lo; c <= c1hi; c++) norm_ctx(h + c * sew, ft + c * sew, sew);
        /* mantissa rows conditioned on SE = v>>mb: real blocks touch only a
           few dozen of 2*ew — norm only used rows (same trick as f16/f32) */
        memset(used, 0, 2 * (size_t)ew);
        for (uint64_t i = 0; i < bn; i++) used[s[b0 + i] >> mb] = 1;
        for (int c = 0; c < 2 * ew; c++) if (used[c])
            norm_ctx(h + tSE + (size_t)c * mw, ft + tSE + (size_t)c * mw, mw);
        uint32_t *cSE = cum, *cM = cum + (size_t)K * (sew + 1);
        for (int64_t c = c0lo; c <= c0hi; c++) { uint32_t *b = cSE + c * (sew + 1); b[0] = 0; for (int i = 0; i < sew; i++) b[i + 1] = b[i] + ft[c * sew + i]; }
        for (int64_t c = c1lo; c <= c1hi; c++) { uint32_t *b = cSE + c * (sew + 1); b[0] = 0; for (int i = 0; i < sew; i++) b[i + 1] = b[i] + ft[c * sew + i]; }
        for (int c = 0; c < 2 * ew; c++) if (used[c]) { uint32_t *b = cM + (size_t)c * (mw + 1); b[0] = 0; for (int i = 0; i < mw; i++) b[i + 1] = b[i] + ft[tSE + (size_t)c * mw + i]; }
        uint8_t *pp = scr + SCRSZ;
        uint64_t x = LOWER;
        /* po = gi%P (column) or gi/D (rowwise), ctx = po*K/P — computed
           incrementally while walking gi backward: one division to seed at
           the block's last element, then pure add/subtract steps (K<=P so
           ctx moves by at most 1 per element). Identical values, no per-
           element division. */
        uint64_t gl = b0 + bn - 1;
        int64_t po = rowwise ? (int64_t)(gl / (uint64_t)D) : (int64_t)(gl % (uint64_t)P);
        int64_t ctx = po * K / P, rem = po * K - ctx * P;
        uint64_t pob = (uint64_t)po * (uint64_t)D;   /* rowwise: next-lower po boundary */
        for (uint64_t i = bn; i-- > 0;) {
            uint64_t gi = b0 + i;
            uint32_t v = s[gi];
            uint32_t SE = v >> mb, Mv = v & (mw - 1);
            x = enc(x, ft[tSE + (size_t)SE * mw + Mv], cM[(size_t)SE * (mw + 1) + Mv], &pp);
            x = enc(x, ft[ctx * sew + SE], cSE[ctx * (sew + 1) + SE], &pp);
            h[ctx * sew + SE]++;
            h[tSE + (size_t)SE * mw + Mv]++;
            if (rowwise) {
                if (gi == pob && po > 0) {   /* po = gi/D steps down */
                    po--; pob -= (uint64_t)D;
                    rem -= K; if (rem < 0) { rem += P; ctx--; }
                }
            } else {
                if (po == 0) { po = P - 1; ctx = K - 1; rem = P - K; }
                else { po--; rem -= K; if (rem < 0) { rem += P; ctx--; } }
            }
        }
        o = emit_blk(o, scr + SCRSZ, pp, x);
    }
    free(ft); free(cum); free(scr); free(used);
    return o - out;
}
/*@ requires 1 <= ew <= 32768;
    requires 1 <= mw <= 32768 && ew * mw == 32768;
    requires 0 <= mb <= 15;
    requires 1 <= sew <= 32768;
    requires 0 <= tSE <= 2097152;
    requires 0 <= tC;
    requires \valid_read(ft + (0 .. tSE + 65535));
    requires \valid_read(cum + (0 .. tC + 65536 + 2 * ew - 1));
    requires \valid(rp);
    requires 0 <= end - *rp;
    requires \valid_read(*rp + (0 .. end - *rp - 1));
    requires \valid(si);
    requires \valid(h + (0 .. 2162687));
    requires \forall integer j; 0 <= j < 2162688 ==>
        h[j] <= 281474976710655;
    terminates \false;
    exits \exit_status == 1;
    assigns *rp \from *rp, x, end, ew, mw, mb, sew, tSE, tC, ctx,
        ft[0 .. tSE + 65535], cum[0 .. tC + 65536 + 2 * ew - 1],
        (*rp)[0 .. end - *rp - 1];
    assigns *si, h[0 .. 2162687];
    ensures 0 <= *rp - \at(*rp, Pre) && *rp - \at(*rp, Pre) <= end - \at(*rp, Pre);
    ensures \base_addr(*rp) == \base_addr(\at(*rp, Pre));
    ensures \forall integer j; 0 <= j < 2162688 ==>
        h[j] <= \at(h[j], Pre) + 1;
*/
static uint64_t pos_elem(uint64_t x, const uint8_t **rp, const uint8_t *end,
                         const uint16_t *ft, const uint32_t *cum,
                         int ew, int mw, int mb,
                         int64_t sew, int64_t tSE, int64_t tC, int64_t ctx,
                         uint16_t *si, uint64_t *h) {
    const uint16_t *fM = ft + tSE;
    const uint32_t *cSE = cum, *cM = cum + tC;
    const uint8_t *r = *rp;
    uint32_t v = (uint32_t)(x & (TOT - 1)), fc;
    if (ctx < 0 || ctx > tSE) die("corrupt pos");
    /*@ assert 0 <= ctx && ctx <= tSE; */
    if ((int64_t)ctx * sew + sew > tSE ||
        (int64_t)ctx * (sew + 1) + sew > tC)
        die("corrupt pos");
    /*@ assert ctx * sew + sew <= tSE && ctx * (sew + 1) + sew <= tC; */
    uint32_t SE = dsym_raw(ft + ctx * sew, cSE + ctx * (sew + 1), sew, v, &fc);
    if (!fc || fc > TOT) die("corrupt pos");
    x = dec(x, fc, cSE[ctx * (sew + 1) + SE], &r, end);
    v = (uint32_t)(x & (TOT - 1));
    if ((int64_t)SE * mw + mw > 65536 ||
        (int64_t)SE * (mw + 1) + mw > 65536 + 2 * ew - 1)
        die("corrupt pos");
    /*@ assert SE * mw + mw <= 65536 && SE * (mw + 1) + mw <= 65536 + 2 * ew - 1; */
    uint32_t M = dsym_raw(fM + SE * mw, cM + SE * (mw + 1), mw, v, &fc);
    if (!fc || fc > TOT) die("corrupt pos");
    x = dec(x, fc, cM[SE * (mw + 1) + M], &r, end);
    *si = (uint16_t)((SE << mb) | M);
    /*@ assert ctx * sew + SE < 2162688; */
    /*@ assert tSE + SE * mw + M < 2162688; */
    h[ctx * sew + SE]++;
    h[tSE + SE * mw + M]++;
    *rp = r;
    return x;
}

/*@ requires 1 <= ew <= 32768;
    requires 1 <= mw <= 32768 && ew * mw == 32768;
    requires 0 <= mb <= 15;
    requires 1 <= sew <= 32768;
    requires 0 <= tSE <= 2097152;
    requires 0 <= tC;
    requires 1 <= K <= 2097152;
    requires 1 <= P <= 1099511627776;
    requires 1 <= D;
    requires 0 <= rowwise <= 1;
    requires \valid_read(ft + (0 .. tSE + 65535));
    requires \valid_read(cum + (0 .. tC + 65536 + 2 * ew - 1));
    requires \valid_read(r + (0 .. end - r - 1));
    requires 0 <= end - r;
    requires \valid(s + (0 .. bn - 1));
    requires \valid(h + (0 .. 2162687));
    requires 1 <= bn <= 131072;
    requires \forall integer j; 0 <= j < 2162688 ==>
        h[j] + bn <= 281474976710655;
    terminates \false;
    exits \exit_status == 1;
    assigns s[0 .. bn - 1], h[0 .. 2162687];
    ensures \forall integer j; 0 <= j < 2162688 ==>
        h[j] <= \at(h[j], Pre) + bn;
*/
static void pos_blk(uint64_t x, const uint8_t *r, const uint8_t *end,
                    const uint16_t *ft, const uint32_t *cum,
                    int ew, int mw, int mb,
                    int64_t sew, int64_t K, int64_t tSE, int64_t tC,
                    int64_t P, int64_t D, int rowwise, uint64_t b0,
                    uint64_t bn, uint16_t *s, uint64_t *h) {
    /*@ loop invariant 0 <= i <= bn;
        loop invariant 0 <= end - r;
        loop invariant \valid_read(r + (0 .. end - r - 1));
        loop invariant \forall integer j; 0 <= j < 2162688 ==>
            h[j] + bn - i <= 281474976710655;
        loop invariant \forall integer j; 0 <= j < 2162688 ==>
            h[j] <= \at(h[j], Pre) + i;
        loop assigns x, r, s[0 .. bn - 1], h[0 .. 2162687], i;
        loop variant bn - i;
    */
    for (uint64_t i = 0; i < bn; i++) {
        uint64_t gi = b0 + i;
        int64_t po = rowwise ? (int64_t)(gi / (uint64_t)D) : (int64_t)(gi % (uint64_t)P);
        if (po < 0 || po >= P) die("corrupt pos");
        int64_t ctx = (int64_t)((__int128)po * K / P);
        x = pos_elem(x, &r, end, ft, cum, ew, mw, mb, sew, tSE, tC, ctx, s + i, h);
    }
}

/*@ requires 1 <= ew <= 32768;
    requires 1 <= mw <= 32768 && ew * mw == 32768;
    requires 1 <= sew <= 32768;
    requires 1 <= K <= 2097152;
    requires 0 <= tSE <= 2097152;
    requires tC == tSE + K;
    requires \valid_read(h + (0 .. 2162687));
    requires \forall integer j; 0 <= j < 2162688 ==>
        h[j] <= 281474976710655;
    requires \valid(ft + (0 .. tSE + 65535));
    requires \valid(cum + (0 .. tC + 65536 + 2 * ew - 1));
    terminates \false;
    assigns ft[0 .. tSE + 65535], cum[0 .. tC + 65536 + 2 * ew - 1];
    exits \exit_status == 1;
*/
static void pos_tab(const uint64_t *h, uint16_t *ft, uint32_t *cum,
                    int ew, int mw, int64_t sew, int64_t K, int64_t tSE, int64_t tC) {
    /*@ loop invariant 0 <= c <= K;
        loop invariant 0 <= off <= tSE;
        loop assigns ft[0 .. tSE + 65535], c, off;
        loop variant K - c;
    */
    for (int64_t c = 0, off = 0; c < K; c++, off += sew) {
        if (off + sew > tSE) die("pos layout");
        /*@ assert off + sew <= tSE; */
        norm_ctx(h + off, ft + off, sew);
    }
    int k = 0;
    /*@ loop invariant 0 <= c <= 2 * ew;
        loop invariant 0 <= k <= 65536;
        loop assigns ft[tSE .. tSE + 65535], c, k;
        loop variant 2 * ew - c;
    */
    for (int c = 0; c < 2 * ew; c++, k += mw) {
        if (k + mw > 65536) die("pos layout");
        /*@ assert k + mw <= 65536; */
        norm_ctx(h + tSE + k, ft + tSE + k, mw);
    }
    uint32_t *cSE = cum, *cM = cum + tC;
    int64_t k2 = 0, fo = 0;
    /*@ loop invariant 0 <= c <= K;
        loop invariant 0 <= k2 <= tC;
        loop invariant 0 <= fo <= tSE;
        loop assigns cSE[0 .. tC - 1], c, k2, fo;
        loop variant K - c;
    */
    for (int64_t c = 0; c < K; c++, k2 += sew + 1, fo += sew) {
        if (fo + sew > tSE || k2 + sew + 1 > tC) die("pos layout");
        /*@ assert fo + sew <= tSE && k2 + sew + 1 <= tC; */
        fill_ctx(cSE + k2, ft + fo, sew);
    }
    k = 0;
    /*@ loop invariant 0 <= c <= 2 * ew;
        loop invariant 0 <= c <= k && k <= 65536 + 2 * ew;
        loop assigns cM[0 .. 65536 + 2 * ew - 1], c, k;
        loop variant 2 * ew - c;
    */
    for (int c = 0; c < 2 * ew; c++, k += mw + 1) {
        if (k + mw + 1 > 65536 + 2 * ew || k - c + mw > 65536) die("pos layout");
        /*@ assert k + mw + 1 <= 65536 + 2 * ew && k - c + mw <= 65536; */
        fill_ctx(cM + k, ft + tSE + k - c, mw);
    }
}

/*@ requires \valid_read(in + (0 .. lim - in - 1));
    requires 0 <= lim - in;
    requires mb == 7 || mb == 10;
    requires 1 <= P <= 1099511627776;
    requires 1 <= D;
    requires 1 <= K <= 2097152;
    requires 0 <= rowwise <= 1;
    requires 0 <= tSE <= 2097152;
    requires tC == tSE + K;
    requires \valid(s + (0 .. n - 1));
    requires \valid(h + (0 .. 2162687));
    requires n <= 281474976710655;
    requires \forall integer j; 0 <= j < 2162688 ==>
        h[j] + n <= 281474976710655;
    requires \valid(ft + (0 .. tSE + 65535));
    requires \valid(cum + (0 .. tC + (mb == 7 ? 66048 : 65600) - 1));
    requires \separated(s + (0 .. n - 1), h + (0 .. 2162687));
    requires \separated(s + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(h + (0 .. 2162687), in + (0 .. lim - in - 1));
    requires \separated(ft + (0 .. tSE + 65535), s + (0 .. n - 1));
    requires \separated(ft + (0 .. tSE + 65535), h + (0 .. 2162687));
    requires \separated(ft + (0 .. tSE + 65535), in + (0 .. lim - in - 1));
    requires \separated(cum + (0 .. tC + (mb == 7 ? 66048 : 65600) - 1), s + (0 .. n - 1));
    requires \separated(cum + (0 .. tC + (mb == 7 ? 66048 : 65600) - 1), h + (0 .. 2162687));
    requires \separated(cum + (0 .. tC + (mb == 7 ? 66048 : 65600) - 1), in + (0 .. lim - in - 1));
    requires \separated(ft + (0 .. tSE + 65535), cum + (0 .. tC + (mb == 7 ? 66048 : 65600) - 1));
    terminates \false;
    assigns s[0 .. n - 1], h[0 .. 2162687],
        ft[0 .. tSE + 65535], cum[0 .. tC + (mb == 7 ? 66048 : 65600) - 1];
    exits \exit_status == 1;
    ensures \forall integer j; 0 <= j < 2162688 ==>
        h[j] <= \at(h[j], Pre) + n;
*/
static void pos_dec_ws(const uint8_t *in, const uint8_t *lim, uint64_t n, int mb,
                       int64_t P, int64_t D, int64_t K, int rowwise,
                       int64_t tSE, int64_t tC,
                       uint16_t *s, uint64_t *h, uint16_t *ft, uint32_t *cum) {
    int ew, mw;
    if (mb == 7) { ew = 256; mw = 128; } else { ew = 32; mw = 1024; }
    int64_t sew = 2 * ew;
    /*@ assert ew * mw == 32768; */
    /*@ assert 65536 + 2 * ew <= (mb == 7 ? 66048 : 65600); */
    const uint8_t *rp = in;
    /*@ loop invariant 0 <= b0;
        loop invariant 0 <= lim - rp;
        loop invariant in <= rp;
        loop invariant \base_addr(rp) == \base_addr(in);
        loop invariant \valid_read(rp + (0 .. lim - rp - 1));
        loop invariant \forall integer j; 0 <= j < 2162688 ==>
            h[j] <= \at(h[j], Pre) + b0;
        loop invariant \forall integer j; 0 <= j < 2162688 ==>
            h[j] <= \at(h[j], Pre) + n;
        loop assigns rp, s[0 .. n - 1], h[0 .. 2162687], b0,
            ft[0 .. tSE + 65535], cum[0 .. tC + (mb == 7 ? 66048 : 65600) - 1];
        loop variant n - b0;
    */
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        pos_tab(h, ft, cum, ew, mw, sew, K, tSE, tC);
        uint64_t x; const uint8_t *end;
        /*@ assert 0 <= lim - rp; */
        rp = read_blk(rp, lim, &x, &end);
        pos_blk(x, rp, end, ft, cum, ew, mw, mb, sew, K, tSE, tC, P, D,
                rowwise, b0, bn, s + b0, h);
        rp = end;
    }
    if (rp != lim) die("corrupt pos");
}

static void pos_dec(const uint8_t *in, const uint8_t *lim, uint64_t n, int mb, int64_t P, int rowwise,
                    uint16_t *s, uint64_t *h) {
    if (P <= 0 || P > ((int64_t)1 << 40)) die("bad shape");
    int ew = 1 << (15 - mb), mw = 1 << mb, sew = 2 * ew;
    int64_t K = P < (int64_t)(PCAP / sew) ? P : (int64_t)(PCAP / sew);
    int64_t D = n / P;
    if (D < 1) D = 1;
    size_t tSE = (size_t)K * sew, tC = (size_t)K * (sew + 1);
    uint16_t *ft = xm((tSE + (size_t)2 * ew * mw) * 2);
    uint32_t *cum = xc(tC + (size_t)2 * ew * (mw + 1), 4);
    pos_dec_ws(in, lim, n, mb, P, D, K, rowwise, (int64_t)tSE, (int64_t)tC, s, h, ft, cum);
    free(ft); free(cum);
}

/* ============ F32: S|E|m1(7)|m2(8)|m3(8) ============ */
static size_t f32_enc(const uint32_t *s, uint64_t n, uint8_t *out, uint64_t *h) {
    uint8_t *o = out;
    size_t nM1 = 512 * 128, nM2 = 512 * 256, nM3 = 512 * 256;
    uint16_t *ft = xm((2 + 512 + nM1 + nM2 + nM3) * 2);
    uint32_t *cum = xm((4 + 2 * 257 + 512 * 129 + 2 * 512 * 257) * 4);
    uint8_t *scr = xm(SCRSZ);
    uint8_t *used = xm(512), urow[2];
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        /* used-se mask: f32 tensors touch a few dozen of the 512 sign×exp
           contexts — norm only those mantissa rows (and the E rows for the
           sign halves actually present); unused rows are never read */
        memset(used, 0, 512); urow[0] = urow[1] = 0;
        for (uint64_t i = 0; i < bn; i++) {
            uint32_t v = s[b0 + i];
            uint32_t se = (v >> 23) & 511;   /* = S*256+E (bit8 is S) */
            used[se] = 1; urow[v >> 31] = 1;
        }
        norm_ctx(h, ft, 2);
        for (int c = 0; c < 2; c++) if (urow[c]) norm_ctx(h + 2 + c * 256, ft + 2 + c * 256, 256);
        uint16_t *f1 = ft + 2 + 512, *f2 = f1 + nM1, *f3 = f2 + nM2;
        for (int c = 0; c < 512; c++) if (used[c]) {
            norm_ctx(h + 2 + 512 + (size_t)c * 128, f1 + (size_t)c * 128, 128);
            norm_ctx(h + 2 + 512 + nM1 + (size_t)c * 256, f2 + (size_t)c * 256, 256);
            norm_ctx(h + 2 + 512 + nM1 + nM2 + (size_t)c * 256, f3 + (size_t)c * 256, 256);
        }
        uint32_t *cS = cum, *cE = cum + 4, *c1 = cE + 2 * 257, *c2 = c1 + 512 * 129, *c3 = c2 + 512 * 257;
        cS[0] = 0; cS[1] = ft[0]; cS[2] = (uint32_t)ft[0] + ft[1];
        for (int c = 0; c < 2; c++) if (urow[c]) { uint32_t *b = cE + c * 257; b[0] = 0; for (int i = 0; i < 256; i++) b[i + 1] = b[i] + ft[2 + c * 256 + i]; }
        for (int c = 0; c < 512; c++) if (used[c]) { uint32_t *b = c1 + c * 129; b[0] = 0; for (int i = 0; i < 128; i++) b[i + 1] = b[i] + f1[c * 128 + i]; }
        for (int c = 0; c < 512; c++) if (used[c]) { uint32_t *b = c2 + c * 257; b[0] = 0; for (int i = 0; i < 256; i++) b[i + 1] = b[i] + f2[c * 256 + i]; }
        for (int c = 0; c < 512; c++) if (used[c]) { uint32_t *b = c3 + c * 257; b[0] = 0; for (int i = 0; i < 256; i++) b[i + 1] = b[i] + f3[c * 256 + i]; }
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
            h[S]++; h[2 + S * 256 + E]++;
            h[2 + 512 + se * 128 + m1]++;
            h[2 + 512 + nM1 + se * 256 + m2]++;
            h[2 + 512 + nM1 + nM2 + se * 256 + m3]++;
        }
        o = emit_blk(o, scr + SCRSZ, pp, x);
    }
    free(ft); free(cum); free(scr); free(used);
    return o - out;
}
/*@ requires \valid_read(ft + (0 .. 328193));
    requires \valid_read(cum + (0 .. 329729));
    requires \valid(rp);
    requires 0 <= end - *rp;
    requires \valid_read(*rp + (0 .. end - *rp - 1));
    requires \valid(si);
    requires \valid(h + (0 .. 328193));
    requires \forall integer j; 0 <= j < 328194 ==>
        h[j] <= 281474976710655;
    terminates \false;
    exits \exit_status == 1;
    assigns *rp \from *rp, x, end, ft[0 .. 328193],
        cum[0 .. 329729], (*rp)[0 .. end - *rp - 1];
    assigns *si, h[0 .. 328193];
    ensures 0 <= *rp - \at(*rp, Pre) && *rp - \at(*rp, Pre) <= end - \at(*rp, Pre);
    ensures \base_addr(*rp) == \base_addr(\at(*rp, Pre));
    ensures \forall integer j; 0 <= j < 328194 ==>
        h[j] <= \at(h[j], Pre) + 1;
*/
static uint64_t f32_elem(uint64_t x, const uint8_t **rp, const uint8_t *end,
                         const uint16_t *ft, const uint32_t *cum,
                         uint32_t *si, uint64_t *h) {
    const uint16_t *fE = ft + 2, *f1 = ft + 514, *f2 = f1 + 65536, *f3 = f2 + 131072;
    const uint32_t *cE = cum, *c1 = cum + 514, *c2 = c1 + 66048, *c3 = c2 + 131584;
    const uint8_t *r = *rp;
    uint32_t v = (uint32_t)(x & (TOT - 1)), fc, S, E, se, m1, m2, m3;
    if (v >= ft[0]) {
        S = 1;
        if (!ft[1] || ft[1] > TOT) die("corrupt f32");
        x = dec(x, ft[1], ft[0], &r, end);
    } else {
        S = 0;
        if (!ft[0] || ft[0] > TOT) die("corrupt f32");
        x = dec(x, ft[0], 0, &r, end);
    }
    v = (uint32_t)(x & (TOT - 1));
    E = dsym_raw(fE + S * 256, cE + S * 257, 256, v, &fc);
    if (!fc || fc > TOT) die("corrupt f32");
    x = dec(x, fc, cE[S * 257 + E], &r, end);
    se = S * 256 + E;
    /*@ assert 0 <= se && se <= 511; */
    v = (uint32_t)(x & (TOT - 1));
    if ((int64_t)se * 128 + 128 > 65536 || (int64_t)se * 129 + 128 > 66048)
        die("corrupt f32");
    m1 = dsym_raw(f1 + se * 128, c1 + se * 129, 128, v, &fc);
    if (!fc || fc > TOT) die("corrupt f32");
    x = dec(x, fc, c1[se * 129 + m1], &r, end);
    v = (uint32_t)(x & (TOT - 1));
    if ((int64_t)se * 256 + 256 > 131072 || (int64_t)se * 257 + 256 > 131584)
        die("corrupt f32");
    m2 = dsym_raw(f2 + se * 256, c2 + se * 257, 256, v, &fc);
    if (!fc || fc > TOT) die("corrupt f32");
    x = dec(x, fc, c2[se * 257 + m2], &r, end);
    v = (uint32_t)(x & (TOT - 1));
    m3 = dsym_raw(f3 + se * 256, c3 + se * 257, 256, v, &fc);
    if (!fc || fc > TOT) die("corrupt f32");
    x = dec(x, fc, c3[se * 257 + m3], &r, end);
    *si = (S << 31) | (E << 23) | (m1 << 16) | (m2 << 8) | m3;
    /*@ assert 2 + S * 256 + E < 514; */
    /*@ assert 514 + se * 128 + m1 < 514 + 65536; */
    /*@ assert 514 + 65536 + se * 256 + m2 < 514 + 65536 + 131072; */
    /*@ assert 514 + 65536 + 131072 + se * 256 + m3 < 328194; */
    h[S]++;
    h[2 + S * 256 + E]++;
    h[514 + se * 128 + m1]++;
    h[514 + 65536 + se * 256 + m2]++;
    h[514 + 65536 + 131072 + se * 256 + m3]++;
    *rp = r;
    return x;
}

/*@ requires \valid_read(ft + (0 .. 328193));
    requires \valid_read(cum + (0 .. 329729));
    requires \valid_read(r + (0 .. end - r - 1));
    requires 0 <= end - r;
    requires \valid(s + (0 .. bn - 1));
    requires \valid(h + (0 .. 328193));
    requires 1 <= bn <= 131072;
    requires \forall integer j; 0 <= j < 328194 ==>
        h[j] + bn <= 281474976710655;
    terminates \false;
    exits \exit_status == 1;
    assigns s[0 .. bn - 1], h[0 .. 328193];
    ensures \forall integer j; 0 <= j < 328194 ==>
        h[j] <= \at(h[j], Pre) + bn;
*/
static void f32_blk(uint64_t x, const uint8_t *r, const uint8_t *end,
                    const uint16_t *ft, const uint32_t *cum,
                    uint64_t bn, uint32_t *s, uint64_t *h) {
    /*@ loop invariant 0 <= i <= bn;
        loop invariant 0 <= end - r;
        loop invariant \valid_read(r + (0 .. end - r - 1));
        loop invariant \forall integer j; 0 <= j < 328194 ==>
            h[j] + bn - i <= 281474976710655;
        loop invariant \forall integer j; 0 <= j < 328194 ==>
            h[j] <= \at(h[j], Pre) + i;
        loop assigns x, r, s[0 .. bn - 1], h[0 .. 328193], i;
        loop variant bn - i;
    */
    for (uint64_t i = 0; i < bn; i++)
        x = f32_elem(x, &r, end, ft, cum, s + i, h);
}

/*@ requires \valid_read(h + (0 .. 328193));
    requires \forall integer j; 0 <= j < 328194 ==>
        h[j] <= 281474976710655;
    requires \valid(ft + (0 .. 328193));
    requires \valid(cum + (0 .. 329729));
    terminates \false;
    assigns ft[0 .. 328193], cum[0 .. 329729];
    exits \exit_status == 1;
*/
static void f32_tab(const uint64_t *h, uint16_t *ft, uint32_t *cum) {
    norm_ctx(h, ft, 2);
    int off = 0;
    /*@ loop invariant 0 <= c <= 2;
        loop invariant 0 <= off <= 512 && off == c * 256;
        loop assigns ft[2 .. 513], c, off;
        loop variant 2 - c;
    */
    for (int c = 0; c < 2; c++, off += 256)
        norm_ctx(h + 2 + off, ft + 2 + off, 256);
    int k = 0;
    /*@ loop invariant 0 <= c <= 512;
        loop invariant 0 <= k <= 65536 && k == c * 128;
        loop assigns ft[514 .. 514 + 65535], c, k;
        loop variant 512 - c;
    */
    for (int c = 0; c < 512; c++, k += 128)
        norm_ctx(h + 514 + k, ft + 514 + k, 128);
    k = 0;
    /*@ loop invariant 0 <= c <= 512;
        loop invariant 0 <= k <= 131072 && k == c * 256;
        loop assigns ft[514 + 65536 .. 514 + 65536 + 131071], c, k;
        loop variant 512 - c;
    */
    for (int c = 0; c < 512; c++, k += 256)
        norm_ctx(h + 514 + 65536 + k, ft + 514 + 65536 + k, 256);
    k = 0;
    /*@ loop invariant 0 <= c <= 512;
        loop invariant 0 <= k <= 131072 && k == c * 256;
        loop assigns ft[514 + 196608 .. 514 + 196608 + 131071], c, k;
        loop variant 512 - c;
    */
    for (int c = 0; c < 512; c++, k += 256)
        norm_ctx(h + 514 + 196608 + k, ft + 514 + 196608 + k, 256);
    if (ft[0] + ft[1] != TOT) die("norm bug");
    uint32_t *cE = cum, *c1 = cum + 514, *c2 = c1 + 66048, *c3 = c2 + 131584;
    fill_ctx(cE, ft + 2, 256);
    fill_ctx(cE + 257, ft + 258, 256);
    int ko = 0, fo = 0;
    /*@ loop invariant 0 <= c <= 512;
        loop invariant 0 <= ko <= 66048 && 0 <= fo <= 65536;
        loop invariant ko == c * 129 && fo == c * 128;
        loop assigns c1[0 .. 66047], c, ko, fo;
        loop variant 512 - c;
    */
    for (int c = 0; c < 512; c++, ko += 129, fo += 128)
        fill_ctx(c1 + ko, ft + 514 + fo, 128);
    ko = 0; fo = 0;
    /*@ loop invariant 0 <= c <= 512;
        loop invariant 0 <= ko <= 131584 && 0 <= fo <= 131072;
        loop invariant ko == c * 257 && fo == c * 256;
        loop assigns c2[0 .. 131583], c, ko, fo;
        loop variant 512 - c;
    */
    for (int c = 0; c < 512; c++, ko += 257, fo += 256)
        fill_ctx(c2 + ko, ft + 514 + 65536 + fo, 256);
    ko = 0; fo = 0;
    /*@ loop invariant 0 <= c <= 512;
        loop invariant 0 <= ko <= 131584 && 0 <= fo <= 131072;
        loop invariant ko == c * 257 && fo == c * 256;
        loop assigns c3[0 .. 131583], c, ko, fo;
        loop variant 512 - c;
    */
    for (int c = 0; c < 512; c++, ko += 257, fo += 256)
        fill_ctx(c3 + ko, ft + 514 + 196608 + fo, 256);
}

/*@ requires \valid_read(in + (0 .. lim - in - 1));
    requires 0 <= lim - in;
    requires \valid(s + (0 .. n - 1));
    requires \valid(h + (0 .. 328193));
    requires n <= 281474976710655;
    requires \forall integer j; 0 <= j < 328194 ==>
        h[j] + n <= 281474976710655;
    requires \valid(ft + (0 .. 328193));
    requires \valid(cum + (0 .. 329729));
    requires \separated(s + (0 .. n - 1), h + (0 .. 328193));
    requires \separated(s + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(h + (0 .. 328193), in + (0 .. lim - in - 1));
    requires \separated(ft + (0 .. 328193), s + (0 .. n - 1));
    requires \separated(ft + (0 .. 328193), h + (0 .. 328193));
    requires \separated(ft + (0 .. 328193), in + (0 .. lim - in - 1));
    requires \separated(cum + (0 .. 329729), s + (0 .. n - 1));
    requires \separated(cum + (0 .. 329729), h + (0 .. 328193));
    requires \separated(cum + (0 .. 329729), in + (0 .. lim - in - 1));
    requires \separated(ft + (0 .. 328193), cum + (0 .. 329729));
    terminates \false;
    assigns s[0 .. n - 1], h[0 .. 328193], ft[0 .. 328193], cum[0 .. 329729];
    exits \exit_status == 1;
    ensures \forall integer j; 0 <= j < 328194 ==>
        h[j] <= \at(h[j], Pre) + n;
*/
static void f32_dec_ws(const uint8_t *in, const uint8_t *lim, uint64_t n,
                       uint32_t *s, uint64_t *h, uint16_t *ft, uint32_t *cum) {
    const uint8_t *rp = in;
    /*@ loop invariant 0 <= b0;
        loop invariant 0 <= lim - rp;
        loop invariant in <= rp;
        loop invariant \base_addr(rp) == \base_addr(in);
        loop invariant \valid_read(rp + (0 .. lim - rp - 1));
        loop invariant \forall integer j; 0 <= j < 328194 ==>
            h[j] <= \at(h[j], Pre) + b0;
        loop invariant \forall integer j; 0 <= j < 328194 ==>
            h[j] <= \at(h[j], Pre) + n;
        loop assigns rp, s[0 .. n - 1], h[0 .. 328193], b0,
            ft[0 .. 328193], cum[0 .. 329729];
        loop variant n - b0;
    */
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        f32_tab(h, ft, cum);
        uint64_t x; const uint8_t *end;
        /*@ assert 0 <= lim - rp; */
        rp = read_blk(rp, lim, &x, &end);
        f32_blk(x, rp, end, ft, cum, bn, s + b0, h);
        rp = end;
    }
    if (rp != lim) die("corrupt f32");
}

static void f32_dec(const uint8_t *in, const uint8_t *lim, uint64_t n, uint32_t *s, uint64_t *h) {
    uint16_t *ft = xm(328194 * 2);
    uint32_t *cum = xc(329730, 4);
    f32_dec_ws(in, lim, n, s, h, ft, cum);
    free(ft); free(cum);
}

/* ============ U8 ============ */
static size_t u8_enc(const uint8_t *s, uint64_t n, uint8_t *out, uint64_t *h) {
    uint8_t *o = out, *scr = xm(SCRSZ);
    uint16_t ft[256]; uint32_t cum[257];
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        norm_ctx(h, ft, 256);
        cum[0] = 0; for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + ft[i];
        /* Σf == TOT exactly (Norm.v, all histograms; aw=256 | TOT so the
           all-zero branch is exact too) — defensive assertion guarding
           enc's c<=TOT-f domain, same pattern as enc's f==0 die */
        if (cum[256] != TOT) die("norm bug");
        uint8_t *pp = scr + SCRSZ;
        uint64_t x = LOWER;
        for (uint64_t i = bn; i-- > 0;) {
            uint8_t sv = s[b0 + i];
            x = enc(x, ft[sv], cum[sv], &pp);
            h[sv]++;
        }
        o = emit_blk(o, scr + SCRSZ, pp, x);
    }
    free(scr);
    return o - out;
}
/*@ requires \valid_read(ft + (0 .. 255));
    requires \valid_read(cum + (0 .. 256));
    requires \forall integer i; 0 <= i < 256 ==> ft[i] <= 32768;
    requires cum[0] == 0 && cum[256] == 32768;
    requires \forall integer i; 0 <= i < 256 ==> cum[i] <= cum[i + 1];
    requires \forall integer i; 0 <= i < 256 && cum[i] < cum[i + 1] ==> ft[i] > 0;
    requires \valid_read(r + (0 .. end - r - 1));
    requires 0 <= end - r;
    requires \valid(s + (0 .. bn - 1));
    requires \valid(h + (0 .. 255));
    requires 1 <= bn <= 131072;
    requires \forall integer j; 0 <= j < 256 ==> h[j] + bn <= 281474976710655;
    assigns s[0 .. bn - 1], h[0 .. 255];
    ensures \forall integer j; 0 <= j < 256 ==> h[j] <= 281474976710655;
    ensures \forall integer j; 0 <= j < 256 ==> h[j] <= \at(h[j], Pre) + bn;
*/
static void u8_blk(uint64_t x, const uint8_t *r, const uint8_t *end,
                   const uint16_t *ft, const uint32_t *cum,
                   uint64_t bn, uint8_t *s, uint64_t *h) {
    /*@ loop invariant 0 <= i <= bn;
        loop invariant 0 <= end - r;
        loop invariant \valid_read(r + (0 .. end - r - 1));
        loop invariant \forall integer j; 0 <= j < 256 ==>
            h[j] + bn - i <= 281474976710655;
        loop invariant \forall integer j; 0 <= j < 256 ==>
            h[j] <= \at(h[j], Pre) + i;
        loop assigns x, r, s[0 .. bn - 1], h[0 .. 255], i;
        loop variant bn - i;
    */
    for (uint64_t i = 0; i < bn; i++) {
        uint32_t v = (uint32_t)(x & (TOT - 1)), fc;
        uint32_t sym = dsym(ft, cum, 256, v, &fc);
        x = dec(x, fc, cum[sym], &r, end);
        s[i] = (uint8_t)sym;
        h[sym]++;
    }
}

/*@ requires \valid_read(in + (0 .. lim - in - 1));
    requires 0 <= lim - in;
    requires \valid(s + (0 .. n - 1));
    requires \valid(h + (0 .. 255));
    requires n <= 281474976710655;
    requires \forall integer j; 0 <= j < 256 ==> h[j] + n <= 281474976710655;
    requires \separated(s + (0 .. n - 1), h + (0 .. 255));
    requires \separated(s + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(h + (0 .. 255), in + (0 .. lim - in - 1));
    terminates \false;
    assigns s[0 .. n - 1], h[0 .. 255];
    assigns \result \from lim;
    exits \exit_status == 1;
    ensures \result == lim;
    ensures \forall integer j; 0 <= j < 256 ==> h[j] <= \at(h[j], Pre) + n;
*/
static const uint8_t *u8_dec(const uint8_t *in, const uint8_t *lim, uint64_t n, uint8_t *s, uint64_t *h) {
    const uint8_t *rp = in;
    uint16_t ft[256]; uint32_t cum[257];
    /*@ loop invariant 0 <= b0;
        loop invariant 0 <= lim - rp;
        loop invariant in <= rp;
        loop invariant \base_addr(rp) == \base_addr(in);
        loop invariant \valid_read(rp + (0 .. lim - rp - 1));
        loop invariant \forall integer j; 0 <= j < 256 ==>
            h[j] <= \at(h[j], Pre) + b0;
        loop invariant \forall integer j; 0 <= j < 256 ==>
            h[j] <= \at(h[j], Pre) + n;
        loop assigns rp, s[0 .. n - 1], h[0 .. 255], b0,
            ft[0 .. 255], cum[0 .. 256];
        loop variant n - b0;
    */
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        norm_ctx(h, ft, 256);
        cum[0] = 0;
        /*@ loop invariant 0 <= i <= 256;
            loop invariant cum[i] <= 32768 * i;
            loop invariant \forall integer j; 0 <= j < i ==>
                cum[j + 1] == cum[j] + ft[j];
            loop assigns cum[1 .. 256], i;
            loop variant 256 - i;
        */
        for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + ft[i];
        /* Σf == TOT exactly (Norm.v); guards dsym's v < cum[aw] domain */
        if (cum[256] != TOT) die("norm bug");
        uint64_t x; const uint8_t *end;
        /*@ assert 0 <= lim - rp; */
        rp = read_blk(rp, lim, &x, &end);
        u8_blk(x, rp, end, ft, cum, bn, s + b0, h);
        rp = end;
    }
    if (rp != lim) die("corrupt u8");
    return rp;
}

/* ============ DELTA (16-bit K-residual) ============ */
/*@ assigns \nothing;
    ensures 0 <= \result <= 0xFFFF;
    ensures \result == ((u & 0x8000) != 0 ?
                        (int64_t)(uint16_t)(~u) : (int64_t)(u ^ 0x8000));
*/
static inline int64_t kmap16(uint16_t u) {
    return (u & 0x8000) ? (int64_t)(uint16_t)(~u) : (int64_t)(u ^ 0x8000);
}
/*@ assigns \nothing;
    ensures \result == (((\at(k,Pre) & 0xFFFF) & 0x8000) != 0 ?
                        (uint16_t)((\at(k,Pre) & 0xFFFF) ^ 0x8000)
                        : (uint16_t)(~(\at(k,Pre) & 0xFFFF)));
*/
static inline uint16_t kmap16_inv(int64_t k) {
    k &= 0xFFFF;
    return (k & 0x8000) ? (uint16_t)(k ^ 0x8000) : (uint16_t)(~k);
}
/*@ assigns \nothing;
    ensures 0 <= \result <= 4294967295;
    ensures \result == ((u & 0x80000000) != 0 ?
                        (int64_t)(uint32_t)(~u) : (int64_t)(u ^ 0x80000000));
*/
static inline int64_t kmap32(uint32_t u) {   /* ordering map for delta_ok pretest */
    return (u & 0x80000000u) ? (int64_t)(uint32_t)(~u) : (int64_t)(u ^ 0x80000000u);
}
#define DR 2048
#define DESC (2 * DR)
#define DSYMS (2 * DR + 1)
#define DCTX 128               /* residual ctx = top 7 bits of ref element */
/* reusable scratch for the DELTA16 codec — PRW runs it once per row, so the
   buffers must persist across calls instead of mmap-churning per row */
typedef struct {
    uint8_t *scr, *cB, *used;
    uint16_t *ft, *sB, *escbuf;
    uint32_t *dcum;
    uint64_t esc_cap;
    uint8_t *escbits; uint64_t esccap;  /* decode side only: escape bitmap */
} DltWs;
static void dltws_init(DltWs *w) {
    w->scr = xm(SCRSZ); w->cB = xm(BLK); w->used = xm(DCTX);
    w->ft = xm(DCTX * DSYMS * 2); w->sB = xm(BLK * 2);
    w->dcum = xm(DCTX * (DSYMS + 1) * 4);
    w->esc_cap = 8192; w->escbuf = xm(w->esc_cap * 2);
    w->escbits = 0; w->esccap = 0;
}
/* decode side never touches the encoder scratch (scr/cB/sB — ~2.5MB) —
   skip it so a DELTA/PRW decode's fixed footprint is ~3.3MB, not ~5.4MB */
static void dltws_init_dec(DltWs *w) {
    w->scr = 0; w->cB = 0; w->sB = 0;
    w->used = xm(DCTX);
    w->ft = xm(DCTX * DSYMS * 2);
    w->dcum = xm(DCTX * (DSYMS + 1) * 4);
    w->esc_cap = 8192; w->escbuf = xm(w->esc_cap * 2);
    w->escbits = 0; w->esccap = 0;
}
static void dltws_free(DltWs *w) {
    free(w->scr); free(w->cB); free(w->used);
    free(w->ft); free(w->sB); free(w->dcum);
    free(w->escbuf); free(w->escbits);
}
/* callers size the escape bitmap/values buffer BEFORE entering the verified
   decoder — keeps realloc out of the WP-checked loops entirely */
static void dltws_need_bits(DltWs *w, uint64_t n) {
    uint64_t need = (n + 7) / 8;
    if (w->esccap < need) { free(w->escbits); w->escbits = xm(need); w->esccap = need; }
}
static void dltws_need_eb(DltWs *w, uint64_t escn) {
    if (w->esc_cap < escn) {
        w->esc_cap = escn;
        w->escbuf = realloc(w->escbuf, w->esc_cap * 2);
        if (!w->escbuf) die("oom");
    }
}
/* block-parallel DELTA16: the element pass (sym/ctx/escapes) depends only on
   cur+ref so it runs in parallel into per-block buffers; the norm+merge of h
   is the serial model chain; blocks then encode in parallel against their
   frozen ft snapshots.  Same ft/cum => same bytes as the serial path. */
typedef struct {
    const uint16_t *cur, *ref; uint64_t n;
    uint64_t blo, bhi;                        /* absolute block range */
    uint8_t *cBs; uint16_t *sBs;              /* [blk][BLK] wave-local */
    uint8_t *useds;                           /* [blk][DCTX] */
    uint16_t *fts;                            /* [blk][DCTX*DSYMS] */
    uint32_t *h32;                            /* [blk][DCTX*DSYMS] */
    uint16_t **escs; uint32_t *escn;          /* per-block escape lists */
    uint8_t **bufs; uint32_t *lens; uint64_t *xs;
    int mode;                                 /* 0 = build+count, 1 = encode */
    int spawned;
} DltW;
static void *dltw_run(void *a) {
    DltW *w = a;
    if (!w->mode) {
        for (uint64_t b = w->blo; b < w->bhi; b++) {
            uint64_t lb = b - w->blo;
            uint64_t b0 = b * BLK, bn = w->n - b0 < BLK ? w->n - b0 : BLK;
            uint8_t *cB = w->cBs + lb * BLK, *used = w->useds + lb * DCTX;
            uint16_t *sB = w->sBs + lb * BLK;
            uint32_t *bh = w->h32 + lb * (size_t)(DCTX * DSYMS);
            /* fused single pass — a full bh memset is cheaper than scanning
               ref twice just to learn which ctx rows need clearing */
            memset(used, 0, DCTX);
            memset(bh, 0, (size_t)DCTX * DSYMS * 4);
            uint16_t *eb = xm(bn * 2 + 2); uint32_t en = 0;
            for (uint64_t i = 0; i < bn; i++) {
                int c = w->ref[b0 + i] >> 9;
                int64_t d = kmap16(w->cur[b0 + i]) - kmap16(w->ref[b0 + i]);
                uint32_t sym = (d < -DR || d >= DR) ? DESC : (uint32_t)(d + DR);
                cB[i] = (uint8_t)c; sB[i] = (uint16_t)sym; used[c] = 1;
                bh[c * DSYMS + sym]++;
                if (sym == DESC) eb[en++] = w->cur[b0 + i];
            }
            w->escs[lb] = eb; w->escn[lb] = en;
        }
        return 0;
    }
    uint8_t *scr = xm(SCRSZ);
    uint32_t *dcum = xm(DCTX * (DSYMS + 1) * 4);
    for (uint64_t b = w->blo; b < w->bhi; b++) {
        uint64_t lb = b - w->blo;
        uint64_t bn = w->n - b * BLK < BLK ? w->n - b * BLK : BLK;
        const uint8_t *cB = w->cBs + lb * BLK, *used = w->useds + lb * DCTX;
        const uint16_t *sB = w->sBs + lb * BLK;
        const uint16_t *ft = w->fts + lb * (size_t)(DCTX * DSYMS);
        for (int c = 0; c < DCTX; c++) if (used[c]) {
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
        uint64_t bl = (uint64_t)(scr + SCRSZ - pp);
        w->bufs[lb] = xm(bl ? bl : 1);
        memcpy(w->bufs[lb], pp, bl);
        w->lens[lb] = (uint32_t)bl; w->xs[lb] = x;
    }
    free(scr); free(dcum);
    return 0;
}
static size_t dlt_enc_ws(const uint16_t *cur, const uint16_t *ref, uint64_t n,
                         uint8_t *out, uint64_t *h, int mb, uint64_t *hesc, DltWs *w,
                         int nthr) {
    uint8_t *o = out, *scr = w->scr;
    uint16_t *ft = w->ft;
    uint32_t *dcum = w->dcum;
    uint16_t *sB = w->sB; uint8_t *cB = w->cB, *used = w->used;
    uint64_t esc_cap = w->esc_cap, esc_n = 0;
    uint16_t *escbuf = w->escbuf;
    uint64_t nblk = (n + BLK - 1) / BLK;
    if (nthr > 1 && nblk >= 2) {
        uint64_t wcap = (uint64_t)nthr * 4;   /* heavier per-block state */
        if (wcap > nblk) wcap = nblk;
        uint8_t *cBs = xm(wcap * BLK), *useds = xm(wcap * DCTX);
        uint16_t *sBs = xm(wcap * BLK * 2), *fts = xm(wcap * (size_t)(DCTX * DSYMS) * 2);
        uint32_t *h32 = xc(wcap * (size_t)(DCTX * DSYMS), 4);
        uint16_t **escs = xc(wcap, sizeof(uint16_t *));
        uint32_t *escn = xc(wcap, 4), *lens = xc(wcap, 4);
        uint8_t **bufs = xc(wcap, sizeof(uint8_t *));
        uint64_t *xs = xc(wcap, 8);
        DltW *wj = xc(nthr, sizeof(DltW));
        pthread_t *th = xc(nthr, sizeof(pthread_t));
        for (uint64_t bs = 0; bs < nblk; bs += wcap) {
            uint64_t wn = nblk - bs < wcap ? nblk - bs : wcap;
            int sp = (uint64_t)nthr < wn ? nthr : (int)wn;
            uint64_t per = (wn + sp - 1) / sp;
            for (int phase = 0; phase < 2; phase++) {
                uint64_t lo = bs;
                for (int k = 0; k < sp; k++) {
                    DltW *q = &wj[k];
                    q->cur = cur; q->ref = ref; q->n = n; q->mode = phase;
                    q->blo = lo; q->bhi = lo + per < bs + wn ? lo + per : bs + wn;
                    q->cBs = cBs + (lo - bs) * BLK; q->sBs = sBs + (lo - bs) * BLK;
                    q->useds = useds + (lo - bs) * DCTX;
                    q->fts = fts + (lo - bs) * (size_t)(DCTX * DSYMS);
                    q->h32 = h32 + (lo - bs) * (size_t)(DCTX * DSYMS);
                    q->escs = escs + (lo - bs); q->escn = escn + (lo - bs);
                    q->bufs = bufs + (lo - bs); q->lens = lens + (lo - bs);
                    q->xs = xs + (lo - bs); q->spawned = 0;
                    lo = q->bhi;
                    if (pthread_create(&th[k], 0, dltw_run, q)) dltw_run(q);
                    else q->spawned = 1;
                }
                for (int k = 0; k < sp; k++) if (wj[k].spawned) pthread_join(th[k], 0);
                if (phase) continue;
                /* serial norm+merge per block, in order (model chain) */
                for (uint64_t b = 0; b < wn; b++) {
                    const uint8_t *u = useds + b * DCTX;
                    uint16_t *f = fts + b * (size_t)(DCTX * DSYMS);
                    const uint32_t *bh = h32 + b * (size_t)(DCTX * DSYMS);
                    for (int c = 0; c < DCTX; c++) if (u[c]) {
                        norm_ctx(h + c * DSYMS, f + c * DSYMS, DSYMS);
                        uint64_t *hp = h + c * DSYMS;
                        const uint32_t *bp = bh + c * DSYMS;
                        for (int i = 0; i < DSYMS; i++) hp[i] += bp[i];
                    }
                }
            }
            for (uint64_t b = 0; b < wn; b++) {          /* emit, in order */
                uint8_t *bf = bufs[b];
                o = emit_blk(o, bf + lens[b], bf, xs[b]);
                free(bf);
                /* escapes concatenate in element order */
                if (escn[b]) {
                    if (esc_n + escn[b] > esc_cap) {
                        while (esc_n + escn[b] > esc_cap) esc_cap *= 2;
                        escbuf = realloc(escbuf, esc_cap * 2);
                        if (!escbuf) die("oom");
                    }
                    memcpy(escbuf + esc_n, escs[b], escn[b] * 2);
                    esc_n += escn[b];
                }
                free(escs[b]);
            }
        }
        free(cBs); free(sBs); free(useds); free(fts); free(h32);
        free(escs); free(escn); free(lens); free(bufs); free(xs);
        free(wj); free(th);
        p64le(o, esc_n); o += 8;
        o += f16_enc(escbuf, esc_n, mb, o, hesc, nthr > 1 ? nthr : 1);
        w->escbuf = escbuf; w->esc_cap = esc_cap;
        return o - out;
    }
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
            h[c * DSYMS + sym]++;
        }
        o = emit_blk(o, scr + SCRSZ, pp, x);
    }
    p64le(o, esc_n); o += 8;
    o += f16_enc(escbuf, esc_n, mb, o, hesc, nthr);  /* escapes through FIELD channel */
    w->escbuf = escbuf; w->esc_cap = esc_cap;
    return o - out;
}
static size_t dlt_enc(const uint16_t *cur, const uint16_t *ref, uint64_t n,
                      uint8_t *out, uint64_t *h, int mb, uint64_t *hesc, int nthr) {
    DltWs w; dltws_init(&w);
    size_t r = dlt_enc_ws(cur, ref, n, out, h, mb, hesc, &w, nthr);
    dltws_free(&w);
    return r;
}
/*@ requires sym < 4096;
    assigns \nothing;
    ensures \result <= 65535;
*/
static uint16_t dlt_val(uint16_t rv, uint32_t sym) {
    return kmap16_inv(kmap16(rv) + (int64_t)sym - DR);
}
/*@ requires \valid_read(ft + (0 .. 524415));
    requires \valid_read(cum + (0 .. 524543));
    requires \valid(rp);
    requires 0 <= end - *rp;
    requires \valid_read(*rp + (0 .. end - *rp - 1));
    requires \valid(ci);
    requires \valid(h + (0 .. 524543));
    requires \forall integer j; 0 <= j < 524544 ==>
        h[j] <= 281474976710655;
    requires \valid(escbits + (0 .. gi / 8));
    terminates \false;
    exits \exit_status == 1;
    assigns *rp \from *rp, x, end, rv, ft[0 .. 524415],
        cum[0 .. 524543], (*rp)[0 .. end - *rp - 1];
    assigns *ci, h[0 .. 524543], escbits[0 .. gi / 8];
    ensures 0 <= *rp - \at(*rp, Pre) && *rp - \at(*rp, Pre) <= end - \at(*rp, Pre);
    ensures \base_addr(*rp) == \base_addr(\at(*rp, Pre));
    ensures \forall integer j; 0 <= j < 524544 ==>
        h[j] <= \at(h[j], Pre) + 1;
*/
static uint64_t dlt_elem(uint64_t x, const uint8_t **rp, const uint8_t *end,
                         uint16_t rv, const uint16_t *ft, const uint32_t *cum,
                         uint16_t *ci, uint64_t *h, uint8_t *escbits,
                         uint64_t gi) {
    uint32_t v = (uint32_t)(x & (TOT - 1)), fc;
    uint32_t c = ((uint32_t)rv / 512) % 128;
    /*@ assert c <= 127; */
    uint32_t fo = c * DSYMS, ko = c * (DSYMS + 1);
    /*@ assert fo + 4096 < 524544 && ko + 4096 < 524544; */
    uint32_t sym = dsym_raw(ft + fo, cum + ko, DSYMS, v, &fc);
    if (!fc || fc > TOT) die("corrupt delta");
    const uint8_t *r = *rp;
    x = dec(x, fc, cum[ko + sym], &r, end);
    uint32_t hidx = fo + sym;
    /*@ assert hidx < 524544; */
    h[hidx]++;
    if (sym == DESC) {
        *ci = 0;
        escbits[gi / 8] |= (uint8_t)(1u << (gi % 8));
    } else {
        *ci = dlt_val(rv, sym);
    }
    *rp = r;
    return x;
}

/*@ requires \valid_read(ft + (0 .. 524415));
    requires \valid_read(cum + (0 .. 524543));
    requires \valid_read(r + (0 .. end - r - 1));
    requires 0 <= end - r;
    requires \valid_read(ref + (0 .. bn - 1));
    requires \valid(cur + (0 .. bn - 1));
    requires \valid(h + (0 .. 524543));
    requires 1 <= bn <= 131072;
    requires \forall integer j; 0 <= j < 524544 ==>
        h[j] + bn <= 281474976710655;
    requires \valid(escbits + (0 .. (b0 + bn - 1) / 8));
    requires b0 + bn <= 281474976710655;
    terminates \false;
    exits \exit_status == 1;
    assigns cur[0 .. bn - 1], h[0 .. 524543],
            escbits[0 .. (b0 + bn - 1) / 8];
    ensures \forall integer j; 0 <= j < 524544 ==>
        h[j] <= \at(h[j], Pre) + bn;
*/
static void dlt_blk(uint64_t x, const uint8_t *r, const uint8_t *end,
                    const uint16_t *ref, uint64_t b0, const uint16_t *ft,
                    const uint32_t *cum, uint64_t bn, uint16_t *cur,
                    uint64_t *h, uint8_t *escbits) {
    /*@ loop invariant 0 <= i <= bn;
        loop invariant 0 <= end - r;
        loop invariant \valid_read(r + (0 .. end - r - 1));
        loop invariant \forall integer j; 0 <= j < 524544 ==>
            h[j] + bn - i <= 281474976710655;
        loop invariant \forall integer j; 0 <= j < 524544 ==>
            h[j] <= \at(h[j], Pre) + i;
        loop assigns x, r, cur[0 .. bn - 1], h[0 .. 524543],
                escbits[0 .. (b0 + bn - 1) / 8], i;
        loop variant bn - i;
    */
    for (uint64_t i = 0; i < bn; i++)
        x = dlt_elem(x, &r, end, ref[i], ft, cum, cur + i, h, escbits,
                     b0 + i);
}

/*@ requires \valid_read(used + (0 .. 127));
    requires \valid_read(h + (0 .. 524543));
    requires \forall integer j; 0 <= j < 524544 ==>
        h[j] <= 281474976710655;
    requires \valid(ft + (0 .. 524415));
    requires \valid(cum + (0 .. 524543));
    terminates \false;
    assigns ft[0 .. 524415], cum[0 .. 524543];
    exits \exit_status == 1;
*/
static void dlt_tab(const uint8_t *used, const uint64_t *h,
                    uint16_t *ft, uint32_t *cum) {
    int fo = 0, ko = 0;
    /*@ loop invariant 0 <= c <= 128;
        loop invariant fo == c * 4097 && ko == c * 4098;
        loop invariant 0 <= fo <= 524416 && 0 <= ko <= 524544;
        loop assigns ft[0 .. 524415], cum[0 .. 524543], c, fo, ko;
        loop variant 128 - c;
    */
    for (int c = 0; c < DCTX; c++, fo += DSYMS, ko += DSYMS + 1) {
        if (used[c]) {
            norm_ctx(h + fo, ft + fo, DSYMS);
            fill_ctx(cum + ko, ft + fo, DSYMS);
        }
    }
}

/*@ requires \valid_read(in + (0 .. lim - in - 1));
    requires 0 <= lim - in;
    requires \valid_read(ref + (0 .. n - 1));
    requires \valid(cur + (0 .. n - 1));
    requires \valid(h + (0 .. 524543));
    requires n <= 281474976710655;
    requires \forall integer j; 0 <= j < 524544 ==>
        h[j] + n <= 281474976710655;
    requires \valid(w);
    requires \valid(w->ft + (0 .. 524415));
    requires \valid(w->dcum + (0 .. 524543));
    requires \valid(w->used + (0 .. 127));
    requires \valid(w->escbits + (0 .. (n + 7) / 8 - 1));
    requires \valid(escn);
    requires \separated(escn, cur + (0 .. n - 1));
    requires \separated(escn, h + (0 .. 524543));
    requires \separated(cur + (0 .. n - 1), h + (0 .. 524543));
    requires \separated(cur + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(cur + (0 .. n - 1), ref + (0 .. n - 1));
    requires \separated(cur + (0 .. n - 1), w->ft + (0 .. 524415));
    requires \separated(cur + (0 .. n - 1), w->dcum + (0 .. 524543));
    requires \separated(cur + (0 .. n - 1), w->used + (0 .. 127));
    requires \separated(cur + (0 .. n - 1), w->escbits + (0 .. (n + 7) / 8 - 1));
    requires \separated(cur + (0 .. n - 1), escn);
    requires \separated(h + (0 .. 524543), in + (0 .. lim - in - 1));
    requires \separated(h + (0 .. 524543), ref + (0 .. n - 1));
    requires \separated(h + (0 .. 524543), w->ft + (0 .. 524415));
    requires \separated(h + (0 .. 524543), w->dcum + (0 .. 524543));
    requires \separated(h + (0 .. 524543), w->used + (0 .. 127));
    requires \separated(h + (0 .. 524543), w->escbits + (0 .. (n + 7) / 8 - 1));
    requires \separated(h + (0 .. 524543), escn);
    requires \separated(w->ft + (0 .. 524415), in + (0 .. lim - in - 1));
    requires \separated(w->ft + (0 .. 524415), ref + (0 .. n - 1));
    requires \separated(w->ft + (0 .. 524415), w->dcum + (0 .. 524543));
    requires \separated(w->ft + (0 .. 524415), w->used + (0 .. 127));
    requires \separated(w->ft + (0 .. 524415), w->escbits + (0 .. (n + 7) / 8 - 1));
    requires \separated(w->ft + (0 .. 524415), escn);
    requires \separated(w->dcum + (0 .. 524543), in + (0 .. lim - in - 1));
    requires \separated(w->dcum + (0 .. 524543), ref + (0 .. n - 1));
    requires \separated(w->dcum + (0 .. 524543), w->used + (0 .. 127));
    requires \separated(w->dcum + (0 .. 524543), w->escbits + (0 .. (n + 7) / 8 - 1));
    requires \separated(w->dcum + (0 .. 524543), escn);
    requires \separated(w->used + (0 .. 127), in + (0 .. lim - in - 1));
    requires \separated(w->used + (0 .. 127), ref + (0 .. n - 1));
    requires \separated(w->used + (0 .. 127), w->escbits + (0 .. (n + 7) / 8 - 1));
    requires \separated(w->used + (0 .. 127), escn);
    requires \separated(w->escbits + (0 .. (n + 7) / 8 - 1), in + (0 .. lim - in - 1));
    requires \separated(w->escbits + (0 .. (n + 7) / 8 - 1), ref + (0 .. n - 1));
    requires \separated(w->escbits + (0 .. (n + 7) / 8 - 1), escn);
    requires \separated(ref + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(ref + (0 .. n - 1), escn);
    requires \separated(in + (0 .. lim - in - 1), escn);
    terminates \false;
    assigns cur[0 .. n - 1], h[0 .. 524543],
            w->escbits[0 .. (n + 7) / 8 - 1], w->ft[0 .. 524415],
            w->dcum[0 .. 524543], w->used[0 .. 127], *escn;
    assigns \result \from in, n, lim, in[0 .. lim - in - 1];
    exits \exit_status == 1;
    ensures \forall integer j; 0 <= j < 524544 ==>
        h[j] <= \at(h[j], Pre) + n;
    ensures *escn <= n;
    ensures 0 <= \result - in && \result - in <= lim - in;
    ensures \base_addr(\result) == \base_addr(in);
*/
static const uint8_t *dlt_dec_ws(const uint8_t *in, const uint8_t *lim,
                                 const uint16_t *ref, uint64_t n,
                                 uint16_t *cur, uint64_t *h, DltWs *w,
                                 uint64_t *escn) {
    const uint8_t *rp = in;
    uint16_t *ft = w->ft;
    uint32_t *cum = w->dcum;
    uint8_t *used = w->used;
    uint8_t *escbits = w->escbits;
    /* escape positions now live in a caller-sized bitmap — no realloc inside
       the verified loop nest */
    /*@ loop invariant 0 <= z <= (n + 7) / 8;
        loop assigns w->escbits[0 .. (n + 7) / 8 - 1], z;
        loop variant (n + 7) / 8 - z;
    */
    for (uint64_t z = 0; z < (n + 7) / 8; z++) escbits[z] = 0;
    /*@ loop invariant 0 <= b0;
        loop invariant 0 <= lim - rp;
        loop invariant in <= rp;
        loop invariant \base_addr(rp) == \base_addr(in);
        loop invariant \valid_read(rp + (0 .. lim - rp - 1));
        loop invariant \forall integer j; 0 <= j < 524544 ==>
            h[j] <= \at(h[j], Pre) + b0;
        loop invariant \forall integer j; 0 <= j < 524544 ==>
            h[j] <= \at(h[j], Pre) + n;
        loop assigns rp, cur[0 .. n - 1], h[0 .. 524543], b0,
                w->escbits[0 .. (n + 7) / 8 - 1], w->ft[0 .. 524415],
                w->dcum[0 .. 524543], w->used[0 .. 127];
        loop variant n - b0;
    */
    for (uint64_t b0 = 0; b0 < n; b0 += BLK) {
        uint64_t bn = n - b0 < BLK ? n - b0 : BLK;
        /*@ assert b0 + bn <= n; */
        /*@ loop invariant 0 <= z <= 128;
            loop assigns w->used[0 .. 127], z;
            loop variant 128 - z;
        */
        for (int z = 0; z < DCTX; z++) used[z] = 0;
        /*@ loop invariant 0 <= i <= bn;
            loop assigns w->used[0 .. 127], i;
            loop variant bn - i;
        */
        for (uint64_t i = 0; i < bn; i++)
            used[((uint32_t)ref[b0 + i] / 512) % 128] = 1;
        dlt_tab(used, h, ft, cum);
        uint64_t x; const uint8_t *end;
        /*@ assert 0 <= lim - rp; */
        rp = read_blk(rp, lim, &x, &end);
        dlt_blk(x, rp, end, ref + b0, b0, ft, cum, bn, cur + b0, h,
                escbits);
        rp = end;
    }
    if ((uint64_t)(lim - rp) < 8) die("corrupt delta");
    uint64_t ne = g64le(rp); rp += 8;
    uint64_t ec = 0;
    /*@ loop invariant 0 <= gi <= n;
        loop invariant ec <= gi;
        loop assigns gi, ec;
        loop variant n - gi;
    */
    for (uint64_t gi = 0; gi < n; gi++)
        ec += (escbits[gi / 8] >> (gi % 8)) & 1;
    *escn = ec;
    if (ne != ec) die("delta esc count mismatch");
    return rp;
}

/*@ requires \valid_read(escbits + (0 .. (n + 7) / 8 - 1));
    requires \valid_read(eb + (0 .. escn - 1));
    requires \valid(cur + (0 .. n - 1));
    requires escn <= n;
    requires n <= 281474976710655;
    terminates \false;
    exits \exit_status == 1;
    assigns cur[0 .. n - 1];
*/
static void dlt_scatter(uint16_t *cur, uint64_t n, const uint8_t *escbits,
                        const uint16_t *eb, uint64_t escn) {
    uint64_t j = 0;
    /*@ loop invariant 0 <= gi <= n;
        loop invariant j <= escn;
        loop assigns cur[0 .. n - 1], gi, j;
        loop variant n - gi;
    */
    for (uint64_t gi = 0; gi < n; gi++) {
        /*@ assert gi / 8 < (n + 7) / 8; */
        if ((escbits[gi / 8] >> (gi % 8)) & 1) {
            if (j >= escn) die("corrupt delta");
            cur[gi] = eb[j++];
        }
    }
    if (j != escn) die("corrupt delta");
}

/*@ requires \valid_read(in + (0 .. lim - in - 1));
    requires 0 <= lim - in;
    requires mb == 7 || mb == 10;
    requires \valid(eb + (0 .. escn - 1));
    requires \valid(hesc + (0 .. (mb == 7 ? 66050 : 65602) - 1));
    requires \forall integer j; 0 <= j < (mb == 7 ? 66050 : 65602) ==>
        hesc[j] + escn <= 281474976710655;
    requires \valid(cur + (0 .. n - 1));
    requires \valid_read(escbits + (0 .. (n + 7) / 8 - 1));
    requires escn <= n;
    requires n <= 281474976710655;
    requires \separated(eb + (0 .. escn - 1), hesc + (0 .. (mb == 7 ? 66050 : 65602) - 1));
    requires \separated(eb + (0 .. escn - 1), in + (0 .. lim - in - 1));
    requires \separated(eb + (0 .. escn - 1), cur + (0 .. n - 1));
    requires \separated(eb + (0 .. escn - 1), escbits + (0 .. (n + 7) / 8 - 1));
    requires \separated(hesc + (0 .. (mb == 7 ? 66050 : 65602) - 1), in + (0 .. lim - in - 1));
    requires \separated(hesc + (0 .. (mb == 7 ? 66050 : 65602) - 1), cur + (0 .. n - 1));
    requires \separated(hesc + (0 .. (mb == 7 ? 66050 : 65602) - 1), escbits + (0 .. (n + 7) / 8 - 1));
    requires \separated(cur + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(cur + (0 .. n - 1), escbits + (0 .. (n + 7) / 8 - 1));
    requires \separated(escbits + (0 .. (n + 7) / 8 - 1), in + (0 .. lim - in - 1));
    terminates \false;
    exits \exit_status == 1;
    assigns eb[0 .. escn - 1], hesc[0 .. (mb == 7 ? 66050 : 65602) - 1],
            cur[0 .. n - 1];
    ensures \forall integer j; 0 <= j < (mb == 7 ? 66050 : 65602) ==>
        hesc[j] <= \at(hesc[j], Pre) + escn;
*/
static void dlt_tail(const uint8_t *in, const uint8_t *lim, uint64_t escn,
                     int mb, uint16_t *eb, uint64_t *hesc,
                     uint16_t *cur, uint64_t n, const uint8_t *escbits) {
    if (escn) {
        f16_dec(in, lim, escn, mb, eb, hesc);   /* escapes from FIELD channel; consumes to lim */
        dlt_scatter(cur, n, escbits, eb, escn);
    } else if (in != lim) die("corrupt delta");
}

static void dlt_dec(const uint8_t *in, const uint8_t *lim, const uint16_t *ref, uint64_t n,
                    uint16_t *cur, uint64_t *h, int mb, uint64_t *hesc) {
    DltWs w; dltws_init_dec(&w);
    dltws_need_bits(&w, n);
    uint64_t escn;
    const uint8_t *rp = dlt_dec_ws(in, lim, ref, n, cur, h, &w, &escn);
    dltws_need_eb(&w, escn);
    dlt_tail(rp, lim, escn, mb, w.escbuf, hesc, cur, n, w.escbits);
    dltws_free(&w);
}

/* ============ DELTA32: xor byte planes (f32 pairs) ============
 * measured on real F32 checkpoint pairs: int32-ordered K-residuals are
 * near-uniform over any window (mantissa diffs are high-entropy); XOR
 * planes win (~16.5b vs ~18b/elem). */
#define D32CN (1u << 24)   /* elements per plane chunk */
/* each XOR plane coded conditioned on ref exponent byte (ref>>23):
 * ~0.6b/elem gain on real F32 checkpoint deltas */
/* block-parallel DELTA32: planes are still serial (h is shared across them),
   but within a plane blocks run build+count in parallel, norm+merge serially
   in order, then encode in parallel.  pl/cb are wave-local. */
typedef struct {
    const uint32_t *cur, *ref; uint64_t n; int sh;
    uint64_t blo, bhi;
    uint8_t *pl, *cb, *useds;                 /* [blk][BLK], [blk][256] */
    uint16_t *fts;                            /* [blk][65536] */
    uint32_t *h32;                            /* [blk][65536] */
    uint8_t **bufs; uint32_t *lens; uint64_t *xs;
    int mode, spawned;
} D32W;
static void *d32w_run(void *a) {
    D32W *w = a;
    if (!w->mode) {
        for (uint64_t b = w->blo; b < w->bhi; b++) {
            uint64_t lb = b - w->blo;
            uint64_t b0 = b * BLK, bn = w->n - b0 < BLK ? w->n - b0 : BLK;
            uint8_t *pl = w->pl + lb * BLK, *cb = w->cb + lb * BLK;
            uint8_t *used = w->useds + lb * 256;
            uint32_t *bh = w->h32 + lb * 65536;
            /* fused single pass per plane (wave-local cb — each plane rebuilds
               it, same reads as the serial chunk pass).  Full bh memset is
               cheaper than a separate ref scan to learn the used mask first */
            memset(used, 0, 256);
            memset(bh, 0, 65536 * 4);
            for (uint64_t i = 0; i < bn; i++) {
                int c = (w->ref[b0 + i] >> 23) & 255;
                uint32_t sym = ((w->cur[b0 + i] ^ w->ref[b0 + i]) >> w->sh) & 255;
                cb[i] = (uint8_t)c; pl[i] = (uint8_t)sym;
                used[c] = 1; bh[c * 256 + sym]++;
            }
        }
        return 0;
    }
    uint8_t *scr = xm(SCRSZ);
    uint32_t *cum = xm(256 * 257 * 4);
    for (uint64_t b = w->blo; b < w->bhi; b++) {
        uint64_t lb = b - w->blo;
        uint64_t bn = w->n - b * BLK < BLK ? w->n - b * BLK : BLK;
        const uint8_t *pl = w->pl + lb * BLK, *cb = w->cb + lb * BLK;
        const uint8_t *used = w->useds + lb * 256;
        const uint16_t *ft = w->fts + lb * 65536;
        for (int c = 0; c < 256; c++) if (used[c]) {
            uint32_t *cu = cum + c * 257;
            cu[0] = 0;
            for (int i = 0; i < 256; i++) cu[i + 1] = cu[i] + ft[c * 256 + i];
        }
        uint8_t *pp = scr + SCRSZ;
        uint64_t x = LOWER;
        for (uint64_t i = bn; i-- > 0;) {
            int c = cb[i], sym = pl[i];
            x = enc(x, ft[c * 256 + sym], cum[c * 257 + sym], &pp);
        }
        uint64_t bl = (uint64_t)(scr + SCRSZ - pp);
        w->bufs[lb] = xm(bl ? bl : 1);
        memcpy(w->bufs[lb], pp, bl);
        w->lens[lb] = (uint32_t)bl; w->xs[lb] = x;
    }
    free(scr); free(cum);
    return 0;
}
static size_t dlt32_enc(const uint32_t *cur, const uint32_t *ref, uint64_t n,
                        uint8_t *out, uint64_t *h, int nthr) {
    uint8_t *o = out, *scr = xm(SCRSZ);
    uint16_t *ft = xm(256 * 256 * 2); uint32_t *cum = xm(256 * 257 * 4);
    uint64_t nblk = (n + BLK - 1) / BLK;
    if (nthr > 1 && nblk >= 2) {
        uint64_t wcap = (uint64_t)nthr * 4;
        if (wcap > nblk) wcap = nblk;
        uint8_t *pl = xm(wcap * BLK), *cb = xm(wcap * BLK), *useds = xm(wcap * 256);
        uint16_t *fts = xm(wcap * 65536 * 2);
        uint32_t *h32 = xc(wcap * 65536, 4);
        uint8_t **bufs = xc(wcap, sizeof(uint8_t *));
        uint32_t *lens = xc(wcap, 4); uint64_t *xs = xc(wcap, 8);
        D32W *wj = xc(nthr, sizeof(D32W));
        pthread_t *th = xc(nthr, sizeof(pthread_t));
        for (int p = 0; p < 4; p++) {
            int sh = p * 8;
            for (uint64_t bs = 0; bs < nblk; bs += wcap) {
                uint64_t wn = nblk - bs < wcap ? nblk - bs : wcap;
                int sp = (uint64_t)nthr < wn ? nthr : (int)wn;
                uint64_t per = (wn + sp - 1) / sp;
                for (int phase = 0; phase < 2; phase++) {
                    uint64_t lo = bs;
                    for (int k = 0; k < sp; k++) {
                        D32W *q = &wj[k];
                        q->cur = cur; q->ref = ref; q->n = n; q->sh = sh;
                        q->mode = phase; q->blo = lo;
                        q->bhi = lo + per < bs + wn ? lo + per : bs + wn;
                        q->pl = pl + (lo - bs) * BLK; q->cb = cb + (lo - bs) * BLK;
                        q->useds = useds + (lo - bs) * 256;
                        q->fts = fts + (lo - bs) * 65536;
                        q->h32 = h32 + (lo - bs) * 65536;
                        q->bufs = bufs + (lo - bs); q->lens = lens + (lo - bs);
                        q->xs = xs + (lo - bs); q->spawned = 0;
                        lo = q->bhi;
                        if (pthread_create(&th[k], 0, d32w_run, q)) d32w_run(q);
                        else q->spawned = 1;
                    }
                    for (int k = 0; k < sp; k++) if (wj[k].spawned) pthread_join(th[k], 0);
                    if (phase) continue;
                    for (uint64_t b = 0; b < wn; b++) {
                        const uint8_t *u = useds + b * 256;
                        uint16_t *f = fts + b * 65536;
                        const uint32_t *bh = h32 + b * 65536;
                        for (int c = 0; c < 256; c++) if (u[c]) {
                            norm_ctx(h + (p * 256 + c) * 256, f + c * 256, 256);
                            uint64_t *hp = h + (p * 256 + c) * 256;
                            const uint32_t *bp = bh + c * 256;
                            for (int i = 0; i < 256; i++) hp[i] += bp[i];
                        }
                    }
                }
                for (uint64_t b = 0; b < wn; b++) {
                    uint8_t *bf = bufs[b];
                    o = emit_blk(o, bf + lens[b], bf, xs[b]);
                    free(bf);
                }
            }
        }
        free(pl); free(cb); free(useds); free(fts); free(h32);
        free(bufs); free(lens); free(xs); free(wj); free(th);
        free(scr); free(ft); free(cum);
        return o - out;
    }
    uint8_t *pl = xm(D32CN), *cb = xm(D32CN);
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
                    h[(p * 256 + c) * 256 + sym]++;
                }
                o = emit_blk(o, scr + SCRSZ, pp, x);
            }
        }
    }
    free(pl); free(cb); free(scr); free(ft); free(cum);
    return o - out;
}
/*@ requires \valid_read(ft + (0 .. 65535));
    requires \valid_read(cum + (0 .. 65791));
    requires \valid(rp);
    requires 0 <= end - *rp;
    requires \valid_read(*rp + (0 .. end - *rp - 1));
    requires \valid(pli);
    requires \valid(hp + (0 .. 65535));
    terminates \false;
    exits \exit_status == 1;
    assigns *rp \from *rp, x, end, cbv, ft[0 .. 65535],
        cum[0 .. 65791], (*rp)[0 .. end - *rp - 1];
    assigns *pli, hp[0 .. 65535];
    ensures 0 <= *rp - \at(*rp, Pre) && *rp - \at(*rp, Pre) <= end - \at(*rp, Pre);
    ensures \base_addr(*rp) == \base_addr(\at(*rp, Pre));
    ensures \forall integer j; 0 <= j < 65536 ==>
        hp[j] <= \at(hp[j], Pre) + 1;
*/
static uint64_t d32_elem(uint64_t x, const uint8_t **rp, const uint8_t *end,
                         uint8_t cbv, const uint16_t *ft,
                         const uint32_t *cum, uint8_t *pli, uint64_t *hp) {
    uint32_t v = (uint32_t)(x & (TOT - 1)), fc;
    uint32_t c = cbv % 256;
    uint32_t fo = c * 256, ko = c * 257;
    /*@ assert fo + 255 < 65536 && ko + 256 < 65792; */
    uint32_t sym = dsym_raw(ft + fo, cum + ko, 256, v, &fc);
    if (!fc || fc > TOT) die("corrupt delta32");
    const uint8_t *r = *rp;
    x = dec(x, fc, cum[ko + sym], &r, end);
    uint32_t hidx = fo + sym;
    /*@ assert hidx < 65536; */
    if (hp[hidx] != 0xFFFFFFFFFFFFFFFFu) hp[hidx]++;
    *pli = (uint8_t)sym;
    *rp = r;
    return x;
}

/*@ requires \valid_read(ft + (0 .. 65535));
    requires \valid_read(cum + (0 .. 65791));
    requires \valid_read(r + (0 .. end - r - 1));
    requires 0 <= end - r;
    requires \valid_read(cb + (0 .. bn - 1));
    requires \valid(pl + (0 .. bn - 1));
    requires \valid(hp + (0 .. 65535));
    requires 1 <= bn <= 131072;
    terminates \false;
    exits \exit_status == 1;
    assigns pl[0 .. bn - 1], hp[0 .. 65535];
    ensures \forall integer j; 0 <= j < 65536 ==>
        hp[j] <= \at(hp[j], Pre) + bn;
*/
static void d32_blk(uint64_t x, const uint8_t *r, const uint8_t *end,
                    const uint8_t *cb, uint64_t bn,
                    const uint16_t *ft, const uint32_t *cum,
                    uint8_t *pl, uint64_t *hp) {
    /*@ loop invariant 0 <= i <= bn;
        loop invariant 0 <= end - r;
        loop invariant \valid_read(r + (0 .. end - r - 1));
        loop invariant \forall integer j; 0 <= j < 65536 ==>
            hp[j] <= \at(hp[j], Pre) + i;
        loop assigns x, r, pl[0 .. bn - 1], hp[0 .. 65535], i;
        loop variant bn - i;
    */
    for (uint64_t i = 0; i < bn; i++)
        x = d32_elem(x, &r, end, cb[i], ft, cum, pl + i, hp);
}

/*@ requires \valid_read(cb + (0 .. bn - 1));
    requires bn <= 131072;
    requires \valid_read(hp + (0 .. 65535));
    requires \forall integer j; 0 <= j < 65536 ==> hp[j] <= 281474976710655;
    requires \valid(ft + (0 .. 65535));
    requires \valid(cum + (0 .. 65791));
    terminates \false;
    assigns ft[0 .. 65535], cum[0 .. 65791];
    exits \exit_status == 1;
*/
static void d32_tab(const uint8_t *cb, uint64_t bn, const uint64_t *hp,
                    uint16_t *ft, uint32_t *cum) {
    uint8_t used[256];
    /*@ loop invariant 0 <= z <= 256;
        loop assigns used[0 .. 255], z;
        loop variant 256 - z;
    */
    for (int z = 0; z < 256; z++) used[z] = 0;
    /*@ loop invariant 0 <= i <= bn;
        loop assigns used[0 .. 255], i;
        loop variant bn - i;
    */
    for (uint64_t i = 0; i < bn; i++) used[cb[i] % 256] = 1;
    int fo = 0, ko = 0;
    /*@ loop invariant 0 <= c <= 256;
        loop invariant fo == c * 256 && ko == c * 257;
        loop invariant 0 <= fo <= 65536 && 0 <= ko <= 65792;
        loop assigns ft[0 .. 65535], cum[0 .. 65791], c, fo, ko;
        loop variant 256 - c;
    */
    for (int c = 0; c < 256; c++, fo += 256, ko += 257) {
        if (used[c]) {
            norm_ctx(hp + c * 256, ft + fo, 256);
            fill_ctx(cum + ko, ft + fo, 256);
        }
    }
}

/*@ requires \valid_read(in + (0 .. lim - in - 1));
    requires 0 <= lim - in;
    requires \valid_read(ref + (0 .. n - 1));
    requires \valid(cur + (0 .. n - 1));
    requires \valid(hp + (0 .. 65535));
    requires n <= 281474976710655;
    requires \forall integer j; 0 <= j < 65536 ==>
        hp[j] + n <= 281474976710655;
    requires \valid(pl + (0 .. 131071));
    requires \valid(cb + (0 .. 131071));
    requires \valid(ft + (0 .. 65535));
    requires \valid(cum + (0 .. 65791));
    requires 0 <= plane <= 3;
    requires \separated(cur + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(cur + (0 .. n - 1), ref + (0 .. n - 1));
    requires \separated(cur + (0 .. n - 1), hp + (0 .. 65535));
    requires \separated(cur + (0 .. n - 1), pl + (0 .. 131071));
    requires \separated(cur + (0 .. n - 1), cb + (0 .. 131071));
    requires \separated(cur + (0 .. n - 1), ft + (0 .. 65535));
    requires \separated(cur + (0 .. n - 1), cum + (0 .. 65791));
    requires \separated(hp + (0 .. 65535), in + (0 .. lim - in - 1));
    requires \separated(hp + (0 .. 65535), ref + (0 .. n - 1));
    requires \separated(hp + (0 .. 65535), ft + (0 .. 65535));
    requires \separated(hp + (0 .. 65535), cum + (0 .. 65791));
    requires \separated(hp + (0 .. 65535), pl + (0 .. 131071));
    requires \separated(hp + (0 .. 65535), cb + (0 .. 131071));
    requires \separated(ft + (0 .. 65535), in + (0 .. lim - in - 1));
    requires \separated(ft + (0 .. 65535), ref + (0 .. n - 1));
    requires \separated(ft + (0 .. 65535), cum + (0 .. 65791));
    requires \separated(ft + (0 .. 65535), pl + (0 .. 131071));
    requires \separated(ft + (0 .. 65535), cb + (0 .. 131071));
    requires \separated(cum + (0 .. 65791), in + (0 .. lim - in - 1));
    requires \separated(cum + (0 .. 65791), ref + (0 .. n - 1));
    requires \separated(cum + (0 .. 65791), pl + (0 .. 131071));
    requires \separated(cum + (0 .. 65791), cb + (0 .. 131071));
    requires \separated(ref + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(ref + (0 .. n - 1), pl + (0 .. 131071));
    requires \separated(ref + (0 .. n - 1), cb + (0 .. 131071));
    requires \separated(pl + (0 .. 131071), in + (0 .. lim - in - 1));
    requires \separated(pl + (0 .. 131071), cb + (0 .. 131071));
    requires \separated(cb + (0 .. 131071), in + (0 .. lim - in - 1));
    terminates \false;
    assigns cur[0 .. n - 1], hp[0 .. 65535], pl[0 .. 131071], cb[0 .. 131071],
            ft[0 .. 65535], cum[0 .. 65791];
    assigns \result \from in, n, lim, in[0 .. lim - in - 1];
    exits \exit_status == 1;
    ensures 0 <= \result - in && \result - in <= lim - in;
    ensures \base_addr(\result) == \base_addr(in);
*/
static const uint8_t *d32_plane(const uint8_t *in, const uint8_t *lim,
                                const uint32_t *ref, uint64_t n,
                                uint32_t *cur, uint64_t *hp, int plane,
                                uint8_t *pl, uint8_t *cb, uint16_t *ft,
                                uint32_t *cum) {
    const uint8_t *rp = in;
    int sh = plane * 8;
    /*@ loop invariant 0 <= c0;
        loop invariant 0 <= lim - rp;
        loop invariant in <= rp;
        loop invariant \base_addr(rp) == \base_addr(in);
        loop invariant \valid_read(rp + (0 .. lim - rp - 1));
        loop invariant \forall integer j; 0 <= j < 65536 ==>
            hp[j] <= \at(hp[j], Pre) + c0;
        loop invariant \forall integer j; 0 <= j < 65536 ==>
            hp[j] <= \at(hp[j], Pre) + n;
        loop assigns rp, cur[0 .. n - 1], hp[0 .. 65535], c0,
                pl[0 .. 131071], cb[0 .. 131071], ft[0 .. 65535],
                cum[0 .. 65791];
        loop variant n - c0;
    */
    for (uint64_t c0 = 0; c0 < n; c0 += D32CN) {
        uint64_t cn = n - c0 < D32CN ? n - c0 : D32CN;
        /*@ assert c0 + cn <= n; */
        /*@ loop invariant 0 <= b0;
            loop invariant 0 <= lim - rp;
            loop invariant in <= rp;
            loop invariant \base_addr(rp) == \base_addr(in);
            loop invariant \valid_read(rp + (0 .. lim - rp - 1));
            loop invariant \forall integer j; 0 <= j < 65536 ==>
                hp[j] <= \at(hp[j], Pre) + c0 + (b0 <= cn ? b0 : cn);
            loop invariant \forall integer j; 0 <= j < 65536 ==>
                hp[j] <= \at(hp[j], Pre) + n;
            loop assigns rp, cur[c0 .. c0 + cn - 1], hp[0 .. 65535], b0,
                    pl[0 .. 131071], cb[0 .. 131071], ft[0 .. 65535],
                    cum[0 .. 65791];
            loop variant cn - b0;
        */
        for (uint64_t b0 = 0; b0 < cn; b0 += BLK) {
            uint64_t bn = cn - b0 < BLK ? cn - b0 : BLK;
            /*@ assert b0 + bn <= cn; */
            /*@ assert bn <= 131072; */
            /*@ assert c0 + b0 + bn <= n; */
            /*@ assert \forall integer j; 0 <= j < 65536 ==>
                hp[j] <= 281474976710655; */
            /*@ loop invariant 0 <= i <= bn;
                loop assigns cb[0 .. bn - 1], i;
                loop variant bn - i;
            */
            for (uint64_t i = 0; i < bn; i++)
                cb[i] = (uint8_t)(ref[c0 + b0 + i] >> 23);
            d32_tab(cb, bn, hp, ft, cum);
            uint64_t x; const uint8_t *end;
            /*@ assert 0 <= lim - rp; */
            rp = read_blk(rp, lim, &x, &end);
            d32_blk(x, rp, end, cb, bn, ft, cum, pl, hp);
            rp = end;
            if (plane == 0) {
                /*@ loop invariant 0 <= i <= bn;
                    loop assigns cur[c0 + b0 .. c0 + b0 + bn - 1], i;
                    loop variant bn - i;
                */
                for (uint64_t i = 0; i < bn; i++)
                    cur[c0 + b0 + i] = ref[c0 + b0 + i] ^ ((uint32_t)pl[i]);
            } else {
                /*@ loop invariant 0 <= i <= bn;
                    loop assigns cur[c0 + b0 .. c0 + b0 + bn - 1], i;
                    loop variant bn - i;
                */
                for (uint64_t i = 0; i < bn; i++)
                    cur[c0 + b0 + i] ^= ((uint32_t)pl[i]) << sh;
            }
        }
    }
    return rp;
}

/*@ requires \valid_read(in + (0 .. lim - in - 1));
    requires 0 <= lim - in;
    requires \valid_read(ref + (0 .. n - 1));
    requires \valid(cur + (0 .. n - 1));
    requires \valid(h + (0 .. 262143));
    requires n <= 281474976710655;
    requires \forall integer k; 0 <= k < 262144 ==>
        h[k] + n <= 281474976710655;
    requires \valid(pl + (0 .. 131071));
    requires \valid(cb + (0 .. 131071));
    requires \valid(ft + (0 .. 65535));
    requires \valid(cum + (0 .. 65791));
    requires \separated(cur + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(cur + (0 .. n - 1), ref + (0 .. n - 1));
    requires \separated(cur + (0 .. n - 1), h + (0 .. 262143));
    requires \separated(cur + (0 .. n - 1), pl + (0 .. 131071));
    requires \separated(cur + (0 .. n - 1), cb + (0 .. 131071));
    requires \separated(cur + (0 .. n - 1), ft + (0 .. 65535));
    requires \separated(cur + (0 .. n - 1), cum + (0 .. 65791));
    requires \separated(h + (0 .. 262143), in + (0 .. lim - in - 1));
    requires \separated(h + (0 .. 262143), ref + (0 .. n - 1));
    requires \separated(h + (0 .. 262143), ft + (0 .. 65535));
    requires \separated(h + (0 .. 262143), cum + (0 .. 65791));
    requires \separated(h + (0 .. 262143), pl + (0 .. 131071));
    requires \separated(h + (0 .. 262143), cb + (0 .. 131071));
    requires \separated(ft + (0 .. 65535), in + (0 .. lim - in - 1));
    requires \separated(ft + (0 .. 65535), ref + (0 .. n - 1));
    requires \separated(ft + (0 .. 65535), cum + (0 .. 65791));
    requires \separated(ft + (0 .. 65535), pl + (0 .. 131071));
    requires \separated(ft + (0 .. 65535), cb + (0 .. 131071));
    requires \separated(cum + (0 .. 65791), in + (0 .. lim - in - 1));
    requires \separated(cum + (0 .. 65791), ref + (0 .. n - 1));
    requires \separated(cum + (0 .. 65791), pl + (0 .. 131071));
    requires \separated(cum + (0 .. 65791), cb + (0 .. 131071));
    requires \separated(ref + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(ref + (0 .. n - 1), pl + (0 .. 131071));
    requires \separated(ref + (0 .. n - 1), cb + (0 .. 131071));
    requires \separated(pl + (0 .. 131071), in + (0 .. lim - in - 1));
    requires \separated(pl + (0 .. 131071), cb + (0 .. 131071));
    requires \separated(cb + (0 .. 131071), in + (0 .. lim - in - 1));
    terminates \false;
    assigns cur[0 .. n - 1], h[0 .. 262143], pl[0 .. 131071], cb[0 .. 131071],
            ft[0 .. 65535], cum[0 .. 65791];
    exits \exit_status == 1;
*/
static void dlt32_dec_ws(const uint8_t *in, const uint8_t *lim,
                         const uint32_t *ref, uint64_t n,
                         uint32_t *cur, uint64_t *h,
                         uint8_t *pl, uint8_t *cb, uint16_t *ft,
                         uint32_t *cum) {
    const uint8_t *rp = in;
    /*@ assert \forall integer j; 0 <= j < 65536 ==>
        h[j] + n <= 281474976710655; */
    /*@ assert \separated(h + (0 .. 65535), in + (0 .. lim - in - 1)); */
    /*@ assert \separated(h + (0 .. 65535), ref + (0 .. n - 1)); */
    /*@ assert \separated(h + (0 .. 65535), ft + (0 .. 65535)); */
    /*@ assert \separated(h + (0 .. 65535), cum + (0 .. 65791)); */
    /*@ assert \separated(h + (0 .. 65535), pl + (0 .. 131071)); */
    /*@ assert \separated(h + (0 .. 65535), cb + (0 .. 131071)); */
    /*@ assert \separated(cur + (0 .. n - 1), h + (0 .. 65535)); */
    /*@ assert 0 <= lim - rp && \base_addr(rp) == \base_addr(in) &&
        \valid_read(rp + (0 .. lim - rp - 1)); */
    rp = d32_plane(rp, lim, ref, n, cur, h, 0, pl, cb, ft, cum);
    /*@ assert \forall integer j; 0 <= j < 65536 ==>
        h[65536 + j] + n <= 281474976710655; */
    /*@ assert \separated(h + 65536 + (0 .. 65535), in + (0 .. lim - in - 1)); */
    /*@ assert \separated(h + 65536 + (0 .. 65535), ref + (0 .. n - 1)); */
    /*@ assert \separated(h + 65536 + (0 .. 65535), ft + (0 .. 65535)); */
    /*@ assert \separated(h + 65536 + (0 .. 65535), cum + (0 .. 65791)); */
    /*@ assert \separated(h + 65536 + (0 .. 65535), pl + (0 .. 131071)); */
    /*@ assert \separated(h + 65536 + (0 .. 65535), cb + (0 .. 131071)); */
    /*@ assert \separated(cur + (0 .. n - 1), h + 65536 + (0 .. 65535)); */
    /*@ assert 0 <= lim - rp && \base_addr(rp) == \base_addr(in) &&
        \valid_read(rp + (0 .. lim - rp - 1)); */
    rp = d32_plane(rp, lim, ref, n, cur, h + 65536, 1, pl, cb, ft, cum);
    /*@ assert \forall integer j; 0 <= j < 65536 ==>
        h[131072 + j] + n <= 281474976710655; */
    /*@ assert \separated(h + 131072 + (0 .. 65535), in + (0 .. lim - in - 1)); */
    /*@ assert \separated(h + 131072 + (0 .. 65535), ref + (0 .. n - 1)); */
    /*@ assert \separated(h + 131072 + (0 .. 65535), ft + (0 .. 65535)); */
    /*@ assert \separated(h + 131072 + (0 .. 65535), cum + (0 .. 65791)); */
    /*@ assert \separated(h + 131072 + (0 .. 65535), pl + (0 .. 131071)); */
    /*@ assert \separated(h + 131072 + (0 .. 65535), cb + (0 .. 131071)); */
    /*@ assert \separated(cur + (0 .. n - 1), h + 131072 + (0 .. 65535)); */
    /*@ assert 0 <= lim - rp && \base_addr(rp) == \base_addr(in) &&
        \valid_read(rp + (0 .. lim - rp - 1)); */
    rp = d32_plane(rp, lim, ref, n, cur, h + 131072, 2, pl, cb, ft, cum);
    /*@ assert \forall integer j; 0 <= j < 65536 ==>
        h[196608 + j] + n <= 281474976710655; */
    /*@ assert \separated(h + 196608 + (0 .. 65535), in + (0 .. lim - in - 1)); */
    /*@ assert \separated(h + 196608 + (0 .. 65535), ref + (0 .. n - 1)); */
    /*@ assert \separated(h + 196608 + (0 .. 65535), ft + (0 .. 65535)); */
    /*@ assert \separated(h + 196608 + (0 .. 65535), cum + (0 .. 65791)); */
    /*@ assert \separated(h + 196608 + (0 .. 65535), pl + (0 .. 131071)); */
    /*@ assert \separated(h + 196608 + (0 .. 65535), cb + (0 .. 131071)); */
    /*@ assert \separated(cur + (0 .. n - 1), h + 196608 + (0 .. 65535)); */
    /*@ assert 0 <= lim - rp && \base_addr(rp) == \base_addr(in) &&
        \valid_read(rp + (0 .. lim - rp - 1)); */
    rp = d32_plane(rp, lim, ref, n, cur, h + 196608, 3, pl, cb, ft, cum);
    if (rp != lim) die("corrupt delta32");
}

static void dlt32_dec(const uint8_t *in, const uint8_t *lim, const uint32_t *ref, uint64_t n,
                      uint32_t *cur, uint64_t *h) {
    uint8_t *pl = xm(BLK), *cb = xm(BLK);
    uint16_t *ft = xm(256 * 256 * 2); uint32_t *cum = xm(256 * 257 * 4);
    dlt32_dec_ws(in, lim, ref, n, cur, h, pl, cb, ft, cum);
    free(pl); free(cb); free(ft); free(cum);
}

/* ============ PACK (dict + bitpack) ============ */
static size_t pack_enc(const uint8_t *data, uint64_t n, int bsz, uint8_t *out) {
    /* gather unique elements (as bsz-byte atoms) */
    int aw;            /* hash space */
    if (bsz == 2) aw = 65536;
    else if (bsz == 1) aw = 256;
    else aw = 65536;   /* wider atoms: hash first 2 bytes — rare, fallback RAW anyway */
    uint64_t *cnt = xc(aw, 8);   /* u64: a single atom can legitimately occur >2^32 times */
    uint64_t ne = n / bsz;
    if (bsz == 2) { const uint16_t *s = (const uint16_t *)data; for (uint64_t i = 0; i < ne; i++) cnt[s[i]]++; }
    else { for (uint64_t i = 0; i < ne; i++) cnt[data[i]]++; }
    int k = 0;
    for (int i = 0; i < aw; i++) if (cnt[i]) k++;
    if (k < 1 || k > 256) { free(cnt); return 0; }
    int ib = 0; while ((1 << ib) < k) ib++;
    uint8_t *o = out;
    p32le(o, (uint32_t)k); o += 4;
    uint16_t dict[256] = { 0 }; int dk = 0;
    for (int i = 0; i < aw; i++) if (cnt[i]) dict[dk++] = (uint16_t)i;
    for (int i = 0; i < k; i++) p16le(o + 2 * i, dict[i]);   /* dict is explicitly little-endian */
    o += k * 2;
    /* build reverse map */
    int *rmap = xc(aw, 4);
    for (int i = 0; i < k; i++) rmap[dict[i]] = i;
    /* bitpack ib-bit indices, MSB-first */
    if (ne > UINT64_MAX / 8) die("pack size");   /* ne*ib must not wrap */
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
/*@ requires \valid_read(in + (0 .. lim - in - 1));
    requires 0 <= lim - in;
    requires \valid(data + (0 .. n - 1));
    requires bsz == 1 || bsz == 2;
    requires n <= 281474976710655;
    requires lim - in <= 281474976710655;
    requires \separated(data + (0 .. n - 1), in + (0 .. lim - in - 1));
    terminates \false;
    exits \exit_status == 1;
    assigns data[0 .. n - 1];
*/
static void pack_dec(const uint8_t *in, const uint8_t *lim, uint64_t n, int bsz, uint8_t *data) {
    if (lim - in < 4) die("corrupt pack");
    uint32_t k = g32le(in); in += 4;
    if (k < 1 || k > 256) die("corrupt pack");
    if ((uint64_t)(lim - in) < (uint64_t)k * 2) die("corrupt pack");
    uint8_t dict[512];              /* raw LE bytes — no host-order cast */
    /*@ loop invariant 0 <= di <= k * 2;
        loop assigns dict[0 .. 511], di;
        loop variant k * 2 - di;
    */
    for (uint32_t di = 0; di < k * 2; di++) dict[di] = in[di];
    in += k * 2;
    uint64_t ib = 0, p2 = 1;
    /*@ loop invariant 0 <= ib <= 8 && p2 >= 1;
        loop assigns ib, p2;
        loop variant 8 - ib;
    */
    while (p2 < k) { if (ib >= 8) die("corrupt pack"); ib++; p2 += p2; }
    uint64_t ne = n / bsz, bit = 0, nb = (uint64_t)(lim - in);
    /*@ loop invariant 0 <= i <= ne;
        loop invariant 0 <= bit <= 8 * nb;
        loop assigns data[0 .. n - 1], i, bit;
        loop variant ne - i;
    */
    for (uint64_t i = 0; i < ne; i++) {
        uint32_t idx = 0;
        if (ib) {   /* k==1: zero-bit indices — the index stream is empty */
            if ((bit + ib - 1) / 8 >= nb) die("corrupt pack");
            /*@ assert bit / 8 <= (bit + ib - 1) / 8; */
            /*@ loop invariant 0 <= b <= ib;
                loop invariant bit + ib - b <= 8 * nb;
                loop assigns idx, bit, b;
                loop variant ib - b;
            */
            for (uint64_t b = 0; b < ib; b++) {
                idx = idx * 2 + ((in[bit / 8] >> (7 - bit % 8)) % 2);
                bit++;
            }
        }
        if (idx >= k) die("corrupt pack");
        /*@ assert 2 * idx + 1 < 512; */
        /*@ assert bsz == 2 ==> 2 * i + 1 < n; */
        if (bsz == 2) { data[2 * i] = dict[2 * idx]; data[2 * i + 1] = dict[2 * idx + 1]; }
        else data[i] = dict[2 * idx];
    }
    /* the index stream must be exactly ceil(ne*ib/8) bytes — no slack */
    if ((uint64_t)(lim - in) != (bit + 7) / 8) die("corrupt pack");
}

/* ================= archive ================= */
/* archive-stored __metadata__ is replayed verbatim into regenerated headers;
   it must be one complete, strictly valid JSON value on a NUL-terminated
   buffer (all callers copy+terminate first). Balance-only checking is NOT
   enough: `{"a" 1}` replayed verbatim would emit malformed output JSON. */
static const char *jws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}
/* strict JSON string: every escape must be valid (incl. 4-hex \uXXXX);
   raw control bytes <0x20 are forbidden by RFC 8259 */
static const char *jstrv(const char *p) {
    if (*p != '"') return 0;
    for (p++;; p++) {
        uint8_t c = (uint8_t)*p;
        if (!c) return 0;
        if (c == '"') return p + 1;
        if (c < 0x20) return 0;
        if (c == '\\') {
            p++;
            if (!*p) return 0;   /* strchr would match the terminator itself */
            if (*p == 'u') {
                for (int k = 1; k <= 4; k++)
                    if (!isxdigit((uint8_t)p[k])) return 0;
                p += 4;
            } else if (!strchr("\"\\/bfnrt", *p)) return 0;
        }
    }
}
/* JSON number: -? (0|[1-9][0-9]*) (\.[0-9]+)? ([eE][+-]?[0-9]+)? */
static const char *jnum(const char *s) {
    if (*s == '-') s++;
    if (*s == '0') s++;
    else if (*s >= '1' && *s <= '9') while (*s >= '0' && *s <= '9') s++;
    else return 0;
    if (*s == '.') {
        s++;
        if (*s < '0' || *s > '9') return 0;
        while (*s >= '0' && *s <= '9') s++;
    }
    if (*s == 'e' || *s == 'E') {
        s++;
        if (*s == '+' || *s == '-') s++;
        if (*s < '0' || *s > '9') return 0;
        while (*s >= '0' && *s <= '9') s++;
    }
    return s;
}
static const char *jval(const char *p, int depth) {
    if (depth > 128) return 0;          /* crafted nesting vs stack */
    p = jws(p);
    if (*p == '"') return jstrv(p);
    if (*p == '{') {
        p = jws(p + 1);
        if (*p == '}') return p + 1;
        for (;;) {
            if (*p != '"') return 0;
            const char *e = jstrv(p);
            if (!e) return 0;
            p = jws(e);
            if (*p != ':') return 0;
            p = jval(p + 1, depth + 1);
            if (!p) return 0;
            p = jws(p);
            if (*p == '}') return p + 1;
            if (*p != ',') return 0;
            p = jws(p + 1);
        }
    }
    if (*p == '[') {
        p = jws(p + 1);
        if (*p == ']') return p + 1;
        for (;;) {
            p = jval(p, depth + 1);
            if (!p) return 0;
            p = jws(p);
            if (*p == ']') return p + 1;
            if (*p != ',') return 0;
            p = jws(p + 1);
        }
    }
    /* strncmp stops at the NUL terminator — memcmp would read past it */
    if (!strncmp(p, "true", 4)) return p + 4;
    if (!strncmp(p, "false", 5)) return p + 5;
    if (!strncmp(p, "null", 4)) return p + 4;
    return jnum(p);
}
static void ck_meta(const uint8_t *m, uint32_t ml) {
    if (!ml) return;
    if (memchr(m, 0, ml)) die("bad metadata");
    const char *e = jval((const char *)m, 0);
    if (!e) die("bad metadata");
    if (jws(e) != (const char *)m + ml) die("bad metadata");  /* trailing junk */
}

/* duplicate (file,name) records would emit duplicate JSON keys in `d` output.
   open-addressed set over (name,file): entry = 2 file bytes + name. */
static uint64_t nfh(const char *name, int file) {
    uint64_t h = 1469598103934665603ull;
    for (const char *p = name; *p; p++) { h ^= (uint8_t)*p; h *= 1099511628211ull; }
    return h ^ (uint64_t)file * 0x9E3779B97F4A7C15ull;
}
static int nf_find(char **seen, uint64_t mask, const char *name, int file) {
    for (uint64_t i = nfh(name, file) & mask;; i = (i + 1) & mask) {
        const char *s = seen[i];
        if (!s) return 0;
        if (s[0] == (char)(file & 255) && s[1] == (char)((file >> 8) & 255) && !strcmp(s + 2, name))
            return 1;
    }
}
static void nf_put(char **seen, uint64_t mask, const char *name, int file) {
    uint64_t i = nfh(name, file) & mask;
    while (seen[i]) i = (i + 1) & mask;
    char *e = xm(strlen(name) + 3);
    e[0] = (char)(file & 255); e[1] = (char)((file >> 8) & 255); strcpy(e + 2, name);
    seen[i] = e;
}
static void ck_dupname(char **seen, const char *name, int file, uint32_t n, uint32_t cap) {
    if (nf_find(seen, cap - 1, name, file)) die("dup tensor in archive");
    nf_put(seen, cap - 1, name, file);
    (void)n;
}

/* append formatted text to a growing JSON header — guards the one
   snprintf failure mode that matters: a negative return would wrap jl */
static void jput(char *j, size_t *jl, size_t hcap, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    int r = vsnprintf(j + *jl, hcap - *jl, fmt, ap);
    va_end(ap);
    if (r < 0) die("header emit");
    *jl += (size_t)r;
}

/* name->source-index map: v looks up every archive tensor in its source
   file — a per-tensor O(n) strcmp scan is quadratic on crafted archives */
typedef struct { char **k; int32_t *v; uint64_t m; } NMap;
static void nm_build(NMap *M, Tensor *t, int n) {
    uint64_t c = 64; while (c < (uint64_t)n * 2) c *= 2;
    M->m = c - 1; M->k = xc(c, 8); M->v = xc(c, 4);
    for (int j = 0; j < n; j++) {
        uint64_t i = nfh(t[j].name, 0) & M->m;
        while (M->k[i]) i = (i + 1) & M->m;
        M->k[i] = t[j].name; M->v[i] = j;
    }
}
static int nm_get(const NMap *M, const char *name) {
    for (uint64_t i = nfh(name, 0) & M->m; M->k[i]; i = (i + 1) & M->m)
        if (!strcmp(M->k[i], name)) return M->v[i];
    return -1;
}

/* archive integers are explicitly little-endian (portable format) */
static void w64(FILE *f, uint64_t v) { uint8_t b[8]; for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i)); fwrite(b, 8, 1, f); }
static void w32(FILE *f, uint32_t v) { uint8_t b[4]; for (int i = 0; i < 4; i++) b[i] = (uint8_t)(v >> (8 * i)); fwrite(b, 4, 1, f); }
static void w16(FILE *f, uint16_t v) { uint8_t b[2] = { (uint8_t)v, (uint8_t)(v >> 8) }; fwrite(b, 2, 1, f); }
static void w8(FILE *f, uint8_t v) { fwrite(&v, 1, 1, f); }
/*@ requires \valid(p);
    requires \valid_read(*p + (0 .. 7));
    assigns *p \from *p;
    ensures *p == \at(*p, Pre) + 8;
    ensures \base_addr(*p) == \base_addr(\at(*p, Pre));
*/
static uint64_t r64(const uint8_t **p) {
    const uint8_t *r = *p;
    uint64_t v = r[7];
    v = v * 256 + r[6]; v = v * 256 + r[5];
    v = v * 256 + r[4]; v = v * 256 + r[3];
    v = v * 256 + r[2]; v = v * 256 + r[1]; v = v * 256 + r[0];
    *p = r + 8;
    return v;
}
/*@ requires \valid(p);
    requires \valid_read(*p + (0 .. 3));
    assigns *p \from *p;
    ensures *p == \at(*p, Pre) + 4;
    ensures \base_addr(*p) == \base_addr(\at(*p, Pre));
*/
static uint32_t r32(const uint8_t **p) {
    const uint8_t *r = *p;
    uint32_t v = r[3];
    v = v * 256 + r[2]; v = v * 256 + r[1]; v = v * 256 + r[0];
    *p = r + 4;
    return v;
}
/*@ requires \valid(p);
    requires \valid_read(*p + (0 .. 1));
    assigns *p \from *p;
    ensures *p == \at(*p, Pre) + 2;
    ensures \base_addr(*p) == \base_addr(\at(*p, Pre));
*/
static uint16_t r16(const uint8_t **p) { uint16_t v = (uint16_t)((*p)[0] + (uint16_t)(*p)[1] * 256); *p += 2; return v; }
/*@ requires \valid(p);
    requires \valid_read(*p);
    assigns *p \from *p;
    ensures *p == \at(*p, Pre) + 1;
    ensures \base_addr(*p) == \base_addr(\at(*p, Pre));
*/
static uint8_t r8(const uint8_t **p) { uint8_t v = **p; *p += 1; return v; }

static void emit_rec(FILE *of, Tensor *t, int bat) {   /* bat: 0 solo, 1 head, 2 member */
    size_t nl_ = strlen(t->name), dl_ = strlen(t->dtype);
    if (nl_ > 65535 || dl_ > 65535) die("name too long for record");
    uint16_t nl = (uint16_t)nl_, dl = (uint16_t)dl_;
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
    w32(of, bat ? t->crc : crc32b(t->data, t->len));   /* batch workers
        precomputed the CRC inside ejob_run; solo tensors pay it inline */
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
    uint64_t got = 0;
    while (got < *sz) {          /* pread may return short on huge files */
        ssize_t r = pread(fd, b + got, *sz - got, (off_t)got);
        if (r <= 0) die("read");
        got += (uint64_t)r;
    }
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

static FILE *xfopen_tmp(const char *p) {
    /* Temp outputs must not follow a planted symlink (overwrite primitive
       in a shared dir) or block on a planted FIFO: O_NOFOLLOW refuses
       symlink finals (ELOOP), O_NONBLOCK fails an unreader'd FIFO (ENXIO)
       and is then cleared so normal writes aren't affected. */
    int fd = open(p, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_NONBLOCK, 0600);
    if (fd < 0) return 0;
    fcntl(fd, F_SETFL, 0);
    FILE *f = fdopen(fd, "wb");
    if (!f) close(fd);
    return f;
}

/* is `out` the same file as `in`?  Plain strcmp misses symlink/hardlink
   aliases and textual aliases like "dir/../out".  stat() catches any alias
   when `out` already exists; otherwise canonicalize the directory part and
   compare resolved paths. */
static int same_out_path(const char *out, const char *in) {
    if (!strcmp(out, in)) return 1;
    struct stat so, si;
    if (!stat(in, &si) && !stat(out, &so) &&
        si.st_dev == so.st_dev && si.st_ino == so.st_ino) return 1;
    char *ri = realpath(in, NULL);
    if (!ri) return 0;                    /* unreadable input dies elsewhere */
    int eq = 0;
    char *od = xstrdup(out);
    const char *d = dirname(od);          /* GNU dirname: may return "." */
    char *rd = realpath(d, NULL);
    if (rd) {
        size_t n = strlen(rd) + strlen(bname(out)) + 2;
        char *full = xm(n);
        snprintf(full, n, "%s/%s", rd, bname(out));
        eq = !strcmp(full, ri);
        free(full); free(rd);
    }
    free(od); free(ri);
    return eq;
}

/* JSON-escape s into o[cap]; returns full needed length (snprintf-style) */
static size_t jesc(char *o, size_t cap, const char *s) {
    size_t n = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        char tmp[8]; const char *rep = tmp;
        switch (*p) {
        case '"': rep = "\\\""; break;  case '\\': rep = "\\\\"; break;
        case '\b': rep = "\\b"; break;  case '\f': rep = "\\f"; break;
        case '\n': rep = "\\n"; break;  case '\r': rep = "\\r"; break;
        case '\t': rep = "\\t"; break;
        default:
            if (*p < 0x20) snprintf(tmp, sizeof tmp, "\\u%04x", *p);
            else { tmp[0] = (char)*p; tmp[1] = 0; }
        }
        size_t rl = strlen(rep);
        if (n + rl + 1 <= cap) memcpy(o + n, rep, rl + 1);
        n += rl;
    }
    if (cap) o[n < cap ? n : cap - 1] = 0;
    return n;
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
/* RAW/PACK carry no rANS streams, so a winning RAW/PACK would otherwise
 * leave the float channels cold forever on many small tensors. Feed the
 * tensor's natural channel instead — identical counting on both sides,
 * so encode and decode stay in lockstep. Runs AFTER winner commit on the
 * encode side / after payload decode on the decode side. */
static void teach(const Tensor *t, const uint8_t *d) {
    /* d may point into the unaligned archive mmap (zero-copy RAW path) —
       all typed loads must go through memcpy */
    int bsz = dtb(t->dtype);
    uint64_t ne = bsz > 1 ? t->len / bsz : 0;
    if (ne && is_flt16(t->dtype)) {
        int mb = mbits_of(t->dtype), ew = 1 << (15 - mb), mw = 1 << mb;
        uint64_t *h = H(is_bf(t->dtype) ? CBF : CFP);
        for (uint64_t i = 0; i < ne; i++) {
            uint16_t v; memcpy(&v, d + i * 2, 2);
            uint32_t S = v >> 15, E = (v >> mb) & (ew - 1), Mv = v & (mw - 1);
            h[S]++; h[2 + S * ew + E]++; h[2 + 2 * ew + (size_t)(S * ew + E) * mw + Mv]++;
        }
    } else if (ne && is_f32(t->dtype)) {
        uint64_t *h = H(C32);
        for (uint64_t i = 0; i < ne; i++) {
            uint32_t v; memcpy(&v, d + i * 4, 4);
            uint32_t S = v >> 31, E = (v >> 23) & 255;
            size_t se = S * 256 + E;
            h[S]++; h[2 + S * 256 + E]++;
            h[2 + 512 + se * 128 + ((v >> 16) & 127)]++;
            h[2 + 512 + 512 * 128 + se * 256 + ((v >> 8) & 255)]++;
            h[2 + 512 + 512 * 128 + 512 * 256 + se * 256 + (v & 255)]++;
        }
    } else {
        uint64_t *h = H(C8);
        for (uint64_t i = 0; i < t->len; i++) h[d[i]]++;
    }
}

/* ============ PREVROW: intra-tensor previous-row delta ============
 * Row i is coded as an ordered kmap16 residual against row i-1 — the
   self-referential analog of DELTA16 for smooth-row tensors (embeddings,
   norm layers, low-rank structure). Payload: [u32 len1][FIELD(row0)] then
   per row [u32 len][DELTA16 stream(row)]. Rows are SEPARATE dlt streams:
   a single stream over the whole tensor would make dlt_dec's block-level
   used-mask read not-yet-written self-referential positions (and deferred
   escape values would still be 0 placeholders when read as refs).
   Per-row calls guarantee the referent row is fully final before use. */
static size_t prw_enc(const uint16_t *s, uint64_t n, uint64_t cols, int mb,
                      uint8_t *out, uint64_t *hf, uint64_t *hd) {
    DltWs w; dltws_init(&w);
    uint8_t *o = out + 4;
    uint32_t l1 = (uint32_t)f16_enc(s, cols, mb, o, hf, 1);
    p32le(out, l1); o += l1;
    for (uint64_t r = 1; r * cols < n; r++) {
        uint64_t rn = n - r * cols < cols ? n - r * cols : cols;
        uint8_t *hdr = o; o += 4;
        uint32_t lr = (uint32_t)dlt_enc_ws(s + r * cols, s + (r - 1) * cols, rn, o, hd, mb, hf, &w, 1);
        p32le(hdr, lr); o += lr;
    }
    dltws_free(&w);
    return o - out;
}
/*@ requires \valid_read(in + (0 .. lim - in - 1));
    requires 0 <= lim - in;
    requires \valid(s + (0 .. n - 1));
    requires 1 <= cols <= n;
    requires n <= 281474976710655;
    requires mb == 7 || mb == 10;
    requires \valid(hf + (0 .. (mb == 7 ? 66050 : 65602) - 1));
    requires \forall integer j; 0 <= j < (mb == 7 ? 66050 : 65602) ==>
        hf[j] + n <= 281474976710655;
    requires \valid(hd + (0 .. 524543));
    requires \forall integer j; 0 <= j < 524544 ==>
        hd[j] + n <= 281474976710655;
    requires \valid(w);
    requires \valid(w->ft + (0 .. 524415));
    requires \valid(w->dcum + (0 .. 524543));
    requires \valid(w->used + (0 .. 127));
    requires \valid(w->escbits + (0 .. (cols + 7) / 8 - 1));
    requires \valid(w->escbuf + (0 .. cols - 1));
    requires \separated(s + (0 .. n - 1), in + (0 .. lim - in - 1));
    requires \separated(s + (0 .. n - 1),
                        hf + (0 .. (mb == 7 ? 66050 : 65602) - 1));
    requires \separated(s + (0 .. n - 1), hd + (0 .. 524543));
    requires \separated(s + (0 .. n - 1), w->ft + (0 .. 524415));
    requires \separated(s + (0 .. n - 1), w->dcum + (0 .. 524543));
    requires \separated(s + (0 .. n - 1), w->used + (0 .. 127));
    requires \separated(s + (0 .. n - 1),
                        w->escbits + (0 .. (cols + 7) / 8 - 1));
    requires \separated(s + (0 .. n - 1), w->escbuf + (0 .. cols - 1));
    requires \separated(s + (0 .. n - 1), w);
    requires \separated(hf + (0 .. (mb == 7 ? 66050 : 65602) - 1),
                        in + (0 .. lim - in - 1));
    requires \separated(hf + (0 .. (mb == 7 ? 66050 : 65602) - 1),
                        hd + (0 .. 524543));
    requires \separated(hf + (0 .. (mb == 7 ? 66050 : 65602) - 1),
                        w->ft + (0 .. 524415));
    requires \separated(hf + (0 .. (mb == 7 ? 66050 : 65602) - 1),
                        w->dcum + (0 .. 524543));
    requires \separated(hf + (0 .. (mb == 7 ? 66050 : 65602) - 1),
                        w->used + (0 .. 127));
    requires \separated(hf + (0 .. (mb == 7 ? 66050 : 65602) - 1),
                        w->escbits + (0 .. (cols + 7) / 8 - 1));
    requires \separated(hf + (0 .. (mb == 7 ? 66050 : 65602) - 1),
                        w->escbuf + (0 .. cols - 1));
    requires \separated(hd + (0 .. 524543), in + (0 .. lim - in - 1));
    requires \separated(hd + (0 .. 524543), w->ft + (0 .. 524415));
    requires \separated(hd + (0 .. 524543), w->dcum + (0 .. 524543));
    requires \separated(hd + (0 .. 524543), w->used + (0 .. 127));
    requires \separated(hd + (0 .. 524543),
                        w->escbits + (0 .. (cols + 7) / 8 - 1));
    requires \separated(hd + (0 .. 524543), w->escbuf + (0 .. cols - 1));
    requires \separated(w->ft + (0 .. 524415), in + (0 .. lim - in - 1));
    requires \separated(w->ft + (0 .. 524415), w->dcum + (0 .. 524543));
    requires \separated(w->ft + (0 .. 524415), w->used + (0 .. 127));
    requires \separated(w->ft + (0 .. 524415),
                        w->escbits + (0 .. (cols + 7) / 8 - 1));
    requires \separated(w->ft + (0 .. 524415), w->escbuf + (0 .. cols - 1));
    requires \separated(w->dcum + (0 .. 524543), in + (0 .. lim - in - 1));
    requires \separated(w->dcum + (0 .. 524543), w->used + (0 .. 127));
    requires \separated(w->dcum + (0 .. 524543),
                        w->escbits + (0 .. (cols + 7) / 8 - 1));
    requires \separated(w->dcum + (0 .. 524543), w->escbuf + (0 .. cols - 1));
    requires \separated(w->used + (0 .. 127), in + (0 .. lim - in - 1));
    requires \separated(w->used + (0 .. 127),
                        w->escbits + (0 .. (cols + 7) / 8 - 1));
    requires \separated(w->used + (0 .. 127), w->escbuf + (0 .. cols - 1));
    requires \separated(w->escbits + (0 .. (cols + 7) / 8 - 1),
                        in + (0 .. lim - in - 1));
    requires \separated(w->escbits + (0 .. (cols + 7) / 8 - 1),
                        w->escbuf + (0 .. cols - 1));
    requires \separated(w->escbuf + (0 .. cols - 1),
                        in + (0 .. lim - in - 1));
    terminates \false;
    assigns s[0 .. n - 1], hf[0 .. (mb == 7 ? 66050 : 65602) - 1],
            hd[0 .. 524543], w->ft[0 .. 524415], w->dcum[0 .. 524543],
            w->used[0 .. 127], w->escbits[0 .. (cols + 7) / 8 - 1],
            w->escbuf[0 .. cols - 1];
    exits \exit_status == 1;
    ensures \forall integer j; 0 <= j < (mb == 7 ? 66050 : 65602) ==>
        hf[j] <= \at(hf[j], Pre) + n;
    ensures \forall integer j; 0 <= j < 524544 ==>
        hd[j] <= \at(hd[j], Pre) + n;
*/
static void prw_dec_ws(const uint8_t *in, const uint8_t *lim, uint64_t n, uint64_t cols,
                       int mb, uint16_t *s, uint64_t *hf, uint64_t *hd, DltWs *w) {
    if ((uint64_t)(lim - in) < 4) die("corrupt prw");
    uint32_t l1 = g32le(in); in += 4;
    if ((uint64_t)l1 > (uint64_t)(lim - in)) die("corrupt prw");
    f16_dec(in, in + l1, cols, mb, s, hf);
    in += l1;
    /*@ loop invariant cols <= pos;
        loop invariant 0 <= lim - in;
        loop invariant 0 <= in - \at(in, Pre);
        loop invariant \base_addr(in) == \base_addr(\at(in, Pre));
        loop invariant \valid_read(in + (0 .. lim - in - 1));
        loop invariant \forall integer j; 0 <= j < (mb == 7 ? 66050 : 65602) ==>
            hf[j] <= \at(hf[j], Pre) + pos;
        loop invariant \forall integer j; 0 <= j < (mb == 7 ? 66050 : 65602) ==>
            hf[j] <= \at(hf[j], Pre) + n;
        loop invariant \forall integer j; 0 <= j < 524544 ==>
            hd[j] <= \at(hd[j], Pre) + pos;
        loop invariant \forall integer j; 0 <= j < 524544 ==>
            hd[j] <= \at(hd[j], Pre) + n;
        loop assigns in, pos, s[0 .. n - 1],
                hf[0 .. (mb == 7 ? 66050 : 65602) - 1], hd[0 .. 524543],
                w->ft[0 .. 524415], w->dcum[0 .. 524543],
                w->used[0 .. 127], w->escbits[0 .. (cols + 7) / 8 - 1],
                w->escbuf[0 .. cols - 1];
        loop variant n - pos;
    */
    for (uint64_t pos = cols; pos < n; pos += cols) {
        uint64_t rn = n - pos < cols ? n - pos : cols;
        /*@ assert 1 <= rn && rn <= cols; */
        /*@ assert pos + rn <= n; */
        /*@ assert pos - cols + rn <= n; */
        /*@ assert (rn + 7) / 8 <= (cols + 7) / 8; */
        if ((uint64_t)(lim - in) < 4) die("corrupt prw");
        uint32_t lr = g32le(in); in += 4;
        if ((uint64_t)lr > (uint64_t)(lim - in)) die("corrupt prw");
        /*@ assert \separated(s + pos + (0 .. rn - 1),
                              s + pos - cols + (0 .. rn - 1)); */
        /*@ assert \separated(s + pos + (0 .. rn - 1),
                              in + (0 .. lim - in - 1)); */
        /*@ assert \separated(s + pos + (0 .. rn - 1), hd + (0 .. 524543)); */
        /*@ assert \separated(s + pos + (0 .. rn - 1), w->ft + (0 .. 524415)); */
        /*@ assert \separated(s + pos + (0 .. rn - 1),
                              w->dcum + (0 .. 524543)); */
        /*@ assert \separated(s + pos + (0 .. rn - 1), w->used + (0 .. 127)); */
        /*@ assert \separated(s + pos + (0 .. rn - 1),
                              w->escbits + (0 .. (rn + 7) / 8 - 1)); */
        /*@ assert \separated(s + pos - cols + (0 .. rn - 1),
                              in + (0 .. lim - in - 1)); */
        /*@ assert \separated(s + pos - cols + (0 .. rn - 1),
                              w->escbits + (0 .. (rn + 7) / 8 - 1)); */
        /*@ assert \separated(s + pos - cols + (0 .. rn - 1),
                              w->ft + (0 .. 524415)); */
        /*@ assert \separated(s + pos - cols + (0 .. rn - 1),
                              w->dcum + (0 .. 524543)); */
        /*@ assert \separated(s + pos - cols + (0 .. rn - 1),
                              w->used + (0 .. 127)); */
        /*@ assert \separated(hd + (0 .. 524543),
                              w->escbits + (0 .. (rn + 7) / 8 - 1)); */
        /*@ assert \separated(hd + (0 .. 524543),
                              s + pos - cols + (0 .. rn - 1)); */
        /*@ assert \separated(w->escbits + (0 .. (rn + 7) / 8 - 1),
                              in + (0 .. lim - in - 1)); */
        uint64_t escn;
        /*@ assert \separated(&escn, s + pos + (0 .. rn - 1)); */
        /*@ assert \separated(&escn, hd + (0 .. 524543)); */
        /*@ assert \separated(&escn, w->ft + (0 .. 524415)); */
        /*@ assert \separated(&escn, w->dcum + (0 .. 524543)); */
        const uint8_t *rp = dlt_dec_ws(in, in + lr, s + pos - cols, rn,
                                     s + pos, hd, w, &escn);
        /*@ assert escn <= rn && escn <= cols; */
        /*@ assert \separated(w->escbuf + (0 .. escn - 1),
                              hf + (0 .. (mb == 7 ? 66050 : 65602) - 1)); */
        /*@ assert \separated(w->escbuf + (0 .. escn - 1),
                              in + (0 .. lim - in - 1)); */
        /*@ assert \separated(w->escbuf + (0 .. escn - 1),
                              s + pos + (0 .. rn - 1)); */
        /*@ assert \separated(w->escbuf + (0 .. escn - 1),
                              w->escbits + (0 .. (rn + 7) / 8 - 1)); */
        /*@ assert \separated(hf + (0 .. (mb == 7 ? 66050 : 65602) - 1),
                              in + (0 .. lim - in - 1)); */
        /*@ assert \separated(hf + (0 .. (mb == 7 ? 66050 : 65602) - 1),
                              s + pos + (0 .. rn - 1)); */
        /*@ assert \separated(hf + (0 .. (mb == 7 ? 66050 : 65602) - 1),
                              w->escbits + (0 .. (rn + 7) / 8 - 1)); */
        /*@ assert \separated(s + pos + (0 .. rn - 1),
                              in + (0 .. lim - in - 1)); */
        /*@ assert \separated(s + pos + (0 .. rn - 1),
                              w->escbits + (0 .. (rn + 7) / 8 - 1)); */
        /*@ assert \separated(w->escbits + (0 .. (rn + 7) / 8 - 1),
                              in + (0 .. lim - in - 1)); */
        dlt_tail(rp, in + lr, escn, mb, w->escbuf, hf, s + pos, rn,
                 w->escbits);
        in += lr;
    }
    if (in != lim) die("corrupt prw");   /* exact payload consumption */
}

static void prw_dec(const uint8_t *in, const uint8_t *lim, uint64_t n, uint64_t cols,
                    int mb, uint16_t *s, uint64_t *hf, uint64_t *hd) {
    DltWs w; dltws_init_dec(&w);
    dltws_need_bits(&w, cols);
    dltws_need_eb(&w, cols);
    prw_dec_ws(in, lim, n, cols, mb, s, hf, hd, &w);
    dltws_free(&w);
}

/* capel: element cap for prefix trials (candidate selection only — a
   prefix-encoded candidate is never emitted; near-ties are re-encoded at
   full length and the winner is always chosen by real full-size bytes).
   ~0 means no cap. */
static uint64_t try_method(int m, Tensor *t, Tensor *all, uint8_t *scr, uint64_t **hout, int *chout, uint32_t ri, uint64_t capel, int nthr) {
    hout[0] = 0; hout[1] = 0; chout[0] = -1; chout[1] = -1;
    int bsz = dtb(t->dtype);
    uint64_t ne = bsz ? t->len / bsz : 0, ne0 = ne;
    if (capel != ~0ull && ne > capel) ne = capel;
    uint64_t nB = t->len;
    if (capel != ~0ull && nB > capel * (uint64_t)bsz) nB = capel * (uint64_t)bsz;
    switch (m) {
    case M_RAW:
        memcpy(scr, t->data, nB);
        return nB;
    case M_PACK:
        if (bsz <= 2 && nB) return pack_enc(t->data, nB, bsz, scr);
        return 0;
    case M_FIELD: {
        if (!is_flt16(t->dtype) || !ne) return 0;
        int c = is_bf(t->dtype) ? CBF : CFP;
        uint64_t *h = hclone(c);
        uint64_t sz = f16_enc((uint16_t *)t->data, ne, mbits_of(t->dtype), scr, h, nthr);
        hout[0] = h; chout[0] = c;
        return sz;
    }
    case M_FIELDPOS: {
        if (!is_flt16(t->dtype) || !ne || t->nd < 1) return 0;
        int c = is_bf(t->dtype) ? CBPOS : CFPOS;
        uint64_t *h = hclone(c);
        uint64_t sz = pos_enc((uint16_t *)t->data, ne, mbits_of(t->dtype), t->shape[t->nd - 1], 0, scr, h);
        hout[0] = h; chout[0] = c;
        return sz;
    }
    case M_FIELDROW: {
        if (!is_flt16(t->dtype) || !ne || t->nd < 2) return 0;
        int c = is_bf(t->dtype) ? CBPOS : CFPOS;
        uint64_t *h = hclone(c);
        /* rowwise prefix: keep the row length, cap the ROW count so the
           position contexts see the same mapping as the full encode */
        int64_t P = t->shape[0];
        if (ne < ne0) { int64_t p2 = (int64_t)((__uint128_t)ne * P / ne0); if (p2 < 1) p2 = 1; P = p2; }
        uint64_t sz = pos_enc((uint16_t *)t->data, ne, mbits_of(t->dtype), P, 1, scr, h);
        hout[0] = h; chout[0] = c;
        return sz;
    }
    case M_F32: {
        if (!is_f32(t->dtype) || !ne) return 0;
        uint64_t *h = hclone(C32);
        uint64_t sz = f32_enc((uint32_t *)t->data, ne, scr, h);
        hout[0] = h; chout[0] = C32;
        return sz;
    }
    case M_U8: {
        if (!nB) return 0;
        uint64_t *h = hclone(C8);
        uint64_t sz = u8_enc(t->data, nB, scr, h);
        hout[0] = h; chout[0] = C8;
        return sz;
    }
    case M_DELTA: {
        if (!ne || ri == 0xFFFFFFFFu) return 0;
        Tensor *r = &all[ri];
        if (r->len != t->len) return 0;   /* precondition: refs are same-size */
        uint8_t *rh = 0;
        const uint8_t *rd = alview(r->data, r->len, bsz, &rh);
        if (is_f32(t->dtype)) {
            uint64_t *h = hclone(C32D);
            uint64_t sz = dlt32_enc((const uint32_t *)t->data, (const uint32_t *)rd, ne, scr, h, nthr);
            free(rh); hout[0] = h; chout[0] = C32D;
            return sz;
        }
        if (!is_flt16(t->dtype)) { free(rh); return 0; }
        {
            int ec = is_bf(t->dtype) ? CBF : CFP;
            uint64_t *h = hclone(CD), *h2 = hclone(ec);
            uint64_t sz = dlt_enc((const uint16_t *)t->data, (const uint16_t *)rd, ne, scr, h, mbits_of(t->dtype), h2, nthr);
            free(rh); hout[0] = h; chout[0] = CD; hout[1] = h2; chout[1] = ec;
            return sz;
        }
    }
    case M_DELTAX: {
        if (!ne || !t->xreft) return 0;
        uint8_t *rh = 0;
        const uint8_t *rd = alview(t->xreft->data, t->xreft->len, bsz, &rh);
        if (is_f32(t->dtype)) {
            uint64_t *h = hclone(C32D);
            uint64_t sz = dlt32_enc((const uint32_t *)t->data, (const uint32_t *)rd, ne, scr, h, nthr);
            free(rh); hout[0] = h; chout[0] = C32D;
            return sz;
        }
        if (!is_flt16(t->dtype)) { free(rh); return 0; }
        {
            int ec = is_bf(t->dtype) ? CBF : CFP;
            uint64_t *h = hclone(CD), *h2 = hclone(ec);
            uint64_t sz = dlt_enc((const uint16_t *)t->data, (const uint16_t *)rd, ne, scr, h, mbits_of(t->dtype), h2, nthr);
            free(rh); hout[0] = h; chout[0] = CD; hout[1] = h2; chout[1] = ec;
            return sz;
        }
    }
    case M_PRW: {
        if (!is_flt16(t->dtype) || !ne || t->nd < 2) return 0;
        uint64_t cols = (uint64_t)t->shape[t->nd - 1];
        /* row>=64 elems keeps the per-row header negligible; row count cap
           keeps per-row stream overhead inside ebound's slack; cols cap
           keeps each row's stream under the u32 length field
           (worst DELTA16 row is ~8*cols bytes) */
        if (cols < 64 || cols > (1u << 28) || ne0 < 2 * cols || ne0 > cols * 65536) return 0;
        if (ne < 2 * cols) return 0;   /* prefix too small to sample a row pair */
        int mb = mbits_of(t->dtype), ec = is_bf(t->dtype) ? CBF : CFP;
        uint64_t *hd = hclone(CD), *hf = hclone(ec);
        uint64_t sz = prw_enc((const uint16_t *)t->data, ne, cols, mb, scr, hf, hd);
        hout[0] = hd; chout[0] = CD; hout[1] = hf; chout[1] = ec;
        return sz;
    }
    }
    return 0;
}

/* cheap pre-check for PRW: sample row-adjacent kmap16 residuals and estimate
   the residual stream's bits/elem (escapes cost a FIELD recode each; in-range
   residuals ~log2|d|). Skip the encode when the estimate can't plausibly beat
   FIELD's ~12b/elem. Candidate pruning only; winner chosen by real size. */
static int prw_gain(const uint16_t *s, uint64_t n, uint64_t cols) {
    uint64_t tot = n - cols, cnt = 0;
    uint64_t st = tot / 8192; if (st < 1) st = 1;
    double est = 0;
    for (uint64_t i = 0; i < tot; i += st) {
        int64_t d = kmap16(s[i + cols]) - kmap16(s[i]);
        uint64_t a = d < 0 ? (uint64_t)(-d) : (uint64_t)d;
        est += (d < -DR || d >= DR) ? 14.0 : 2.0 + (double)(64 - __builtin_clzll(a | 1));
        cnt++;
    }
    return cnt && est / cnt < 9.0;
}

/* floor(log2(n) * 2^20) — pure integer, so candidate pruning is
   deterministic across libm implementations/CPUs (a libm log2 differs in
   the last ulp and could flip a marginal gate → different archive bytes
   for identical input on different machines). Verified exact over the
   entire reachable domain [1, 2^22] and at every 2^k boundary; deep-domain
   values may underestimate by ~1 ulp (bounded, still deterministic). */
static uint64_t flog2(uint64_t n) {
    if (!n) return 0;                /* log2(0): defined as 0 here */
    int k = 63 - __builtin_clzll(n);
    uint64_t x = n << (63 - k);          /* m·2^63 for m = n/2^k ∈ [1,2) */
    uint64_t r = (uint64_t)k << 20;
    for (int i = 19; i >= 0; i--) {
        __uint128_t x2 = (__uint128_t)x * x;   /* m² · 2^126 */
        if (x2 >= ((__uint128_t)1 << 127)) {   /* m ≥ √2 → next bit is 1 */
            r |= 1ull << i;
            x = (uint64_t)(x2 >> 64);          /* (m²/2) · 2^63 */
        } else {
            x = (uint64_t)(x2 >> 63);          /* m² · 2^63 */
        }
    }
    return r;
}

/* cheap pre-check: estimate H(SE) - H(SE|pos) on a sample; used to prune
 * FIELDPOS/FIELDROW candidates so ordinary matrices skip the slow encodes */
/* H0 - H1 = [ns·log2 ns - Σmh·log2 mh - Σtot·log2 tot + Σjh·log2 jh]/ns
   — all in Q20 fixed point via flog2; deterministic everywhere. The
   ns/mh terms are shared between the column and row gates (pos_gain2
   computes both in one pass) — base is passed in precomputed. */
static double pos_ent(const uint32_t *jh, int64_t K, int sew, uint64_t ns, __int128_t base) {
    __int128_t sum = base;
    for (int64_t c = 0; c < K; c++) {
        uint64_t tot = 0;
        for (int i = 0; i < sew; i++) tot += jh[c * sew + i];
        if (!tot) continue;
        sum -= (__int128_t)tot * (__int128_t)flog2(tot);
        for (int i = 0; i < sew; i++)
            if (jh[c * sew + i]) sum += (__int128_t)jh[c * sew + i] * (__int128_t)flog2(jh[c * sew + i]);
    }
    /* IEEE double division is correctly rounded → still deterministic */
    return sum > 0 ? (double)(uint64_t)sum / ((double)ns * 1048576.0) : 0;
}
/* column (po=i%Pc) and rowwise (po=i/Dr) position-gain in ONE sample pass —
   same counts as two separate scans (the marginal mh is shared), so gate
   decisions and archive bytes are unchanged. Prow<=0 disables the row gate
   (nd<2). */
static void pos_gain2(const uint16_t *s, uint64_t n, int mb, int64_t Pc, int64_t Pr,
                      double *gc, double *gr) {
    *gc = *gr = 0;
    if (!n) return;
    if (Pc <= 0 || Pc > ((int64_t)1 << 40)) Pc = 0;   /* invalid → gate off */
    if (Pr <= 0 || Pr > ((int64_t)1 << 40)) Pr = 0;
    if (!Pc && !Pr) return;
    int ew = 1 << (15 - mb), sew = 2 * ew;
    int64_t Kc = Pc ? (Pc < (int64_t)(PCAP / sew) ? Pc : (int64_t)(PCAP / sew)) : 0;
    int64_t Kr = Pr ? (Pr < (int64_t)(PCAP / sew) ? Pr : (int64_t)(PCAP / sew)) : 0;
    int64_t Dr = Pr ? n / Pr : 1; if (Dr < 1) Dr = 1;
    uint64_t ns = n < (4u << 20) ? n : (4u << 20);
    /* u32 counts are safe: every cell counts at most ns <= 4M samples.
       po/ctx are stepped incrementally (K<=P → ctx moves <=1 per po step),
       removing the per-sample divisions; identical counts either way. */
    uint32_t *jc = Pc ? xc((size_t)Kc * sew, 4) : 0;
    uint32_t *jr = Pr ? xc((size_t)Kr * sew, 4) : 0;
    uint32_t *mh = xc(sew, 4);
    int64_t poc = 0, ctxc = 0, remc = 0;
    int64_t por = 0, ctxr = 0, remr = 0;
    uint64_t pob = (uint64_t)Dr;   /* rowwise: next i where por increments */
    for (uint64_t i = 0; i < ns; i++) {
        uint32_t SE = s[i] >> mb;
        mh[SE]++;
        if (jc) {
            jc[ctxc * sew + SE]++;
            if (++poc == Pc) { poc = 0; ctxc = 0; remc = 0; }
            else { remc += Kc; if (remc >= Pc) { remc -= Pc; ctxc++; } }
        }
        if (jr) {
            jr[ctxr * sew + SE]++;
            if (i + 1 == pob) {       /* por = i/Dr steps up at i = por*Dr */
                por++; pob += (uint64_t)Dr;
                remr += Kr; if (remr >= Pr) { remr -= Pr; ctxr++; }
            }
        }
    }
    __int128_t base = (__int128_t)ns * (__int128_t)flog2(ns);
    for (int i = 0; i < sew; i++) if (mh[i]) base -= (__int128_t)mh[i] * (__int128_t)flog2(mh[i]);
    if (jc) *gc = pos_ent(jc, Kc, sew, ns, base);
    if (jr) *gr = pos_ent(jr, Kr, sew, ns, base);
    free(jc); free(jr); free(mh);
}

static Tensor *ref_find(const char *name, const Tensor *t) {
    /* first name+len+dtype match — a same-named incompatible tensor must
       not shadow a compatible one later in the file */
    for (int r = 0; r < g_nref; r++)
        for (int j = 0; j < g_refs[r]->n; j++) {
            Tensor *xr = &g_refs[r]->t[j];
            if (!strcmp(xr->name, name) && xr->len == t->len &&
                !strcmp(xr->dtype, t->dtype)) return xr;
        }
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
#define PREF_EL (1u << 20)                /* prefix-trial length, elements */

typedef struct {
    int m; Tensor *t; Tensor *all; uint32_t ri;
    uint64_t **ghs, **snp;               /* owner channel state (read-only here) */
    uint64_t sz; uint8_t *buf;
    uint64_t capel;                       /* element cap: 0/~0 = full encode */
    uint64_t *ho[2]; int ch[2];
    int nthr;                             /* inner block-parallelism budget */
    int spawned;                          /* pthread_create succeeded -> join required */
} TJob;
/* worst-case payload for method m on a len-byte tensor. Every enc() call
   emits <=2B (x<2^39 vs XMAX>=2^24). Per-element call counts: FIELD 3
   (=>3*len), POS 2, DELTA16 1 call + escapes recoded through f16_enc
   (<=3*2B/elem => sym len + esc 3*len = 4*len), DELTA32 4 planes
   (=>2*len), F32 5 (=>2.5*len), U8 1 (=>2*len), PACK <=len+dict. Plus
   <=12B per emitted block and small fixed tails (esc_n, dict).
   4*len + len/1024 covers every method incl. DELTA's escape channel. */
/*@ terminates len <= 18446744073709551615ULL / 5;
    assigns \nothing;
    exits \exit_status == 1;
    ensures \result == 4 * len + len / 1024 + 67108864;
*/
static uint64_t ebound(uint64_t len) {
    /* 4*len + len/1024 + 64MB must not wrap u64 — len > 2^64/5 keeps the
       total comfortably below 2^64 (unreachable on real files, off_t-bound) */
    if (len > 3689348814741910323ULL) die("tensor too large");
    return 4 * len + len / 1024 + 67108864u;
}
static void *tjob_run(void *a) {
    TJob *j = a;
    uint64_t **oh = gh, **os = gsnp;   /* restore on inline fallback so the
                                        caller's channel pointers survive */
    gh = j->ghs; gsnp = j->snp;
    int bsz = dtb(j->t->dtype);
    uint64_t bl = j->t->len;
    if (j->capel != ~0ull && j->capel < bl / (bsz ? bsz : 1)) bl = j->capel * (bsz ? bsz : 1);
    j->buf = xm(ebound(bl));
    j->sz = try_method(j->m, j->t, j->all, j->buf, j->ho, j->ch, j->ri, j->capel, j->nthr);
    gh = oh; gsnp = os;
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
        {   /* one sample pass produces both position gates */
            double gc = 0, gr = 0;
            pos_gain2((uint16_t *)t->data, ne, mb,
                      t->nd >= 1 ? t->shape[t->nd - 1] : 0,
                      t->nd >= 2 ? t->shape[0] : 0, &gc, &gr);
            if (gc > 0.06) cand[nc++] = M_FIELDPOS;
            if (t->nd >= 2 && gr > 0.06) cand[nc++] = M_FIELDROW;
        }
        if (t->nd >= 2) {
            uint64_t cl = (uint64_t)t->shape[t->nd - 1];
            if (cl >= 64 && cl <= (1u << 28) && ne >= 2 * cl && ne <= cl * 65536 &&
                prw_gain((uint16_t *)t->data, ne, cl)) cand[nc++] = M_PRW;
        }
    }
    else if (is_f32(t->dtype)) { cand[nc++] = M_F32; cand[nc++] = M_U8; }
    else cand[nc++] = M_U8;
    cand[nc++] = M_PACK;
    for (int k = 0; k < t->ncand; k++)
        if (t->candref[k] < refcut) { cref[nc] = t->candref[k]; cand[nc++] = M_DELTA; }
    if (t->xreft) { cref[nc] = 0xFFFFFFFFu; cand[nc++] = M_DELTAX; }

    TJob *jb = xc(nc, sizeof(TJob));
    pthread_t *th = xc(nc, sizeof(pthread_t));

    /* prefix pre-filter (big tensors only): encode the first cap elements
       of every candidate, project to full size, and drop candidates
       projected >1/6 + 256KB behind the best. Candidates within the margin
       are re-encoded at full size below — the winner is still chosen by
       real full-size bytes; a prefix-failing candidate is kept. */
    {
        int bsz = dtb(t->dtype);
        uint64_t ne0 = bsz ? t->len / bsz : t->len;
        uint64_t cap = ne0 / 8;                    /* ~12.5% of the tensor */
        if (cap > PREF_EL) cap = PREF_EL;          /* cap absolute work */
        if (nc > 1 && cap >= (1u << 16)) {         /* engage at >=512K elems */
            uint64_t nBk = cap * (bsz ? bsz : 1);
            if (nBk > t->len) nBk = t->len;
            int psp = ntry < nc ? ntry : nc;
            uint64_t pper = ebound(nBk);
            if (psp > 1 && pper > TRYBUD / (uint64_t)psp) psp = (int)(TRYBUD / pper);
            if (psp < 1) psp = 1;
            uint64_t *psz = xc(nc, 8);
            for (int k = 0; k < nc; k += psp) {
                int e = k + psp < nc ? k + psp : nc;
                for (int k2 = k; k2 < e; k2++) {
                    TJob *j = &jb[k2];
                    j->m = cand[k2]; j->t = t; j->all = all; j->ri = cref[k2];
                    j->capel = cap; j->spawned = 0;
                    j->nthr = ntry / psp;   /* wave runs psp jobs; split budget */
                    j->ghs = gh; j->snp = gsnp;
                    if (psp == 1) tjob_run(j);
                    else if (pthread_create(&th[k2], 0, tjob_run, j)) tjob_run(j);
                    else j->spawned = 1;
                }
                for (int k2 = k; k2 < e; k2++) if (jb[k2].spawned) pthread_join(th[k2], 0);
                for (int k2 = k; k2 < e; k2++) {
                    TJob *j = &jb[k2];
                    psz[k2] = j->sz;
                    free(j->buf); free(j->ho[0]); free(j->ho[1]);
                    j->ho[0] = j->ho[1] = 0;
                }
            }
            __uint128_t bp = ~(__uint128_t)0;
            for (int k = 0; k < nc; k++)
                if (psz[k]) { __uint128_t p = (__uint128_t)psz[k] * t->len / nBk; if (p < bp) bp = p; }
            int w = 0;
            for (int k = 0; k < nc; k++) {
                if (psz[k]) {
                    __uint128_t p = (__uint128_t)psz[k] * t->len / nBk;
                    if (p > bp + bp / 6 + 262144) continue;   /* clearly beaten */
                }
                cand[w] = cand[k]; cref[w] = cref[k]; w++;
            }
            nc = w;   /* argmin always survives its own test */
            free(psz);
        }
    }

    int nsp = ntry < nc ? ntry : nc;
    uint64_t per = ebound(t->len);
    /* division form — per*nsp itself could wrap u64 on absurd lens */
    if (nsp > 1 && per > TRYBUD / (uint64_t)nsp) nsp = (int)(TRYBUD / per);
    if (nsp < 1) nsp = 1;
    /* rolling best: keep only the current winner's buffer+hist clones so peak
       memory is ~(nsp+1) x ebound, not nc x ebound */
    uint64_t best = t->len; int wm = -1; uint32_t wri = 0;
    uint8_t *wbuf = 0; uint64_t *who[2] = {0, 0}; int wch[2] = {-1, -1};
    for (int k = 0; k < nc; k += nsp) {
        int e = k + nsp < nc ? k + nsp : nc;
        for (int k2 = k; k2 < e; k2++) {
            TJob *j = &jb[k2];
            j->m = cand[k2]; j->t = t; j->all = all; j->ri = cref[k2];
            j->capel = ~0ull;   /* full-size competition */
            j->spawned = 0;
            j->nthr = ntry / nsp;   /* a lone survivor gets the whole budget */
            j->ghs = gh; j->snp = gsnp;
            if (nsp == 1) tjob_run(j);
            else if (pthread_create(&th[k2], 0, tjob_run, j)) tjob_run(j);
            else j->spawned = 1;
        }
        for (int k2 = k; k2 < e; k2++) if (jb[k2].spawned) pthread_join(th[k2], 0);
        for (int k2 = k; k2 < e; k2++) {
            TJob *j = &jb[k2];
            if (j->sz && j->sz < best) {   /* promote: free old best, keep this */
                free(wbuf); free(who[0]); free(who[1]);
                best = j->sz; wm = j->m; wri = j->ri; wbuf = j->buf;
                who[0] = j->ho[0]; who[1] = j->ho[1];
                wch[0] = j->ch[0]; wch[1] = j->ch[1];
            } else {
                free(j->buf);
                for (int s = 0; s < 2; s++) free(j->ho[s]);
            }
        }
    }
    if (wm >= 0) {
        t->method = wm; t->plen = best;
        t->ref = (wm == M_DELTA) ? wri : 0xFFFFFFFFu;
        *outp = wbuf;
        for (int s = 0; s < 2; s++) if (who[s]) hcommit(wch[s], who[s]);
    }
    free(jb); free(th);
    /* RAW/PACK write no rANS streams — feed the natural channel anyway so
       short tensors still train the model (decode replays the same count) */
    if (t->method == M_RAW || t->method == M_PACK) teach(t, t->data);
    t->data = aorig; free(ahold);
}

/* a tensor may join the batch starting at bstart only if every possible
   intra-archive ref (exact-dup target and all DELTA candidates) lies in
   already-committed ranges — in-batch refs can't be resolved in parallel */
/*@ requires \valid_read(t);
    requires 0 <= t->ncand <= 4;
    assigns \nothing;
    ensures \result == 0 || \result == 1;
*/
static int joinable(const Tensor *t, uint32_t bstart) {
    if (t->ref != 0xFFFFFFFFu && t->ref >= bstart) return 0;
    /*@ loop invariant 0 <= k <= t->ncand;
        loop assigns k;
        loop variant t->ncand - k;
    */
    for (int k = 0; k < t->ncand; k++) if (t->candref[k] >= bstart) return 0;
    return 1;
}

/* batch worker: encode one tensor with hist state lazily cloned from the
   batch snapshot; its hist delta is merged into gch by the main thread. */
typedef struct {
    Tensor *t; Tensor *all; uint32_t refcut;
    uint64_t **snp;
    uint64_t *ch[NCH];
    uint8_t *out;
    int ntry;                             /* candidate-trial thread budget */
    int spawned;
} EJob;
static void *ejob_run(void *a) {
    EJob *j = a;
    uint64_t **oh = gh, **os = gsnp;
    gh = j->ch; gsnp = j->snp;
    compete(j->t, j->all, j->refcut, &j->out, j->ntry);
    j->t->crc = crc32b(j->t->data, j->t->len);   /* overlap the record CRC with
                                                  sibling workers; emit_rec
                                                  reads t->crc for bat>0 */
    gh = oh; gsnp = os;
    return 0;
}
/* serial emit of a joined+merged batch — record order is tensor order.
   Deferred one batch so it overlaps the NEXT batch's encode. */
static void emit_batch(FILE *of, EJob *ej, uint32_t nm, InFile **ins,
                       uint64_t *tin, uint64_t *tout) {
    for (uint32_t k = 0; k < nm; k++) {
        Tensor *t = ej[k].t;
        emit_rec(of, t, k ? 2 : 1);
        if (t->plen) fwrite(ej[k].out ? ej[k].out : t->data, 1, t->plen, of);
        free(ej[k].out);
        *tin += t->len; *tout += t->plen;
        drop_pages(t->data, t->len, ins[t->file]->mapped);
    }
}

/* merge worker hist deltas: gch += (worker - snap), channel-wise.
   counts commute, so member order is irrelevant.  The batch-start snapshot
   is materialized lazily per channel at its FIRST merge: gch is never
   written while a batch is in flight, so gch[c] still holds batch-start
   state at that point.  Workers therefore clone directly from gch and the
   unconditional ~44MB hsnap copy disappears — only channels a member
   actually committed get snapshotted at all. */
static void hmerge(uint64_t *const wch[NCH], uint64_t *snap[NCH]) {
    for (int c = 0; c < NCH; c++) if (wch[c]) {
        uint64_t *w = wch[c], *g = gch[c];
        if (!snap[c]) { snap[c] = xm(csz[c] * 8); memcpy(snap[c], g, csz[c] * 8); }
        uint64_t *s = snap[c];
        for (int i = 0; i < csz[c]; i++) g[i] += w[i] - s[i];
        free(w);
    }
}

/* ================= decode ================= */
/* Per-in-flight decode scratch the tensor buffer itself doesn't cover:
   DELTA16: escape values <= len (escn <= n u16's) + bitmap <= len/16 +
   possible misaligned-ref copy <= len + ~3.3MB decode workspace.
   DELTA32: plane+table scratch ~0.65MB + ref copy <= len.
   DELTAX: same + aligned external-ref copy <= len (ref_resolve enforces
   equal len).  PRW: escbuf = cols*2 <= len/2 (rows >= 2) + escbits <=
   len/32 + workspace.  FIELDPOS/FIELDROW: K-context ft/cum <= ~14MB.
   Charged alongside t->len so g_dlive reflects real peak RSS, not just
   outputs. */
/*@ requires \valid_read(t);
    requires valid_read_string(t->dtype);
    requires t->len <= 3074457345617559551ULL;
    assigns \nothing;
*/
static uint64_t dec_aux(const Tensor *t) {
    int bsz = dtb(t->dtype);
    switch (t->method) {
    case M_DELTA:  return (bsz == 2 ? 6 : 1) * t->len + 4194304u;
    case M_DELTAX: return (bsz == 2 ? 6 : 2) * t->len + 4194304u;
    case M_PRW:    return t->len / 2 + t->len / 16 + 4194304u;
    case M_FIELDPOS: case M_FIELDROW: return 35651584ull;
    case M_F32:    return 4194304ull;
    default:       return 1048576u;
    }
}

static void dec_tensor(Tensor *t, const uint8_t *payload, Tensor *all) {
    uint64_t ch = t->len + dec_aux(t);
    t->dlivc = ch;
    if (__sync_add_and_fetch(&g_dlive, ch) > g_dlim)
        die("decoded set exceeds available memory");
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
        if (t->plen) die("corrupt ref");
        if (all[t->ref].len != t->len) die("bad ref");
        memcpy(t->data, all[t->ref].data, t->len); break;
    case M_U8:
        /* encoder only omits U8 for flt16 — reject that non-canonical pair */
        if (is_flt16(t->dtype)) die("bad u8 dtype");
        u8_dec(payload, lim, t->len, t->data, H(C8)); break;
    case M_FIELD:
        if (!is_flt16(t->dtype)) die("bad field dtype");
        f16_dec(payload, lim, ne, mbits_of(t->dtype), (uint16_t *)t->data, H(is_bf(t->dtype) ? CBF : CFP)); break;
    case M_FIELDPOS:
        if (!is_flt16(t->dtype) || t->nd < 1) die("bad fieldpos");
        pos_dec(payload, lim, ne, mbits_of(t->dtype), t->shape[t->nd-1], 0, (uint16_t *)t->data, H(is_bf(t->dtype) ? CBPOS : CFPOS)); break;
    case M_FIELDROW:
        if (!is_flt16(t->dtype) || t->nd < 2) die("bad fieldrow");
        pos_dec(payload, lim, ne, mbits_of(t->dtype), t->shape[0], 1, (uint16_t *)t->data, H(is_bf(t->dtype) ? CBPOS : CFPOS)); break;
    case M_F32:
        if (!is_f32(t->dtype)) die("bad f32 dtype");
        f32_dec(payload, lim, ne, (uint32_t *)t->data, H(C32)); break;
    case M_DELTA: {
        if (all[t->ref].len != t->len) die("bad ref");
        if (!is_flt16(t->dtype) && !is_f32(t->dtype)) die("bad delta dtype");
        /* the referent may be a zero-copy RAW pointer into the archive
           mmap — not necessarily bsz-aligned; copy if misaligned */
        uint8_t *rh = 0;
        const uint8_t *rd = alview(all[t->ref].data, t->len, bsz, &rh);
        if (is_f32(t->dtype))
            dlt32_dec(payload, lim, (const uint32_t *)rd, ne, (uint32_t *)t->data, H(C32D));
        else
            dlt_dec(payload, lim, (const uint16_t *)rd, ne, (uint16_t *)t->data, H(CD), mbits_of(t->dtype), H(is_bf(t->dtype) ? CBF : CFP));
        free(rh);
        break;
    }
    case M_DELTAX: {
        if (!is_flt16(t->dtype) && !is_f32(t->dtype)) die("bad delta dtype");
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
    case M_PRW: {
        if (!is_flt16(t->dtype) || t->nd < 2) die("bad prw dtype");
        uint64_t cols = (uint64_t)t->shape[t->nd - 1];
        if (cols < 64 || cols > (1u << 28) || ne < 2 * cols || ne > cols * 65536) die("bad prw shape");
        prw_dec(payload, lim, ne, cols, mbits_of(t->dtype), (uint16_t *)t->data,
                H(is_bf(t->dtype) ? CBF : CFP), H(CD));
        break;
    }
    }
    if (t->method == M_RAW || t->method == M_PACK) teach(t, t->data);
}

typedef struct {
    Tensor *t; const uint8_t *pl; Tensor *all;
    uint64_t **snp;
    uint64_t *ch[NCH];
    int spawned;
    int keep;                       /* tensor must stay resident for REF/DELTA */
} DJob;
static void *djob_run(void *a) {
    DJob *j = a;
    uint64_t **oh = gh, **os = gsnp;
    gh = j->ch; gsnp = j->snp;
    if (j->t->method == M_RAW && !j->keep) {
        /* zero-copy: payload is the tensor; t->data stays NULL */
        if (j->t->plen != j->t->len) die("corrupt raw");
        if (crc32b(j->pl, j->t->len) != j->t->crc)
            die("crc mismatch: archive corrupt");
        teach(j->t, j->pl);
    } else {
        dec_tensor(j->t, j->pl, j->all);
        if (crc32b(j->t->data, j->t->len) != j->t->crc)
            die("crc mismatch: archive corrupt");
    }
    gh = oh; gsnp = os;
    return 0;
}
/* decode a flagged batch run all[i..j): members decode against the
   batch-start snapshot in parallel; hist deltas merge order-free. */
static void dec_batch(Tensor *all, uint32_t i, uint32_t j, const uint8_t *buf,
                      const uint8_t *keep) {
    for (uint32_t k = i; k < j; k++)
        if ((all[k].method == M_REF || all[k].method == M_DELTA) && all[k].ref >= i)
            die("batch ref into open batch");
    uint64_t *snap[NCH] = {0};   /* lazy: frozen per channel at first merge */
    for (uint32_t k = i; k < j; k += (uint32_t)g_threads) {
        uint32_t e = k + (uint32_t)g_threads < j ? k + (uint32_t)g_threads : j;
        uint32_t cnt = e - k;
        /* batch-start view: gch[c] still holds batch-start state unless a
           previous chunk merged channel c — then snap[c] (frozen at that
           first merge) is the correct batch-start view instead */
        uint64_t *view[NCH];
        for (int c = 0; c < NCH; c++) view[c] = snap[c] ? snap[c] : gch[c];
        DJob *dj = xc(cnt, sizeof(DJob));
        pthread_t *th = xc(cnt, sizeof(pthread_t));
        for (uint32_t m = 0; m < cnt; m++) {
            dj[m].t = &all[k + m]; dj[m].pl = buf + all[k + m].off;
            dj[m].all = all; dj[m].snp = view; dj[m].keep = keep[k + m];
            if (pthread_create(&th[m], 0, djob_run, &dj[m])) djob_run(&dj[m]);
            else dj[m].spawned = 1;
        }
        for (uint32_t m = 0; m < cnt; m++) if (dj[m].spawned) pthread_join(th[m], 0);
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
            memmove(&argv[i], &argv[i + 2], (argc - i - 1) * sizeof(char *));  /* keep trailing NULL */
            argc -= 2; i--;
        } else if (!strncmp(argv[i], "-j", 2) &&
                   (!argv[i][2] || (argv[i][2] >= '0' && argv[i][2] <= '9'))) {
            const char *v = argv[i] + 2; int eat = 1;
            if (!*v) { if (i + 1 >= argc) die("-j needs a count"); v = argv[i + 1]; eat = 2; }
            for (const char *d = v; *d; d++) if (*d < '0' || *d > '9') die("-j needs digits");
            unsigned nj = 0;   /* atoi on an over-long digit string is UB —
                                  accumulate with an early cap instead */
            for (const char *d = v; *d; d++) {
                nj = nj * 10 + (unsigned)(*d - '0');
                if (nj > 64) { nj = 64; break; }
            }
            g_threads = nj < 1 ? 1 : (int)nj;
            memmove(&argv[i], &argv[i + eat], (argc - i - eat + 1) * sizeof(char *));
            argc -= eat; i--;
        }
    gh = gch;          /* main thread uses the committed state directly */
    crc_setup();
    if (!strcmp(argv[1], "c")) {
        if (argc < 4) die("caiw c out.caiw in.st...");
        /* write to <out>.caiwtmp and rename on success — a crash mid-encode
           must not leave a complete-looking truncated archive */
        char ctmp[8192];
        int cw = snprintf(ctmp, sizeof ctmp, "%s.caiwtmp", argv[2]);
        if (cw < 0 || (size_t)cw >= sizeof ctmp) die("output path too long");
        int nf = argc - 3;
        if (nf > 65535) die("too many inputs");   /* file_idx field is u16 */
        /* path-identity checks BEFORE opening ctmp: fopen would truncate a
           same-named input (c foo foo.caiwtmp), and a rename could later
           clobber inputs/refs via aliases (links, "dir/../x", symlinked
           parents). Both the final name and the temp name are checked. */
        for (int i = 0; i < nf; i++) {
            if (same_out_path(argv[2], argv[3 + i])) die("input == output path");
            if (same_out_path(ctmp, argv[3 + i])) die("input == output tmp path");
        }
        for (int r = 0; r < g_nref; r++) {
            if (same_out_path(argv[2], g_refs[r]->path)) die("ref == output path");
            if (same_out_path(ctmp, g_refs[r]->path)) die("ref == output tmp path");
        }
        /* g_outn stays 0 until the tmp is actually ours — a die on a
           pre-existing foreign/unwritable tmp must not unlink it */
        g_outp = xc(1, sizeof(char *)); g_outp[0] = xstrdup(ctmp);
        g_outn = 0; g_outok = 0; atexit(out_cleanup);
        FILE *of = xfopen_tmp(ctmp);
        if (!of) die("out");
        g_outn = 1;
        InFile **ins = xc(nf, sizeof(InFile *));
        int64_t NT64 = 0;
        for (int i = 0; i < nf; i++) {
            ins[i] = st_load(argv[3 + i], i); NT64 += ins[i]->n;
        }
        /* decoder caps NT at 2^29 (dseen pow2 table ≤ 2^30) — enforce the
           same bound here so encode can't emit an archive it can't read */
        if (NT64 > (1ll << 29)) die("too many tensors");
        int NT = (int)NT64;
        Tensor *all = xc(NT, sizeof(Tensor));
        int ti = 0;
        for (int i = 0; i < nf; i++)
            for (int j = 0; j < ins[i]->n; j++) all[ti++] = ins[i]->t[j];
        /* dedup + delta ref resolution (earlier tensors only).
           hash chains keep newest-first index lists per key, reproducing the
           original "latest match wins" scan semantics in near-linear time:
           dup key = (len,thash), delta key = name, family key = nnorm */
        uint64_t *thash = xc(NT, 8);
        for (int i = 0; i < NT; i++) thash[i] = fnv(all[i].data, all[i].len);
        /* normalized names for family matching (experts.N.*, layers.N.*) */
        char **nn = xc(NT, sizeof(char *));
        for (int i = 0; i < NT; i++) nn[i] = nnorm(all[i].name);
        uint64_t mc64 = 1024; while (mc64 < (uint64_t)NT * 2) mc64 *= 2;
        if (mc64 > (1ull << 31)) die("too many tensors");
        uint32_t mmask = (uint32_t)mc64 - 1;
        uint32_t *dtab = xm((size_t)mc64 * 4), *ntab = xm((size_t)mc64 * 4), *ftab = xm((size_t)mc64 * 4);
        memset(dtab, 0xFF, (size_t)mc64 * 4); memset(ntab, 0xFF, (size_t)mc64 * 4); memset(ftab, 0xFF, (size_t)mc64 * 4);
        uint32_t *dnext = xc(NT ? NT : 1, 4), *nnext = xc(NT ? NT : 1, 4), *fnext = xc(NT ? NT : 1, 4);
        for (int i = 0; i < NT; i++) {
            all[i].ref = 0xFFFFFFFFu;
            all[i].xreft = NULL; all[i].xfile = 0;
            if (g_nref) {
                int ri = 0;
                Tensor *xr = g_nref > 1 ? ref_find_best(&all[i], &ri) : ref_find(all[i].name, &all[i]);
                if (xr && xr->len == all[i].len && !strcmp(xr->dtype, all[i].dtype)) {
                    all[i].xreft = xr; all[i].xfile = ri;
                }
            }
            int dup = -1, dref = -1, dref1 = -1, fref = -1;
            uint64_t dh = thash[i] ^ all[i].len * 0x9E3779B97F4A7C15ull;
            for (uint32_t j = dtab[dh & mmask]; j != 0xFFFFFFFFu; j = dnext[j])
                if (all[j].len == all[i].len && thash[j] == thash[i] &&
                    !memcmp(all[j].data, all[i].data, all[i].len)) { dup = (int)j; break; }
            for (uint32_t j = ntab[nfh(all[i].name, 0) & mmask]; j != 0xFFFFFFFFu && dref1 < 0; j = nnext[j])
                if (!strcmp(all[j].name, all[i].name) && all[j].nd == all[i].nd &&
                    all[j].len == all[i].len && !strcmp(all[j].dtype, all[i].dtype)) {
                    if (dref < 0) dref = (int)j; else dref1 = (int)j;
                }
            if (is_flt16(all[i].dtype) || is_f32(all[i].dtype))
                for (uint32_t j = ftab[nfh(nn[i], 0) & mmask]; j != 0xFFFFFFFFu; j = fnext[j])
                    if (strcmp(all[j].name, all[i].name) && !strcmp(nn[j], nn[i]) &&
                        all[j].nd == all[i].nd && all[j].len == all[i].len &&
                        !strcmp(all[j].dtype, all[i].dtype)) { fref = (int)j; break; }
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
            dnext[i] = dtab[dh & mmask]; dtab[dh & mmask] = (uint32_t)i;
            uint32_t b = (uint32_t)nfh(all[i].name, 0) & mmask;
            nnext[i] = ntab[b]; ntab[b] = (uint32_t)i;
            b = (uint32_t)nfh(nn[i], 0) & mmask;
            fnext[i] = ftab[b]; ftab[b] = (uint32_t)i;
        }
        free(dtab); free(ntab); free(ftab); free(dnext); free(nnext); free(fnext);
        for (int i = 0; i < NT; i++) free(nn[i]);
        free(nn);
        model_init();
        /* header */
        fwrite("CAI5", 4, 1, of);
        w32(of, nf);
        /* basename set: dup names and <name>.caiwtmp collisions become
           membership tests — a pairwise scan is O(nf^2) on huge file lists */
        uint32_t bcap = 64;
        while (bcap < 2 * (uint32_t)nf + 1) bcap <<= 1;
        char **bset = xc(bcap, sizeof(char *));
        for (int i = 0; i < nf; i++) {
            const char *bn = bname(ins[i]->path);
            /* keep enc-side names inside what `d` accepts: no escapes,
               no dot-specials, no backslash, <=3000 bytes */
            if (!*bn || strlen(bn) > 3000 || strchr(bn, '\\') ||
                !strcmp(bn, ".") || !strcmp(bn, ".."))
                die("bad input basename");
            if (nf_find(bset, bcap - 1, bn, 0)) die("duplicate input basename");
            /* decoder writes <name>.caiwtmp then renames — a pair like
               (a, a.caiwtmp) would clobber; refuse to emit it */
            char tmpn[3016];
            size_t tl = strlen(bn);
            memcpy(tmpn, bn, tl); memcpy(tmpn + tl, ".caiwtmp", 9);
            if (nf_find(bset, bcap - 1, tmpn, 0))
                die("basename collides with temp name");
            if (tl > 8 && !memcmp(bn + tl - 8, ".caiwtmp", 8)) {
                /* bn itself is "<pfx>.caiwtmp" — collides if pfx is a member */
                memcpy(tmpn, bn, tl - 8); tmpn[tl - 8] = 0;
                if (nf_find(bset, bcap - 1, tmpn, 0))
                    die("basename collides with temp name");
            }
            nf_put(bset, bcap - 1, bn, 0);
            w32(of, strlen(bn)); fwrite(bn, 1, strlen(bn), of);
            uint32_t ml = ins[i]->meta ? (uint32_t)strlen(ins[i]->meta) : 0;
            w32(of, ml);
            if (ml) fwrite(ins[i]->meta, 1, ml, of);   /* verbatim __metadata__ object */
        }
        for (uint32_t i = 0; i < bcap; i++) free(bset[i]);
        free(bset);
        w32(of, NT);
        uint64_t tin = 0, tout = 0;
        uint32_t i = 0;
        EJob *pej = 0; uint32_t pnm = 0;   /* pending emit (joined+merged) */
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
                uint64_t *snap[NCH] = {0};   /* lazy: frozen at first merge */
                EJob *ej = xc(nm, sizeof(EJob));
                pthread_t *th = xc(nm, sizeof(pthread_t));
                for (uint32_t k = 0; k < nm; k++) {
                    ej[k].t = &all[i + k]; ej[k].all = all;
                    ej[k].refcut = i; ej[k].snp = gch;
                    /* split the machine's thread budget across the batch:
                       when nm < g_threads each worker's candidate trials
                       (and lone-survivor block encodes) parallelize into
                       the spare slots. Sum of budgets <= g_threads. */
                    ej[k].ntry = g_threads / (int)nm
                               + ((int)k < g_threads % (int)nm ? 1 : 0);
                    if (pthread_create(&th[k], 0, ejob_run, &ej[k])) ejob_run(&ej[k]);
                    else ej[k].spawned = 1;
                }
                /* the pending batch's serial emit overlaps this batch's
                   encode; then join+merge so gch is post-batch before the
                   next batch is spawned (workers clone gch directly) */
                if (pej) { emit_batch(of, pej, pnm, ins, &tin, &tout); free(pej); }
                for (uint32_t k = 0; k < nm; k++) if (ej[k].spawned) pthread_join(th[k], 0);
                for (uint32_t k = 0; k < nm; k++) hmerge(ej[k].ch, snap);
                for (int c = 0; c < NCH; c++) free(snap[c]);
                free(th);
                pej = ej; pnm = nm;          /* emit deferred to next batch */
                i += nm;
                continue;
            }
            if (pej) { emit_batch(of, pej, pnm, ins, &tin, &tout); free(pej); pej = 0; }
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
        if (pej) { emit_batch(of, pej, pnm, ins, &tin, &tout); free(pej); }
        if (ferror(of) || fclose(of)) die("write failed (disk full?)");
        if (rename(ctmp, argv[2])) die("rename failed");
        g_outok = 1;
        static const char *mn[] = {"RAW","PACK","REF","FIELD","FIELDPOS","DELTA","U8","F32","FIELDROW","DELTAX","PRW"};
        uint64_t mc[11] = {0}, mb2[11] = {0};
        for (int ti = 0; ti < NT; ti++) { mc[all[ti].method]++; mb2[all[ti].method] += all[ti].plen; }
        fprintf(stderr, "in=%llu out=%llu ratio=%.3f\n",
                (unsigned long long)tin, (unsigned long long)tout,
                tin ? (double)tout / tin : 0);
        for (int m = 0; m < 11; m++) if (mc[m])
            fprintf(stderr, "  %-8s n=%-5llu payload=%.1fMB\n", mn[m],
                    (unsigned long long)mc[m], (double)mb2[m] / 1048576);
        return 0;
    }
    if (!strcmp(argv[1], "d")) {
        if (argc < 4) die("caiw d out.caiw outdir/");
        mkpath(argv[3]);
        dlim_init();
        uint64_t fsz; uint8_t *buf = slurp(argv[2], &fsz);
        const uint8_t *p = buf, *bend = buf + fsz;
#define BND(q, need) do { if ((q) > bend || (uint64_t)(need) > (uint64_t)(bend - (q))) die("truncated archive"); } while (0)
        BND(p, 4);
        int cver;
        if (!memcmp(p, "CAI5", 4)) cver = 5;
        else if (!memcmp(p, "CAI4", 4)) cver = 4;
        else if (!memcmp(p, "CAI3", 4)) cver = 3;
        else die("bad magic");
        p += 4;
        BND(p, 4); uint32_t nf = r32(&p);
        if (nf > 65535) die("too many files");
        char **fnames = xc(nf, sizeof(char *)), **fmeta = xc(nf, sizeof(char *));
        uint64_t fcap = 64; while (fcap < (uint64_t)nf * 2) fcap *= 2;
        char **fset = xc(fcap, sizeof(char *));   /* name set: dup + tmp-collision */
        for (uint32_t i = 0; i < nf; i++) {
            BND(p, 4); uint32_t l = r32(&p); BND(p, l);
            fnames[i] = xm(l + 1); memcpy(fnames[i], p, l); fnames[i][l] = 0; p += l;
            /* output filename must not escape outdir or overflow the path buf */
            if (!l || l > 3000 || memchr(fnames[i], 0, l) ||
                strchr(fnames[i], '/') || strchr(fnames[i], '\\') ||
                !strcmp(fnames[i], ".") || !strcmp(fnames[i], ".."))
                die("bad filename");
            if (nf_find(fset, fcap - 1, fnames[i], 0)) die("dup filename");
            nf_put(fset, fcap - 1, fnames[i], 0);
            if (cver >= 5) {   /* CAI5: verbatim __metadata__ per file */
                BND(p, 4); uint32_t ml = r32(&p); BND(p, ml);
                if (ml) { fmeta[i] = xm(ml + 1); memcpy(fmeta[i], p, ml); fmeta[i][ml] = 0; ck_meta((const uint8_t *)fmeta[i], ml); }
                p += ml;
            }
        }
        /* tmp output is <name>.caiwtmp in the same dir — a member name equal
           to another member's temp name would make the final rename clobber
           the wrong file; reject the collision up front */
        {
            char tb[4096];
            for (uint32_t i = 0; i < nf; i++) {
                snprintf(tb, sizeof tb, "%s.caiwtmp", fnames[i]);
                if (nf_find(fset, fcap - 1, tb, 0)) die("filename collides with temp name");
                /* a member whose output path IS the archive would clobber
                   its own input mid-decode; likewise a member landing on a
                   --ref source silently destroys a file the decoder is
                   still reading from */
                int w = snprintf(tb, sizeof tb, "%s/%s", argv[3], fnames[i]);
                /* truncation skips the check, but is unreachable in practice:
                   the longer "%s/%s.caiwtmp" path dies at xfopen_tmp first */
                if (w > 0 && (size_t)w < sizeof tb) {
                    if (same_out_path(tb, argv[2])) die("output overwrites archive");
                    for (int r = 0; r < g_nref; r++)
                        if (same_out_path(tb, g_refs[r]->path))
                            die("output overwrites --ref input");
                }
                /* the .caiwtmp path itself must also be checked: the archive
                   could BE <outdir>/<name>.caiwtmp — O_TRUNC would destroy
                   the mmap'd input mid-decode (SIGBUS, not a clean die) */
                w = snprintf(tb, sizeof tb, "%s/%s.caiwtmp", argv[3], fnames[i]);
                if (w > 0 && (size_t)w < sizeof tb) {
                    if (same_out_path(tb, argv[2])) die("output overwrites archive");
                    for (int r = 0; r < g_nref; r++)
                        if (same_out_path(tb, g_refs[r]->path))
                            die("output overwrites --ref input");
                }
            }
        }
        BND(p, 4); uint32_t NT = r32(&p);
        BND(p, (uint64_t)NT * 28);  /* minimum record size */
        uint64_t dc64 = 64; while (dc64 < (uint64_t)NT * 2) dc64 *= 2;
        if (dc64 > (1ull << 30)) die("too many tensors");
        uint32_t dcap = (uint32_t)dc64;   /* open-addressed set, pow2, load<0.5 */
        /* the record table amplifies the archive ~22x (≈640B/Tensor +
           names + the dup-name set) — count it against the residency cap
           before allocating, else a small archive can calloc its way into
           OOM territory */
        uint64_t meta = (uint64_t)NT * (sizeof(Tensor) + 512) + (uint64_t)dcap * 16;
        if (__sync_add_and_fetch(&g_dlive, meta) > g_dlim)
            die("archive metadata exceeds memory");
        Tensor *all = xc(NT, sizeof(Tensor));
        char **dseen = xc(dcap, sizeof(char *));
        /* pass 1: metadata walk — no decode */
        const uint8_t *q = p;
        uint32_t bstart = 0;   /* head index of currently open batch */
        for (uint32_t i = 0; i < NT; i++) {
            Tensor *t = &all[i];
            uint16_t nl = 0, dl = 0;
            BND(q, 2); nl = r16(&q);
            BND(q, nl); t->name = xm(nl + 1); memcpy(t->name, q, nl); t->name[nl] = 0; q += nl;
            if (memchr(t->name, 0, nl) || !utf8_ok(t->name, nl) ||
                !strcmp(t->name, "__metadata__")) die("bad name");
            BND(q, 2); dl = r16(&q);
            BND(q, dl); t->dtype = xm(dl + 1); memcpy(t->dtype, q, dl); t->dtype[dl] = 0; q += dl;
            if (memchr(t->dtype, 0, dl) || !utf8_ok(t->dtype, dl)) die("bad dtype");
            BND(q, 1); t->nd = r8(&q);
            if (t->nd > 64) die("bad nd");
            BND(q, 8 * t->nd);
            for (int d = 0; d < t->nd; d++) t->shape[d] = (int64_t)r64(&q);
            BND(q, 2); t->file = r16(&q);
            if (t->file >= (int)nf) die("bad file idx");
            ck_dupname(dseen, t->name, t->file, i, dcap);
            BND(q, 1); int mraw = r8(&q);
            if ((mraw & MF_BH) && !(mraw & MF_BAT)) die("bad method flags");
            t->bat = (mraw & MF_BAT) != 0 ? ((mraw & MF_BH) ? 1 : 2) : 0;
            t->method = mraw & 0x3F;
            if (t->method > M_PRW) die("bad method");
            BND(q, 8); t->len = r64(&q);
            ck_shape_len(t);
            t->ref = 0xFFFFFFFFu;
            if (t->method == M_REF || t->method == M_DELTA) {
                BND(q, 4); t->ref = r32(&q);
                if (t->ref >= i) die("bad ref");
                if (t->bat == 2 && t->ref >= bstart) die("ref into open batch");
                if (t->method == M_DELTA && strcmp(all[t->ref].dtype, t->dtype))
                    die("bad delta ref");
            } else if (t->method == M_DELTAX) { BND(q, 4); t->ref = r32(&q); }
            BND(q, 8); t->plen = r64(&q);
            BND(q, 4); t->crc = r32(&q);
            t->off = q - buf;              /* payload offset */
            BND(q, t->plen); q += t->plen;
            if (t->bat == 1) bstart = i;
            else if (!t->bat) bstart = i + 1;
        }
        if (q != bend) die("trailing data after last record");
        /* per-file record order: single pass buckets — an nf*NT scan would
           be quadratic on crafted archives */
        uint32_t *head = xc(nf ? nf : 1, 4), *tail = xc(nf ? nf : 1, 4),
                 *nxt = xc(NT ? NT : 1, 4);
        for (uint32_t i = 0; i < nf; i++) head[i] = tail[i] = 0xFFFFFFFFu;
        for (uint32_t k = 0; k < NT; k++) {
            uint32_t f = (uint32_t)all[k].file; nxt[k] = 0xFFFFFFFFu;
            if (head[f] == 0xFFFFFFFFu) head[f] = k; else nxt[tail[f]] = k;
            tail[f] = k;
        }
        /* per-file offsets + write headers + open outfiles */
        uint64_t *foff = xc(nf, 8), *hbase = xc(nf, 8);
        FILE **ofs = xc(nf, sizeof(FILE *));
        char tmp[4096];
        g_outp = xc(nf, sizeof(char *)); g_outn = nf; g_outok = 0;
        atexit(out_cleanup);
        for (uint32_t i = 0; i < nf; i++) {
            size_t hcap = 1 << 20; char *j = xm(hcap); size_t jl = 0;
            jput(j, &jl, hcap, "{");
            int first = 1;
            if (fmeta[i]) {   /* CAI5: replay __metadata__ verbatim */
                while (jl + strlen(fmeta[i]) + 64 > hcap) {
                    if (hcap >= (1u << 30)) die("header too large");   /* keep every
                        snprintf under INT_MAX — an EOVERFLOW return of -1 would
                        silently misplace all following bytes */
                    hcap *= 2; j = realloc(j, hcap); if (!j) die("oom");
                }
                jput(j, &jl, hcap, "\"__metadata__\":%s", fmeta[i]);
                first = 0;
            }
            for (uint32_t k = head[i]; k != 0xFFFFFFFFu; k = nxt[k]) {
                Tensor *t = &all[k];
                /* whole-record bound: 6x escape growth + dims + fixed fields;
                   must cover EVERYTHING the snprintf calls below emit, since a
                   truncated snprintf returns its full would-be length and jl
                   would overflow hcap on the next call. */
                size_t need = 7 * (strlen(t->name) + strlen(t->dtype)) +
                              24 * ((size_t)t->nd + 1) + 128;
                while (jl + need > hcap) {
                    if (hcap >= (1u << 30)) die("header too large");
                    hcap *= 2; j = realloc(j, hcap); if (!j) die("oom");
                }
                /* off_t is signed: data_offset + header base must stay <
                   INT64_MAX — leave 2^40 headroom for the JSON header */
                if (t->len > (uint64_t)INT64_MAX - foff[i] - (1ull << 40))
                    die("size overflow");
                t->dataoff = foff[i];
                char *en = xm(6 * strlen(t->name) + 1), *ed = xm(6 * strlen(t->dtype) + 1);
                jesc(en, 6 * strlen(t->name) + 1, t->name);
                jesc(ed, 6 * strlen(t->dtype) + 1, t->dtype);
                jput(j, &jl, hcap, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                     first ? "" : ",", en, ed);
                free(en); free(ed);
                for (int d = 0; d < t->nd; d++) jput(j, &jl, hcap, "%s%lld", d ? "," : "", (long long)t->shape[d]);
                jput(j, &jl, hcap, "],\"data_offsets\":[%llu,%llu]}",
                     (unsigned long long)foff[i], (unsigned long long)(foff[i] + t->len));
                foff[i] += t->len;
                first = 0;
            }
            jput(j, &jl, hcap, "}");
            /* dataoff keeps 2^40 of headroom under INT64_MAX for exactly
               this header — enforce it so hbase+dataoff stays a valid
               positive off_t at every fseeko */
            if (jl > (1ull << 40) - 8) die("regenerated header too large");
            /* write to .caiwtmp then rename: a crash mid-run must never
               leave a complete-looking truncated .st behind */
            int wr = snprintf(tmp, sizeof tmp, "%s/%s.caiwtmp", argv[3], fnames[i]);
            if (wr < 0 || (size_t)wr >= sizeof tmp) die("output path too long");
            g_outp[i] = xstrdup(tmp);
            ofs[i] = xfopen_tmp(tmp);
            if (!ofs[i]) die("out file");
            w64(ofs[i], jl); fwrite(j, 1, jl, ofs[i]);
            hbase[i] = 8 + jl;
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
                dec_batch(all, i, j, buf, keep);
                for (uint32_t k = i; k < j; k++) {
                    Tensor *t = &all[k];
                    /* streamed RAW members never materialized t->data —
                       the payload IS the tensor bytes */
                    const uint8_t *wd = t->data ? t->data : buf + t->off;
                    FILE *of = ofs[t->file];
                    if (fseeko(of, hbase[t->file] + t->dataoff, SEEK_SET)) die("seek");
                    if (t->len && fwrite(wd, 1, t->len, of) != t->len) die("write");
                    drop_pages(buf + t->off, t->plen, g_amap);
                    if (!keep[k] && t->data) {
                        __sync_sub_and_fetch(&g_dlive, t->dlivc);
                        free(t->data); t->data = 0;
                    }
                }
                i = j; continue;
            }
            Tensor *t = &all[i];
            if (t->method == M_RAW && !keep[i]) {
                /* zero-copy: payload bytes are the tensor — no alloc */
                if (t->plen != t->len) die("corrupt raw");
                const uint8_t *pd = buf + t->off;
                if (crc32b(pd, t->len) != t->crc) die("crc mismatch: archive corrupt");
                teach(t, pd);
                FILE *of = ofs[t->file];
                if (fseeko(of, hbase[t->file] + t->dataoff, SEEK_SET)) die("seek");
                if (t->len && fwrite(pd, 1, t->len, of) != t->len) die("write");
                drop_pages(pd, t->plen, g_amap);
                i++; continue;
            }
            dec_tensor(t, buf + t->off, all);
            if (crc32b(t->data, t->len) != t->crc) die("crc mismatch: archive corrupt");
            FILE *of = ofs[t->file];
            if (fseeko(of, hbase[t->file] + t->dataoff, SEEK_SET)) die("seek");
            if (t->len && fwrite(t->data, 1, t->len, of) != t->len) die("write");
            drop_pages(buf + t->off, t->plen, g_amap);
            if (!keep[i] && t->data) {
                __sync_sub_and_fetch(&g_dlive, t->dlivc);
                free(t->data); t->data = 0;
            }
            i++;
        }
        for (uint32_t i = 0; i < nf; i++)
            if (ferror(ofs[i]) || fclose(ofs[i])) die("write");
        for (uint32_t i = 0; i < nf; i++) {
            char fin[4096];
            int wr = snprintf(fin, sizeof fin, "%s/%s", argv[3], fnames[i]);
            if (wr < 0 || (size_t)wr >= sizeof fin || rename(g_outp[i], fin))
                die("rename failed");
            /* track the final name too: a later failure unlinks it, so the
               output set stays all-or-nothing. strdup FIRST — if it dies,
               g_outp[i] must not be left dangling for out_cleanup's unlink */
            { char *n2 = xstrdup(fin); free(g_outp[i]); g_outp[i] = n2; }
            fprintf(stderr, "wrote %s\n", fin);
        }
        g_outok = 1;
        return 0;
    }
    if (!strcmp(argv[1], "v")) {
        /* verify: decode archive in-memory, memcmp each tensor vs source files */
        if (argc < 4) die("caiw v out.caiw in.st...");
        dlim_init();
        uint64_t fsz; uint8_t *buf = slurp(argv[2], &fsz);
        int nf = argc - 3;
        if (nf > 65535) die("too many inputs");   /* positional match to u16 file idx */
        InFile **ins = xc(nf, sizeof(InFile *));
        for (int i = 0; i < nf; i++) ins[i] = st_load(argv[3 + i], i);
        NMap *nmap = xc(nf, sizeof(NMap));
        for (int i = 0; i < nf; i++) nm_build(&nmap[i], ins[i]->t, ins[i]->n);
        const uint8_t *p = buf, *bend = buf + fsz;
        BND(p, 4);
        int cver;
        if (!memcmp(p, "CAI5", 4)) cver = 5;
        else if (!memcmp(p, "CAI4", 4)) cver = 4;
        else if (!memcmp(p, "CAI3", 4)) cver = 3;
        else die("bad magic");
        p += 4;
        BND(p, 4); uint32_t nfa = r32(&p);
        if (nfa > 65535) die("too many files");
        BND(p, (uint64_t)nfa * 4);
        char **fnames = xc(nfa ? nfa : 1, sizeof(char *)), **fmeta = xc(nfa ? nfa : 1, sizeof(char *));
        uint64_t fcap = 64; while (fcap < (uint64_t)nfa * 2) fcap *= 2;
        char **fset = xc(fcap, sizeof(char *));
        for (uint32_t i = 0; i < nfa; i++) {
            BND(p, 4); uint32_t l = r32(&p); BND(p, l);
            fnames[i] = xm(l + 1); memcpy(fnames[i], p, l); fnames[i][l] = 0; p += l;
            if (memchr(fnames[i], 0, l)) die("bad filename");
            if (nf_find(fset, fcap - 1, fnames[i], 0)) die("dup filename");
            nf_put(fset, fcap - 1, fnames[i], 0);
            if (cver >= 5) {
                BND(p, 4); uint32_t ml = r32(&p); BND(p, ml);
                if (ml) { fmeta[i] = xm(ml + 1); memcpy(fmeta[i], p, ml); fmeta[i][ml] = 0; ck_meta((const uint8_t *)fmeta[i], ml); }
                p += ml;
            }
        }
        /* completeness: file set must match positionally (archives store
           positional file indices) */
        uint64_t bad = 0;
        if ((int)nfa != nf) { fprintf(stderr, "FILECOUNT archive=%u source=%d\n", nfa, nf); bad++; }
        for (uint32_t i = 0; i < nfa && i < (uint32_t)nf; i++) {
            if (strcmp(fnames[i], bname(ins[i]->path))) {
                fprintf(stderr, "FILE %u archive=%s source=%s\n", i, fnames[i], bname(ins[i]->path));
                bad++;
            }
            const char *am = fmeta[i], *sm = ins[i]->meta;
            if ((am != 0) != (sm != 0) || (am && strcmp(am, sm))) {
                fprintf(stderr, "META %s\n", fnames[i]);
                bad++;
            }
        }
        BND(p, 4); uint32_t NT = r32(&p);
        BND(p, (uint64_t)NT * 28);
        model_init();
        uint64_t dc64 = 64; while (dc64 < (uint64_t)NT * 2) dc64 *= 2;
        if (dc64 > (1ull << 30)) die("too many tensors");
        uint32_t dcap = (uint32_t)dc64;
        uint64_t meta = (uint64_t)NT * (sizeof(Tensor) + 512) + (uint64_t)dcap * 16;
        if (__sync_add_and_fetch(&g_dlive, meta) > g_dlim)
            die("archive metadata exceeds memory");
        Tensor *all = xc(NT, sizeof(Tensor));
        char **dseen = xc(dcap, sizeof(char *));
        /* pass 1: metadata + payload offsets, keep flags */
        const uint8_t *q = p;
        uint32_t bstart = 0;   /* head index of currently open batch */
        for (uint32_t i = 0; i < NT; i++) {
            Tensor *t = &all[i];
            uint16_t nl, dl;
            BND(q, 2); nl = r16(&q);
            BND(q, nl); t->name = xm(nl + 1); memcpy(t->name, q, nl); t->name[nl] = 0; q += nl;
            if (memchr(t->name, 0, nl) || !utf8_ok(t->name, nl) ||
                !strcmp(t->name, "__metadata__")) die("bad name");
            BND(q, 2); dl = r16(&q);
            BND(q, dl); t->dtype = xm(dl + 1); memcpy(t->dtype, q, dl); t->dtype[dl] = 0; q += dl;
            if (memchr(t->dtype, 0, dl) || !utf8_ok(t->dtype, dl)) die("bad dtype");
            BND(q, 1); t->nd = r8(&q);
            if (t->nd > 64) die("bad nd");
            BND(q, 8 * t->nd);
            for (int d = 0; d < t->nd; d++) t->shape[d] = (int64_t)r64(&q);
            BND(q, 2); t->file = r16(&q);
            if (t->file >= (int)nfa) die("bad file idx");
            ck_dupname(dseen, t->name, t->file, i, dcap);
            BND(q, 1); int mraw = r8(&q);
            if ((mraw & MF_BH) && !(mraw & MF_BAT)) die("bad method flags");
            t->bat = (mraw & MF_BAT) != 0 ? ((mraw & MF_BH) ? 1 : 2) : 0;
            t->method = mraw & 0x3F;
            if (t->method > M_PRW) die("bad method");
            BND(q, 8); t->len = r64(&q);
            ck_shape_len(t);
            t->ref = 0xFFFFFFFFu;
            if (t->method == M_REF || t->method == M_DELTA) {
                BND(q, 4); t->ref = r32(&q);
                if (t->ref >= i) die("bad ref");
                if (t->bat == 2 && t->ref >= bstart) die("ref into open batch");
                if (t->method == M_DELTA && strcmp(all[t->ref].dtype, t->dtype))
                    die("bad delta ref");
            } else if (t->method == M_DELTAX) { BND(q, 4); t->ref = r32(&q); }
            BND(q, 8); t->plen = r64(&q);
            BND(q, 4); t->crc = r32(&q);
            t->off = q - buf;
            BND(q, t->plen); q += t->plen;
            if (t->bat == 1) bstart = i;
            else if (!t->bat) bstart = i + 1;
        }
        if (q != bend) die("trailing data after last record");
        uint8_t *keep = xc(NT ? NT : 1, 1);
        for (uint32_t i = 0; i < NT; i++)
            if ((all[i].method == M_REF || all[i].method == M_DELTA) && all[i].ref < NT)
                keep[all[i].ref] = 1;
        uint64_t checked = 0;
        for (uint32_t i = 0; i < NT;) {
            uint32_t i0 = i, i1 = i + 1;
            if (all[i].bat) {
                if (all[i].bat != 1) die("orphan batch member");
                i1 = i + 1; while (i1 < NT && all[i1].bat == 2) i1++;
                dec_batch(all, i, i1, buf, keep);
            } else {
                Tensor *t = &all[i];
                if (t->method == M_RAW && !keep[i]) {
                    /* zero-copy: payload IS the tensor; compare direct */
                    if (t->plen != t->len) die("corrupt raw");
                    if (crc32b(buf + t->off, t->len) != t->crc)
                        die("crc mismatch: archive corrupt");
                    teach(t, buf + t->off);
                } else {
                    dec_tensor(t, buf + t->off, all);
                    if (crc32b(t->data, t->len) != t->crc) die("crc mismatch: archive corrupt");
                }
            }
            /* post: compare each decoded tensor against the source */
            for (uint32_t k = i0; k < i1; k++) {
                Tensor *t = &all[k];
                if (t->file >= nf) { fprintf(stderr, "BADFILE %s\n", t->name); bad++; continue; }
                InFile *src = ins[t->file];
                int ji = nm_get(&nmap[t->file], t->name);
                Tensor *st = ji >= 0 ? &src->t[ji] : 0;
                if (!st) { fprintf(stderr, "MISS %s\n", t->name); bad++; continue; }
                if (st->matched) { fprintf(stderr, "DUP %s\n", t->name); bad++; }
                st->matched = 1;
                checked++;
                if (strcmp(st->dtype, t->dtype) || st->nd != t->nd ||
                    memcmp(st->shape, t->shape, (size_t)t->nd * 8)) {
                    fprintf(stderr, "META %s\n", t->name);
                    bad++;
                }
                /* streamed RAW: t->data is NULL, payload holds the bytes */
                const uint8_t *dd = t->data ? t->data : buf + t->off;
                if (st->len != t->len || memcmp(st->data, dd, t->len)) {
                    fprintf(stderr, "DIFF %s method=%d\n", t->name, t->method);
                    bad++;
                }
                drop_pages(buf + t->off, t->plen, g_amap);
                drop_pages(st->data, st->len, src->mapped);
                if (!keep[k] && t->data) {
                    __sync_sub_and_fetch(&g_dlive, t->dlivc);
                    free(t->data); t->data = 0;
                }
            }
            i = i1;
        }
        /* reverse direction: every source tensor must appear in the archive */
        for (int i = 0; i < nf; i++)
            for (int j = 0; j < ins[i]->n; j++)
                if (!ins[i]->t[j].matched) {
                    fprintf(stderr, "MISS-SRC %s\n", ins[i]->t[j].name);
                    bad++;
                }
        fprintf(stderr, "verify: %llu tensors, %llu bad\n",
                (unsigned long long)checked, (unsigned long long)bad);
        return bad ? 1 : 0;
    }
    die("unknown cmd");
}
