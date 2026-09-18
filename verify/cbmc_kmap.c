/* CBMC harness: kmap16 is a bijection — inv(kmap(u))==u over all u16.
 * Fully nondet scalar: complete domain coverage in one instance. */
#define main caiw_main_
#include "../caiw.c"
#undef main

int main() {
    uint16_t u;
    __CPROVER_assert(kmap16_inv(kmap16(u)) == u, "kmap16 inverse");
    return 0;
}
