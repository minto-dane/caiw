CC      ?= cc
CFLAGS  ?= -O3 -march=native -Wall -Wextra
LDLIBS  := -lm -pthread

caiw: caiw.c
	$(CC) $(CFLAGS) -pthread -o $@ $< $(LDLIBS)

test: caiw
	./test.sh

fuzz: caiw
	./fuzz.sh

verify:
	./verify/verify.sh

clean:
	rm -f caiw

.PHONY: test fuzz verify clean
